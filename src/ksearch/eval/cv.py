"""Cross-validated model vs baselines, with event-clustered paired inference.

Out-of-fold predictions are pooled across folds. The headline comparison is a
paired bootstrap over event groups: resample groups with replacement, compute
macro F1 for both predictors on the same rows, take the difference. The
legacy keep rule (mean fold delta > threshold and >= 80% of folds won) is
reported alongside it for continuity with the Spring work.
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from ksearch.eval.folds import walk_forward_folds
from ksearch.model.direction import (flip_cost_decision, flip_penalty_proba, train_weights,
                                     two_stage_proba)

LABELS = ["DOWN", "FLAT", "UP"]
PREV_TO_LABEL = {-1.0: "DOWN", 0.0: "FLAT", 1.0: "UP"}


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


# ── baselines ────────────────────────────────────────────────────────────────
def majority(train: pd.DataFrame, val: pd.DataFrame, rng=None) -> np.ndarray:
    return np.full(len(val), train["label"].mode().iloc[0], dtype=object)


def no_change(train: pd.DataFrame, val: pd.DataFrame, rng=None) -> np.ndarray:
    return np.full(len(val), "FLAT", dtype=object)


def momentum(train: pd.DataFrame, val: pd.DataFrame, rng=None) -> np.ndarray:
    """Same-market previous label, only if it resolved by t0; else train majority."""
    fallback = train["label"].mode().iloc[0]
    return val["prev_label"].map(PREV_TO_LABEL).fillna(fallback).to_numpy(dtype=object)


def random_prior(train: pd.DataFrame, val: pd.DataFrame, rng=None) -> np.ndarray:
    """Random guess at the training fold's class rates (never the validation rates)."""
    prior = train["label"].value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
    return rng.choice(np.array(LABELS, dtype=object), size=len(val), p=prior.to_numpy())


BASELINES = {"majority": majority, "no_change": no_change, "momentum": momentum,
             "random": random_prior}


# ── model ────────────────────────────────────────────────────────────────────
def fit_xgb(train, features, params, weights: str = "balanced") -> XGBClassifier:
    y = train["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    clf = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                        learning_rate=params["learning_rate"], subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        min_child_weight=params["min_child_weight"], random_state=params["seed"],
                        verbosity=0)
    clf.fit(train[features], y, sample_weight=train_weights(y, weights))
    return clf


def predict_proba(clf, val, features) -> np.ndarray:
    proba = clf.predict_proba(val[features]).astype(float)
    return proba / proba.sum(axis=1, keepdims=True)  # float32 softmax drifts off 1


def fit_predict_xgb(train, val, features, params, model=None) -> tuple[np.ndarray, np.ndarray]:
    """model: None or {"kind": "3class"} is the default 3-class XGBoost; "two_stage" (with
    "stage2" overrides) and "flip_penalty" (with "lam") are in ksearch.model.direction.
    "flip_cost" > 1 replaces the argmax call with the expected-cost call. "weights":
    "balanced" (default) or "none", for every stage (see train_weights)."""
    model = model or {}
    kind = model.get("kind", "3class")
    weights = model.get("weights", "balanced")
    if kind == "3class":
        proba = predict_proba(fit_xgb(train, features, params, weights), val, features)
    elif kind == "two_stage":
        proba = two_stage_proba(train, val, features, params, model.get("stage2", {}), weights)
    elif kind == "flip_penalty":
        proba = flip_penalty_proba(train, val, features, params, model["lam"], weights)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    proba = proba / proba.sum(axis=1, keepdims=True)
    return flip_cost_decision(proba, model.get("flip_cost", 1.0)), proba


def direction_breakdown(y, pred, proba) -> dict:
    """Split the 3-class problem into "did it move" and "which way".

    move_auc: P(UP)+P(DOWN) ranking moves above FLAT. dir_auc / dir_acc: on true
    moves only, P(UP)/(P(UP)+P(DOWN)) for UP vs DOWN. Call counts: flips are moves
    called in the wrong direction; false moves are FLATs called UP or DOWN."""
    y, pred = np.asarray(y), np.asarray(pred)
    moved, called = y != "FLAT", pred != "FLAT"
    p_move = proba[:, 0] + proba[:, 2]
    p_up = proba[moved, 2] / np.maximum(p_move[moved], 1e-12)
    up = y[moved] == "UP"
    return {
        "move_auc": float(roc_auc_score(moved, p_move)),
        "dir_auc": float(roc_auc_score(up, p_up)),
        "dir_acc": float(((p_up > 0.5) == up).mean()),
        "n_moves": int(moved.sum()), "n_called_move": int(called.sum()),
        "right_direction": int((called & moved & (pred == y)).sum()),
        "flips": int((called & moved & (pred != y)).sum()),
        "false_moves": int((called & ~moved).sum()),
        "missed_moves": int((~called & moved).sum()),
    }


# ── inference ────────────────────────────────────────────────────────────────
def _group_confusions(y, pred, inv, n_groups) -> np.ndarray:
    """Per-group 3x3 confusion counts, shape (n_groups, 3, 3)."""
    li = {l: i for i, l in enumerate(LABELS)}
    yi = np.array([li[v] for v in y])
    pi = np.array([li[v] for v in pred])
    cm = np.zeros((n_groups, 3, 3))
    np.add.at(cm, (inv, yi, pi), 1)
    return cm


def _macro_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    """Macro F1 from confusion matrices (..., 3, 3), zero_division=0 like sklearn."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(-1) + cm.sum(-2)  # (tp+fn) + (tp+fp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(-1)


def clustered_paired_bootstrap(y, pred_a, pred_b, groups, n_boot=2000, seed=0) -> dict:
    """CI for macro_f1(a) - macro_f1(b), resampling event groups.

    A resample of groups is a vector of group multiplicities, so each replicate's
    confusion matrix is the multiplicity-weighted sum of per-group matrices.
    """
    y, pred_a, pred_b, groups = map(np.asarray, (y, pred_a, pred_b, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    cm_a = _group_confusions(y, pred_a, inv, len(uniq)).reshape(len(uniq), 9)
    cm_b = _group_confusions(y, pred_b, inv, len(uniq)).reshape(len(uniq), 9)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                  for _ in range(n_boot)]).astype(float)
    deltas = (_macro_f1_from_cm((w @ cm_a).reshape(-1, 3, 3))
              - _macro_f1_from_cm((w @ cm_b).reshape(-1, 3, 3)))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": macro_f1(y, pred_a) - macro_f1(y, pred_b),
            "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0),
            "n_groups": int(len(uniq)), "n_boot": n_boot}


def legacy_keep_rule(cand_folds, base_folds, threshold) -> dict:
    delta = float(np.mean(cand_folds) - np.mean(base_folds))
    won = int(sum(c > b for c, b in zip(cand_folds, base_folds)))
    required = math.ceil(0.8 * len(cand_folds))
    verdict = "KEPT" if delta > threshold and won >= required else (
        "REGRESSED" if delta < -threshold else "NEUTRAL")
    return {"verdict": verdict, "delta": delta, "folds_won": won, "required": required,
            "n_folds": len(cand_folds)}


def run_cv(events: pd.DataFrame, features: list[str], cv_cfg: dict, clf_params: dict,
           reference_features: list[str] | None = None, model: dict | None = None) -> dict:
    """reference_features: also fit a second XGBoost on these features ("reference")
    on the same folds, and test model vs reference with the same paired bootstrap.
    model: see fit_predict_xgb; applies to the model and the reference."""
    folds = walk_forward_folds(events, cv_cfg["n_splits"], cv_cfg["embargo_h"])
    refs = {"reference": reference_features} if reference_features else {}
    names = ["model"] + list(refs) + list(BASELINES)
    oof = {n: np.empty(len(events), dtype=object) for n in names}
    proba = np.full((len(events), 3), np.nan)
    in_val = np.zeros(len(events), dtype=bool)
    per_fold = []
    rng = np.random.default_rng(clf_params["seed"])  # random baseline draws, in fold order
    for f in folds:
        tr, va = events.iloc[f.train], events.iloc[f.val]
        preds = {"model": None}
        preds["model"], p = fit_predict_xgb(tr, va, features, clf_params, model)
        for n, feats in refs.items():
            preds[n], _ = fit_predict_xgb(tr, va, feats, clf_params, model)
        for n, fn in BASELINES.items():
            preds[n] = fn(tr, va, rng)
        for n in names:
            oof[n][f.val] = preds[n]
        proba[f.val] = p
        in_val[f.val] = True
        row = {"fold": f.k, "n_train": len(f.train), "n_val": len(f.val),
               "val_start": int(va.t0.min()), "val_end": int(va.t0.max())}
        row.update({f"f1_{n}": macro_f1(va.label, preds[n]) for n in names})
        row["logloss_model"] = float(log_loss(va.label, p, labels=LABELS))
        per_fold.append(row)

    ev = events[in_val]
    y, groups = ev["label"].to_numpy(), ev["event_group"].to_numpy()
    pooled = {n: macro_f1(y, oof[n][in_val]) for n in names}
    comparisons = {
        f"model_vs_{b}": clustered_paired_bootstrap(y, oof["model"][in_val], oof[b][in_val], groups,
                                                    cv_cfg["n_bootstrap"])
        for b in list(refs) + list(BASELINES)
    }
    pf = pd.DataFrame(per_fold)
    return {
        "features": features,
        "n_events_scored": int(in_val.sum()),
        "pooled_macro_f1": pooled,
        "pooled_logloss_model": float(log_loss(y, proba[in_val], labels=LABELS)),
        "direction_breakdown_model": direction_breakdown(y, oof["model"][in_val], proba[in_val]),
        "model_spec": model or {"kind": "3class"},
        "comparisons": comparisons,
        "legacy_keep_rule_vs_momentum": legacy_keep_rule(pf.f1_model, pf.f1_momentum,
                                                         cv_cfg["keep_threshold"]),
        "per_fold": per_fold,
        "label_dist_scored": ev["label"].value_counts().to_dict(),
    }
