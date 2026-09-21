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
from sklearn.metrics import f1_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ksearch.eval.folds import walk_forward_folds

LABELS = ["DOWN", "FLAT", "UP"]
PREV_TO_LABEL = {-1.0: "DOWN", 0.0: "FLAT", 1.0: "UP"}


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


# ── baselines ────────────────────────────────────────────────────────────────
def majority(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    return np.full(len(val), train["label"].mode().iloc[0], dtype=object)


def no_change(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    return np.full(len(val), "FLAT", dtype=object)


def momentum(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Same-market previous label, only if it resolved by t0; else train majority."""
    fallback = train["label"].mode().iloc[0]
    return val["prev_label"].map(PREV_TO_LABEL).fillna(fallback).to_numpy(dtype=object)


BASELINES = {"majority": majority, "no_change": no_change, "momentum": momentum}


# ── model ────────────────────────────────────────────────────────────────────
def fit_predict_xgb(train, val, features, params) -> tuple[np.ndarray, np.ndarray]:
    y = train["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    clf = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                        learning_rate=params["learning_rate"], subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        min_child_weight=params["min_child_weight"], random_state=params["seed"],
                        verbosity=0)
    clf.fit(train[features], y, sample_weight=compute_sample_weight("balanced", y))
    proba = clf.predict_proba(val[features]).astype(float)
    proba /= proba.sum(axis=1, keepdims=True)  # float32 softmax drifts off 1
    return np.array(LABELS, dtype=object)[proba.argmax(1)], proba


# ── inference ────────────────────────────────────────────────────────────────
def clustered_paired_bootstrap(y, pred_a, pred_b, groups, n_boot=2000, seed=0) -> dict:
    """CI for macro_f1(a) - macro_f1(b), resampling event groups."""
    y, pred_a, pred_b, groups = map(np.asarray, (y, pred_a, pred_b, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        deltas[b] = macro_f1(y[idx], pred_a[idx]) - macro_f1(y[idx], pred_b[idx])
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


def run_cv(events: pd.DataFrame, features: list[str], cv_cfg: dict, clf_params: dict) -> dict:
    folds = walk_forward_folds(events, cv_cfg["n_splits"], cv_cfg["embargo_h"])
    names = ["model"] + list(BASELINES)
    oof = {n: np.empty(len(events), dtype=object) for n in names}
    proba = np.full((len(events), 3), np.nan)
    in_val = np.zeros(len(events), dtype=bool)
    per_fold = []
    for f in folds:
        tr, va = events.iloc[f.train], events.iloc[f.val]
        preds = {"model": None}
        preds["model"], p = fit_predict_xgb(tr, va, features, clf_params)
        for n, fn in BASELINES.items():
            preds[n] = fn(tr, va)
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
        for b in BASELINES
    }
    pf = pd.DataFrame(per_fold)
    return {
        "features": features,
        "n_events_scored": int(in_val.sum()),
        "pooled_macro_f1": pooled,
        "pooled_logloss_model": float(log_loss(y, proba[in_val], labels=LABELS)),
        "comparisons": comparisons,
        "legacy_keep_rule_vs_momentum": legacy_keep_rule(pf.f1_model, pf.f1_momentum,
                                                         cv_cfg["keep_threshold"]),
        "per_fold": per_fold,
        "label_dist_scored": ev["label"].value_counts().to_dict(),
    }
