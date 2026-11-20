"""Alternatives to the 3-class XGBoost aimed at direction flips (UP called DOWN, or the reverse).

All return probabilities over LABELS = [DOWN, FLAT, UP], so CV, the
significance gate and the reports treat them like the 3-class model.

  two_stage      stage 1: MOVE vs FLAT on every training event; stage 2: UP vs
                 DOWN on training-fold moves only. Combined as
                 P(UP) = P(move) * P(UP | move), P(DOWN) = P(move) * (1 - P(UP | move)).
  flip_penalty   3-class XGBoost trained on log loss + lam * P(opposite direction):
                 a true UP also pays for probability put on DOWN, and vice versa.
                 lam = 0 is plain softmax log loss.
  flip_cost_decision
                 not a model: given probabilities, pick the call with the lowest
                 expected cost when a flip costs `flip_cost` and any other error 1.

Labels are read only from training rows (stage 2's filter included).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def train_weights(y, weights: str = "balanced"):
    """Training sample weights: "balanced" (sklearn: every class the same total weight) or
    "none" (each event 1, so probabilities follow the training class rates)."""
    if weights == "balanced":
        return compute_sample_weight("balanced", y)
    if weights == "none":
        return None
    raise ValueError(f"unknown weights {weights!r}")

LABELS = ["DOWN", "FLAT", "UP"]
DOWN, FLAT, UP = range(3)
OPPOSITE = np.array([UP, -1, DOWN])  # -1: FLAT has no opposite


def _binary(params: dict) -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                         n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                         learning_rate=params["learning_rate"], subsample=params["subsample"],
                         colsample_bytree=params["colsample_bytree"],
                         min_child_weight=params["min_child_weight"], random_state=params["seed"],
                         verbosity=0)


def two_stage_proba(train: pd.DataFrame, val: pd.DataFrame, features: list[str], params: dict,
                    stage2: dict | None = None, weights: str = "balanced") -> np.ndarray:
    """Both stages use `params` (stage2: optional overrides for the direction model only)
    and the same sample weighting as fit_xgb (balanced by default, "none" for unweighted)."""
    y = train["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    move = (y != FLAT).astype(int)
    s1 = _binary(params)
    s1.fit(train[features], move, sample_weight=train_weights(move, weights))
    p_move = s1.predict_proba(val[features])[:, 1].astype(float)

    moved = move == 1
    up = (y[moved] == UP).astype(int)
    s2 = _binary({**params, **(stage2 or {})})
    s2.fit(train.loc[moved, features], up, sample_weight=train_weights(up, weights))
    p_up = s2.predict_proba(val[features])[:, 1].astype(float)
    return np.c_[p_move * (1 - p_up), 1 - p_move, p_move * p_up]


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def flip_penalty_objective(lam: float):
    """Gradient and diagonal Hessian of  -log p_y + lam * p_opp(y)  in the softmax margins.

    For the log-loss part this matches XGBoost's multi:softprob (grad p - onehot,
    hess 2p(1-p)). For the penalty, d p_o / d z_k = p_o (1[k=o] - p_k); its second
    derivative can be negative, so it is clipped at 0 to keep Newton steps stable.
    """
    def obj(margins: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label().astype(int)
        w = dtrain.get_weight()
        w = np.ones(len(y)) if w.size == 0 else w
        p = _softmax(margins.reshape(len(y), 3))
        onehot = np.eye(3)[y]
        grad = p - onehot
        hess = np.maximum(2.0 * p * (1.0 - p), 1e-6)
        opp = OPPOSITE[y]
        has = opp >= 0
        if lam and has.any():
            po = np.zeros(len(y))
            po[has] = p[has, opp[has]]
            ind = np.zeros_like(p)
            ind[has, opp[has]] = 1.0
            d = ind - p
            grad = grad + lam * po[:, None] * d
            hess = hess + lam * np.maximum(po[:, None] * (d * d - p * (1.0 - p)), 0.0)
        return grad * w[:, None], hess * w[:, None]
    return obj


def flip_penalty_proba(train: pd.DataFrame, val: pd.DataFrame, features: list[str], params: dict,
                       lam: float, weights: str = "balanced") -> np.ndarray:
    """Same hyperparameters and weights as fit_xgb, custom objective."""
    y = train["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    dtrain = xgb.DMatrix(train[features], label=y, weight=train_weights(y, weights))
    booster = xgb.train(
        {"num_class": 3, "max_depth": params["max_depth"], "eta": params["learning_rate"],
         "subsample": params["subsample"], "colsample_bytree": params["colsample_bytree"],
         "min_child_weight": params["min_child_weight"], "seed": params["seed"],
         "disable_default_eval_metric": 1, "verbosity": 0},
        dtrain, num_boost_round=params["n_estimators"], obj=flip_penalty_objective(lam))
    return _softmax(booster.predict(xgb.DMatrix(val[features]), output_margin=True).reshape(-1, 3))


def flip_cost_decision(proba: np.ndarray, flip_cost: float) -> np.ndarray:
    """Call with the lowest expected cost: a flip costs flip_cost, any other error 1.
    flip_cost = 1 is the argmax."""
    cost = 1.0 - np.eye(3)  # cost[call, truth]
    cost[DOWN, UP] = cost[UP, DOWN] = flip_cost
    return np.array(LABELS, dtype=object)[(proba @ cost.T).argmin(1)]
