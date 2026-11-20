"""Two-stage model, flip-penalized objective, and flip-cost decision (ksearch.model.direction)."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from ksearch.eval.cv import direction_breakdown, fit_predict_xgb
from ksearch.model.direction import (_softmax, flip_cost_decision, flip_penalty_objective,
                                     two_stage_proba)

PARAMS = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0,
          "colsample_bytree": 1.0, "min_child_weight": 1, "seed": 0}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=(n, 3))
    move = np.abs(x[:, 0]) > 1.0
    label = np.where(~move, "FLAT", np.where(x[:, 1] + 0.5 * rng.normal(size=n) > 0, "UP", "DOWN"))
    df = pd.DataFrame(x, columns=["a", "b", "c"]).assign(label=label)
    return df.iloc[:400].reset_index(drop=True), df.iloc[400:].reset_index(drop=True)


def _loss(z, y, lam):
    p = _softmax(z)
    opp = {0: 2, 1: None, 2: 0}[y]
    return -np.log(p[0, y]) + (lam * p[0, opp] if opp is not None else 0.0)


@pytest.mark.parametrize("y", [0, 1, 2])
def test_flip_penalty_gradient_matches_finite_difference(y):
    z = np.array([[0.3, -0.2, 0.5]])
    d = xgb.DMatrix(np.zeros((1, 1)), label=[y])
    grad, _ = flip_penalty_objective(2.0)(z.ravel(), d)
    eps = 1e-6
    for k in range(3):
        e = np.zeros_like(z)
        e[0, k] = eps
        num = (_loss(z + e, y, 2.0) - _loss(z - e, y, 2.0)) / (2 * eps)
        assert grad[0, k] == pytest.approx(num, abs=1e-6)


def test_flip_penalty_zero_is_softmax_log_loss():
    z = np.array([0.3, -0.2, 0.5])
    grad, hess = flip_penalty_objective(0.0)(z, xgb.DMatrix(np.zeros((1, 1)), label=[2]))
    p = _softmax(z[None])
    assert np.allclose(grad, p - np.eye(3)[2])
    assert np.allclose(hess, 2 * p * (1 - p))


def test_flip_penalty_moves_mass_off_the_opposite_direction(data):
    tr, va = data
    _, p0 = fit_predict_xgb(tr, va, ["a", "b", "c"], PARAMS, {"kind": "flip_penalty", "lam": 0.0})
    _, p5 = fit_predict_xgb(tr, va, ["a", "b", "c"], PARAMS, {"kind": "flip_penalty", "lam": 5.0})
    up, down = (va.label == "UP").to_numpy(), (va.label == "DOWN").to_numpy()
    assert p5[up, 0].mean() < p0[up, 0].mean() and p5[down, 2].mean() < p0[down, 2].mean()


def test_two_stage_is_a_distribution_and_ignores_val_labels(data):
    tr, va = data
    p = two_stage_proba(tr, va, ["a", "b", "c"], PARAMS, {"max_depth": 2})
    assert p.shape == (len(va), 3) and np.allclose(p.sum(1), 1) and (p >= 0).all()
    shuffled = va.assign(label=np.random.default_rng(1).permutation(va.label.to_numpy()))
    assert np.array_equal(p, two_stage_proba(tr, shuffled, ["a", "b", "c"], PARAMS, {"max_depth": 2}))


def test_two_stage_learns_both_parts(data):
    tr, va = data
    pred, p = fit_predict_xgb(tr, va, ["a", "b", "c"], PARAMS, {"kind": "two_stage", "stage2": {}})
    d = direction_breakdown(va.label, pred, p)
    assert d["move_auc"] > 0.9 and d["dir_auc"] > 0.8


def test_flip_cost_decision():
    p = np.array([[0.45, 0.10, 0.45 - 1e-9], [0.60, 0.30, 0.10], [0.2, 0.7, 0.1]])
    assert list(flip_cost_decision(p, 1.0)) == ["DOWN", "DOWN", "FLAT"]
    # cost 3, row 0: DOWN 0.10 + 3*0.45 > FLAT 0.90; row 1: DOWN 0.30 + 3*0.10 = 0.60 < FLAT 0.70
    assert list(flip_cost_decision(p, 3.0)) == ["FLAT", "DOWN", "FLAT"]
    assert np.array_equal(flip_cost_decision(p, 1.0), np.array(["DOWN", "FLAT", "UP"])[p.argmax(1)])


def test_direction_breakdown_counts():
    y = np.array(["UP", "UP", "DOWN", "FLAT", "FLAT"])
    pred = np.array(["UP", "DOWN", "FLAT", "UP", "FLAT"])
    p = np.array([[.1, .2, .7], [.6, .1, .3], [.3, .6, .1], [.1, .3, .6], [.1, .8, .1]])
    d = direction_breakdown(y, pred, p)
    assert (d["right_direction"], d["flips"], d["false_moves"], d["missed_moves"]) == (1, 1, 1, 1)
    assert d["n_moves"] == 3 and d["n_called_move"] == 3


def test_flip_penalized_loss_reduces_to_gate_loss_and_charges_only_directional_events():
    from ksearch.eval.direction_metrics import direction_loss, flip_penalized_loss
    from ksearch.search.gate import balanced_log_loss
    y = np.array(["UP", "DOWN", "FLAT", "UP"], dtype=object)
    p = np.array([[.2, .2, .6], [.5, .3, .2], [.1, .8, .1], [.3, .3, .4]])
    assert np.allclose(flip_penalized_loss(y, p, 0.0), balanced_log_loss(y, p))
    extra = flip_penalized_loss(y, p, 2.0) - flip_penalized_loss(y, p, 0.0)
    w = len(y) / (3 * np.array([2, 1, 1, 2]))
    assert np.allclose(extra, w * 2.0 * np.array([.2, .2, 0.0, .3]))  # P(opposite); FLAT pays nothing
    loss, moved = direction_loss(y, p)
    assert moved.tolist() == [True, True, False, True]
    assert np.allclose(loss, -np.log([.6 / .8, .5 / .7, .4 / .7]))


@pytest.mark.parametrize("kind", ["3class", "two_stage"])
def test_unweighted_training_follows_class_rates(data, kind):
    """Balanced weights inflate move probabilities; "none" keeps the mean P(FLAT) near
    the training FLAT rate. Default stays balanced, so older results reproduce."""
    tr, va = data
    feats = ["a", "b", "c"]
    _, p_default = fit_predict_xgb(tr, va, feats, PARAMS, {"kind": kind})
    _, p_bal = fit_predict_xgb(tr, va, feats, PARAMS, {"kind": kind, "weights": "balanced"})
    _, p_none = fit_predict_xgb(tr, va, feats, PARAMS, {"kind": kind, "weights": "none"})
    assert np.allclose(p_default, p_bal)
    flat_rate = (tr.label == "FLAT").mean()
    assert abs(p_none[:, 1].mean() - flat_rate) < abs(p_bal[:, 1].mean() - flat_rate)
    with pytest.raises(ValueError):
        fit_predict_xgb(tr, va, feats, PARAMS, {"kind": kind, "weights": "inverse"})
