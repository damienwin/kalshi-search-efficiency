"""Per-event losses that single out direction, for paired tests with ksearch.search.gate.dm_test.

  flip_penalized_loss   the gate's class-balanced log loss plus lam * P(opposite direction)
                        on true UP / DOWN events. lam = 0 is the gate's loss.
  direction_loss        on events that moved only: log loss of P(UP) / (P(UP) + P(DOWN))
                        against the true direction. Ignores the FLAT probability, so it
                        isolates the UP-vs-DOWN call.
"""

import numpy as np

from ksearch.eval.cv import LABELS
from ksearch.model.direction import OPPOSITE
from ksearch.search.gate import EPS, _check, balanced_weights


def flip_penalized_loss(y, proba, lam: float, eps: float = EPS) -> np.ndarray:
    y, proba = _check(y, proba)
    yi = np.searchsorted(LABELS, y.astype(str))
    rows = np.arange(len(y))
    opp = OPPOSITE[yi]
    p_opp = np.where(opp >= 0, proba[rows, np.maximum(opp, 0)], 0.0)
    return balanced_weights(y) * (-np.log(np.clip(proba[rows, yi], eps, 1.0)) + lam * p_opp)


def direction_loss(y, proba, eps: float = EPS) -> tuple[np.ndarray, np.ndarray]:
    """(loss on each moved event, boolean mask of the moved events)."""
    y, proba = _check(y, proba)
    moved = y != "FLAT"
    p_move = np.maximum(proba[moved, 0] + proba[moved, 2], eps)
    p_true = np.where(y[moved] == "UP", proba[moved, 2], proba[moved, 0]) / p_move
    return -np.log(np.clip(p_true, eps, 1.0)), moved
