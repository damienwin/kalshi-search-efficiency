"""Class-balance sweeps: fixed-size training subsamples at a chosen class composition,
test-side importance weights, and weighted metrics for comparing models across compositions.

A composition is (P(DOWN), P(FLAT), P(UP)) in LABELS order, usually built from a FLAT
share f and an UP share of moves u.

Training side. Within each walk-forward fold, the model trains on a subsample of the
fold's training events with exact class counts and a size n_k that is the same at every
composition, so composition is not confounded with sample size. Each class's training
indices are shuffled once per (replicate, fold) and every composition takes a prefix of
each class's order (common random numbers: neighbouring compositions share most events,
so differences between them are not swamped by which events were drawn). Only training-
fold labels are read, and the subsample is a subset of the fold's purged training rows,
so L8-L10 hold as for the full fold.

Test side. Validation events are never dropped. Each gets an importance weight
target[c] / share_k[c] for its class c and fold k, so the weighted class shares of every
fold equal the target (pure label shift: P(x | y) fixed, P(y) changed) and weights average
1 within a fold. Every metric here takes those weights.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from ksearch.eval.cv import LABELS

DOWN, FLAT, UP = range(3)
EPS = 1e-6  # same log loss cap as the gate


# ── compositions ─────────────────────────────────────────────────────────────
def composition(flat_share: float, up_share: float) -> np.ndarray:
    m = 1.0 - flat_share
    return np.array([m * (1.0 - up_share), flat_share, m * up_share])


def label_index(y) -> np.ndarray:
    return np.searchsorted(LABELS, np.asarray(y).astype(str))  # LABELS is sorted


def class_counts(n: int, comp) -> np.ndarray:
    """Integer counts summing to n, closest to n * comp (largest remainder)."""
    raw = n * np.asarray(comp, dtype=float)
    out = np.floor(raw).astype(int)
    out[np.argsort(-(raw - out), kind="stable")[: n - out.sum()]] += 1
    return out


def fixed_sizes(fold_counts, comps, frac: float = 0.8) -> list[int]:
    """Per-fold training size feasible at every composition in `comps`, times `frac`.

    fold_counts: per fold, available (DOWN, FLAT, UP) training counts. frac < 1 keeps
    replicates from drawing every event of the scarcest class at the extreme composition."""
    sizes = []
    for avail in fold_counts:
        avail = np.asarray(avail, dtype=float)
        cap = min(min(avail[j] / c[j] for j in range(3) if c[j] > 0) for c in map(np.asarray, comps))
        sizes.append(int(np.floor(frac * cap)))
    return sizes


def class_orders(yi: np.ndarray, idx: np.ndarray, rng) -> list[np.ndarray]:
    """A random order of `idx` within each class (indices into the events frame)."""
    return [rng.permutation(idx[yi[idx] == j]) for j in range(3)]


def subsample(orders: list[np.ndarray], n: int, comp) -> np.ndarray:
    """Prefix of each class order with exact counts class_counts(n, comp), sorted."""
    counts = class_counts(n, comp)
    for j in range(3):
        if counts[j] > len(orders[j]):
            raise ValueError(f"need {counts[j]} {LABELS[j]} events, fold has {len(orders[j])}")
    return np.sort(np.concatenate([orders[j][: counts[j]] for j in range(3)]))


def target_weights(yi: np.ndarray, fold: np.ndarray, target) -> np.ndarray:
    """Importance weight per scored event: target[c] / share of c in its fold."""
    target = np.asarray(target, dtype=float)
    w = np.empty(len(yi))
    for k in np.unique(fold):
        m = fold == k
        share = np.bincount(yi[m], minlength=3) / m.sum()
        if ((share == 0) & (target > 0)).any():
            raise ValueError(f"fold {k} has no events of a class the target needs")
        w[m] = np.divide(target, share, out=np.zeros(3), where=share > 0)[yi[m]]
    return w


# ── probabilities and calls ──────────────────────────────────────────────────
def effective_prior(train_comp, kind: str, weights: str) -> np.ndarray:
    """Class rates a model's probabilities are calibrated to. Unweighted: the training
    composition. Balanced: uniform for the 3-class model; for two-stage, each binary stage
    is balanced, so P(move) = 1/2 and P(UP | move) = 1/2."""
    if weights == "none":
        return np.asarray(train_comp, dtype=float)
    return np.array([1 / 3] * 3) if kind == "3class" else np.array([0.25, 0.5, 0.25])


def prior_shift(proba: np.ndarray, from_prior, to_prior) -> np.ndarray:
    """Known-prior label-shift correction (Saerens et al. 2002): p'_c ∝ p_c * to_c / from_c."""
    p = proba * (np.asarray(to_prior) / np.asarray(from_prior))
    return p / p.sum(-1, keepdims=True)


def calls(proba: np.ndarray, prior=None) -> np.ndarray:
    """argmax of proba / prior (class indices). prior=None is the plain argmax; the
    model's effective prior gives the call that weighs every class equally."""
    return (proba if prior is None else proba / np.asarray(prior)).argmax(-1)


# ── weighted metrics ─────────────────────────────────────────────────────────
def per_event_losses(yi: np.ndarray, proba: np.ndarray) -> dict[str, np.ndarray]:
    """Additive per-event losses (lower is better). dir_logloss is NaN on FLAT events."""
    rows = np.arange(len(yi))
    onehot = np.eye(3)[yi]
    moved = yi != FLAT
    p_move = np.maximum(proba[:, DOWN] + proba[:, UP], EPS)
    p_dir = np.where(yi == UP, proba[:, UP], proba[:, DOWN]) / p_move
    return {
        "logloss": -np.log(np.clip(proba[rows, yi], EPS, 1.0)),
        "brier": ((proba - onehot) ** 2).sum(1),
        "dir_logloss": np.where(moved, -np.log(np.clip(p_dir, EPS, 1.0)), np.nan),
    }


def _wauc(y_bin, score, w) -> float:
    if w[y_bin].sum() == 0 or w[~y_bin].sum() == 0:
        return float("nan")
    return float(roc_auc_score(y_bin, score, sample_weight=w))


def confusion(yi, ci, w) -> np.ndarray:
    cm = np.zeros((3, 3))
    np.add.at(cm, (yi, ci), w)
    return cm


def confusion_metrics(cm: np.ndarray) -> dict:
    """From a (weighted) 3x3 confusion matrix [truth, call]; works on (..., 3, 3)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    truth, called, n = cm.sum(-1), cm.sum(-2), cm.sum((-2, -1))
    denom = truth + called
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    recall = np.divide(tp, truth, out=np.zeros_like(tp), where=truth > 0)
    # Gorodkin's multiclass MCC
    c, s = tp.sum(-1), n
    num = c * s - (called * truth).sum(-1)
    den = np.sqrt((s ** 2 - (called ** 2).sum(-1)) * (s ** 2 - (truth ** 2).sum(-1)))
    right = cm[..., DOWN, DOWN] + cm[..., UP, UP]
    flips = cm[..., DOWN, UP] + cm[..., UP, DOWN]
    moves = truth[..., DOWN] + truth[..., UP]
    return {
        "macro_f1": f1.mean(-1),
        "balanced_acc": recall.mean(-1),
        "mcc": np.divide(num, den, out=np.zeros_like(num), where=den > 0),
        "flips_per_call": np.divide(flips, flips + right, out=np.full_like(flips, np.nan),
                                    where=(flips + right) > 0),
        "flip_rate": np.divide(flips, moves, out=np.full_like(flips, np.nan), where=moves > 0),
        "missed_rate": np.divide(cm[..., DOWN, FLAT] + cm[..., UP, FLAT], moves,
                                 out=np.full_like(flips, np.nan), where=moves > 0),
        "false_move_rate": np.divide(cm[..., FLAT, DOWN] + cm[..., FLAT, UP], truth[..., FLAT],
                                     out=np.full_like(flips, np.nan), where=truth[..., FLAT] > 0),
    }


def weighted_metrics(yi: np.ndarray, proba: np.ndarray, w: np.ndarray, ci: np.ndarray) -> dict:
    """All sweep metrics for one model on one weighted test set.

    Skill scores are against predicting the weighted class shares themselves (the best
    constant forecast at that composition), so they are comparable across compositions:
    ll_skill = 1 - logloss / H(shares), brier_skill = 1 - brier / (1 - sum shares^2).
    balanced_logloss is the mean over classes of the class-mean log loss (the gate's loss),
    which test weights do not change."""
    loss = per_event_losses(yi, proba)
    shares = np.bincount(yi, weights=w, minlength=3) / w.sum()
    moved = yi != FLAT
    out = {
        "logloss": float(np.average(loss["logloss"], weights=w)),
        "brier": float(np.average(loss["brier"], weights=w)),
        "dir_logloss": float(np.average(loss["dir_logloss"][moved], weights=w[moved])),
        "balanced_logloss": float(np.mean([loss["logloss"][yi == j].mean()
                                           for j in range(3) if (yi == j).any()])),
    }
    h = -(shares[shares > 0] * np.log(shares[shares > 0])).sum()
    out["ll_skill"] = 1 - out["logloss"] / h
    out["brier_skill"] = 1 - out["brier"] / (1 - (shares ** 2).sum())
    p_move = proba[:, DOWN] + proba[:, UP]
    out["move_auc"] = _wauc(moved, p_move, w)
    p_up = proba[moved, UP] / np.maximum(p_move[moved], EPS)
    out["dir_auc"] = _wauc(yi[moved] == UP, p_up, w[moved])
    out["dir_acc"] = float(np.average((p_up > 0.5) == (yi[moved] == UP), weights=w[moved]))
    out.update({k: float(v) for k, v in confusion_metrics(confusion(yi, ci, w)).items()})
    return out


# ── inference helpers ────────────────────────────────────────────────────────
def group_multiplicities(groups: np.ndarray, n_boot: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(inverse group index per event, (n_boot, n_groups) bootstrap multiplicities)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    m = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
                  for _ in range(n_boot)]).astype(float)
    return inv, m


def boot_weighted_auc(y_bin: np.ndarray, score: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted ROC AUC for each row of event weights W (n_boot, n); ties count 1/2."""
    order = np.argsort(score, kind="mergesort")
    s, yb, W = score[order], y_bin[order], W[:, order]
    wn = np.where(yb, 0.0, W)
    wp = np.where(yb, W, 0.0)
    # negatives strictly below each event, plus half the negatives tied with it
    _, first, cnt = np.unique(s, return_index=True, return_counts=True)
    cum = np.concatenate([np.zeros((len(W), 1)), np.cumsum(wn, 1)], 1)
    start = np.repeat(first, cnt)
    end = start + np.repeat(cnt, cnt)
    below = cum[:, start] + 0.5 * (cum[:, end] - cum[:, start])
    num = (wp * below).sum(1)
    den = wp.sum(1) * wn.sum(1)
    return np.divide(num, den, out=np.full(len(W), np.nan), where=den > 0)


def ols_contrast(x) -> np.ndarray:
    """c with sum_f c_f * y_f = OLS slope of y on x."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    return xc / (xc ** 2).sum()
