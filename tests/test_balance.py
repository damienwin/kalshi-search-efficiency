"""Class-balance sweep utilities (ksearch.eval.balance)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, log_loss, matthews_corrcoef, roc_auc_score

from ksearch.eval.balance import (boot_weighted_auc, class_counts, class_orders, composition,
                                  confusion, confusion_metrics, fixed_sizes, label_index,
                                  ols_contrast, prior_shift, subsample, target_weights,
                                  weighted_metrics)
from ksearch.eval.cv import LABELS
from ksearch.eval.folds import walk_forward_folds


def _events(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t0 = np.sort(rng.integers(0, 400, n)) * 4 * 3600
    label = rng.choice(LABELS, n, p=[0.07, 0.86, 0.07])
    return pd.DataFrame({"t0": t0, "t_end": t0 + 4 * 3600, "label": label,
                         "event_group": (t0 // (4 * 3600 * 8)).astype(str)})


@pytest.mark.parametrize("n,comp", [(100, [0.1, 0.8, 0.1]), (7, [0.3, 0.4, 0.3]), (999, composition(0.95, 0.2))])
def test_class_counts_sum_and_round(n, comp):
    c = class_counts(n, comp)
    assert c.sum() == n and np.all(np.abs(c - n * np.asarray(comp)) < 1)


def test_subsample_exact_fixed_size_nested_and_inside_training_fold():
    ev = _events()
    yi = label_index(ev.label)
    folds = walk_forward_folds(ev, 5, 4)
    comps = [composition(f, 0.5) for f in (0.5, 0.7, 0.9, 0.95)] + [composition(0.85, 0.2)]
    sizes = fixed_sizes([np.bincount(yi[f.train], minlength=3) for f in folds], comps)
    rng = np.random.default_rng(0)
    for f, n in zip(folds, sizes):
        orders = class_orders(yi, f.train, rng)
        subs = {}
        for c in comps:
            s = subsample(orders, n, c)
            assert len(s) == n and np.isin(s, f.train).all()  # never a validation or purged row
            assert (np.bincount(yi[s], minlength=3) == class_counts(n, c)).all()
            subs[tuple(c)] = s
        # common random numbers: the FLATs at 0.9 are a subset of the FLATs at 0.95
        a, b = subs[tuple(comps[2])], subs[tuple(comps[3])]
        assert set(a[yi[a] == 1]) <= set(b[yi[b] == 1])
    with pytest.raises(ValueError):
        subsample(orders, 10 * len(f.train), comps[0])


def test_target_weights_hit_target_in_every_fold():
    rng = np.random.default_rng(1)
    yi = rng.choice(3, 2000, p=[0.1, 0.8, 0.1])
    fold = rng.integers(1, 4, 2000)
    target = composition(0.5, 0.3)
    w = target_weights(yi, fold, target)
    for k in (1, 2, 3):
        m = fold == k
        assert np.allclose(np.bincount(yi[m], weights=w[m], minlength=3) / w[m].sum(), target)
        assert w[m].mean() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        target_weights(np.ones(10, int), np.ones(10, int), target)


def test_prior_shift_identity_and_direction():
    p = np.array([[0.2, 0.6, 0.2]])
    assert np.allclose(prior_shift(p, [0.3, 0.4, 0.3], [0.3, 0.4, 0.3]), p)
    assert prior_shift(p, [1 / 3] * 3, composition(0.9, 0.5))[0, 1] > 0.6


def test_metrics_match_sklearn():
    rng = np.random.default_rng(2)
    n = 1500
    yi = rng.choice(3, n, p=[0.15, 0.7, 0.15])
    proba = rng.dirichlet([1, 1, 1], n) * 0.5 + np.eye(3)[yi] * 0.5
    w = rng.uniform(0.2, 3, n)
    ci = proba.argmax(1)
    cm = confusion_metrics(confusion(yi, ci, w))
    assert cm["macro_f1"] == pytest.approx(f1_score(yi, ci, average="macro", sample_weight=w))
    assert cm["mcc"] == pytest.approx(matthews_corrcoef(yi, ci, sample_weight=w))
    m = weighted_metrics(yi, proba, np.ones(n), ci)
    assert m["logloss"] == pytest.approx(log_loss(yi, proba))
    moved = yi != 1
    assert m["move_auc"] == pytest.approx(roc_auc_score(moved, proba[:, 0] + proba[:, 2]))


def test_boot_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(3)
    y = rng.random(400) < 0.3
    s = np.round(rng.random(400) + 0.3 * y, 1)  # many ties
    W = rng.uniform(0, 2, (4, 400))
    got = boot_weighted_auc(y, s, W)
    for b in range(4):
        assert got[b] == pytest.approx(roc_auc_score(y, s, sample_weight=W[b]))


def test_ols_contrast_is_slope():
    x, y = np.array([0.5, 0.6, 0.7, 0.9]), np.array([1.0, 1.5, 1.7, 3.0])
    assert ols_contrast(x) @ y == pytest.approx(np.polyfit(x, y, 1)[0])
