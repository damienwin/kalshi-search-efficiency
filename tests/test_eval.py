"""Leakage tests L8-L11, L13, plus inference sanity checks."""

import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ksearch.data.manifest import Manifest, REPO_ROOT, verify
from ksearch.eval.cv import clustered_paired_bootstrap, legacy_keep_rule, momentum
from ksearch.eval.folds import HOUR, FoldScoped, walk_forward_folds


def ladder_events(n_groups=60, strikes=5, seed=0) -> pd.DataFrame:
    """Strike ladders: every market in a group shares t0s, like INX-24FEB23-*."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        start = 1_700_000_000 + g * 13 * HOUR + int(rng.integers(0, 4)) * 4 * HOUR
        for s in range(strikes):
            for j in range(int(rng.integers(2, 8))):
                t0 = start + j * 4 * HOUR
                rows.append({"market_ticker": f"G{g}-T{s}", "event_group": f"G{g}", "t0": t0,
                             "t_end": t0 + 4 * HOUR, "label": rng.choice(["DOWN", "FLAT", "UP"]),
                             "prev_label": np.nan})
    return pd.DataFrame(rows).sort_values(["t0", "market_ticker"], kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="module")
def ev():
    return ladder_events()


def test_L9_no_event_group_on_both_sides(ev):
    for f in walk_forward_folds(ev, 5, 4):
        assert not set(ev.event_group.iloc[f.train]) & set(ev.event_group.iloc[f.val])


def test_L10_purge_and_embargo(ev):
    folds = walk_forward_folds(ev, 5, embargo_h=4)
    assert len(folds) == 5
    for f in folds:
        assert ev.t_end.iloc[f.train].max() + 4 * HOUR <= ev.t0.iloc[f.val].min()


def test_folds_are_walk_forward(ev):
    folds = walk_forward_folds(ev, 5, 4)
    starts = [ev.t0.iloc[f.val].min() for f in folds]
    assert starts == sorted(starts)
    assert all(ev.t0.iloc[f.train].max() < ev.t0.iloc[f.val].min() for f in folds)


def test_L8_fold_scoped_transform_refuses_out_of_fold_rows(ev):
    f = walk_forward_folds(ev, 5, 4)[2]
    X = pd.DataFrame({"x": np.arange(len(ev), dtype=float)})
    FoldScoped(StandardScaler(), f.train).fit(X.iloc[f.train])  # fine
    with pytest.raises(ValueError, match="outside the training fold"):
        FoldScoped(StandardScaler(), f.train).fit(X.iloc[np.r_[f.train, f.val[:1]]])


def test_L11_run_cv_refuses_sealed_data(tmp_path):
    sealed = os.path.join(REPO_ROOT, "sealed", "sealed.parquet")
    r = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "run_cv.py"), "--data", sealed],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "refuses sealed data" in (r.stderr + r.stdout)


def test_L13_manifest_detects_changed_raw_file(tmp_path, monkeypatch):
    import ksearch.data.manifest as man
    monkeypatch.setattr(man, "REPO_ROOT", str(tmp_path))
    raw = tmp_path / "data" / "raw" / "x.json.gz"
    raw.parent.mkdir(parents=True)
    with gzip.open(raw, "wt") as f:
        json.dump({"candlesticks": []}, f)
    m = Manifest(str(tmp_path / "data" / "manifests" / "k.jsonl"))
    m.record(str(raw), "/endpoint", {"a": 1})
    assert man.verify([m.path]) == []
    with gzip.open(raw, "wt") as f:
        json.dump({"candlesticks": [1]}, f)
    assert man.verify([m.path]) == ["hash mismatch: data/raw/x.json.gz"]
    raw.unlink()
    assert man.verify([m.path]) == ["missing: data/raw/x.json.gz"]


def test_momentum_uses_only_resolved_prev_label():
    train = pd.DataFrame({"label": ["FLAT", "FLAT", "UP"]})
    val = pd.DataFrame({"prev_label": [1.0, -1.0, np.nan]})
    assert list(momentum(train, val)) == ["UP", "DOWN", "FLAT"]


def test_bootstrap_ci_brackets_zero_for_identical_predictors(ev):
    y = ev.label.to_numpy()
    r = clustered_paired_bootstrap(y, y, y, ev.event_group, n_boot=200)
    assert r["delta"] == 0 and r["ci95"] == [0.0, 0.0] and not r["excludes_zero"]


def test_bootstrap_detects_a_real_difference(ev):
    y = ev.label.to_numpy()
    bad = np.where(np.arange(len(y)) % 2, y, "FLAT")
    r = clustered_paired_bootstrap(y, y, bad, ev.event_group, n_boot=200)
    assert r["delta"] > 0 and r["excludes_zero"]


def test_legacy_keep_rule_matches_spring_semantics():
    assert legacy_keep_rule([.40] * 5, [.39] * 5, 0.005)["verdict"] == "KEPT"
    assert legacy_keep_rule([.40, .40, .40, .38, .38], [.39] * 5, 0.005)["verdict"] == "NEUTRAL"
    assert legacy_keep_rule([.38] * 5, [.39] * 5, 0.005)["verdict"] == "REGRESSED"


def test_seal_split_is_disjoint_in_time_and_groups(ev):
    from ksearch.eval.folds import seal_split
    dev, sealed, purged = seal_split(ev, 0.15, embargo_h=4)
    assert not set(dev.event_group) & set(sealed.event_group)
    assert dev.t_end.max() + 4 * HOUR <= sealed.t0.min()
    assert len(dev) + len(sealed) + purged == len(ev)
    assert 0.08 < len(sealed) / len(ev) < 0.25
