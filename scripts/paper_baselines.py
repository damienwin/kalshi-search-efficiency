"""Baselines on the search's own folds and scored rows, for the paper's Setup section.

Usage: python scripts/paper_baselines.py [--out results/paper_baselines.json]

Same events, folds and scored rows as scripts/run_search.py (scope "all"), so the
numbers sit on the same footing as the base model's balanced log loss (0.858) and
pooled macro F1 (0.500) in runs/<arm>/seed<k>/run.json["initial_reference"].

Each baseline gives hard calls (for macro F1, as in ksearch.eval.cv.BASELINES) and
probabilities (for log loss), both fitted on the training fold only:

  uniform    probability 1/3 per class (log loss only; its calls would be the majority's)
  majority   calls the training majority; probabilities = training class rates
             ("rates") or, with the model's balanced class weights, 1/3 each ("balanced")
  random     calls drawn at the training class rates; same probabilities as majority
  momentum   calls the market's previous label if resolved by t0, else the majority;
             probabilities = training P(label | previous label), with missing previous
             label as its own level, m-estimate smoothed toward the class rates (m = 10).
             "balanced" weights each training event by n / (3 n_class), as the model does.

Balanced log loss is the gate's loss (ksearch.search.gate.balanced_log_loss).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import yaml  # noqa: E402

from ksearch.eval.cv import BASELINES, LABELS, PREV_TO_LABEL, macro_f1  # noqa: E402
from ksearch.search.gate import balanced_log_loss  # noqa: E402
from ksearch.search.loop import make_folds  # noqa: E402
from run_search import load_events  # noqa: E402

M_SMOOTH = 10.0


def class_rates(y, w) -> np.ndarray:
    tot = np.array([w[y == c].sum() for c in LABELS])
    return tot / tot.sum()


def weights_for(y, mode: str) -> np.ndarray:
    if mode == "rates":
        return np.ones(len(y))
    cls, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    return len(y) / (len(cls) * cnt[inv])


def momentum_proba(train, val, mode: str) -> np.ndarray:
    y = train["label"].to_numpy(dtype=object)
    w = weights_for(y, mode)
    prior = class_rates(y, w)
    prev_tr = train["prev_label"].map(PREV_TO_LABEL).fillna("NA").to_numpy(dtype=object)
    prev_va = val["prev_label"].map(PREV_TO_LABEL).fillna("NA").to_numpy(dtype=object)
    out = np.tile(prior, (len(val), 1))
    for level in np.unique(np.concatenate([prev_tr, prev_va])):
        m = prev_tr == level
        counts = np.array([w[m & (y == c)].sum() for c in LABELS])
        p = (counts + M_SMOOTH * prior) / (counts.sum() + M_SMOOTH)
        out[prev_va == level] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "paper_baselines.json"))
    args = ap.parse_args()

    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    events = load_events(args.data, "all")
    folds = make_folds(events, cfg)

    rng = np.random.default_rng(0)
    calls = {k: np.empty(len(events), dtype=object) for k in ("majority", "random", "momentum")}
    proba = {k: np.full((len(events), 3), np.nan) for k in
             ("uniform", "majority_rates", "momentum_rates", "momentum_balanced")}
    scored = np.zeros(len(events), dtype=bool)
    for f in folds:
        tr, va = events.iloc[f.train], events.iloc[f.val]
        scored[f.val] = True
        calls["majority"][f.val] = BASELINES["majority"](tr, va)
        calls["random"][f.val] = BASELINES["random"](tr, va, rng)
        calls["momentum"][f.val] = BASELINES["momentum"](tr, va)
        y_tr = tr["label"].to_numpy(dtype=object)
        proba["uniform"][f.val] = 1.0 / 3.0
        proba["majority_rates"][f.val] = class_rates(y_tr, np.ones(len(y_tr)))
        proba["momentum_rates"][f.val] = momentum_proba(tr, va, "rates")
        proba["momentum_balanced"][f.val] = momentum_proba(tr, va, "balanced")

    y = events["label"].to_numpy(dtype=object)[scored]
    out = {"n_scored": int(scored.sum()), "data": args.data, "m_smooth": M_SMOOTH,
           "macro_f1": {k: macro_f1(y, v[scored]) for k, v in calls.items()},
           "balanced_logloss": {k: float(balanced_log_loss(y, p[scored]).mean()) for k, p in proba.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
