"""Controlled class-balance sweep: 3-class vs two-stage across training compositions.

    OMP_NUM_THREADS=1 python scripts/run_balance_sweep.py [--reps 20] [--jobs 8]

Pool: the current-scope development events of the week 4 two-stage results (all
events, same walk-forward folds). Within every fold, each model trains on a subsample
of the fold's training events with a fixed size per fold and an exact composition
(ksearch.eval.balance), then predicts every validation event of the fold. Test-side
compositions are applied later, by importance weights, in analyze_balance_sweep.py.

Designs (training compositions):
  flat   FLAT share 0.50 ... 0.95, UP share of moves = the fold's own; plus "nat" (the
         fold's own composition at the fixed size) and "full" (the whole fold, no
         subsample: the week 4 setting)
  up     FLAT share 0.85, UP share of moves 0.20 ... 0.80

Every (design, level, replicate) uses one subsample for all models, weightings and
feature sets, and XGBoost seed = replicate, so comparisons are paired. Per-event
probabilities go to runs/balance_sweep/ (gitignored: dev labels and tickers stay local).
Refuses the sealed slice like run_cv.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker; parallelism is across tasks

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from ksearch.data.manifest import REPO_ROOT, sha256_file
from ksearch.eval.balance import class_orders, composition, fixed_sizes, label_index, subsample
from ksearch.eval.cv import fit_predict_xgb
from ksearch.eval.folds import walk_forward_folds
from ksearch.features.price import PRICE_FEATURES, PROVISIONAL_FEATURES
from ksearch.features.sentiment import SENTIMENT_FEATURES
from direction_significance import load_events
from run_cv import guard_not_sealed

FLAT_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
UP_LEVELS = [0.20, 0.35, 0.50, 0.65, 0.80]
UP_DESIGN_FLAT = 0.85
MODELS = ["3class", "two_stage"]
WEIGHTS = ["balanced", "none"]
SIZE_FRAC = 0.8


def levels() -> list[tuple[str, str]]:
    return ([("flat", f"{f:.2f}") for f in FLAT_LEVELS] + [("flat", "nat"), ("flat", "full")]
            + [("up", f"{u:.2f}") for u in UP_LEVELS])


def train_comp(design: str, level: str, natural: np.ndarray) -> np.ndarray | None:
    """Target training composition for a fold whose own composition is `natural`."""
    up_nat = natural[2] / (natural[0] + natural[2])
    if level == "full":
        return None
    if level == "nat":
        return natural
    x = float(level)
    return composition(x, up_nat) if design == "flat" else composition(UP_DESIGN_FLAT, x)


def run_task(ev, yi, folds, sizes, design, level, rep, feature_sets, params, val_index):
    """One (design, level, replicate): fit every model x weighting x feature set."""
    di, li = ["flat", "up"].index(design), [l for _, l in levels()].index(level)
    preds = {(fs, w, m): np.zeros(((val_index >= 0).sum(), 3), np.float32)
             for fs in feature_sets for w in WEIGHTS for m in MODELS}
    counts = []
    for f, n in zip(folds, sizes):
        rng = np.random.default_rng([rep, f.k])  # same class orders at every level: common random numbers
        orders = class_orders(yi, f.train, rng)
        natural = np.bincount(yi[f.train], minlength=3) / len(f.train)
        comp = train_comp(design, level, natural)
        tr_idx = f.train if comp is None else subsample(orders, n, comp)
        counts.append(np.bincount(yi[tr_idx], minlength=3).tolist())
        tr, va = ev.iloc[tr_idx], ev.iloc[f.val]
        rows = val_index[f.val]
        for fs, feats in feature_sets.items():
            for w in WEIGHTS:
                for m in MODELS:
                    _, p = fit_predict_xgb(tr, va, feats, {**params, "seed": rep}, {"kind": m, "weights": w})
                    preds[(fs, w, m)][rows] = p
    return design, level, rep, preds, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "balance_sweep"))
    args = ap.parse_args()
    guard_not_sealed(args.data)
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    price = [c for c in PRICE_FEATURES if c not in PROVISIONAL_FEATURES]
    feature_sets = {"price": price,
                    "price_sentiment": price + [c for c in SENTIMENT_FEATURES if c != "event_hour"]}

    ev = load_events(args.data, news_only=False)
    yi = label_index(ev.label)
    folds = walk_forward_folds(ev, cfg["cv"]["n_splits"], cfg["cv"]["embargo_h"])
    fold_counts = [np.bincount(yi[f.train], minlength=3) for f in folds]
    # one size per fold, feasible for every composition of both designs
    sizes = [fixed_sizes([c], [train_comp(d, l, c / c.sum()) for d, l in levels()
                               if l not in ("nat", "full")], SIZE_FRAC)[0] for c in fold_counts]

    val = np.concatenate([f.val for f in folds])
    val_index = np.full(len(ev), -1)
    val_index[val] = np.arange(len(val))
    fold_of = np.zeros(len(val), int)
    for f in folds:
        fold_of[val_index[f.val]] = f.k

    os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)
    ev.iloc[val][["market_ticker", "event_group", "series", "t0", "label", "dmid"]].assign(
        fold=fold_of).reset_index(drop=True).to_parquet(os.path.join(args.out, "events.parquet"))

    tasks = [(d, l, r) for d, l in levels() for r in range(args.reps)]
    print(f"{len(ev)} events, {len(val)} scored, fold train sizes {sizes}, {len(tasks)} tasks", flush=True)
    t = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_task)(ev, yi, folds, sizes, d, l, r, feature_sets, cfg["classifier"], val_index)
        for d, l, r in tasks)

    store, counts = {}, {}
    for d, l, r, preds, c in results:
        counts[f"{d}|{l}|{r}"] = c
        for (fs, w, m), p in preds.items():
            store.setdefault((d, l, fs, w, m), np.zeros((args.reps, len(val), 3), np.float32))[r] = p
    for (d, l, fs, w, m), arr in store.items():
        np.save(os.path.join(args.out, "preds", f"{d}_{l}_{fs}_{w}_{m}.npy"), arr)
    meta = {"levels": levels(), "flat_levels": FLAT_LEVELS, "up_levels": UP_LEVELS,
            "up_design_flat": UP_DESIGN_FLAT, "models": MODELS, "weights": WEIGHTS,
            "feature_sets": list(feature_sets), "reps": args.reps, "size_frac": SIZE_FRAC,
            "fold_train_sizes": sizes, "fold_train_counts_full": [c.tolist() for c in fold_counts],
            "train_counts": counts, "classifier": cfg["classifier"], "cv": cfg["cv"],
            "data_sha256": sha256_file(args.data), "minutes": round((time.time() - t) / 60, 1)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done in {meta['minutes']} min -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
