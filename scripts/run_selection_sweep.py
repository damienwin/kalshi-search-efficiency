"""Other ways to pick a fixed-size training set: 3-class vs two-stage when the subsample
is chosen by a criterion instead of uniformly at random.

    OMP_NUM_THREADS=1 python scripts/run_selection_sweep.py [--reps 20] [--jobs 8]

Same pool, folds and per-fold training sizes as run_balance_sweep.py (run that first:
its meta.json supplies the sizes and its random-drop predictions are the baseline).

Designs:
  select   FLAT share 0.50 ... 0.95 as in the "flat" design, but the FLAT events kept are
           the ones ranked highest by a criterion known at t0 (n_articles in the 6h window,
           or vol_24h), ties broken by the random order. Moves are drawn exactly as in the
           random sweep (same per-(replicate, fold) class orders), so each cell is paired
           with its random-drop twin and differs only in which FLATs were dropped.
  filter   keep only training events passing a filter (n_articles >= k, or vol_24h above a
           pool quantile), then draw a fixed-size uniform subsample of them. The size per
           fold is the same at every filter level (0.8 x the smallest filtered pool), so
           the filter is not confounded with sample size. The class balance is whatever the
           filter produces. Predictions cover every validation event; the analysis scores
           each model on the validation events passing the same filter.

Price + sentiment features, both weightings, XGBoost seed = replicate.
Output: runs/balance_sweep/selection/ (gitignored). Refuses the sealed slice like run_cv.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time

import numpy as np
import yaml
from joblib import Parallel, delayed

from ksearch.data.manifest import REPO_ROOT, sha256_file
from ksearch.eval.balance import FLAT, class_orders, composition, label_index, subsample
from ksearch.eval.cv import fit_predict_xgb
from ksearch.eval.folds import walk_forward_folds
from ksearch.features.price import PRICE_FEATURES, PROVISIONAL_FEATURES
from ksearch.features.sentiment import SENTIMENT_FEATURES
from direction_significance import load_events
from run_cv import guard_not_sealed

FLAT_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
CRITERIA = ["articles", "vol"]
FILTERS = [("all", 0.0), ("articles", 1), ("articles", 2), ("articles", 5), ("articles", 10),
           ("vol", 0.25), ("vol", 0.50), ("vol", 0.75)]
MODELS = ["3class", "two_stage"]
WEIGHTS = ["balanced", "none"]
SIZE_FRAC = 0.8


def scores(ev) -> dict[str, np.ndarray]:
    """Selection criteria, higher = kept first. Missing vol_24h ranks last."""
    return {"articles": ev["n_articles"].fillna(0).to_numpy(float),
            "vol": ev["vol_24h"].fillna(-np.inf).to_numpy(float)}


def filter_masks(ev) -> dict[str, np.ndarray]:
    s = scores(ev)
    out = {}
    for kind, v in FILTERS:
        name = f"{kind}_{v:g}"
        if kind == "all":
            out[name] = np.ones(len(ev), bool)
        elif kind == "articles":
            out[name] = s["articles"] >= v
        else:  # vol_24h above the pool quantile, as in run_natural_sweeps.py
            out[name] = s["vol"] > np.nanquantile(ev["vol_24h"], v)
    return out


def ranked(order: np.ndarray, score: np.ndarray) -> np.ndarray:
    """`order` re-sorted by score, highest first; the random order breaks ties."""
    return order[np.argsort(-score[order], kind="stable")]


def run_task(ev, yi, folds, sizes, filt_sizes, masks, crit, design, level, rep, feats, params, n_val, val_index):
    preds = {(w, m): np.zeros((n_val, 3), np.float32) for w in WEIGHTS for m in MODELS}
    counts = []
    for f, n, nf in zip(folds, sizes, filt_sizes):
        if design == "select":
            rng = np.random.default_rng([rep, f.k])  # the random sweep's class orders
            orders = class_orders(yi, f.train, rng)
            orders[FLAT] = ranked(orders[FLAT], crit[level.split("|")[0]])
            natural = np.bincount(yi[f.train], minlength=3)
            comp = composition(float(level.split("|")[1]), natural[2] / (natural[0] + natural[2]))
            tr_idx = subsample(orders, n, comp)
        else:
            rng = np.random.default_rng([rep, f.k, 1])
            perm = rng.permutation(f.train)
            tr_idx = np.sort(perm[masks[level][perm]][:nf])
        counts.append(np.bincount(yi[tr_idx], minlength=3).tolist())
        tr, va = ev.iloc[tr_idx], ev.iloc[f.val]
        for w in WEIGHTS:
            for m in MODELS:
                _, p = fit_predict_xgb(tr, va, feats, {**params, "seed": rep}, {"kind": m, "weights": w})
                preds[(w, m)][val_index[f.val]] = p
    return design, level, rep, preds, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--runs", default=os.path.join(REPO_ROOT, "runs", "balance_sweep"))
    args = ap.parse_args()
    guard_not_sealed(args.data)
    out = os.path.join(args.runs, "selection")
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.runs, "meta.json")) as f:
        base = json.load(f)
    feats = [c for c in PRICE_FEATURES if c not in PROVISIONAL_FEATURES]
    feats += [c for c in SENTIMENT_FEATURES if c != "event_hour"]

    ev = load_events(args.data, news_only=False)
    yi = label_index(ev.label)
    folds = walk_forward_folds(ev, cfg["cv"]["n_splits"], cfg["cv"]["embargo_h"])
    sizes = base["fold_train_sizes"]
    masks = filter_masks(ev)
    filt_sizes = [int(SIZE_FRAC * min(masks[k][f.train].sum() for k in masks)) for f in folds]
    crit = scores(ev)

    val = np.concatenate([f.val for f in folds])
    val_index = np.full(len(ev), -1)
    val_index[val] = np.arange(len(val))

    levels = [("select", f"{c}|{x:.2f}") for c in CRITERIA for x in FLAT_LEVELS] + [("filter", k) for k in masks]
    tasks = [(d, l, r) for d, l in levels for r in range(args.reps)]
    print(f"{len(val)} scored, select sizes {sizes}, filter sizes {filt_sizes}, {len(tasks)} tasks", flush=True)
    t = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_task)(ev, yi, folds, sizes, filt_sizes, masks, crit, d, l, r, feats, cfg["classifier"],
                          len(val), val_index)
        for d, l, r in tasks)

    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    store, counts = {}, {}
    for d, l, r, preds, c in results:
        counts[f"{d}|{l}|{r}"] = c
        for (w, m), p in preds.items():
            store.setdefault((d, l, w, m), np.zeros((args.reps, len(val), 3), np.float32))[r] = p
    for (d, l, w, m), arr in store.items():
        np.save(os.path.join(out, "preds", f"{d}_{l.replace('|', '_')}_{w}_{m}.npy"), arr)
    np.save(os.path.join(out, "val_masks.npy"), np.stack([masks[k][val] for k in masks]))
    meta = {"levels": levels, "flat_levels": FLAT_LEVELS, "criteria": CRITERIA, "filters": list(masks),
            "models": MODELS, "weights": WEIGHTS, "reps": args.reps, "select_sizes": sizes,
            "filter_sizes": filt_sizes, "train_counts": counts, "classifier": cfg["classifier"],
            "data_sha256": sha256_file(args.data), "minutes": round((time.time() - t) / 60, 1)}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done in {meta['minutes']} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
