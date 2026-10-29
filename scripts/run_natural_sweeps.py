"""Natural class-balance levers: 3-class vs two-stage where the balance changes for a reason.

    OMP_NUM_THREADS=1 python scripts/run_natural_sweeps.py [--seeds 0 1 2 3 4] [--jobs 8]

Unlike run_balance_sweep.py, each lever changes more than the class balance, which is
the point: these are the ways a different balance would actually arise.

  threshold   move threshold 0.005 ... 0.08 on the same events. Labels are recomputed
              from the stored dmid (rounded to 1e-6) as |dmid| > threshold, and prev_label
              from the previous event in the market with the same rule. The stored labels
              decide events sitting exactly on 0.03 by float rounding, so the 0.03 point here
              differs from the week 4 labels on about 1% of events
  activity    keep events whose 24h volatility (vol_24h, known at t0) is above a quantile
              of the pool: "only predict markets that have been moving"
  articles    keep events with at least k articles in the 6h before t0 (k = 2 is the
              week 4 news-events set)

Every set gets its own walk-forward folds; out-of-fold probabilities are averaged over
the training seeds, as in direction_significance.py. Output: runs/balance_sweep/natural/.
Refuses the sealed slice like run_cv.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from ksearch.data.manifest import REPO_ROOT
from ksearch.eval.cv import LABELS, fit_predict_xgb
from ksearch.eval.folds import walk_forward_folds
from ksearch.features.price import PRICE_FEATURES, PROVISIONAL_FEATURES
from ksearch.features.sentiment import SENTIMENT_FEATURES
from direction_significance import load_events
from run_cv import guard_not_sealed

THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
VOL_QUANTILES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_ARTICLES = [0, 1, 2, 5, 10]
MODELS = ["3class", "two_stage"]
WEIGHTS = ["balanced", "none"]
TOL = 1e-9  # dmid is stored rounded to 1e-6


def relabel(ev: pd.DataFrame, threshold: float) -> pd.DataFrame:
    ev = ev.copy()
    d = ev["dmid"].to_numpy()
    ev["label"] = np.where(d > threshold + TOL, "UP", np.where(d < -threshold - TOL, "DOWN", "FLAT"))
    prev = ev.sort_values(["market_ticker", "t0"]).groupby("market_ticker")["dmid"].shift(1)
    lab = np.where(prev > threshold + TOL, 1.0, np.where(prev < -threshold - TOL, -1.0, 0.0))
    ev["prev_label"] = pd.Series(np.where(prev.isna(), np.nan, lab), index=prev.index)
    return ev


def event_sets(pool: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    sets = {("threshold", f"{t:g}"): relabel(pool, t) for t in THRESHOLDS}
    for q in VOL_QUANTILES:
        th = pool["vol_24h"].quantile(q)
        sets[("activity", f"{q:.1f}")] = pool if q == 0 else pool[pool["vol_24h"] > th]
    for k in MIN_ARTICLES:
        sets[("articles", str(k))] = pool[pool["n_articles"].fillna(0) >= k]
    return {key: ev.sort_values(["t0", "market_ticker"], kind="mergesort").reset_index(drop=True)
            for key, ev in sets.items()}


def run_task(ev, folds, seed, feature_sets, params):
    out = {}
    for fs, feats in feature_sets.items():
        for w in WEIGHTS:
            for m in MODELS:
                p = np.full((len(ev), 3), np.nan)
                for f in folds:
                    _, p[f.val] = fit_predict_xgb(ev.iloc[f.train], ev.iloc[f.val], feats,
                                                  {**params, "seed": seed}, {"kind": m, "weights": w})
                out[(fs, w, m)] = p
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "balance_sweep", "natural"))
    args = ap.parse_args()
    guard_not_sealed(args.data)
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    price = [c for c in PRICE_FEATURES if c not in PROVISIONAL_FEATURES]
    feature_sets = {"price": price,
                    "price_sentiment": price + [c for c in SENTIMENT_FEATURES if c != "event_hour"]}
    sets = event_sets(load_events(args.data, news_only=False))
    folds = {k: walk_forward_folds(ev, cfg["cv"]["n_splits"], cfg["cv"]["embargo_h"]) for k, ev in sets.items()}

    tasks = [(k, s) for k in sets for s in args.seeds]
    res = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_task)(sets[k], folds[k], s, feature_sets, cfg["classifier"]) for k, s in tasks)

    os.makedirs(args.out, exist_ok=True)
    for key, ev in sets.items():
        runs = [r for (k, _), r in zip(tasks, res) if k == key]
        fold = np.zeros(len(ev), int)
        for f in folds[key]:
            fold[f.val] = f.k
        cols = {"market_ticker": ev.market_ticker, "event_group": ev.event_group, "series": ev.series,
                "t0": ev.t0, "label": ev.label, "fold": fold}
        for combo in runs[0]:
            p = np.mean([r[combo] for r in runs], axis=0)
            for i, l in enumerate(LABELS):
                cols[f"p_{'_'.join(combo)}_{l}"] = p[:, i]
        df = pd.DataFrame(cols)
        df[df.fold > 0].reset_index(drop=True).to_parquet(os.path.join(args.out, f"{key[0]}_{key[1]}.parquet"))
        print(f"{key}: {len(ev)} events, {(df.fold > 0).sum()} scored, "
              f"FLAT {(df.label[df.fold > 0] == 'FLAT').mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
