"""Per-event predictions of price vs price+sentiment, to see where they disagree.

    python scripts/compare_predictions.py [--scope current] [--news-only] [--seeds 0 1 2 3 4]

Both models are fit on the same walk-forward folds; each event's out-of-fold
probabilities are averaged over the seeds. For the price+sentiment model the
per-feature contributions to each class margin (XGBoost pred_contribs) are kept
too, so a changed prediction can be traced to the sentiment features behind it.
Development data only; refuses the sealed slice like run_cv.py.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from ksearch.data.manifest import REPO_ROOT
from ksearch.eval.cv import LABELS, fit_xgb, predict_proba
from ksearch.eval.folds import walk_forward_folds
from ksearch.features.price import PRICE_FEATURES, PROVISIONAL_FEATURES
from ksearch.features.sentiment import SENTIMENT_FEATURES
from run_cv import guard_not_sealed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"))
    ap.add_argument("--scope", choices=["all", "spring", "current"], default="current")
    ap.add_argument("--news-only", action="store_true",
                    help="train and score only on events with >= 2 articles, as run_cv.py --news-only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    guard_not_sealed(args.data)
    out = args.out or os.path.join(REPO_ROOT, "results", f"predictions_v2_{args.scope}{'_newsonly' if args.news_only else ''}.parquet")

    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    price = [c for c in PRICE_FEATURES if c not in PROVISIONAL_FEATURES]
    sentiment = [c for c in SENTIMENT_FEATURES if c != "event_hour"]
    both = price + sentiment
    events = pd.read_parquet(args.data)
    sent = pd.read_parquet(os.path.join(os.path.dirname(args.data), "sentiment.parquet"))
    events = events.merge(sent, on=["market_ticker", "t0"], how="left", validate="one_to_one")
    if args.news_only:
        events = events[events["n_articles"] >= 2]
    if args.scope != "all":
        with open(os.path.join(os.path.dirname(args.data), "universe.jsonl")) as f:
            scope = {m["ticker"]: m.get("_scope") for m in map(json.loads, f)}
        want = "spring" if args.scope == "spring" else "sample"
        events = events[events["market_ticker"].map(scope) == want]
    events = events.sort_values(["t0", "market_ticker"], kind="mergesort").reset_index(drop=True)

    folds = walk_forward_folds(events, cfg["cv"]["n_splits"], cfg["cv"]["embargo_h"])
    n = len(events)
    p_price, p_both = np.zeros((n, 3)), np.zeros((n, 3))
    contrib = np.zeros((n, 3, len(both) + 1))  # last column is the bias
    fold_of = np.full(n, -1)
    for seed in args.seeds:
        params = {**cfg["classifier"], "seed": seed}
        for f in folds:
            tr, va = events.iloc[f.train], events.iloc[f.val]
            p_price[f.val] += predict_proba(fit_xgb(tr, price, params), va, price)
            clf = fit_xgb(tr, both, params)
            p_both[f.val] += predict_proba(clf, va, both)
            contrib[f.val] += clf.get_booster().predict(xgb.DMatrix(va[both]), pred_contribs=True)
            fold_of[f.val] = f.k
    k = len(args.seeds)
    p_price, p_both, contrib = p_price / k, p_both / k, contrib / k

    keep = fold_of >= 0
    res = events.loc[keep, ["market_ticker", "event_group", "series", "t0", "label", "mid_t0", "dmid",
                            "n_articles"] + both].copy()
    res["fold"] = fold_of[keep]
    lab = np.array(LABELS, dtype=object)
    res["pred_price"], res["pred_both"] = lab[p_price[keep].argmax(1)], lab[p_both[keep].argmax(1)]
    for i, l in enumerate(LABELS):
        res[f"p_price_{l}"], res[f"p_both_{l}"] = p_price[keep, i], p_both[keep, i]
        # margin contribution of all sentiment features together, and of each one
        res[f"sent_push_{l}"] = contrib[keep][:, i, len(price):-1].sum(1)
        for j, c in enumerate(sentiment):
            res[f"c_{l}__{c}"] = contrib[keep][:, i, len(price) + j]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.to_parquet(out, index=False)
    print(f"wrote {out}: {len(res)} events, {len(args.seeds)} seeds, {len(folds)} folds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
