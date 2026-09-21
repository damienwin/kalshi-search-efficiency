"""Walk-forward CV on the development set: price-only XGBoost vs baselines.

    python scripts/run_cv.py [--with-provisional] [--out results/cv_price_v1.json]

Refuses any input under sealed/ (L11). The sealed slice is scored only by
scripts/confirm_sealed.py, once.
"""

import argparse
import json
import os
import sys

import pandas as pd
import yaml

from ksearch.data.manifest import REPO_ROOT, sha256_file
from ksearch.eval.cv import run_cv
from ksearch.features.price import PRICE_FEATURES, PROVISIONAL_FEATURES
from ksearch.features.sentiment import SENTIMENT_FEATURES

SEALED_DIR = os.path.realpath(os.path.join(REPO_ROOT, "sealed"))


def guard_not_sealed(path: str) -> None:
    if os.path.realpath(path).startswith(SEALED_DIR + os.sep):
        raise SystemExit(f"run_cv refuses sealed data: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "dev.parquet"))
    ap.add_argument("--with-provisional", action="store_true",
                    help="include hours_to_close (L7 ablation)")
    ap.add_argument("--features", choices=["price", "price+sentiment", "sentiment"], default="price")
    ap.add_argument("--news-only", action="store_true",
                    help="restrict to events with >= 2 articles (the Spring inclusion rule)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "cv_price_v1.json"))
    args = ap.parse_args()
    guard_not_sealed(args.data)

    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    price = [c for c in PRICE_FEATURES if args.with_provisional or c not in PROVISIONAL_FEATURES]
    sentiment = [c for c in SENTIMENT_FEATURES if c != "event_hour"]  # t0 is grid-aligned; hour is schedule
    features = {"price": price, "price+sentiment": price + sentiment, "sentiment": sentiment}[args.features]
    events = pd.read_parquet(args.data)
    if args.features != "price" or args.news_only:
        sent = pd.read_parquet(os.path.join(os.path.dirname(args.data), "sentiment.parquet"))
        events = events.merge(sent, on=["market_ticker", "t0"], how="left", validate="one_to_one")
        if args.news_only:
            events = events[events["n_articles"] >= 2]
    events = events.sort_values(["t0", "market_ticker"], kind="mergesort")
    events = events.reset_index(drop=True)

    res = run_cv(events, features, cfg["cv"], cfg["classifier"])
    res["data_sha256"] = sha256_file(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"features={args.features} news_only={args.news_only}  "
          f"scored {res['n_events_scored']} events  labels {res['label_dist_scored']}")
    for name, v in res["pooled_macro_f1"].items():
        print(f"  macro F1  {name:<10} {v:.4f}")
    print(f"  log loss  model      {res['pooled_logloss_model']:.4f}")
    for name, c in res["comparisons"].items():
        print(f"  {name:<22} delta {c['delta']:+.4f}  95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"  {'SIGNIFICANT' if c['excludes_zero'] else 'n.s.'}  ({c['n_groups']} groups)")
    k = res["legacy_keep_rule_vs_momentum"]
    print(f"  legacy keep rule vs momentum: {k['verdict']} ({k['folds_won']}/{k['n_folds']} folds, "
          f"delta {k['delta']:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
