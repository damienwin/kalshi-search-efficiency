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
    real = os.path.realpath(path)
    if real.startswith(SEALED_DIR + os.sep) or os.path.basename(real) == "events_index.parquet":
        raise SystemExit(f"run_cv refuses sealed data: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "dev.parquet"))
    ap.add_argument("--with-provisional", action="store_true",
                    help="include hours_to_close (L7 ablation)")
    ap.add_argument("--features", choices=["price", "price+sentiment", "sentiment"], default="price")
    ap.add_argument("--reference", choices=["price", "price+sentiment", "sentiment"], default=None,
                    help="also fit this feature set on the same folds and test model vs reference")
    ap.add_argument("--news-only", action="store_true",
                    help="restrict to events with >= 2 articles (the Spring inclusion rule)")
    ap.add_argument("--scope", choices=["all", "spring", "current"], default="all",
                    help="spring = the 304 Spring markets, current = post-Spring markets only")
    ap.add_argument("--seed", type=int, default=None, help="override classifier.seed (model + random baseline)")
    ap.add_argument("--sentiment-file", default="sentiment.parquet",
                    help="sentiment parquet next to --data (e.g. sentiment_rel.parquet, topic-filtered)")
    ap.add_argument("--model", choices=["3class", "two_stage", "flip_penalty"], default="3class",
                    help="ksearch.model.direction; two_stage uses the classifier hyperparameters for both stages")
    ap.add_argument("--flip-lambda", type=float, default=1.0,
                    help="flip_penalty: weight on P(opposite direction) added to the log loss")
    ap.add_argument("--weights", choices=["balanced", "none"], default="balanced",
                    help="training sample weights for every model stage")
    ap.add_argument("--flip-cost", type=float, default=1.0,
                    help="cost of a flip relative to any other error in the call; 1 = argmax")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "cv_price_v1.json"))
    args = ap.parse_args()
    guard_not_sealed(args.data)

    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    price = [c for c in PRICE_FEATURES if args.with_provisional or c not in PROVISIONAL_FEATURES]
    sentiment = [c for c in SENTIMENT_FEATURES if c != "event_hour"]  # t0 is grid-aligned; hour is schedule
    sets = {"price": price, "price+sentiment": price + sentiment, "sentiment": sentiment}
    features = sets[args.features]
    reference = sets[args.reference] if args.reference else None
    events = pd.read_parquet(args.data)
    if "sentiment" in args.features or (args.reference and "sentiment" in args.reference) or args.news_only:
        sent = pd.read_parquet(os.path.join(os.path.dirname(args.data), args.sentiment_file))
        events = events.merge(sent, on=["market_ticker", "t0"], how="left", validate="one_to_one")
        if args.news_only:
            events = events[events["n_articles"] >= 2]
    if args.scope != "all":
        with open(os.path.join(os.path.dirname(args.data), "universe.jsonl")) as f:
            scope = {m["ticker"]: m.get("_scope") for m in map(json.loads, f)}
        want = "spring" if args.scope == "spring" else "sample"
        events = events[events["market_ticker"].map(scope) == want]
    if args.seed is not None:
        cfg["classifier"]["seed"] = args.seed
    events = events.sort_values(["t0", "market_ticker"], kind="mergesort")
    events = events.reset_index(drop=True)

    model = {"kind": args.model, "flip_cost": args.flip_cost, "weights": args.weights}
    if args.model == "flip_penalty":
        model["lam"] = args.flip_lambda
    res = run_cv(events, features, cfg["cv"], cfg["classifier"], reference, model)
    res["feature_set"], res["reference_set"], res["news_only"] = args.features, args.reference, args.news_only
    res["scope"], res["seed"] = args.scope, cfg["classifier"]["seed"]
    res["data_sha256"] = sha256_file(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)

    print(f"features={args.features} news_only={args.news_only}  "
          f"scored {res['n_events_scored']} events  labels {res['label_dist_scored']}")
    for name, v in res["pooled_macro_f1"].items():
        print(f"  macro F1  {name:<10} {v:.4f}")
    print(f"  log loss  model      {res['pooled_logloss_model']:.4f}")
    d = res["direction_breakdown_model"]
    print(f"  move AUC {d['move_auc']:.3f}  direction AUC {d['dir_auc']:.3f}  flips {d['flips']}"
          f"  right direction {d['right_direction']}  false moves {d['false_moves']}")
    for name, c in res["comparisons"].items():
        print(f"  {name:<22} delta {c['delta']:+.4f}  95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
              f"  {'SIGNIFICANT' if c['excludes_zero'] else 'n.s.'}  ({c['n_groups']} groups)")
    k = res["legacy_keep_rule_vs_momentum"]
    print(f"  legacy keep rule vs momentum: {k['verdict']} ({k['folds_won']}/{k['n_folds']} folds, "
          f"delta {k['delta']:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
