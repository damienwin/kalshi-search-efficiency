"""Score a frozen model on the sealed test slice. Meant to run once per claim.

    python scripts/confirm_sealed.py --features price --reason "..."            # shows the plan, reads nothing
    python scripts/confirm_sealed.py --features price --reason "..." --confirm  # reads sealed, logs, scores

Trains the configured XGBoost on ALL development events, then scores the
sealed slice once against the same baselines, with the event-clustered paired
bootstrap. The sealed file is read through read_sealed(), which appends to the
committed access log (data/manifests/sealed_access.jsonl). If earlier
confirmations exist, they are listed and --confirm must be passed again
knowingly: each extra look at the sealed slice weakens it.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

from ksearch.data.manifest import REPO_ROOT, sha256_file
from ksearch.eval.cv import BASELINES, LABELS, clustered_paired_bootstrap, fit_predict_xgb, macro_f1
from ksearch.eval.sealed import ACCESS_LOG, SEALED_PATH, access_log, read_sealed
from ksearch.features.price import PRICE_FEATURES, PROVISIONAL_FEATURES
from ksearch.features.sentiment import SENTIMENT_FEATURES
from sklearn.metrics import log_loss


def feature_list(name: str) -> list[str]:
    price = [c for c in PRICE_FEATURES if c not in PROVISIONAL_FEATURES]
    sentiment = [c for c in SENTIMENT_FEATURES if c != "event_hour"]
    return {"price": price, "price+sentiment": price + sentiment}[name]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["price", "price+sentiment"], required=True)
    ap.add_argument("--reason", required=True, help="why this sealed look is justified (logged)")
    ap.add_argument("--confirm", action="store_true", help="actually read and score the sealed slice")
    ap.add_argument("--build-dir", default=os.path.join(REPO_ROOT, "data", "build", "v1"))
    ap.add_argument("--sealed-path", default=SEALED_PATH)
    ap.add_argument("--log", default=ACCESS_LOG)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
    args = ap.parse_args(argv)

    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    features = feature_list(args.features)
    dev_path = os.path.join(args.build_dir, "dev.parquet")
    prior = [e for e in access_log(args.log) if e.get("purpose") == "confirm"]

    print(f"plan: train XGBoost on {dev_path} ({args.features}, {len(features)} features), score sealed once")
    print(f"prior confirmations on the sealed slice: {len(prior)}")
    for e in prior:
        print(f"  {e['at']}  {e.get('features')}  {e['reason']}")
    if not args.confirm:
        print("dry run: nothing read. Re-run with --confirm to score the sealed slice.")
        return 0

    dev = pd.read_parquet(dev_path)
    sealed = read_sealed(args.reason, args.sealed_path, args.log, purpose="confirm",
                         features=args.features, dev_sha256=sha256_file(dev_path))
    if "sentiment" in args.features:
        sent = pd.read_parquet(os.path.join(args.build_dir, "sentiment.parquet"))
        dev = dev.merge(sent, on=["market_ticker", "t0"], how="left", validate="one_to_one")
        sealed = sealed.merge(sent, on=["market_ticker", "t0"], how="left", validate="one_to_one")

    pred, proba = fit_predict_xgb(dev, sealed, features, cfg["classifier"])
    y, groups = sealed["label"].to_numpy(), sealed["event_group"].to_numpy()
    base = {n: fn(dev, sealed) for n, fn in BASELINES.items()}
    res = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": args.reason, "features": args.features,
        "n_sealed": len(sealed), "n_groups": int(pd.Series(groups).nunique()),
        "label_dist": sealed["label"].value_counts().to_dict(),
        "macro_f1": {"model": macro_f1(y, pred), **{n: macro_f1(y, p) for n, p in base.items()}},
        "logloss_model": float(log_loss(y, proba, labels=LABELS)),
        "comparisons": {f"model_vs_{n}": clustered_paired_bootstrap(y, pred, p, groups, cfg["cv"]["n_bootstrap"])
                        for n, p in base.items()},
        "prior_confirmations": len(prior),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"sealed_{args.features.replace('+', '_')}_{res['at'][:10]}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    for n, v in res["macro_f1"].items():
        print(f"  sealed macro F1  {n:<10} {v:.4f}")
    c = res["comparisons"]["model_vs_momentum"]
    print(f"  model vs momentum  {c['delta']:+.4f}  95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    print(f"written {out}; access logged to {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
