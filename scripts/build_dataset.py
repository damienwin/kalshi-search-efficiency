"""Build v1 events + price features from raw candles, then seal the test slice.

    python scripts/build_dataset.py

Reads   data/build/universe.jsonl, data/raw/kalshi/*/candles/**
Writes  data/build/dev.parquet          development events (CV only)
        sealed/sealed.parquet           groups starting in the last time slice, read once
        data/build/attrition.json       per-filter drop counts
        data/manifests/dataset_v1.json  counts + input/output hashes (committed)
"""

import argparse
import collections
import glob
import json
import logging
import os
import sys

import pandas as pd
import yaml

from ksearch.data.grid import MarketCandles
from ksearch.data.kalshi import RAW_DIR, iso_to_ts, load_gz, series_of
from ksearch.data.labels import build_market_events, events_frame
from ksearch.data.manifest import REPO_ROOT, sha256_file
from ksearch.data.parse import parse_candles
from ksearch.eval.folds import seal_split
from ksearch.features.price import PRICE_FEATURES, market_features

log = logging.getLogger("build_dataset")
BUILD = os.path.join(REPO_ROOT, "data", "build")
SEALED = os.path.join(REPO_ROOT, "sealed")


def load_market_candles(market: dict) -> pd.DataFrame:
    pattern = os.path.join(RAW_DIR, market["_source"], "candles", series_of(market), f"{market['ticker']}__*.json.gz")
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(parse_candles(load_gz(path).get("candlesticks", []), market["_source"]))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=BUILD)
    ap.add_argument("--sealed-dir", default=SEALED)
    ap.add_argument("--summary", default=os.path.join(REPO_ROOT, "data", "manifests", "dataset_v1.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    lcfg, ccfg = cfg["labels"], cfg["cv"]

    with open(os.path.join(args.build_dir, "universe.jsonl")) as f:
        markets = [json.loads(line) for line in f]

    attrition = collections.Counter()
    frames, n_no_candles = [], 0
    for i, m in enumerate(markets, 1):
        candles = load_market_candles(m)
        if candles.empty:
            n_no_candles += 1
            continue
        open_ts, close_ts = iso_to_ts(m["open_time"]), iso_to_ts(m["close_time"])
        mc = MarketCandles(candles)
        ev = events_frame(build_market_events(mc, m, open_ts, close_ts, lcfg, attrition))
        if ev.empty:
            continue
        feats = market_features(mc, ev, open_ts, close_ts, lcfg["max_staleness_h"])
        frames.append(pd.concat([ev, feats.loc[ev.index]], axis=1))
        if i % 500 == 0:
            log.info("%d/%d markets, %d events", i, len(markets), attrition["events"])

    events = pd.concat(frames, ignore_index=True).sort_values(["t0", "market_ticker"], kind="mergesort")
    events = events.reset_index(drop=True)
    dev, sealed, purged = seal_split(events, ccfg["sealed_fraction"], ccfg["embargo_h"])
    attrition["purged_at_seal"] = purged

    os.makedirs(args.sealed_dir, exist_ok=True)
    dev_path = os.path.join(args.build_dir, "dev.parquet")
    sealed_path = os.path.join(args.sealed_dir, "sealed.parquet")
    dev.to_parquet(dev_path, index=False)
    sealed.to_parquet(sealed_path, index=False)

    attrition["markets_without_candles"] = n_no_candles
    with open(os.path.join(args.build_dir, "attrition.json"), "w") as f:
        json.dump(dict(attrition), f, indent=2, sort_keys=True)

    def span(df):
        return [pd.Timestamp(df.t0.min(), unit="s", tz="UTC").isoformat(),
                pd.Timestamp(df.t0.max(), unit="s", tz="UTC").isoformat()] if len(df) else None

    summary = {
        "version": "v1",
        "label_config": lcfg,
        "markets_in_universe": len(markets),
        "attrition": dict(sorted(attrition.items())),
        "features": PRICE_FEATURES,
        "splits": {
            name: {"events": len(df), "event_groups": int(df.event_group.nunique()),
                   "markets": int(df.market_ticker.nunique()), "span": span(df),
                   "labels": df.label.value_counts().to_dict(), "sha256": sha256_file(p)}
            for name, df, p in [("dev", dev, dev_path), ("sealed", sealed, sealed_path)]
        },
        "inputs": {os.path.basename(p): sha256_file(p) for p in
                   sorted(glob.glob(os.path.join(REPO_ROOT, "data", "manifests", "kalshi_*.jsonl")))},
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("attrition", "splits")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
