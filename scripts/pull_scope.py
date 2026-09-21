"""Pull the v1 market scope: Spring's markets, plus a same-rule sample after them.

    python scripts/pull_scope.py spring            # the 304 Spring markets, by ticker
    python scripts/pull_scope.py sample            # 2025-01 -> now, from data/build/universe.jsonl
    python scripts/pull_scope.py merge             # -> data/build/universe_v1.jsonl

Scope rule (Spring's): settled markets with volume >= $5k in the Spring series
universe, stratified by close month, kept if they have >= 3 directional (UP/DOWN)
events. Spring density was 304 markets over 2023-10 -> 2024-12, about 22/month;
the sample draws candidates uniformly at random within each month (seeded) until
22 pass the activity rule or the month is exhausted.
"""

import argparse
import collections
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone

import yaml

from ksearch.data.grid import MarketCandles
from ksearch.data.kalshi import RAW_DIR, KalshiClient, iso_to_ts
from ksearch.data.labels import build_market_events
from ksearch.data.manifest import Manifest, REPO_ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import load_market_candles  # noqa: E402

log = logging.getLogger("pull_scope")
BUILD = os.path.join(REPO_ROOT, "data", "build")
SPRING_END = "2025-01-01T00:00:00Z"


def load_cfg() -> dict:
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        return yaml.safe_load(f)


def fetched_paths() -> set:
    """Raw files already recorded in any Kalshi manifest (skipped on re-runs)."""
    d = os.path.join(REPO_ROOT, "data", "manifests")
    done = set()
    for name in os.listdir(d):
        if name.startswith("kalshi_") and name.endswith(".jsonl"):
            done |= Manifest(os.path.join(d, name)).recorded_paths()
    return done


def client_for(tag: str, delay: float) -> KalshiClient:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return KalshiClient(Manifest(os.path.join(REPO_ROOT, "data", "manifests", f"kalshi_{tag}_{stamp}.jsonl")),
                        request_delay=delay)


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def directional_events(market: dict, lcfg: dict) -> int:
    candles = load_market_candles(market)
    if candles.empty:
        return 0
    ev = build_market_events(MarketCandles(candles), market, iso_to_ts(market["open_time"]),
                             iso_to_ts(market["close_time"]), lcfg, collections.Counter())
    return sum(e["label"] != "FLAT" for e in ev)


def pull_spring(cfg: dict) -> None:
    client = client_for("spring", cfg["kalshi"]["request_delay"])
    done = fetched_paths()
    cutoff_ts = iso_to_ts(client.get_cutoff()["market_settled_ts"])
    with open(os.path.join(REPO_ROOT, "config", "spring_markets.txt")) as f:
        tickers = [t.strip() for t in f if t.strip() and not t.startswith("#")]
    markets = []
    for t in tickers:
        out = os.path.join(RAW_DIR, "historical", "market", f"{t}.json.gz")
        m = client._get_saved(f"/historical/markets/{t}", {}, out)["market"]
        m["_source"] = "historical" if iso_to_ts(m["close_time"]) < cutoff_ts else "live"
        m["_scope"] = "spring"
        client.fetch_candles(m, m["_source"], done, cfg["kalshi"]["period_interval"])
        markets.append(m)
    write_jsonl(os.path.join(BUILD, "universe_spring.jsonl"), markets)
    log.info("spring scope: %d markets, %d requests", len(markets), client.n_requests)


def pull_sample(cfg: dict, per_month: int, seed: int) -> None:
    client = client_for("sample", cfg["kalshi"]["request_delay"])
    done = fetched_paths()
    frame = [m for m in read_jsonl(os.path.join(BUILD, "universe.jsonl"))
             if iso_to_ts(m["close_time"]) >= iso_to_ts(SPRING_END)]
    by_month = collections.defaultdict(list)
    for m in frame:
        by_month[m["close_time"][:7]].append(m)
    rng = random.Random(seed)
    kept, report = [], {}
    for month in sorted(by_month):
        cands = sorted(by_month[month], key=lambda m: m["ticker"])
        rng.shuffle(cands)
        got = tried = 0
        for m in cands:
            if got == per_month:
                break
            tried += 1
            client.fetch_candles(m, m["_source"], done, cfg["kalshi"]["period_interval"])
            if directional_events(m, cfg["labels"]) >= 3:
                m["_scope"] = "sample"
                kept.append(m)
                got += 1
        report[month] = {"frame": len(cands), "tried": tried, "kept": got}
        log.info("%s: kept %d of %d tried (frame %d)", month, got, tried, len(cands))
    write_jsonl(os.path.join(BUILD, "universe_sample.jsonl"), kept)
    with open(os.path.join(REPO_ROOT, "data", "manifests", "scope_sample.json"), "w") as f:
        json.dump({"per_month": per_month, "seed": seed, "months": report,
                   "total_kept": len(kept)}, f, indent=2)


def merge() -> None:
    rows = read_jsonl(os.path.join(BUILD, "universe_spring.jsonl"))
    sample_path = os.path.join(BUILD, "universe_sample.jsonl")
    if os.path.exists(sample_path):
        rows += read_jsonl(sample_path)
    rows.sort(key=lambda m: (m["close_time"], m["ticker"]))
    out = os.path.join(BUILD, "v1")
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "universe.jsonl"), rows)
    log.info("merged scope: %d markets -> %s", len(rows), out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["spring", "sample", "merge"])
    ap.add_argument("--per-month", type=int, default=22)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_cfg()
    if args.what == "spring":
        pull_spring(cfg)
    elif args.what == "sample":
        pull_sample(cfg, args.per_month, args.seed)
    else:
        merge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
