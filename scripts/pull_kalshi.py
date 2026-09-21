"""Pull the Kalshi market universe and hourly candles, verbatim and hashed.

    python scripts/pull_kalshi.py --dry-run          # enumerate + estimate, no candles
    python scripts/pull_kalshi.py --limit 50         # candles for a 50-market sample
    python scripts/pull_kalshi.py                    # everything (resumable)

Enumeration always runs (it is cheap) and writes data/build/universe.jsonl:
one line per market that passed the filters, tagged with its source endpoint.
Candle files already recorded in the manifest are skipped, so an interrupted
pull resumes where it stopped.
"""

import argparse
import collections
import json
import logging
import os
import sys
from datetime import datetime, timezone

import yaml

from ksearch.data.kalshi import KalshiClient, iso_to_ts, now_ts
from ksearch.data.manifest import Manifest, REPO_ROOT

log = logging.getLogger("pull_kalshi")
UNIVERSE_PATH = os.path.join(REPO_ROOT, "data", "build", "universe.jsonl")


def read_universe_file(path: str) -> list[str]:
    with open(os.path.join(REPO_ROOT, path)) as f:
        return [s.strip() for s in f if s.strip() and not s.startswith("#")]


def keep(m: dict, min_close_ts: int, min_volume: float, exclude_mve: bool) -> bool:
    if iso_to_ts(m["close_time"]) < min_close_ts:
        return False
    if float(m.get("volume_fp") or 0) < min_volume:
        return False
    if exclude_mve and m.get("mve_collection_ticker"):
        return False
    return bool(m.get("open_time"))


def enumerate_universe(client: KalshiClient, cfg: dict, cutoff_ts: int) -> list[dict]:
    series = read_universe_file(cfg["universe_file"])
    min_close_ts = iso_to_ts(cfg["min_close_date"] + "T00:00:00Z")
    by_ticker = {}
    for i, s in enumerate(series, 1):
        for source in ("historical", "live"):
            lo = min_close_ts if source == "historical" else max(min_close_ts, cutoff_ts)
            passes = lambda m: keep(m, min_close_ts, cfg["min_volume_dollars"], cfg["exclude_multivariate"])
            for m in client.list_series_markets(s, source, lo, keep=passes):
                # A market can surface on both endpoints around the cutoff; the
                # endpoint that holds its candles is decided by its settlement.
                settled_before = iso_to_ts(m["close_time"]) < cutoff_ts
                m["_source"] = "historical" if settled_before else "live"
                by_ticker.setdefault(m["ticker"], m)
        if i % 50 == 0:
            log.info("enumerated %d/%d series, %d markets kept, %d requests",
                     i, len(series), len(by_ticker), client.n_requests)
    markets = sorted(by_ticker.values(), key=lambda m: (m["close_time"], m["ticker"]))
    os.makedirs(os.path.dirname(UNIVERSE_PATH), exist_ok=True)
    with open(UNIVERSE_PATH, "w") as f:
        for m in markets:
            f.write(json.dumps(m, sort_keys=True) + "\n")
    return markets


def estimate(markets: list[dict], period_min: int, delay: float) -> dict:
    per_source = collections.Counter()
    per_month = collections.Counter()
    requests = 0
    for m in markets:
        per_source[m["_source"]] += 1
        per_month[m["close_time"][:7]] += 1
        requests += len(KalshiClient.candle_windows(iso_to_ts(m["open_time"]), iso_to_ts(m["close_time"]), period_min))
    return {
        "markets": len(markets),
        "per_source": dict(per_source),
        "per_month": dict(sorted(per_month.items())),
        "candle_requests": requests,
        "eta_hours_at_delay": round(requests * (delay + 0.25) / 3600, 2),  # +~0.25s latency
    }


def sample(markets: list[dict], n: int) -> list[dict]:
    """Deterministic sample spread evenly across close time."""
    if n >= len(markets):
        return markets
    step = len(markets) / n
    return [markets[int(i * step)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse-universe", action="store_true",
                    help="skip enumeration and read data/build/universe.jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)["kalshi"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    manifest = Manifest(os.path.join(REPO_ROOT, "data", "manifests", f"kalshi_{stamp}.jsonl"))
    client = KalshiClient(manifest, request_delay=cfg["request_delay"])

    cutoff = client.get_cutoff()
    cutoff_ts = iso_to_ts(cutoff["market_settled_ts"])
    log.info("historical cutoff: %s", cutoff["market_settled_ts"])

    if args.reuse_universe and os.path.exists(UNIVERSE_PATH):
        with open(UNIVERSE_PATH) as f:
            markets = [json.loads(line) for line in f]
    else:
        markets = enumerate_universe(client, cfg, cutoff_ts)

    est = estimate(markets, cfg["period_interval"], cfg["request_delay"])
    est["cutoff"] = cutoff["market_settled_ts"]
    est["enumeration_requests"] = client.n_requests
    est["as_of"] = datetime.fromtimestamp(now_ts(), timezone.utc).isoformat()
    with open(os.path.join(REPO_ROOT, "data", "manifests", "universe_estimate.json"), "w") as f:
        json.dump(est, f, indent=2)
    log.info("universe: %d markets %s, %d candle requests, ETA %.1f h",
             est["markets"], est["per_source"], est["candle_requests"], est["eta_hours_at_delay"])
    if args.dry_run:
        print(json.dumps(est, indent=2))
        return 0

    todo = sample(markets, args.limit) if args.limit else markets
    done = set()
    for mp in sorted(os.listdir(os.path.dirname(manifest.path))):
        if mp.startswith("kalshi_") and mp.endswith(".jsonl"):
            done |= Manifest(os.path.join(os.path.dirname(manifest.path), mp)).recorded_paths()
    for i, m in enumerate(todo, 1):
        try:
            client.fetch_candles(m, m["_source"], done, cfg["period_interval"])
        except Exception as e:  # one bad market must not stop a multi-hour pull
            log.warning("candles failed for %s: %s", m["ticker"], e)
        if i % 100 == 0 or i == len(todo):
            log.info("candles: %d/%d markets, %d requests total", i, len(todo), client.n_requests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
