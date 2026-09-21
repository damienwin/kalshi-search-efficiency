"""Fetch Guardian searches for the non-Spring markets in a build (Spring method).

    python scripts/pull_news.py --build-dir data/build/v1 [--estimate]

One search per market: query from the title, window from the first event's
lookback to the last event (search_window). Pages already on disk are never
re-fetched, so the script resumes across days. It stops cleanly when both
keys reach the daily cap; re-run it the next UTC day.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

from ksearch.data.manifest import Manifest, REPO_ROOT
from ksearch.data.news import GuardianClient, QuotaExhausted, market_query, search_window

log = logging.getLogger("pull_news")


def load_env(path: str) -> list[str]:
    keys = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip().removeprefix("export ")
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    keys[k.strip()] = v.strip().strip('"')
    return [keys[k] for k in ("GUARDIAN_API_KEY", "GUARDIAN_API_KEY_2") if keys.get(k)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--sealed-dir", default=os.path.join(REPO_ROOT, "sealed"))
    ap.add_argument("--estimate", action="store_true", help="list pending searches, fetch nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f).get("news", {})
    lookback_h = cfg.get("lookback_h", 6)

    events = pd.concat([pd.read_parquet(os.path.join(args.build_dir, "dev.parquet")),
                        pd.read_parquet(os.path.join(args.sealed_dir, "sealed.parquet"))])
    with open(os.path.join(args.build_dir, "universe.jsonl")) as f:
        markets = {m["ticker"]: m for m in map(json.loads, f)}
    todo = []
    for ticker, ev in events.groupby("market_ticker"):
        m = markets[ticker]
        if m.get("_scope") == "spring":
            continue  # covered by the imported Spring cache
        frm, to = search_window(ev.t0.tolist(), lookback_h)
        todo.append((ticker, market_query(m), frm, to))
    pending = [t for t in todo if not os.path.isdir(GuardianClient.key_dir(f"{t[1]}|{t[2]}|{t[3]}"))]
    log.info("%d market searches, %d not started", len(todo), len(pending))
    if args.estimate:
        for t in pending[:10]:
            print(t)
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    client = GuardianClient(load_env(os.path.join(REPO_ROOT, ".env")),
                            Manifest(os.path.join(REPO_ROOT, "data", "manifests", f"guardian_{stamp}.jsonl")),
                            daily_cap=cfg.get("daily_cap", 450))
    done = 0
    try:
        for ticker, q, frm, to in todo:
            client.search(q, frm, to)  # no-op for pages already on disk
            done += 1
    except QuotaExhausted:
        log.warning("daily quota reached after %d/%d searches; re-run tomorrow (UTC)", done, len(todo))
        return 0
    log.info("all %d searches complete", len(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
