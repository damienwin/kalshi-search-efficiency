"""Import the Spring Guardian cache as a hashed, deduplicated seed.

    python scripts/import_spring_news.py \
        --cache ../kalshi-sentiment-predictor/data/guardian_cache.json \
        --markets ../kalshi-sentiment-predictor/data/raw/markets.json \
                  ../kalshi-sentiment-predictor/data/raw/markets_full_backup.json

The Spring cache maps "query|from|to" to processed article lists (the raw API
pages were not kept). This writes, under data/raw/guardian/spring_seed/:
  articles.jsonl.gz   one line per unique URL
  searches.jsonl.gz   {key, query, from, to, urls}
  spring_markets.jsonl  for each Spring market: its title, reconstructed query,
                        and the cached search key that query maps to
and records the source file's sha256 alongside the outputs in the manifest.
"""

import argparse
import gzip
import json
import os
import sys
from datetime import datetime, timezone

import ijson

from ksearch.data.manifest import Manifest, REPO_ROOT, sha256_file
from ksearch.data.news import RAW_DIR, market_query

SEED = os.path.join(RAW_DIR, "spring_seed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--markets", nargs="+", required=True,
                    help="Spring market lists; earlier files win on duplicate tickers")
    args = ap.parse_args()
    os.makedirs(SEED, exist_ok=True)

    seen, searches = set(), []
    art_path = os.path.join(SEED, "articles.jsonl.gz")
    with open(args.cache, "rb") as src, gzip.open(art_path, "wt") as out:
        for key, arts in ijson.kvitems(src, ""):
            q, frm, to = key.rsplit("|", 2)
            urls = []
            for a in arts:
                urls.append(a["url"])
                if a["url"] not in seen:
                    seen.add(a["url"])
                    out.write(json.dumps(a) + "\n")
            searches.append({"key": key, "query": q, "from": frm, "to": to, "urls": urls})
    search_path = os.path.join(SEED, "searches.jsonl.gz")
    with gzip.open(search_path, "wt") as f:
        for s in searches:
            f.write(json.dumps(s) + "\n")

    with open(os.path.join(REPO_ROOT, "config", "spring_markets.txt")) as f:
        spring = [t.strip() for t in f if t.strip() and not t.startswith("#")]
    by_ticker = {}
    for path in args.markets:
        with open(path) as f:
            for m in json.load(f):
                by_ticker.setdefault(m["ticker"], m)
    by_query = {}
    for s in searches:
        by_query.setdefault(s["query"], []).append(s["key"])
    rows, unmatched = [], []
    for t in spring:
        m = by_ticker.get(t, {})
        q = market_query(m) if m else None
        keys = by_query.get(q, []) if q else []
        rows.append({"ticker": t, "title": m.get("title"), "query": q, "keys": keys})
        if not keys:
            unmatched.append(t)
    map_path = os.path.join(SEED, "spring_markets.jsonl")
    with open(map_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    man = Manifest(os.path.join(REPO_ROOT, "data", "manifests", f"guardian_{stamp}.jsonl"))
    src_meta = {"source": os.path.basename(args.cache), "source_sha256": sha256_file(args.cache),
                "markets_sha256": {os.path.basename(p): sha256_file(p) for p in args.markets}}
    for p in (art_path, search_path, map_path):
        man.record(p, "import/spring_cache", src_meta)

    multi = sum(len(r["keys"]) > 1 for r in rows)
    print(json.dumps({"searches": len(searches), "unique_articles": len(seen),
                      "spring_markets": len(spring), "matched": len(spring) - len(unmatched),
                      "matched_multiple_keys": multi, "unmatched_examples": unmatched[:5],
                      **src_meta}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
