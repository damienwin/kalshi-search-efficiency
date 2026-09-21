"""Score every unique article once with a frozen sentiment model.

    python scripts/score_articles.py [--model finbert] [--limit N]

Reads   data/raw/guardian/spring_seed/articles.jsonl.gz and any fetched API pages
Writes  data/build/scores_<model>.parquet   (url, date, label, positive, negative, neutral, confidence)
Already-scored URLs are skipped, so new articles can be added incrementally.
"""

import argparse
import glob
import gzip
import json
import logging
import os
import sys

import pandas as pd

from ksearch.data.manifest import REPO_ROOT
from ksearch.data.news import RAW_DIR, parse_article
from ksearch.model.sentiment import SentimentScorer

log = logging.getLogger("score_articles")


def iter_articles():
    with gzip.open(os.path.join(RAW_DIR, "spring_seed", "articles.jsonl.gz"), "rt") as f:
        for line in f:
            yield json.loads(line)
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "api", "*", "page_*.json.gz"))):
        with gzip.open(path, "rt") as f:
            for item in json.load(f).get("response", {}).get("results", []):
                yield parse_article(item)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="finbert")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = os.path.join(REPO_ROOT, "data", "build", f"scores_{args.model}.parquet")
    done = pd.read_parquet(out) if os.path.exists(out) else pd.DataFrame()
    seen = set(done["url"]) if len(done) else set()

    todo, urls = [], set()
    for a in iter_articles():
        if a.get("text") and a["url"] not in seen and a["url"] not in urls:
            urls.add(a["url"])
            todo.append(a)
    if args.limit:
        todo = todo[:args.limit]
    log.info("%d already scored, %d to score", len(seen), len(todo))
    if not todo:
        return 0

    scorer = SentimentScorer(args.model)
    frames = [done] if len(done) else []
    for i in range(0, len(todo), args.chunk):
        batch = todo[i:i + args.chunk]
        scores = scorer.score_texts([a["text"] for a in batch])
        frames.append(pd.DataFrame([{"url": a["url"], "date": a["date"], **s} for a, s in zip(batch, scores)]))
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)  # checkpoint each chunk
        log.info("scored %d/%d", min(i + args.chunk, len(todo)), len(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
