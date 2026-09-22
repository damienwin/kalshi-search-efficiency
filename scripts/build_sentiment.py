"""Sentiment features for v1 events, by the Spring per-market query method.

    python scripts/build_sentiment.py --build-dir data/build/spring_only [--model finbert]

For each market in <build-dir>/universe.jsonl:
  articles  Spring markets: the union of the cached Spring searches for the
            market's reconstructed query (flagged provisional_sentiment=1).
            Other markets: the market's own Guardian search, fetched by
            scripts/pull_news.py.
  assign    an event at t0 gets the articles published in [t0 - 6h, t0)   (L12)
  features  Spring's SentimentFeatureEngineer on the pre-computed article scores
Event list comes from <build-dir>/events_index.parquet (no labels), so the
sealed slice is never opened. Writes <build-dir>/sentiment.parquet keyed by
(market_ticker, t0) for dev and sealed events alike, and prints coverage.
"""

import argparse
import gzip
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from ksearch.data.manifest import REPO_ROOT
from ksearch.data.news import RAW_DIR, GuardianClient, in_window, market_query, search_key, search_window
from ksearch.features.sentiment import SENTIMENT_FEATURES, SentimentFeatureEngineer

log = logging.getLogger("build_sentiment")
SEED = os.path.join(RAW_DIR, "spring_seed")


def load_seed_index() -> tuple[dict, dict]:
    """ticker -> [urls] for Spring markets (union over the query's cached searches)."""
    urls_by_key = {}
    with gzip.open(os.path.join(SEED, "searches.jsonl.gz"), "rt") as f:
        for line in f:
            s = json.loads(line)
            urls_by_key[s["key"]] = s["urls"]
    by_ticker = {}
    with open(os.path.join(SEED, "spring_markets.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            seen, urls = set(), []
            for k in r["keys"]:
                for u in urls_by_key[k]:
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)
            by_ticker[r["ticker"]] = urls
    return by_ticker, urls_by_key


def api_urls(market: dict, t0s, lookback_h: float) -> list[str] | None:
    """URLs from a market's own fetched Guardian search, or None if not fetched yet."""
    frm, to = search_window(t0s, lookback_h)
    d = GuardianClient.key_dir(search_key(market_query(market), frm, to))
    if not os.path.isdir(d):
        return None
    urls = []
    for p in sorted(os.listdir(d)):
        if p.startswith("page_"):
            with gzip.open(os.path.join(d, p), "rt") as f:
                urls += [i.get("webUrl", "") for i in json.load(f).get("response", {}).get("results", [])]
    return urls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--model", default="finbert")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        lookback_h = yaml.safe_load(f).get("news", {}).get("lookback_h", 6)

    events = pd.read_parquet(os.path.join(args.build_dir, "events_index.parquet"))
    with open(os.path.join(args.build_dir, "universe.jsonl")) as f:
        markets = {m["ticker"]: m for m in map(json.loads, f)}

    scores = pd.read_parquet(os.path.join(REPO_ROOT, "data", "build", f"scores_{args.model}.parquet"))
    scores = scores.drop_duplicates("url").set_index("url")
    seed_urls, _ = load_seed_index()
    eng = SentimentFeatureEngineer()

    rows, status = [], {"spring_seed": 0, "api": 0, "no_search": 0}
    for ticker, ev in events.groupby("market_ticker"):
        m = markets[ticker]
        if m.get("_scope") == "spring" and ticker in seed_urls:
            urls, provisional = seed_urls[ticker], 1
            status["spring_seed"] += 1
        else:
            urls, provisional = api_urls(m, ev.t0.tolist(), lookback_h), 0
            status["api" if urls is not None else "no_search"] += 1
            urls = urls or []
        known = scores.reindex([u for u in urls if u in scores.index])
        art_ts = np.array([datetime.fromisoformat(d.replace("Z", "+00:00")).timestamp() for d in known["date"]])
        for t0 in ev.t0:
            window = known[in_window(art_ts, t0, lookback_h)] if len(known) else known
            arts = [dict(r, date=r["date"]) for r in window.reset_index().to_dict("records")]
            iso = datetime.fromtimestamp(int(t0), timezone.utc).isoformat()
            feats = {**eng.compute_event_features(arts), **eng.compute_temporal_features(arts, iso)}
            rows.append({"market_ticker": ticker, "t0": int(t0), "n_articles": len(arts),
                         "provisional_sentiment": provisional, **feats})

    df = pd.DataFrame(rows).sort_values(["market_ticker", "t0"], kind="mergesort").reset_index(drop=True)
    # No articles means sentiment is absent, not neutral: blank it before the
    # cross-event shock so zeros never enter another event's rolling history.
    no_news = df["n_articles"] == 0
    blank = [c for c in SENTIMENT_FEATURES if c not in ("event_hour", "sentiment_shock")]
    df.loc[no_news, blank] = np.nan
    df = df.rename(columns={"t0": "event_timestamp"})
    df = eng.compute_cross_event_features(df).rename(columns={"event_timestamp": "t0"})
    df.loc[no_news, "sentiment_shock"] = np.nan
    out = os.path.join(args.build_dir, "sentiment.parquet")
    df.to_parquet(out, index=False)
    print(json.dumps({"events": len(df), "with_>=1_article": int((~no_news).sum()),
                      "with_>=2_articles (Spring rule)": int((df.n_articles >= 2).sum()),
                      "markets_by_source": status}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
