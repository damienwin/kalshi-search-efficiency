"""Guardian news: the Spring 2026 per-market query method, with raw pages kept.

Method (unchanged from Spring, kalshi-sentiment-predictor src/data/news.py):
  query   " ".join(keywords[:3]), keywords from the market title (market_keywords)
  window  one search per market, from (first event t0 - lookback) to last t0
  search  show-fields=body,headline,trailText, newest first, 50/page, max 20 pages
  assign  an event at t0 gets the articles published in [t0 - lookback, t0)

Changes from Spring: every API page is saved verbatim (gzip) and hashed in a
manifest instead of being merged into one rewritten JSON cache, and events
with fewer than 2 articles are kept (sentiment is NaN) rather than dropped.
"""

import gzip
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests

from ksearch.data.manifest import Manifest, REPO_ROOT

log = logging.getLogger(__name__)

GUARDIAN_URL = "https://content.guardianapis.com/search"
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "guardian")
USAGE_FILE = os.path.join(REPO_ROOT, "data", "build", "guardian_usage.json")

# Ported verbatim from DatasetBuilder.generate_keywords (Spring)
STOPWORDS = {
    "will", "the", "a", "an", "in", "by", "of", "to", "be", "is", "are",
    "was", "were", "at", "on", "for", "with", "or", "and", "not", "no",
    "it", "its", "this", "that", "than", "more", "less", "if", "which",
    "who", "when", "where", "how", "what", "as", "from", "up", "about",
    "into", "through", "during", "before", "after", "between", "out",
    "do", "does", "did", "can", "could", "would", "should", "may",
}


def market_keywords(market: dict) -> list[str]:
    """2-4 search phrases from a market's title and category (Spring logic, verbatim)."""
    title = market.get("title", "")
    category = market.get("category", "")
    tokens = []
    for w in title.split():
        cleaned = re.sub(r"[^\w]", "", w)
        if cleaned and cleaned.lower() not in STOPWORDS and len(cleaned) > 1 and not cleaned.isdigit():
            tokens.append(cleaned)
    phrases = [f"{tokens[j]} {tokens[j + 1]}" for j in range(len(tokens) - 1)]
    if len(phrases) < 2 and tokens:
        phrases.extend(tokens[:3])
    if category and category.lower() not in " ".join(phrases).lower():
        phrases.append(category)
    return phrases[:4]


def market_query(market: dict) -> str:
    return " ".join(market_keywords(market)[:3])


def search_window(t0s, lookback_h: float) -> tuple[str, str]:
    """(from_date, to_date) covering every event's lookback window, as Spring did."""
    first = datetime.fromtimestamp(min(t0s), timezone.utc) - timedelta(hours=lookback_h)
    last = datetime.fromtimestamp(max(t0s), timezone.utc)
    return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")


def search_key(query: str, from_date: str, to_date: str) -> str:
    return f"{query}|{from_date}|{to_date}"


def parse_article(item: dict) -> dict:
    from bs4 import BeautifulSoup  # optional dependency, only needed to parse

    fields = item.get("fields", {})
    text = BeautifulSoup(fields.get("body", "") or "", "lxml").get_text(" ", strip=True)
    if not text:
        text = BeautifulSoup(fields.get("trailText", "") or "", "lxml").get_text(" ", strip=True)
    return {"title": item.get("webTitle", ""), "text": text, "date": item.get("webPublicationDate", ""),
            "url": item.get("webUrl", ""), "source": "guardian"}


def in_window(article_ts, t0: int, lookback_h: float):
    """Published in [t0 - lookback, t0): strictly before t0 (L12). Works on arrays."""
    return (article_ts >= t0 - lookback_h * 3600) & (article_ts < t0)


def assign_articles(articles: list[dict], t0: int, lookback_h: float) -> list[dict]:
    """Articles published in [t0 - lookback, t0), deduplicated by URL (L12)."""
    out, seen = [], set()
    for a in articles:
        if not a.get("text") or not a.get("date"):
            continue
        ts = datetime.fromisoformat(a["date"].replace("Z", "+00:00")).timestamp()
        if in_window(ts, t0, lookback_h) and a.get("url") not in seen:
            seen.add(a.get("url"))
            out.append(a)
    return out


class UsageLimiter:
    """Per-key daily request counts (UTC), persisted; picks the least-used key."""

    def __init__(self, n_keys: int, daily_cap: int, path: str = USAGE_FILE):
        self.n_keys, self.daily_cap, self.path = n_keys, daily_cap, path

    def _read(self) -> dict:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with open(self.path) as f:
                d = json.load(f)
            if d.get("date") == today:
                return d
        except (OSError, json.JSONDecodeError):
            pass
        return {"date": today, "counts": [0] * self.n_keys}

    def take(self) -> int | None:
        """Reserve one request on the least-used key; None if every key is at the cap."""
        d = self._read()
        k = min(range(self.n_keys), key=lambda i: d["counts"][i])
        if d["counts"][k] >= self.daily_cap:
            return None
        d["counts"][k] += 1
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(d, f)
        return k


class QuotaExhausted(Exception):
    pass


class GuardianError(Exception):
    """An API error whose message is safe to log (no key, no signed URL)."""


class GuardianClient:
    def __init__(self, api_keys: list[str], manifest: Manifest, daily_cap: int = 450,
                 delay: float = 1.1, max_pages: int = 20):
        if not api_keys:
            raise ValueError("no Guardian API keys; set GUARDIAN_API_KEY in .env")
        self.keys, self.manifest, self.delay, self.max_pages = api_keys, manifest, delay, max_pages
        self.limiter = UsageLimiter(len(api_keys), daily_cap)

    @staticmethod
    def key_dir(key: str) -> str:
        return os.path.join(RAW_DIR, "api", hashlib.sha1(key.encode()).hexdigest()[:16])

    def _get(self, params: dict) -> dict:
        for attempt in range(4):
            k = self.limiter.take()
            if k is None:
                raise QuotaExhausted("all Guardian keys at the daily cap")
            time.sleep(self.delay)
            try:
                r = requests.get(GUARDIAN_URL, params=dict(params, **{"api-key": self.keys[k]}), timeout=20)
            except requests.RequestException as e:
                log.warning("guardian %s; retry", e.__class__.__name__)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429:
                time.sleep(60 * 2 ** attempt)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                # Never raise r.raise_for_status(): its message embeds the URL, api-key included
                raise GuardianError(f"HTTP {r.status_code} on key #{k + 1} for q={params.get('q')!r} "
                                    f"page {params.get('page')}" + (" (key invalid or expired)" if r.status_code in (401, 403) else ""))
            return r.json()
        raise RuntimeError(f"guardian gave up on {params.get('q')!r} page {params.get('page')}")

    def search(self, query: str, from_date: str, to_date: str) -> list[dict]:
        """All pages for one search, fetching only pages not already on disk."""
        key = search_key(query, from_date, to_date)
        d = self.key_dir(key)
        items, page, pages = [], 1, 1
        while page <= min(pages, self.max_pages):
            path = os.path.join(d, f"page_{page:03d}.json.gz")
            if os.path.exists(path):
                with gzip.open(path, "rt") as f:
                    data = json.load(f)
            else:
                params = {"q": query, "from-date": from_date, "to-date": to_date, "page": page,
                          "page-size": 50, "order-by": "newest", "show-fields": "body,headline,trailText"}
                data = self._get(params)
                os.makedirs(d, exist_ok=True)
                with gzip.open(path, "wt") as f:
                    json.dump(data, f)
                self.manifest.record(path, "guardian/search", {"key": key, "page": page})
            resp = data.get("response", {})
            items.extend(resp.get("results", []))
            pages = resp.get("pages", 1) or 1
            page += 1
        with open(os.path.join(d, "key.txt"), "w") as f:
            f.write(key + "\n")
        return [parse_article(i) for i in items]
