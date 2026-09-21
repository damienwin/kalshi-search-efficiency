"""Kalshi public market-data client. Writes every response verbatim.

Kalshi splits data at a moving cutoff (GET /historical/cutoff):
  historical  markets settled before the cutoff -> /historical/markets...
  live        markets settled after it          -> /markets..., /series/...

/historical/markets ignores time filters and returns newest-first across all
series (millions of multivariate combo markets), so the universe is
enumerated per series_ticker on both endpoints.

Every response is saved gzip-compressed under data/raw/kalshi/ and recorded in
a hash manifest before it is parsed. Parsing lives in ksearch.data.parse.
"""

import gzip
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from ksearch.data.manifest import Manifest, REPO_ROOT

log = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MAX_CANDLES_PER_REQUEST = 5000  # API: "max candlesticks: 5000"
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "kalshi")


def iso_to_ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def series_of(market: dict) -> str:
    return market["event_ticker"].split("-")[0]


class KalshiClient:
    def __init__(self, manifest: Manifest, request_delay: float = 0.15, retries: int = 6):
        self.manifest = manifest
        self.request_delay = request_delay
        self.retries = retries
        self.session = requests.Session()
        self.n_requests = 0

    # ── transport ────────────────────────────────────────────────────────────
    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = BASE_URL + endpoint
        for attempt in range(self.retries):
            time.sleep(self.request_delay)
            self.n_requests += 1
            try:
                r = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                wait = 2 ** attempt
                log.warning("%s: %s; retry in %ds", endpoint, e.__class__.__name__, wait)
                time.sleep(wait)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = 2 ** attempt
                log.warning("%s: HTTP %d; retry in %ds", endpoint, r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"gave up on {endpoint} {params} after {self.retries} attempts")

    def _get_saved(self, endpoint: str, params: dict, out_path: str) -> dict:
        """GET, write the raw body to out_path (gzip JSON), record it, return it."""
        data = self._get(endpoint, params)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with gzip.open(out_path, "wt") as f:
            json.dump(data, f)
        self.manifest.record(out_path, endpoint, params)
        return data

    # ── enumeration ──────────────────────────────────────────────────────────
    def get_cutoff(self) -> dict:
        return self._get("/historical/cutoff")

    def list_series_markets(self, series: str, source: str, min_close_ts: int, keep=None) -> list[dict]:
        """Markets in one series from one source that pass `keep`, saving every page.

        Filtering happens per page so memory stays flat on series with hundreds
        of thousands of markets (e.g. hourly crypto ranges). historical pages
        come newest-first; paging stops once a full page closed before
        min_close_ts. live uses server-side min_close_ts.
        """
        if source == "historical":
            endpoint, params = "/historical/markets", {"series_ticker": series, "limit": 1000}
        else:
            endpoint = "/markets"
            params = {"series_ticker": series, "status": "settled", "limit": 1000,
                      "min_close_ts": min_close_ts}
        markets, cursor, page = [], None, 0
        while True:
            p = dict(params, **({"cursor": cursor} if cursor else {}))
            out = os.path.join(RAW_DIR, source, "markets", series, f"page_{page:04d}.json.gz")
            data = self._get_saved(endpoint, p, out)
            batch = data.get("markets", [])
            markets.extend(m for m in batch if keep is None or keep(m))
            cursor = data.get("cursor") or None
            page += 1
            if not cursor or not batch:
                break
            if source == "historical" and all(iso_to_ts(m["close_time"]) < min_close_ts for m in batch):
                break
        return markets

    # ── candles ──────────────────────────────────────────────────────────────
    @staticmethod
    def candle_windows(open_ts: int, close_ts: int, period_min: int = 60) -> list[tuple[int, int]]:
        """Split [open, close] into request windows under the 5000-candle cap."""
        span = MAX_CANDLES_PER_REQUEST * period_min * 60 - period_min * 60
        windows, start = [], open_ts
        while start < close_ts:
            end = min(start + span, close_ts)
            windows.append((start, end))
            start = end
        return windows

    def candle_path(self, market: dict, source: str, start_ts: int) -> str:
        return os.path.join(RAW_DIR, source, "candles", series_of(market),
                            f"{market['ticker']}__{start_ts}.json.gz")

    def fetch_candles(self, market: dict, source: str, done: set, period_min: int = 60) -> int:
        """Fetch and save all candle windows for a market. Returns requests made."""
        made = 0
        open_ts, close_ts = iso_to_ts(market["open_time"]), iso_to_ts(market["close_time"])
        for start, end in self.candle_windows(open_ts, close_ts, period_min):
            out = self.candle_path(market, source, start)
            if os.path.relpath(out, REPO_ROOT) in done and os.path.exists(out):
                continue
            params = {"period_interval": period_min, "start_ts": start, "end_ts": end}
            if source == "historical":
                endpoint = f"/historical/markets/{market['ticker']}/candlesticks"
            else:
                endpoint = f"/series/{series_of(market)}/markets/{market['ticker']}/candlesticks"
            self._get_saved(endpoint, params, out)
            made += 1
        return made


def load_gz(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())
