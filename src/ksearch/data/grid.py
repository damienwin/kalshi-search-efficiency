"""Point-in-time view of one market's candles.

A candle's values become observable at its end_period_ts, never before (L2).
`quote_at(t)` answers "what was the book at time t" using only candles with
end_period_ts <= t, so every caller is point-in-time by construction.

Candles are sparse: Kalshi emits one only for hours with activity, so the
latest candle's quote is taken to stand until the next one, up to a staleness
limit.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOUR = 3600

# Quote states, used by the label builder's attrition report
VALID, NO_QUOTE, STALE, ONE_SIDED, CROSSED = "valid", "no_quote", "stale", "one_sided", "crossed"


@dataclass
class Quote:
    state: str
    bid: float = np.nan
    ask: float = np.nan
    age_h: float = np.nan

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.state == VALID else np.nan

    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.state == VALID else np.nan


def classify(bid: float, ask: float) -> str:
    """Joint-quote validity for one candle: 0 < bid < ask < 1, both from the same candle."""
    if np.isnan(bid) or np.isnan(ask):
        return NO_QUOTE
    if bid <= 0 or ask >= 1:
        return ONE_SIDED  # 0.00 bid = empty bid side; 1.00 ask = empty ask side
    if bid >= ask:
        return CROSSED
    return VALID


class MarketCandles:
    def __init__(self, candles: pd.DataFrame):
        c = candles.sort_values("end_period_ts", kind="mergesort").drop_duplicates("end_period_ts", keep="last")
        self.ts = c["end_period_ts"].to_numpy(dtype=np.int64)
        self.bid = c["bid_close"].to_numpy(dtype=float)
        self.ask = c["ask_close"].to_numpy(dtype=float)
        self.volume = c["volume"].to_numpy(dtype=float)
        self.oi = c["open_interest"].to_numpy(dtype=float)
        self.state = np.array([classify(b, a) for b, a in zip(self.bid, self.ask)], dtype=object)
        self.valid = self.state == VALID
        self.mid = np.where(self.valid, (self.bid + self.ask) / 2, np.nan)

    def last_index(self, t: int) -> int:
        """Index of the latest candle observable at t (end_period_ts <= t), or -1."""
        return int(np.searchsorted(self.ts, t, side="right")) - 1

    def quote_at(self, t: int, max_staleness_h: float) -> Quote:
        i = self.last_index(t)
        if i < 0:
            return Quote(NO_QUOTE)
        age_h = (t - self.ts[i]) / HOUR
        if age_h > max_staleness_h:
            return Quote(STALE, age_h=age_h)
        return Quote(self.state[i], self.bid[i], self.ask[i], age_h)

    def last_valid_mid(self, t: int, max_age_h: float) -> float:
        """Latest valid mid observable at t and no older than max_age_h."""
        i = self.last_index(t)
        lo = int(np.searchsorted(self.ts, t - max_age_h * HOUR, side="left"))
        while i >= lo:
            if self.valid[i]:
                return float(self.mid[i])
            i -= 1
        return np.nan

    def window(self, t: int, hours: float) -> slice:
        """Candles with t - hours < end_period_ts <= t."""
        lo = int(np.searchsorted(self.ts, t - hours * HOUR, side="right"))
        hi = int(np.searchsorted(self.ts, t, side="right"))
        return slice(lo, hi)
