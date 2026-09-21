"""Price-derived features (Plan of Study item 1): bid-ask proxies, momentum,
volatility, activity, schedule.

Every feature for an event at t0 reads only candles with end_period_ts <= t0,
through MarketCandles, so future candles cannot reach it (tests L1, L2).
"""

import numpy as np
import pandas as pd

from ksearch.data.grid import HOUR, MarketCandles
from ksearch.data.labels import LABEL_VALUES

PRICE_FEATURES = [
    # bid-ask proxies
    "mid", "spread", "rel_spread", "dist_from_half",
    # momentum
    "ret_1h", "ret_4h", "ret_24h", "prev_label",
    # volatility
    "vol_24h", "range_24h",
    # activity
    "log_volume_24h", "active_hours_24h", "oi_change_24h",
    # schedule
    "hours_since_open", "hours_to_close",
]

# hours_to_close reads close_time, which Kalshi can revise after the fact.
# It stays out of the default set until the L7 ablation clears it.
PROVISIONAL_FEATURES = {"hours_to_close"}


def _last_value(arr: np.ndarray, i: int) -> float:
    return float(arr[i]) if i >= 0 else np.nan


def event_features(mc: MarketCandles, t0: int, open_ts: int, close_ts: int,
                   max_staleness_h: float, prev: dict | None) -> dict:
    q = mc.quote_at(t0, max_staleness_h)
    mid = q.mid

    def ret(hours: float) -> float:
        past = mc.last_valid_mid(t0 - int(hours * HOUR), max_age_h=max(hours, max_staleness_h))
        return mid - past

    w = mc.window(t0, 24)
    mids = mc.mid[w]
    mids = mids[~np.isnan(mids)]
    vols = mc.volume[w]

    i_now, i_past = mc.last_index(t0), mc.last_index(t0 - 24 * HOUR)

    # prev_label is usable only if the previous event had resolved by t0 (L6)
    prev_label = np.nan
    if prev is not None and prev["t_end"] <= t0:
        prev_label = float(LABEL_VALUES[prev["label"]])

    return {
        "mid": mid,
        "spread": q.spread,
        "rel_spread": q.spread / mid if mid and not np.isnan(mid) else np.nan,
        "dist_from_half": abs(mid - 0.5),
        "ret_1h": ret(1),
        "ret_4h": ret(4),
        "ret_24h": ret(24),
        "prev_label": prev_label,
        "vol_24h": float(np.std(np.diff(mids))) if len(mids) >= 3 else np.nan,
        "range_24h": float(mids.max() - mids.min()) if len(mids) else np.nan,
        "log_volume_24h": float(np.log1p(np.nansum(vols))),
        "active_hours_24h": float(np.sum(np.nan_to_num(vols) > 0)),
        "oi_change_24h": _last_value(mc.oi, i_now) - _last_value(mc.oi, i_past),
        "hours_since_open": (t0 - open_ts) / HOUR,
        "hours_to_close": (close_ts - t0) / HOUR,
    }


def market_features(mc: MarketCandles, events: pd.DataFrame, open_ts: int, close_ts: int,
                    max_staleness_h: float) -> pd.DataFrame:
    """Features for one market's events (sorted by t0)."""
    rows, prev = [], None
    for ev in events.sort_values("t0").to_dict("records"):
        rows.append(event_features(mc, int(ev["t0"]), open_ts, close_ts, max_staleness_h, prev))
        prev = ev
    return pd.DataFrame(rows, columns=PRICE_FEATURES, index=events.sort_values("t0").index)
