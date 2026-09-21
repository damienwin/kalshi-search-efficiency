"""v1 labels: direction of the bid/ask mid over t0 -> t0 + horizon real hours.

Candidate t0s sit on a fixed UTC grid (00:00, 04:00, ... for a 4h horizon).
The grid is chosen independently of the data, so whether one event is valid
never moves where the next one starts, and events in a market never overlap.

A candidate becomes an event only if it passes every filter, checked in
this order (the first failure is recorded in the attrition report):
  before_open   t0 earlier than the market's open_time
  near_close    t0 + horizon later than close_time - settle_buffer
  t0_<state>    no valid quote at t0         (no_quote / stale / one_sided / crossed)
  end_<state>   no valid quote at t0+horizon
  wide_spread   spread at either end above max_spread
"""

import collections

import numpy as np
import pandas as pd

from ksearch.data.grid import HOUR, VALID, MarketCandles

LABEL_VALUES = {"DOWN": -1, "FLAT": 0, "UP": 1}


def candidate_t0s(open_ts: int, close_ts: int, horizon_h: int) -> np.ndarray:
    step = horizon_h * HOUR
    first = -(-open_ts // step) * step  # ceil to grid
    return np.arange(first, close_ts, step, dtype=np.int64)


def label_of(dmid: float, threshold: float) -> str:
    if dmid > threshold:
        return "UP"
    if dmid < -threshold:
        return "DOWN"
    return "FLAT"


def build_market_events(mc: MarketCandles, market: dict, open_ts: int, close_ts: int,
                        cfg: dict, attrition: collections.Counter) -> list[dict]:
    h = cfg["horizon_h"]
    stale = cfg["max_staleness_h"]
    rows = []
    for t0 in candidate_t0s(open_ts, close_ts, h):
        t1 = int(t0 + h * HOUR)
        attrition["candidates"] += 1
        if t1 > close_ts - cfg["settle_buffer_h"] * HOUR:
            attrition["near_close"] += 1
            continue
        q0 = mc.quote_at(int(t0), stale)
        if q0.state != VALID:
            attrition[f"t0_{q0.state}"] += 1
            continue
        q1 = mc.quote_at(t1, stale)
        if q1.state != VALID:
            attrition[f"end_{q1.state}"] += 1
            continue
        if max(q0.spread, q1.spread) > cfg["max_spread"]:
            attrition["wide_spread"] += 1
            continue
        dmid = q1.mid - q0.mid
        attrition["events"] += 1
        rows.append({
            "market_ticker": market["ticker"],
            "event_group": market["event_ticker"],  # strike ladders share this (L9)
            "series": market["event_ticker"].split("-")[0],
            "t0": int(t0),
            "t_end": t1,
            "mid_t0": q0.mid,
            "mid_end": q1.mid,
            "dmid": round(dmid, 6),
            "spread_t0": q0.spread,
            "spread_end": q1.spread,
            "quote_age_t0_h": q0.age_h,
            "quote_age_end_h": q1.age_h,
            "label": label_of(dmid, cfg["move_threshold"]),
        })
    return rows


def events_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["t0", "market_ticker"], kind="mergesort").reset_index(drop=True)
