"""Leakage tests L1, L2, L4, L5, L6 on synthetic markets."""

import collections

import numpy as np
import pandas as pd
import pytest

from ksearch.data.grid import HOUR, MarketCandles, classify
from ksearch.data.labels import build_market_events, candidate_t0s, events_frame
from ksearch.features.price import PRICE_FEATURES, event_features

CFG = {"horizon_h": 4, "move_threshold": 0.03, "max_staleness_h": 2,
       "max_spread": 0.20, "settle_buffer_h": 4, "min_event_spacing_h": 4}
OPEN = 1_700_000_000 - (1_700_000_000 % (4 * HOUR))  # grid-aligned
CLOSE = OPEN + 10 * 24 * HOUR
MARKET = {"ticker": "KXTEST-26JAN01-T50", "event_ticker": "KXTEST-26JAN01"}


def synthetic_candles(seed: int = 0, gaps: bool = True) -> pd.DataFrame:
    """Hourly random-walk book with occasional gaps and one-sided hours."""
    rng = np.random.default_rng(seed)
    ts = np.arange(OPEN + HOUR, CLOSE + 1, HOUR)
    if gaps:
        ts = ts[rng.random(len(ts)) > 0.2]
    mid = np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, len(ts))), 0.1, 0.9)
    half = rng.uniform(0.005, 0.03, len(ts))
    bid, ask = mid - half, mid + half
    bid[rng.random(len(ts)) < 0.05] = 0.0  # empty bid side
    return pd.DataFrame({
        "end_period_ts": ts, "bid_close": bid.round(4), "ask_close": ask.round(4),
        "volume": rng.integers(0, 50, len(ts)).astype(float),
        "open_interest": np.cumsum(rng.integers(0, 10, len(ts))).astype(float),
    })


def build(candles):
    mc = MarketCandles(candles)
    att = collections.Counter()
    ev = events_frame(build_market_events(mc, MARKET, OPEN, CLOSE, CFG, att))
    return mc, ev, att


def feats(mc, ev, k):
    rows = ev.to_dict("records")
    prev = rows[k - 1] if k > 0 else None
    return event_features(mc, int(rows[k]["t0"]), OPEN, CLOSE, CFG["max_staleness_h"], prev)


def same(a: dict, b: dict) -> bool:
    return all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in PRICE_FEATURES)


# ── L1: future candles cannot change features ────────────────────────────────
@pytest.mark.parametrize("seed", range(5))
def test_L1_future_perturbation_leaves_features_unchanged(seed):
    candles = synthetic_candles(seed)
    mc, ev, _ = build(candles)
    assert len(ev) > 5
    rng = np.random.default_rng(100 + seed)
    for k in rng.choice(np.arange(1, len(ev)), size=min(10, len(ev) - 1), replace=False):
        t0 = int(ev.loc[k, "t0"])
        before = feats(mc, ev, k)

        future = candles["end_period_ts"] > t0
        pert = candles.copy()
        pert.loc[future, ["bid_close", "ask_close"]] = rng.uniform(0.01, 0.99, (future.sum(), 2))
        pert.loc[future, ["volume", "open_interest"]] = rng.uniform(0, 1e6, (future.sum(), 2))
        mc2, ev2, _ = build(pert)

        # Every event that resolved by t0 is identical, so prev_label is too
        resolved = ev[ev.t_end <= t0].reset_index(drop=True)
        resolved2 = ev2[ev2.t_end <= t0].reset_index(drop=True)
        pd.testing.assert_frame_equal(resolved, resolved2)
        assert same(before, feats(mc2, ev, k)), f"feature leak at event {k}"


# ── L2: a candle is invisible until its end_period_ts ────────────────────────
def test_L2_candle_not_observable_before_it_closes():
    t0 = OPEN + 40 * HOUR
    base = pd.DataFrame({"end_period_ts": [t0 - HOUR], "bid_close": [0.40], "ask_close": [0.42],
                         "volume": [1.0], "open_interest": [1.0]})
    late = pd.DataFrame({"end_period_ts": [t0 + 60], "bid_close": [0.80], "ask_close": [0.82],
                         "volume": [9.0], "open_interest": [9.0]})
    mc = MarketCandles(pd.concat([base, late]))
    assert mc.quote_at(t0, 2).mid == pytest.approx(0.41)
    f = event_features(mc, t0, OPEN, CLOSE, 2, None)
    assert f["mid"] == pytest.approx(0.41) and f["log_volume_24h"] == pytest.approx(np.log1p(1.0))


# ── L3 (label side): no event uses a zero or one-sided quote ────────────────
@pytest.mark.parametrize("seed", range(5))
def test_L3_no_event_uses_one_sided_or_zero_quote(seed):
    _, ev, att = build(synthetic_candles(seed))
    assert (ev[["mid_t0", "mid_end"]] > 0).all().all()
    assert (ev[["spread_t0", "spread_end"]] > 0).all().all()
    assert att["t0_one_sided"] + att["end_one_sided"] > 0  # the fixture has empty bids


def test_classify_quotes():
    assert classify(0.40, 0.42) == "valid"
    assert classify(0.0, 0.42) == "one_sided"
    assert classify(0.40, 1.0) == "one_sided"
    assert classify(0.45, 0.42) == "crossed"
    assert classify(np.nan, 0.42) == "no_quote"


# ── L4 / L5: exact 4h windows on a fixed grid, never overlapping ────────────
@pytest.mark.parametrize("seed", range(5))
def test_L4_L5_windows_exact_and_non_overlapping(seed):
    _, ev, _ = build(synthetic_candles(seed))
    assert ((ev.t_end - ev.t0) == 4 * HOUR).all()
    assert (ev.t0 % (4 * HOUR) == 0).all()
    assert (ev.t0.diff().dropna() >= 4 * HOUR).all()


def test_grid_is_data_independent():
    # The same candidate t0s whatever the candles contain
    a = build(synthetic_candles(1))[1]
    b = build(synthetic_candles(2))[1]
    grid = set(candidate_t0s(OPEN, CLOSE, 4))
    assert set(a.t0) <= grid and set(b.t0) <= grid


def test_attrition_accounts_for_every_candidate():
    _, ev, att = build(synthetic_candles(3))
    dropped = sum(v for k, v in att.items() if k not in ("candidates", "events"))
    assert att["events"] == len(ev)
    assert att["events"] + dropped == att["candidates"]


def test_stale_quote_is_not_a_price():
    t0 = OPEN + 40 * HOUR
    c = pd.DataFrame({"end_period_ts": [t0 - 3 * HOUR], "bid_close": [0.40], "ask_close": [0.42],
                      "volume": [1.0], "open_interest": [1.0]})
    assert MarketCandles(c).quote_at(t0, 2).state == "stale"


# ── L6: prev_label only once the previous event has resolved ─────────────────
def test_L6_prev_label_requires_resolution():
    mc, ev, _ = build(synthetic_candles(0))
    rows = ev.to_dict("records")
    t0 = int(rows[1]["t0"])
    unresolved = dict(rows[0], t_end=t0 + HOUR)
    assert np.isnan(event_features(mc, t0, OPEN, CLOSE, 2, unresolved)["prev_label"])
    assert not np.isnan(event_features(mc, t0, OPEN, CLOSE, 2, rows[0])["prev_label"])
