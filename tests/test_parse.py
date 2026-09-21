"""L3: a missing value parses to NaN, never 0.0. Fixtures are real API responses."""

import math

from ksearch.data.kalshi import KalshiClient, MAX_CANDLES_PER_REQUEST
from ksearch.data.parse import CANDLE_FIELDS, parse_candles

# /historical/markets/INX-24FEB23-B5137/candlesticks — no trades this hour
HISTORICAL_NO_TRADE = {
    "end_period_ts": 1708142400, "open_interest": "0.00",
    "price": {"close": None, "high": None, "low": None, "mean": None, "open": None, "previous": None},
    "volume": "0.00",
    "yes_ask": {"close": "0.2100", "high": "0.4800", "low": "0.2100", "open": "0.4800"},
    "yes_bid": {"close": "0.0000", "high": "0.0000", "low": "0.0000", "open": "0.0000"},
}

# /series/KXINX/markets/KXINX-26SEP18H1600-T7225/candlesticks — live schema
LIVE_NO_TRADE = {
    "end_period_ts": 1789480800, "open_interest_fp": "0.00", "price": {}, "volume_fp": "0.00",
    "yes_ask": {"close_dollars": "0.0500", "high_dollars": "0.0500", "low_dollars": "0.0400", "open_dollars": "0.0400"},
    "yes_bid": {"close_dollars": "0.0000", "high_dollars": "0.0000", "low_dollars": "0.0000", "open_dollars": "0.0000"},
}

LIVE_TRADED = {
    "end_period_ts": 1789484400, "open_interest_fp": "120.00", "volume_fp": "35.00",
    "price": {"close_dollars": "0.4400", "high_dollars": "0.4500", "low_dollars": "0.4100", "open_dollars": "0.4100"},
    "yes_ask": {"close_dollars": "0.4500", "high_dollars": "0.4600", "low_dollars": "0.4200", "open_dollars": "0.4200"},
    "yes_bid": {"close_dollars": "0.4300", "high_dollars": "0.4400", "low_dollars": "0.4000", "open_dollars": "0.4000"},
}


def test_historical_null_price_is_nan():
    (row,) = parse_candles([HISTORICAL_NO_TRADE], "historical")
    assert math.isnan(row["price_close"])  # the Spring parser returned 0.0 here
    assert row["ask_close"] == 0.21
    assert row["bid_close"] == 0.0  # a real empty bid side is data, not missing


def test_live_empty_price_block_is_nan():
    (row,) = parse_candles([LIVE_NO_TRADE], "live")
    assert math.isnan(row["price_close"])
    assert row["ask_close"] == 0.05


def test_live_schema_reads_dollar_suffixed_fields():
    # The Spring parser read yes_ask.close on live data -> every quote was $0.00
    (row,) = parse_candles([LIVE_TRADED], "live")
    assert (row["bid_close"], row["ask_close"], row["price_close"]) == (0.43, 0.45, 0.44)
    assert row["volume"] == 35.0 and row["open_interest"] == 120.0


def test_schema_mismatch_yields_nan_not_zero():
    # Reading a live candle with the historical parser must not invent zeros
    (row,) = parse_candles([LIVE_TRADED], "historical")
    assert math.isnan(row["ask_close"]) and math.isnan(row["bid_close"])


def test_parsed_rows_have_every_field():
    for c, src in [(HISTORICAL_NO_TRADE, "historical"), (LIVE_TRADED, "live")]:
        (row,) = parse_candles([c], src)
        assert list(row) == CANDLE_FIELDS


def test_candle_windows_respect_api_cap():
    open_ts, close_ts = 0, 400 * 24 * 3600  # 400 days of hourly candles
    windows = KalshiClient.candle_windows(open_ts, close_ts, 60)
    assert windows[0][0] == open_ts and windows[-1][1] == close_ts
    assert all(b == c for (_, b), (c, _) in zip(windows, windows[1:]))  # contiguous
    assert all((e - s) / 3600 < MAX_CANDLES_PER_REQUEST for s, e in windows)
