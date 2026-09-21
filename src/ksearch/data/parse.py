"""Null-preserving parsers for Kalshi candlestick responses.

Kalshi serves two candle schemas:

  historical  /historical/markets/{t}/candlesticks   (settled before the cutoff)
      yes_bid.close = "0.2100"        price.close = null when no trades
      volume, open_interest

  live        /markets/candlesticks                   (after the cutoff)
      yes_bid.close_dollars = "0.2100" price = {} when no trades
      volume_fp, open_interest_fp

A missing value is NaN, never 0.0. The Spring pipeline mapped None -> 0.0,
which turned every no-trade hour into a $0.00 "price" (see DATA_CARD.md).
A real "0.0000" bid (an empty bid side) is kept as 0.0: that is data, and the
label builder treats it as a one-sided book.
"""

import math

NAN = float("nan")

CANDLE_FIELDS = [
    "end_period_ts",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
    "price_open", "price_high", "price_low", "price_close",
    "volume", "open_interest",
]

_OHLC = ("open", "high", "low", "close")


def to_float(val) -> float:
    """Parse a dollar/count string. None, '' and missing -> NaN."""
    if val is None or val == "":
        return NAN
    return float(val)


def _ohlc(block, suffix: str) -> dict:
    block = block or {}
    return {k: to_float(block.get(k + suffix)) for k in _OHLC}


def _flatten(ts: int, bid: dict, ask: dict, price: dict, volume, oi) -> dict:
    row = {"end_period_ts": int(ts)}
    for name, vals in (("bid", bid), ("ask", ask), ("price", price)):
        for k in _OHLC:
            row[f"{name}_{k}"] = vals[k]
    row["volume"] = to_float(volume)
    row["open_interest"] = to_float(oi)
    return row


def parse_historical_candle(c: dict) -> dict:
    return _flatten(
        c["end_period_ts"],
        _ohlc(c.get("yes_bid"), ""),
        _ohlc(c.get("yes_ask"), ""),
        _ohlc(c.get("price"), ""),
        c.get("volume"),
        c.get("open_interest"),
    )


def parse_live_candle(c: dict) -> dict:
    return _flatten(
        c["end_period_ts"],
        _ohlc(c.get("yes_bid"), "_dollars"),
        _ohlc(c.get("yes_ask"), "_dollars"),
        _ohlc(c.get("price"), "_dollars"),
        c.get("volume_fp"),
        c.get("open_interest_fp"),
    )


PARSERS = {"historical": parse_historical_candle, "live": parse_live_candle}


def parse_candles(candles: list, source: str) -> list[dict]:
    parse = PARSERS[source]
    return [parse(c) for c in candles]


def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)
