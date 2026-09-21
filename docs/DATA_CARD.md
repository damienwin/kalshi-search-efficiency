# Data card: Kalshi v1 events

**Status:** draft for progress report #1 (Sep 25, 2026). Numbers below come
from the Spring-scope build (`data/build/spring_only`). The full v1 scope,
Spring markets plus the 2025–26 sample, replaces them when `data-v1` is tagged.
Every count on this page is read from `data/manifests/dataset_v1.json` or
`attrition.json`, never typed by hand.

## What an example is

One row is one **market at one time t0**. t0 values sit on a fixed UTC grid
(00:00, 04:00, …, 20:00).

| Field | Meaning |
|---|---|
| `label` | Direction of the bid/ask **mid** from t0 to t0 + 4 real hours: `UP` if Δmid > +0.03, `DOWN` if < −0.03, else `FLAT` |
| `mid_t0`, `mid_end` | (bid + ask)/2 from the latest candle closed by t0 (and by t0 + 4h) |
| `event_group` | Kalshi `event_ticker`. All strikes of one ladder share it and are never split across folds |
| price features | Bid-ask proxies, momentum, volatility, activity, schedule. See `src/ksearch/features/price.py` |
| sentiment features | FinBERT article scores aggregated over [t0 − 6h, t0), using the Spring method |

## Sources

| Source | Endpoint | Stored as |
|---|---|---|
| Kalshi markets | `/historical/markets` (before cutoff), `/markets` (after); cutoff read from `/historical/cutoff` (2026-07-23) | gzip JSON per response, sha256 in `data/manifests/kalshi_*.jsonl` |
| Kalshi candles | `/historical/markets/{t}/candlesticks`, `/series/{s}/markets/{t}/candlesticks`; hourly, ≤ 5,000 per request | same |
| Guardian (2023-07 → 2024-12) | Spring cache, imported once: 521 searches, 45,009 unique articles | `spring_seed/`, source sha256 `8812fed2…` |
| Guardian (2025 →) | `content.guardianapis.com/search`, per-market query | gzip JSON per page |

## Market scope

The scope matches Spring's, with current data:
- **2023-10 → 2024-12:** exactly the 304 markets in the Spring dataset (`config/spring_markets.txt`).
- **2025-01 → now:** Spring's rule applied to later months. That means settled markets with volume ≥ $5k in the Spring series universe and ≥ 3 directional events, drawn uniformly at random within each close month (seed 0) at Spring's density of about 22 markets/month.

## Label filters and attrition (Spring scope)

Candidates are every grid t0 in each market's life. A candidate becomes an
event only if it passes every filter below, checked in order. The first failure is counted.

| Filter | Dropped |
|---|---|
| near settlement (t0 + 4h within 4h of close) | 549 |
| no quote at t0 | 1,080 |
| **quote at t0 older than 2h** | **22,762** |
| one-sided book at t0 (bid = 0 or ask = 1) | 972 |
| quote at t0 + 4h older than 2h | 3,154 |
| one-sided book at t0 + 4h | 264 |
| spread > 0.20 at either end | 553 |
| **events** | **3,170** of 32,504 candidates |
| dev events purged because they run into the sealed period | 370 |

Splits: **dev 2,680** events (218 markets, 2023-07-17 → 2024-11-12; FLAT 2,043 / UP 327 / DOWN 310).
**Sealed 120** (23 markets, 2024-11-13 → 2024-12-28). The median |Δmid| is 0.01, the median spread 0.05.

### Why stale quotes are dropped rather than carried forward

Kalshi normally emits a candle for idle hours: 68% of candles in a 200-market
sample show no quote change and no volume. A gap of more than 2h is therefore
not "nothing happened", and the book during it is unknown. Candle `open` can't
settle the question, because it is always the previous `close`: 100% of 34,806
consecutive candles, including the 9,090 where the quote moved. Gaps are spread
evenly across weekdays and hours, with a median of 6h and a 99th percentile of
167h.

## Point-in-time guarantees

Each is a test in `tests/`; see README (L1–L14). The most important:
- A candle's values are observable only from its `end_period_ts` (L2). Every
  feature reads through `MarketCandles.quote_at` / `window`, which enforce that.
- Randomizing every candle after t0 leaves t0's features unchanged (L1,
  mutation-checked).
- An article counts only if published strictly before t0 (L12).
- `hours_to_close` reads `close_time`, which Kalshi can revise, so it is excluded by default (L7).

## Differences from the Spring 2026 dataset

| | Spring | v1 |
|---|---|---|
| Price | last trade; **null parsed as $0.00** | bid/ask mid; null is NaN |
| Window | 4 candles (median 10h, 26.8% > 24h) | exactly 4h |
| Event spacing | greedy from the first candle | fixed UTC grid, independent of the data |
| Events without news | dropped (< 2 articles) | kept; sentiment is NaN (`n_articles` recorded) |
| Momentum baseline | previous row (crosses markets) | same market, previous event, only if already resolved |
| Split | 70/15/15 by row time | walk-forward by event group, purged, plus a time-disjoint sealed slice |

In the Spring labels, 69.5% of UP/DOWN events (1,723 of 2,480) touch a $0.00
price. Details in [SPRING_ARTIFACT.md](SPRING_ARTIFACT.md).

## Known limitations

- **Sentiment for Spring-era markets is provisional** (`provisional_sentiment = 1`).
  It reuses the Spring cache, whose searches were windowed on Spring's event
  timestamps. v1 events outside those windows get no articles.
- **The query method is weak.** Keywords come from overlapping title bigrams,
  giving queries like `"Bitcoin price price range range Oct"`, and Guardian
  search returns off-topic articles (a birdwatching piece for an economy
  query). This is kept on purpose for comparability with Spring, and is not
  claimed as good retrieval.
- **Class balance shifts** between the Spring scope (76% FLAT) and later
  samples, because Kalshi's market mix changed (hourly crypto and index
  ladders). Folds and inference are grouped by event, but a shifted base rate
  still affects macro F1.
- **Classification is not profit.** Fees, spread crossing, and latency are
  outside this dataset. The cost-aware backtest (PoS item 5) comes later.
