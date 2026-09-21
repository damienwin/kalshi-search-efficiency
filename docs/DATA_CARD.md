# Data card: Kalshi v1 events

**Version:** `data-v1` (Sep 2026). Counts are copied from
`data/manifests/dataset_v1.json` (committed) and `scope_sample.json`.

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

## Size and splits

**766 markets** (304 Spring + 462 sampled; the 2025–26 sample kept 22 of 94–276
candidates tried per month, from a frame of 190,413 settled markets) and **21,587 events**.

| Split | Events | Markets | Event groups | Span | FLAT / UP / DOWN |
|---|---|---|---|---|---|
| dev | 18,335 | 557 | 534 | 2023-07-17 → 2026-02-24 | 15,788 / 1,320 / 1,227 |
| sealed | 1,965 | 145 | 137 | 2026-02-27 → 2026-09-21 | 1,159 / 440 / 366 |

1,287 dev events that would have run into the sealed period were purged. The
sealed slice is much less FLAT (59%) than dev (86%): later markets are more
volatile. That is a distribution shift the sealed evaluation must report.

## Label filters and attrition

Candidates are every grid t0 in each market's life. A candidate becomes an
event only if it passes every filter below, checked in order. The first failure is counted.

| Filter | Dropped |
|---|---|
| near settlement (t0 + 4h within 4h of close) | 1,451 |
| no quote at t0 | 1,637 |
| **quote at t0 older than 2h** | **40,316** |
| one-sided book at t0 (bid = 0 or ask = 1) | 2,471 |
| quote at t0 + 4h older than 2h | 6,332 |
| one-sided book at t0 + 4h | 677 |
| spread > 0.20 at either end | 1,710 |
| **events** | **21,587** of 76,181 candidates |

### Why stale quotes are dropped rather than carried forward

Kalshi normally emits a candle for idle hours: 68% of candles in a 200-market
sample show no quote change and no volume. A gap of more than 2h is therefore
not "nothing happened", and the book during it is unknown. Candle `open` can't
settle the question, because it is always the previous `close`: 100% of 34,806
consecutive candles, including the 9,090 where the quote moved. Gaps are spread
evenly across weekdays and hours, with a median of 6h and a 99th percentile of
167h.

## News coverage

| | Events |
|---|---|
| with ≥ 1 article in [t0 − 6h, t0) | 2,380 of 20,300 |
| with ≥ 2 articles (Spring's inclusion rule) | 1,979 |

All current coverage comes from the Spring cache (242 Spring markets with
events, `provisional_sentiment = 1`). The 460 sampled markets have no search
yet, pending new Guardian API keys. FinBERT rates 55% of the 45,009 articles
neutral, 43% negative, and 2% positive.

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
