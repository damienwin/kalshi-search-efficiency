# Progress report 1 — dataset, features, leakage audit

**Deliverable (Plan of Study, due Sep 25, 2026):** versioned dataset,
feature-extraction code with price-derived features alongside sentiment,
a data card, and this report.

## Summary

1. **The Spring 2026 labels were mostly artifacts.** Three defects, all fixed
   in the rebuild:
   - The Kalshi parser read a missing trade price as $0.00, so 69.5% of UP/DOWN
     labels touch a $0.00 price.
   - The "4-hour" window was really 4 candles, with a median span of 10 hours.
   - After Kalshi's 2026 API change, the same parser would have read every
     quote as $0.00.

   The honest Spring result reproduces (0.390 vs. momentum 0.395). The
   published 0.4179 cannot be regenerated from its own commit.
   See [SPRING_ARTIFACT.md](../SPRING_ARTIFACT.md).
2. **A new, public, leakage-tested pipeline:**
   [github.com/damienwin/kalshi-search-efficiency](https://github.com/damienwin/kalshi-search-efficiency).
   Labels use the bid/ask mid over exactly 4 hours. 14 leakage tests cover
   everything from future candles to secrets in the public repo, and the two
   core look-ahead tests were confirmed to fail when a leak is injected.
3. **Dataset v1:** 766 markets and 21,587 events, 2023-07 → 2026-09. The market
   scope matches Spring's: the same 304 markets for 2023–24, plus a same-rule
   monthly sample for 2025–26. A sealed test slice of 1,965 events has been
   set aside and not touched.
4. **Price features beat momentum in cross-validation:** macro F1 0.499 vs.
   0.455, Δ +0.044 [95% CI +0.024, +0.063], clustered by event. The most recent
   fold is the exception, which points to drift over time.
5. **Sentiment adds nothing significant over price.** In a paired test on the
   same folds, +0.008 [−0.001, +0.017] over all dev events and +0.016
   [−0.012, +0.045] over events with news. Sentiment alone scores below
   momentum. This agrees with the Spring conclusion, now on correct labels.

## 1. What was wrong before

| Defect | Where | Effect |
|---|---|---|
| Null price parsed as $0.00 | `src/data/kalshi.py:13-14` | 1,723 of 2,480 directional labels touch $0.00 |
| 4 candles ≠ 4 hours | `src/data/dataset.py:59` | 29.7% of windows are 4h; median 10h; 26.8% > 24h |
| Live-schema field rename | `_parse_candlesticks` | Every post-cutoff quote would read $0.00 |
| Momentum across markets | `src/backtest/engine.py:87` | Each market's first event takes the previous market's label |
| Published model ≠ committed features | tag `v-spring2026-published` | 0.4179 is not reproducible from the commit |

The honest post-leak result (`v-spring2026-honest`) reproduces exactly: test
F1 0.390 vs. momentum 0.395.

## 2. Dataset v1

- **Label:** direction of the bid/ask mid from t0 to t0 + 4h (±0.03), with t0
  on a fixed UTC 4-hour grid. Both ends need a two-sided quote ≤ 2h old and a
  spread ≤ 0.20. Events near settlement are excluded.
- **Attrition:** 76,181 candidates → 21,587 events. The main filter is the
  staleness rule (40,316 candidates had no quote in the 2h before t0). The data
  card explains why stale quotes cannot be carried forward: Kalshi normally
  emits idle-hour candles, so a gap means the book is unknown.
- **Splits:** dev has 18,335 events (557 markets, 2023-07 → 2026-02). Sealed
  has 1,965 events (145 markets, 2026-02-27 → 2026-09-21), split off by time
  with purging.
- **Provenance:** every API response is stored verbatim and hashed. 14,765 raw
  files verify against the committed manifests.

Full details: [DATA_CARD.md](../DATA_CARD.md).

## 3. Features

- **Price-derived (15):** bid-ask proxies (mid, spread, relative spread,
  distance from 0.5), momentum (1h/4h/24h Δmid, the previous label once it has
  resolved), volatility (24h std and range of Δmid), activity (24h volume,
  active hours, open-interest change), and schedule (hours since open, plus
  hours to close, which is held out; see below).
- **Sentiment (24):** Spring's FinBERT features, ported verbatim. The port
  reproduces Spring's article scores to within 2×10⁻⁶.

## 4. First results (dev CV only; sealed untouched)

Walk-forward, 5 folds, grouped by event (strike ladders never split), 4h
purge and embargo. Paired bootstrap over 445 event groups.

| | Macro F1 |
|---|---|
| **price-only XGBoost** | **0.499** |
| same-market momentum | 0.455 |
| majority / no change | 0.309 |

Model − momentum: **+0.044 [+0.024, +0.063]**. Per fold: 0.418/0.386,
0.506/0.457, 0.477/0.437, 0.560/0.488, **0.436/0.447**. The model wins the
first four folds and loses the most recent one.

On the 304 Spring markets alone, the edge is +0.034 [−0.003, +0.070]: **not
significant**, even though the Spring-era keep rule (mean > 0.005 and 4 of 5
folds won) would have accepted it. That is a direct example of the old
acceptance gate being looser than clustered inference.

### Sentiment (provisional, 2023–24 markets only)

Paired model-vs-model tests on identical folds:

| Events | price + sentiment | price only | Δ [95% CI] |
|---|---|---|---|
| with ≥ 2 articles (1,581 scored) | 0.450 | 0.434 | +0.016 [−0.012, +0.045] |
| all dev (16,363 scored) | 0.507 | 0.499 | +0.008 [−0.001, +0.017] |
| sentiment only, news events | 0.338 | — | vs momentum −0.058 [−0.088, −0.027] |

Coverage is limited: 2,380 of 20,300 events have any article in the 6h
lookback, and all of them are Spring-era markets. **The news-only row is the
real test of sentiment.** In the all-dev row, about 85% of scored events have
no articles, so sentiment is blank for them and that model is mostly the price
model. Its +0.008 should not be read as a test of sentiment over the full dataset.

These sentiment numbers come from articles already downloaded in Spring 2026
(the local Guardian cache). No live Guardian calls were needed.

`hours_to_close` is held out because Kalshi can revise `close_time` after the
fact. Adding it raises the Spring-scope edge to significance, so it needs the
point-in-time check before it can be used.

### What these numbers are, and are not

- All F1 values above are **walk-forward CV on the development set**. The
  sealed slice has not been scored. It is read only through a logged gateway;
  its one read so far fetched event keys for sentiment (no labels, no
  scoring) and is recorded in `data/manifests/sealed_access.jsonl`.
- **The markets were selected on outcomes.** Spring's rule, reused here, keeps
  a market only if it has ≥ 3 UP/DOWN events. That over-represents markets that
  moved and inflates the absolute F1 of every predictor, momentum included.
  The model-vs-momentum *difference* is the quantity to read.

## 5. Open items

| Item | Status |
|---|---|
| Sentiment for 2023–24 markets | Provisional, from the Spring cache (flagged); results above |
| Sentiment for 2025–26 markets | **Blocked: both Guardian API keys return 401.** Needs new keys; about 3–4 days of pulls once available |
| Market-category encoding | Returns only as a fold-scoped transform (it was the Spring leak) |
| `hours_to_close` | Point-in-time check of `close_time` revisions |
| Drift in the latest fold | Examine before any sealed evaluation |

## 6. Next (progress report 2, Oct 9 / Oct 30)

The contracted comparisons on this harness: the LLM-driven search vs.
matched-budget random and Bayesian (TPE) search over a shared feature grammar,
with repeated seeds; the knowledge-cutoff contamination check using the
post-May-2026 slice; then calibration, confidence gating, and a spread-aware
backtest.
