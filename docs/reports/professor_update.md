# Kalshi Prediction Market Model — Progress Report

## Overview

While rebuilding the pipeline for this semester, I found that the Spring
dataset's price labels were mostly data errors, not real price movements.
I corrected the underlying data, added new price-based features, and
re-ran all evaluations. The corrected model now beats the momentum
baseline; sentiment does not add a measurable improvement on top of price.

## Data issues found and fixed

| Issue | Effect on Spring dataset |
|---|---|
| Missing trade prices were recorded as $0.00 instead of "no data" | 70% of UP/DOWN labels were based on a $0.00 price |
| The "4-hour" prediction window was measured in candles, not real time | Actual windows ranged from under 1 hour to several days (median 10 hours) |
| Momentum baseline compared events across different markets | Baseline was computed incorrectly for the first event in each market |

All three were corrected in the new dataset. Prices now use the bid/ask
midpoint, and each prediction window is exactly 4 real hours.

## Results: before and after

| Version | Model Macro F1 | Momentum Baseline | Beats Momentum? |
|---|---|---|---|
| Spring, as published | 0.418 | 0.395 | Yes (result not reproducible — see note) |
| Spring, after leak fix | 0.390 | 0.395 | No |
| **New dataset, price features only** | **0.499** | **0.455** | **Yes, statistically significant** |
| New dataset, price + sentiment | 0.507 | 0.455 | Yes (sentiment adds no significant gain) |

*Note on the published result:* the model and feature files saved from that
run don't match each other, so the 0.418 score cannot be regenerated and is
not used going forward.

## New price-based features

| Feature | Description |
|---|---|
| Mid price | Midpoint between best bid and ask |
| Spread | Best ask minus best bid |
| Relative spread | Spread as a fraction of mid price |
| Distance from 0.5 | How far the mid price is from an even-odds market |
| 1h / 4h / 24h return | Price change over each lookback window |
| Previous outcome | The market's last resolved UP/DOWN/FLAT result |
| 24h volatility | Standard deviation of hourly price changes |
| 24h price range | High minus low over the past 24 hours |
| 24h volume (log) | Trading volume over the past 24 hours |
| 24h active hours | Number of hours with any trading activity |
| 24h open-interest change | Change in outstanding contracts |
| Hours since market opened | Market age at prediction time |

## Sentiment feature results

News-sentiment coverage is currently limited to 2023–2024 markets (article
data for 2025–2026 markets is pending). Within that coverage:

| Comparison | Price Only | Price + Sentiment | Improvement |
|---|---|---|---|
| Events with news coverage | 0.434 | 0.450 | Not statistically significant |
| Sentiment features alone | — | 0.338 (below momentum) | — |

This is consistent with the Spring finding: news sentiment does not add
measurable predictive value beyond price and market-structure information.

## Dataset overview

| | Count |
|---|---|
| Total markets | 766 |
| Total labeled events | 21,587 |
| Development (training/validation) events | 18,335 |
| Held-out test events (not yet evaluated) | 1,965 |
| Date range | Oct 2023 – Sep 2026 |

## Outstanding item

News-article collection for 2025–2026 markets is paused pending renewed API
access; all results above use available 2023–2024 sentiment data only.
