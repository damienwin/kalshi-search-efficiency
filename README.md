# kalshi-search-efficiency

Does LLM-driven feature search beat matched-budget random and Bayesian search
at finding features that generalize? The testbed is short-horizon price moves
in Kalshi prediction markets.

This repo is the rebuild of the CSC 4444 Spring 2026 project
([`kalshi-sentiment-predictor`](https://github.com/damienwin/kalshi-sentiment-predictor)).
It starts from corrected labels and a leakage-tested pipeline. See
[docs/SPRING_ARTIFACT.md](docs/SPRING_ARTIFACT.md) for what was wrong before and why.

**Status (Sep 2026):** data layer, v1 labels, price features, and walk-forward
CV are in place. Sentiment features and the search arms come next.

## Pipeline

```
pull_kalshi.py   universe per series (historical + live endpoints) -> raw candles, verbatim + hashed
build_dataset.py v1 labels + price features -> data/build/dev.parquet, sealed/sealed.parquet
run_cv.py        walk-forward CV, price-only XGBoost vs baselines, clustered paired bootstrap
```

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
git config core.hooksPath .githooks        # pre-push guard against secrets/PII/raw data

.venv/bin/python scripts/pull_kalshi.py --dry-run     # enumerate + estimate
.venv/bin/python scripts/pull_kalshi.py --reuse-universe
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/run_cv.py
.venv/bin/python -m ksearch.data.manifest verify      # raw data unchanged since pull
.venv/bin/python -m pytest
```

## What the label means

An event is a market at a time t0 on a fixed UTC 4-hour grid. The label is
the direction of the bid/ask mid from t0 to t0 + 4 real hours: UP or DOWN
beyond ±0.03, otherwise FLAT. Both ends need a two-sided, uncrossed quote no
more than 2h old, with a spread of at most 0.20. Last-trade prices are never
used. Full definition, filters, and attrition: [docs/DATA_CARD.md](docs/DATA_CARD.md).

## Leakage controls

Each is a test in `tests/`:

| | |
|---|---|
| L1 | Randomizing every candle after t0 leaves t0's features unchanged |
| L2 | A candle is invisible until its `end_period_ts` |
| L3 | A missing value parses to NaN, never 0; no label uses a zero or one-sided quote |
| L4–L5 | Label windows are exactly 4h, on a data-independent grid, never overlapping |
| L6 | `prev_label` only once the previous event has resolved |
| L7 | `hours_to_close` (revisable metadata) is excluded by default and ablated |
| L8 | Fold-scoped transforms refuse out-of-fold rows |
| L9 | A strike ladder (event group) never straddles a split |
| L10 | Training labels end ≥ 4h before validation starts |
| L11 | `run_cv.py` refuses the sealed slice |
| L13 | The raw-data hash manifest detects any change |
| L14 | The pre-push hook blocks `.env`, `private/`, raw data, and files over 5 MB |

L1 and L2 were mutation-checked: injecting a 1-hour look-ahead or a 30-minute
early read makes them fail.

## Layout

```
src/ksearch/data/      kalshi client, parsers, point-in-time grid, labels, manifest
src/ksearch/features/  price-derived features (sentiment port next)
src/ksearch/eval/      folds, baselines, CV + inference
scripts/               pull, build, cv
data/manifests/        committed hashes and dataset summaries (raw data is not committed)
docs/                  data card, plans, Spring artifact, progress reports
```
