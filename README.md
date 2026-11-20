# kalshi-search-efficiency

Does LLM-driven feature search beat matched-budget random and Bayesian search
at finding features that generalize? The testbed is short-horizon price moves
in Kalshi prediction markets.

This repo is the rebuild of the CSC 4444 Spring 2026 project
([`kalshi-sentiment-predictor`](https://github.com/damienwin/kalshi-sentiment-predictor)).
It starts from corrected labels and a leakage-tested pipeline. See
[docs/SPRING_ARTIFACT.md](docs/SPRING_ARTIFACT.md) for what was wrong before and why.

**Status (Nov 2026):** all four arms (random, TPE, LLM full, LLM no-history)
have finished on the dev set, 5 seeds × 100 trials each, on dataset v2
(`data/build/v2/dev.parquet`, 19,867 dev events; 2,048 sealed, not yet scored).
The null calibration is done. Still open: one sealed opening for all arms, the
contamination split, and the paper (`paper/`).

**Continuing this work?** Start with [docs/HANDOFF.md](docs/HANDOFF.md) (setup,
every step, sealed-slice rules, open items), then
[docs/PROVENANCE.md](docs/PROVENANCE.md) (which code produced which run).

## Pipeline

```
pull_kalshi.py         universe per series (historical + live endpoints): the sampling frame
pull_scope.py          the 304 Spring markets + a same-rule monthly sample; candles, verbatim + hashed
build_dataset.py       v1 labels + price features -> dev.parquet, sealed/sealed.parquet
import_spring_news.py  Spring Guardian cache -> deduplicated, hashed seed
pull_news.py           per-market Guardian searches for non-Spring markets
score_articles.py      FinBERT over every unique article, once
build_sentiment.py     Spring sentiment features per event
run_cv.py              walk-forward CV vs baselines / a reference feature set, clustered paired bootstrap
confirm_sealed.py      score a frozen model on the sealed slice once (dry run by default; logged)
run_search.py          one feature search: --arm {random,tpe,llm_full,llm_no_history,llm_blind} --seed K
run_null_calibration.py  null and positive-control probes scored by the search gate
summarize_arms.py      per-arm summary over seeds, Holm-corrected permutation tests
paper_baselines.py     baselines on the search's folds (paper Setup)
paper_figures.py       Figure 1, running gain per trial
```

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,search]'
git config core.hooksPath .githooks        # pre-push guard against secrets/PII/raw data

.venv/bin/python scripts/pull_kalshi.py --dry-run     # enumerate + estimate
.venv/bin/python scripts/pull_kalshi.py --reuse-universe
.venv/bin/python scripts/build_dataset.py --build-dir data/build/v2 --summary data/manifests/dataset_v2.json
.venv/bin/python scripts/run_cv.py
.venv/bin/python -m ksearch.data.manifest verify      # raw data unchanged since pull
.venv/bin/python -m pytest
.venv/bin/python scripts/run_search.py --arm random --seed 0 --dry-run
tectonic paper/main.tex
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
| L11 | Only `ksearch.eval.sealed.read_sealed` opens the sealed slice, and every read is logged to the committed `data/manifests/sealed_access.jsonl`. `run_cv.py` refuses it; other scripts use a label-free event index |
| L13 | The raw-data hash manifest detects any change |
| L14 | The pre-push hook blocks `.env`, `private/`, raw data, and files over 5 MB |

L1 and L2 were mutation-checked: injecting a 1-hour look-ahead or a 30-minute
early read makes them fail.

## Layout

```
src/ksearch/data/      kalshi client, parsers, point-in-time grid, labels, manifest
src/ksearch/features/  price-derived features (sentiment is parked)
src/ksearch/eval/      folds, baselines, CV + inference, sealed gateway
src/ksearch/model/     two-stage / direction models (finished side study)
src/ksearch/search/    feature grammar, significance gate, search loop, arms/
scripts/               pull, build, cv, search, calibration, summaries, paper tables/figures
data/manifests/        committed hashes and dataset summaries (raw data is not committed)
runs/                  search runs and calibration output (local only, gitignored)
paper/                 LaTeX source; build with `tectonic paper/main.tex`
docs/                  data card, Spring artifact, HANDOFF, PROVENANCE, reports/ (local drafts)
```
