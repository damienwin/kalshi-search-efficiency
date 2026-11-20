# Handoff: continuing the search-efficiency study

This document is for a student picking up the work without the author around.
It covers where things are, how to rerun every step, the rules that protect the
sealed test slice, and what is still open. Read [PROVENANCE.md](PROVENANCE.md)
before relying on any number in `runs/`.

## Credit

The study, code, dataset and paper draft are the work of the original author
(GitHub `damienwin`), supervised by their advisor. If you continue it:

- Keep the author as an author on any paper, preprint or poster that uses this
  code, dataset or these runs, and cite the preprint once it exists.
- Work on a branch and record what you added in a short "Contributions" list at
  the bottom of this file, so it is clear who did what.
- Ask the author before pushing anything to the shared remote (see "Repo rules").

## The question and the current answer

Does an LLM proposing features beat random and TPE search over the same feature
grammar, with the same trial budget and the same acceptance gate? The testbed is
4-hour price direction (UP / FLAT / DOWN, ±0.03 on the bid/ask mid) in Kalshi
markets. The dev set is 19,867 events; the sealed slice is 2,048 events and has
not been scored.

All four arms have finished on dev: 5 seeds × 100 trials each
(`runs/summary/arms.md`). In short: TPE has the largest mean gain in balanced log
loss, LLM full is the most consistent across seeds, LLM no-history accepts
nothing (58-64 of its 100 trials per seed are duplicates), and no full arm beats another
after Holm correction. These are dev numbers and measure search behaviour, not
out-of-sample skill.

## Repo layout

```
config/settings.yaml   every knob: labels, CV, classifier, search budget and gate, LLM model
config/*.txt           Kalshi series universe, the 304 Spring markets
src/ksearch/data/      Kalshi client, candle parsing, point-in-time grid, labels, hash manifest
src/ksearch/features/  price features (sentiment is parked)
src/ksearch/eval/      walk-forward folds, baselines, CV, sealed gateway (sealed.py)
src/ksearch/model/     two-stage / direction models (finished side study, appendix)
src/ksearch/search/    grammar.py + spec.py (feature language), trial.py + gate.py (scoring
                       and the DM + placebo gate), loop.py (one run), arms/ (random_, tpe, llm)
scripts/               one entry point per step (table below)
tests/                 pytest suite, including the leakage tests L1-L14
data/manifests/        committed hashes of every raw file and dataset build
data/raw/, data/build/ raw pulls and built parquet files: local only, never committed
sealed/                the sealed slice: local only, read through the gateway (below)
runs/                  search runs, calibration, summaries: local only (gitignored)
results/               CV and baseline JSON (small, committable)
paper/                 LaTeX source (NeurIPS 2025 style), sections/, figures/, outline.md
docs/                  data card, Spring artifact, this file, PROVENANCE.md, reports/ (local drafts)
```

`runs/` and `data/` are not in git. To continue on another machine, copy
`data/raw/`, `data/build/v2/`, `sealed/` and `runs/` from the author's machine
privately (they hold Kalshi data and per-event labels; do not upload them anywhere
public), then check them with `python -m ksearch.data.manifest verify`.

## Environment

Python 3.12 (the runs used 3.12.13; `pyproject.toml` allows ≥ 3.11).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev,search]'       # core + pytest + optuna/anthropic
.venv/bin/pip install matplotlib               # paper_figures.py (not yet in pyproject.toml)
.venv/bin/pip install -e '.[sentiment]'        # only for the parked news/sentiment path
git config core.hooksPath .githooks            # pre-push guard (L14)
brew install tectonic                          # paper build
```

Package versions the runs used are in every `runs/<arm>/seed<k>/run.json` under
`versions` (anthropic 1.11.0, numpy 2.5.3, optuna 5.0.0, pandas 3.0.6,
scikit-learn 1.9.1, scipy 1.18.1, xgboost 3.4.1). Pin to these to reproduce
bit-for-bit.

Secrets go in `.env` (gitignored, never commit): `ANTHROPIC_API_KEY` for the LLM
arms, the Guardian key for the news pull. Nothing else needs a key; the Kalshi
endpoints used are public.

## Pipeline

| Step | Command | Notes |
|---|---|---|
| Pull universe and candles | `scripts/pull_kalshi.py --dry-run`, then `--reuse-universe` | writes `data/raw/kalshi/**` and a hash manifest `data/manifests/kalshi_<date>.jsonl` |
| Pull scope | `scripts/pull_scope.py spring \| sample \| merge` | Spring markets + same-rule monthly sample |
| Build dataset | see below | labels + price features, seals the test slice |
| CV vs baselines | `scripts/run_cv.py` | refuses sealed inputs |
| Search run | `scripts/run_search.py --arm A --seed K` | one arm, one seed, 100 trials |
| Null calibration | `scripts/run_null_calibration.py run` / `summary` | gate false-accept rate |
| Summarize arms | `scripts/summarize_arms.py` | `runs/summary/arms.{md,json}` |
| Paper baselines | `scripts/paper_baselines.py` | `results/paper_baselines.json` |
| Paper figure | `scripts/paper_figures.py` | `paper/figures/fig1_running_gain.pdf` |
| Paper | `tectonic paper/main.tex` | `paper/main.pdf` (never commit it) |
| Sealed score | `scripts/confirm_sealed.py` | once, at the end; see "Sealed slice" |

All commands below assume `.venv/bin/python` and the repo root as working directory.

### Build the dataset

The search runs use **`data/build/v2/dev.parquet`** (sha256 `be87830a…`, recorded
in `data/manifests/dataset_v2.json` and in every `run.json`). v2 is the v1
pipeline re-run on the 2026-09-24 pull; the labelling and feature code has not
changed since commit `5292d64`.

The exact v2 build command was not logged. From the code it was, in effect:

```bash
python scripts/pull_kalshi.py --reuse-universe              # 2026-09-24 pull
python scripts/pull_scope.py sample && python scripts/pull_scope.py merge
# build_dataset.py reads <build-dir>/universe.jsonl: data/build/v2/universe.jsonl is the merged universe
python scripts/build_dataset.py --build-dir data/build/v2 --summary data/manifests/dataset_v2.json
python -m ksearch.data.manifest verify
shasum -a 256 data/build/v2/dev.parquet     # must be be87830aeed4da9f…
```

`build_dataset.py` also writes `sealed/sealed.parquet` (overwriting it) and
`data/build/v2/events_index.parquet` (sealed events without labels). Rebuilding
from a fresh pull will not reproduce the hash, because Kalshi's live endpoint
changes; reuse the author's `data/raw/` to reproduce.

### Run a search arm

```bash
python scripts/run_search.py --arm random --seed 0 --dry-run    # plan only: folds, fit counts, sample proposals
python scripts/run_search.py --arm random --seed 0              # -> runs/random/seed0/
python scripts/run_search.py --arm tpe --seed 0
python scripts/run_search.py --arm llm_full --seed 0            # needs ANTHROPIC_API_KEY; costs money
python scripts/run_search.py --arm llm_no_history --seed 0
```

Arms: `random`, `tpe`, `llm_full`, `llm_no_history`, `llm_blind` (dropped for
cost). A run writes `run.json` (config, data hash, git commit, versions, totals,
accepted features), `trials.jsonl` (one line per trial with the gate's verdict),
`oof/` (out-of-fold predictions) and, for LLM arms, `transcript.jsonl` (every
request and response). It refuses an existing run directory unless `--force`;
`--resume` (uncommitted, see PROVENANCE.md) continues a stopped LLM run by
replaying its transcript.

Cost per seed on an 8-core laptop: random 25-60 min, TPE 50-90 min, LLM full
45-110 min and about 1M tokens, LLM no-history about 35 min. Commit before you
start a run: every run records `git.dirty`, and a dirty run cannot be traced to
code (this is the problem PROVENANCE.md works through).

### Null calibration

How often the gate accepts a feature that cannot help. Probes are deterministic
in `(seed, id)`, so shards can run in parallel and a killed run resumes.

```bash
for i in 0 1 2; do python scripts/run_null_calibration.py run --n 1000 --shard $i/3 & done; wait
python scripts/run_null_calibration.py summary                        # runs/null_calibration/summary.json
python scripts/run_null_calibration.py run --control --n 24 --out runs/null_calibration_control   # positive control
```

### Summaries, baselines, figure, paper

```bash
python scripts/summarize_arms.py                 # per-arm mean ± sd, sign consistency, overlap, Holm-corrected permutation tests
python scripts/paper_baselines.py                # uniform / majority / random / momentum on the search's folds
python scripts/paper_figures.py                  # Figure 1: running gain per trial
tectonic paper/main.tex                          # writes paper/main.pdf
```

`paper_figures.py` hard-codes the null best-of-100 95th percentile (0.0041) from
`runs/null_calibration/summary.json`; update it if the calibration is rerun.

## Sealed slice (L11)

The sealed slice is the last 15% of time by event group (2026-03-14 to
2026-09-23, 2,048 events in v2). It is the only out-of-sample evidence the paper
will have, so it is opened **once**, at the end, for all arms together.

- Only `ksearch.eval.sealed.read_sealed(reason)` may open `sealed/sealed.parquet`.
  Every read appends a line (time, commit, script, reason, file hash) to
  `data/manifests/sealed_access.jsonl`, which is committed.
- `scripts/confirm_sealed.py` is the only script that calls it. It is a dry run
  unless `--confirm`; it lists earlier confirmations and makes you confirm again
  knowingly.
- `run_cv.py`, `run_search.py`, the search loop and the null calibration refuse
  any path under `sealed/` and `events_index.parquet`.
- Scripts that need to know which events exist (news, sentiment) read the
  label-free `data/build/<build>/events_index.parquet`.
- The access log has one entry: a backfilled, label-free read of the **v1**
  sealed file (sha `22e99576…`) on 2026-09-21. The current v2 sealed file
  (sha `8f3cac3c…`) has never been read.

Tests that enforce this (`python -m pytest tests/test_eval.py tests/test_search_loop.py -k "L11 or sealed"`):

- `test_eval.py::test_L11_only_the_gateway_opens_sealed_parquet`: no file in `src/` or `scripts/` other than the gateway and `build_dataset.py` (the writer) names `sealed.parquet`
- `test_eval.py::test_L11_read_sealed_requires_reason_and_logs`
- `test_eval.py::test_L11_confirm_sealed_dry_run_reads_nothing`
- `test_eval.py::test_L11_run_cv_refuses_sealed_data`
- `test_search_loop.py::test_never_touches_sealed`, `::test_cli_refuses_sealed_and_missing_arm`

Do not open the sealed slice to "check something". If a look is ever needed
before the final scoring, discuss it with the author and advisor first, and say
so in the paper.

## Tests

```bash
python -m pytest -q                 # 586 tests at ace7d29, ~90 s; no network
```

The leakage controls L1-L14 are listed in the README with the test that enforces
each. The `live` marker exists, but no test currently uses it.

## Repo rules

- Never commit PDFs (`paper/*.pdf`, `docs/reports/*.pdf` are gitignored; so
  should `paper/figures/*.pdf` be, see open items).
- `docs/reports/` holds progress reports to the advisor. Keep them local: do not
  commit new ones or push them.
- Never commit `.env`, `private/`, `data/raw/`, `data/build/`, `sealed/` or
  `runs/`. The pre-push hook blocks most of these and any file over 5 MB.
- Ask the author before pushing to `origin`.
- Commit code before running anything whose numbers will be reported.

## Open items

Provenance (details and commands in [PROVENANCE.md](PROVENANCE.md)):

1. *(Done: `748b15f`, `254e8a1`.)* Commit the code the runs actually used: the uncommitted LLM refusal policy and
   `--resume` (`llm.py`, `run_search.py`, `test_arms_llm.py`), and the untracked
   `scripts/paper_baselines.py`, `scripts/paper_figures.py`,
   `src/ksearch/data/relevance.py` + `tests/test_relevance.py`.
2. *(Done: `74101d7`.)* Commit the data manifests the runs depend on: `data/manifests/dataset_v2.json`,
   `kalshi_20260924.jsonl`, `kalshi_sample_20260924.jsonl`.
3. *(Done: Appendix E and Limitations.)* Disclose in the paper that 4 of 500 LLM-full trials were charged as failed
   because the API refused the request, not because the proposal was bad.
4. Add a replay script to the repo and record git commit and dirty state in the
   null calibration's `meta.json`.

Study:

5. **Sealed scoring of the arms.** `confirm_sealed.py` scores only the fixed
   `price` / `price+sentiment` sets, and its `--build-dir` still defaults to
   `data/build/v1`. The paper needs each arm's accepted features (per seed)
   scored on the v2 sealed slice with the same DM test, in one opening. This
   script does not exist yet.
6. **Contamination split (C4).** Split the sealed slice at the proposer's
   training cutoff and compare LLM-full gains before and after it.
7. Paper sections still missing or not started: setup, significance, results
   (sealed), abstract; related work is new. See `paper/outline.md` for the
   section-by-section plan and what the author approved.
8. *(Done: the paper now includes a 400-dpi PNG; figure PDFs are gitignored.)* `paper/figures/fig1_running_gain.pdf` is a PDF the paper includes. Either
   gitignore `paper/figures/*.pdf` and build the figure before the paper, or
   switch the figure to PNG.
9. *(Done: `paper` extra.)* Add `matplotlib` to `pyproject.toml` (a `paper` extra).

Scope already settled with the advisor: one testbed (no replication datasets),
sentiment parked until spring, no more two-stage variants, blind ablation
dropped for API cost.

## Contributions

- Original study, code, data and runs: `damienwin`.
