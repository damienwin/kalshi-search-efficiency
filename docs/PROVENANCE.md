# Provenance of the dev-set search runs

Every run under `runs/{random,tpe,llm_full,llm_no_history}/seed*/` records
`git.dirty = true`. This note works out what code each run actually used, which
differences could change a result, and how to tie every run to a commit. It was
written on 2026-11-19 from the run records, `git log`/`git diff`, file
timestamps, and no-API reruns (section 4). No LLM arm was called and `sealed/`
was not touched.

**Status, 2026-11-19 (later the same day).** Step 1 of section 5 is done: the code the runs used is commit `748b15f` (call it `R`), the v2 manifests are `74101d7`, and the paper's scripts are `254e8a1`. Steps 2-5 (replay every LLM run at `R`, `results/provenance.json`, the dirty-tree guard) are not done.

Times are UTC. `run.json` start and end times are UTC; commit times are converted.

## 1. Summary

- **Config and data are the same for all 20 runs.** `search`, `cv`,
  `classifier`, `plan`, package `versions` and the dev-set hash
  (`be87830a…`, `data/build/v2/dev.parquet`) are identical in every `run.json`.
- **Random and TPE are not affected by the commit split.** Between `7a4a020`
  and `ace7d29` only `src/ksearch/search/arms/llm.py`, its tests and the new
  `scripts/summarize_arms.py` changed. No code that random or TPE import
  changed, so TPE seed 0 (at `7a4a020`) and seeds 1-4 (at `ace7d29`) ran the
  same code path.
- **"Dirty" mostly meant report drafts.** `dirty` is `git status --porcelain
  --untracked-files=no`, so it counts only modified tracked files. Most runs
  were dirty because of edits to `docs/reports/*.md`, `.gitignore` and
  `scripts/make_pdf.py`, which no run imports.
- **Five LLM runs used uncommitted code that changes behaviour**: llm_full
  seeds 2 and 3 (resumed with `--resume`) and llm_no_history seeds 2-4. The
  uncommitted `llm.py` changes the refusal policy. In llm_full seeds 2 and 3 this
  charged **4 trials as failed** where committed code would have stopped the
  run. In llm_no_history seeds 2-4 no refusal occurred, so the policy never fired.
- **The paper's scripts and dataset manifest are not committed.**
  `scripts/paper_baselines.py`, `scripts/paper_figures.py`,
  `data/manifests/dataset_v2.json` and the 2026-09-24 raw-pull manifests are
  untracked.
- **The null calibration ran before its script was committed**, and its
  `meta.json` records no commit.
- **Reruns at clean commits match the records exactly**, bit for bit (section
  4): random seed 4 at `7a4a020`; llm_full seed 0 and llm_no_history seed 4
  replayed from their transcripts at `ace7d29` with no API calls; 24 sampled
  null-calibration probes at `082e952` and `7a4a020`. So the dirty trees did
  not change those results; the only behavioural difference is item 1 in
  section 3.

## 2. What each run used

| Group | Runs | Recorded commit | Code actually used | Evidence |
|---|---|---|---|---|
| A | random 0-4, tpe 0 | `7a4a020` | `7a4a020`; dirt in files these arms don't import | 7a4a020..ace7d29 touches only `llm.py`, its tests, `summarize_arms.py`; random seed 4 rerun at clean `7a4a020` (section 4) |
| B | tpe 1-4, llm_full 0, 1, 4, llm_no_history 0, 1 | `ace7d29` | `ace7d29`; dirt in report drafts, `.gitignore`, `make_pdf.py` | the uncommitted `llm.py` edits are dated after these runs started (below); the _stopped runs that started with seed 4 crashed with the *committed* refusal message; llm replays at clean `ace7d29` (section 4) |
| C | llm_full 2, 3 | `ace7d29` | `ace7d29` + uncommitted `llm.py` (refusal policy, `ResumeClient`) + `run_search.py --resume` | `argv` has `--resume`; `run.json` has `resumed_from`, `replayed_calls`; failed trials carry the new policy's error text |
| C | llm_no_history 2-4 | `ace7d29` | same uncommitted code as above, policy never triggered (0 refusals) | started 2026-11-16, after the edits; `--force`; seed 4 replays identically at clean `ace7d29` (section 4), so in effect these ran committed code |

Timeline:

| When (UTC) | Event |
|---|---|
| 11-10 18:30-21:40 | null calibration, 1,000 noise/shuffle probes (script uncommitted) |
| 11-11 03:31 | `082e952` commits `run_null_calibration.py` |
| 11-11 05:58-06:43 | positive-control probes (`--control`, uncommitted) |
| 11-11 06:43:08 | `7a4a020` commits `--control` |
| 11-11 06:43-09:40 | random seeds 0-4 (seed 0 starts 2 s after the commit) |
| 11-12 21:33-22:23 | tpe seed 0 at `7a4a020` |
| 11-12 21:37, 21:50, 22:04 | `5f4c768`, `5cb3a3d`, `ace7d29` (LLM refusal handling, summarize_arms) |
| 11-12 22:23 - 11-13 03:17 | tpe seeds 1-4 |
| 11-13 03:18-05:50 | llm_full 0, 1; llm_no_history 0, 1 |
| 11-13 05:02 | `.gitignore` edited (paper build outputs) |
| 11-13 17:26 | llm_full 2, 3, 4 start in parallel (`--force` on 4) |
| 11-13 17:30 | `scripts/make_pdf.py` edited |
| 11-13 18:20, 18:58 | llm_full 2 and 3 stop at trials 42 and 77: "21 consecutive refusals of an unchanged request" (committed policy); moved to `runs/_stopped/` |
| 11-13 18:40 | llm_full 4 completes |
| 11-13 19:00-19:02 | uncommitted edits to `llm.py`, `test_arms_llm.py`, `run_search.py` (last-modified times) |
| 11-13 19:02 | llm_full 2, 3 resumed with `--resume` |
| 11-13 20:13 | llm_full 3 completes |
| 11-16 17:45 | llm_full 2 completes (`wall_s` 6,495: the process was suspended in between, likely sleep) |
| 11-16 17:46-18:21 | llm_no_history 2-4 (`--force`) |

The working tree's earlier states are not recoverable: git keeps only the
current dirty diff, and file timestamps show only the last edit. The groups
above rest on what was committed when, which files each arm imports, the
error messages in the stopped runs, and the reruns.

## 3. Differences that could affect results

1. **Refusal policy (llm_full seeds 2 and 3).** Committed `ace7d29`: after an
   API refusal, resend the same request up to 20 times, then stop the run
   (`ProposerUnavailable`) so the arm is never charged for a provider outage.
   Uncommitted: resend up to 8 times, then charge the trial as failed
   ("refused 9 times in a row; no proposal"); stop only after 3 such trials in a
   row. Trials charged this way: llm_full seed 2 trials 42, 68, 72; seed 3 trial
   77. These are the 4 "failed" trials for llm_full in `runs/summary/arms.md`.
   llm_full therefore made 496 proposals against a budget of 500, a small bias
   *against* the LLM arm. Under the committed code these two seeds could not have
   finished: the same exact request for trial 42 was refused 21 times in the
   first attempt and 9 more times after resuming. **The paper should say this.**

2. **Resume (llm_full seeds 2 and 3).** `--resume` replays the stopped run's
   transcript (42 and 79 calls, no API) and the loop re-fits those trials, then
   calls the API from the stop point on. Since fitting is deterministic and
   `ReplayClient` refuses any request that differs from the recording, the
   resumed run equals an unbroken run that met the same refusals. The stopped
   attempts are kept in `runs/_stopped/llm_full/`.

3. **Overwritten attempts (`--force`).** llm_full seed 4 and llm_no_history
   seeds 2-4 were started with `--force`, which deletes an earlier run directory.
   The run records do not say what those earlier attempts were. **Resolved
   (2026-11-19):** the author confirmed that none of them had completed ("None
   were completed"). Every deleted attempt stopped early; no completed run was
   discarded or replaced, so the reported seeds are not a selection.

4. **Null calibration.** `runs/null_calibration*/meta.json` has no git field.
   The noise probes ran before `082e952` and the control probes before
   `7a4a020`, on the then-uncommitted script. The scoring code they call
   (`gate.py`, `trial.py`, `loop.py`) has not changed since `c62ec11`. A sample
   rerun at the commits matches (section 4).

5. **Dataset manifest.** `dev.parquet` matches the hash in
   `data/manifests/dataset_v2.json`, but that file and the raw-pull manifests it
   lists (`kalshi_20260924.jsonl`, `kalshi_sample_20260924.jsonl`) are
   untracked. The label and feature code has not changed since `5292d64`
   (2026-09-22), before the v2 build (2026-09-24).

6. **Paper numbers from untracked code.** `scripts/paper_baselines.py` and
   `scripts/paper_figures.py` (created 2026-11-19) are untracked;
   `paper_figures.py` also hard-codes the null threshold 0.0041.
   `src/ksearch/data/relevance.py` and `tests/test_relevance.py` (sentiment,
   parked) are untracked too; no search run imports them.

7. **Not result-affecting, but stale.** `scripts/confirm_sealed.py` defaults
   `--build-dir` to `data/build/v1`; the runs and the current
   `sealed/sealed.parquet` are v2. Fix before the sealed opening.

## 4. Reruns at clean commits (no API calls)

Each check extracted the commit with `git archive` into a scratch directory, put
its `src/` first on `PYTHONPATH`, read the same `dev.parquet`, wrote to scratch,
and compared every trial's status, formula, key, gate reason, fit count, and
candidate/gate numbers with the original `trials.jsonl`. LLM runs were
replayed: `ReplayClient` serves `transcript.jsonl` and raises if the code builds a
request that differs from the recorded one, so a full replay also checks the
prompts (which, for llm_full, contain every earlier trial's outcome).
`ANTHROPIC_API_KEY` was unset.

| Check | Commit | Result |
|---|---|---|
| llm_no_history seed 4, full replay | `ace7d29` | **identical**: 101/101 recorded calls matched; all 100 trials equal (max numeric difference 0); totals, final reference and `oof/final.parquet` equal |
| random seed 4, full rerun | `7a4a020` | **identical**: all 100 trials (status, formula, gate numbers; max numeric difference 0), totals and final reference |
| llm_full seed 0, full replay | `ace7d29` | **identical**: 100/100 recorded calls matched (so every prompt, including the trial history, was rebuilt exactly); all 100 trials equal, same accepted feature at trial 74; totals, final reference and `oof/final.parquet` equal |
| null calibration, noise/shuffle probes 0, 50, …, 950 (20) | `082e952` | **identical**: every field of all 20 probes (gain, DM p, legacy verdict, gate verdict) |
| null calibration control, probes 0, 6, 12, 18 | `7a4a020` | **identical**: all 4, including the one accepted signal probe and its placebo threshold |

Not checked (CPU time): TPE (50-90 min per seed), the other seeds. llm_full
seeds 2 and 3 cannot replay at `ace7d29`: their transcripts contain 9 refusals
in a row, and the committed code would ask for a 10th call the recording does
not have. They replay only at a commit that contains the uncommitted `llm.py`.

## 5. Making the runs reproducible from a commit

No rerun of an LLM arm is needed. Recommended order:

1. **Commit the code as it ran.** One commit with the current `llm.py`,
   `run_search.py` and `tests/test_arms_llm.py` unchanged (all 596 tests pass on
   the working tree), message naming the runs it produced (llm_full 2, 3;
   llm_no_history 2-4). Commit the untracked code the paper uses
   (`paper_baselines.py`, `paper_figures.py`) and the data manifests
   (`dataset_v2.json`, `kalshi_20260924.jsonl`, `kalshi_sample_20260924.jsonl`;
   all under the 5 MB hook limit). Leave `docs/reports/` and every PDF out.
   Call that commit `R`.

2. **Add a replay script and a run-to-commit table.** Add
   `scripts/replay_run.py` (the scratch version is below) and a committed
   `results/provenance.json` mapping each run directory to the commit that
   reproduces it (group A → `7a4a020`, group B → `ace7d29`, group C → `R`).
   `R` also reproduces groups A and B: it contains their code, and the new
   refusal policy only acts when a refusal happens, which never did in those
   runs. Check that by replaying at `R`.

3. **Replay every LLM run at `R`** (no API; about 4 CPU-hours for all 10, from
   the fit counts) and store the comparison output in `results/`. llm_full seeds
   2 and 3 must pass here, since `R` is the only commit they can replay at.

4. **Rerun one cheap seed per classic arm at `R`.** Random seed 4 (23 min) is
   the cheapest; add TPE seed 0 (50 min) so TPE has one check too. Deterministic
   arms that match on one seed at a commit are strong evidence for the rest.

5. **Stop this happening again.** In `loop.py`, refuse to start a run from a
   tree with modified tracked files under `src/`, `scripts/` or `config/`
   unless `--allow-dirty`, and then write `git diff HEAD` into the run directory
   with its hash in `run.json`. Add the same `git` block to the null
   calibration's `meta.json`.

The scratch replay driver used in section 4 (run with the checked-out commit's
`src/` first on `PYTHONPATH`):

```python
import os, sys, yaml
os.environ.pop("ANTHROPIC_API_KEY", None)
src_root, arm_name, seed, data, run_dir, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), *sys.argv[4:7]
sys.path.insert(0, os.path.join(src_root, "scripts"))
from run_search import load_events, make_arm
from ksearch.search.loop import make_folds, run_search, search_config
from ksearch.search.arms.llm import ReplayClient
cfg_path = os.path.join(src_root, "config", "settings.yaml")
cfg = yaml.safe_load(open(cfg_path)); s = search_config(cfg)
events = load_events(data, "all"); folds = make_folds(events, cfg)
arm, _ = make_arm(arm_name, seed, s, cfg_path)
arm.client = ReplayClient.from_jsonl(os.path.join(run_dir, "transcript.jsonl"))
run_search(arm, events, cfg, int(s["budget"]), seed, out, data_path=data, folds=folds,
           extra={"scope": "all", "replay_of": run_dir})
```
