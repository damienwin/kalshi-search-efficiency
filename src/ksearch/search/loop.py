"""The search loop: greedy forward selection with a fixed trial budget.

Kept from the Spring loop: one candidate per trial; every trial is logged,
kept or not; a kept feature joins the reference for every later trial;
rejected features never accumulate; failed trials count. Changed: the arm
only proposes JSON. Parsing, evaluation, the budget and the gate belong to
this loop and are identical for every arm.

Trial statuses: accepted | rejected (gate said no) | failed | duplicate.

Errors. Invalid *content* from an arm is a failed trial and costs budget:
unparseable JSON, a spec outside the grammar (SpecError), or a degenerate
column. Anything else stops the run: an exception raised by arm.propose or
arm.observe (an arm must do its own retries; an API still down afterwards
is an outage, not a bad proposal, and counting it as a failed trial would
charge the arm for infrastructure), a propose() that does not return a
Proposal, and any error in evaluation or the gate (a bug or a data problem).
A stopped run keeps every trial logged so far and its run.json says
status "crashed" with the traceback; it is not a valid run and is excluded,
never padded out. So a completed run always has exactly `budget` trials.
Not resumable: a trial's line in trials.jsonl and its LLM calls in
transcript.jsonl are written together after arm.observe, so the logs are
consistent up to the last completed trial (calls made during a trial whose
propose() raised are lost with it); rerun with --force.

LLM transcripts. Proposal.meta["calls"] (full request/response records) is
moved out of the trial line into transcript.jsonl, one call per line in order,
readable by arms.llm.read_transcript / ReplayClient.from_jsonl; the trial line
keeps the rest of meta and n_calls.

Duplicates. A spec whose canonical key was already proposed and parsed in
this run (accepted, rejected or degenerate) is logged as "duplicate", costs a
trial, and is not re-evaluated; the arm is fed the earlier trial's numbers.
Those numbers were measured against the reference at that time, which the
outcome's reason states.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import pandas as pd

from ksearch.data.manifest import REPO_ROOT, sha256_file
from ksearch.eval.folds import Fold, walk_forward_folds
from ksearch.search import DEFAULT_COLUMNS, DEFAULT_MAX_DEPTH, FeatureSpec, SpecError, parse
from ksearch.search.arms.base import Arm, Proposal, SearchState, TrialOutcome
from ksearch.search.gate import benjamini_hochberg
from ksearch.search.trial import DegenerateFeature, Evaluator, TrialEval

SEALED_DIR = os.path.realpath(os.path.join(REPO_ROOT, "sealed"))
TIMING_FIELDS = ("wall_s", "started_at")  # the only fields allowed to differ between identical runs
DEFAULTS = {"budget": 100, "n_seeds": 5, "train_seeds": None, "alpha": 0.05, "n_placebos": 19,
            "placebo_min_shift": 6, "placebo_seed": 0, "max_depth": DEFAULT_MAX_DEPTH,
            "base_columns": None, "grammar": {"include_cross": False}, "min_valid": 100, "gate": {},
            "llm": {}}
PARSE_ERRORS = (SpecError, TypeError, ValueError, RecursionError)


# ── config ───────────────────────────────────────────────────────────────────
def search_config(cfg: dict) -> dict:
    """settings.yaml `search:` with defaults filled and the seed list resolved.

    Training seeds are the same for every arm and every run seed, so a trial's
    verdict depends only on (reference, candidate); the run seed seeds the arm alone.
    """
    s = {**DEFAULTS, **(cfg.get("search") or {})}
    s["gate"] = dict(s.get("gate") or {})
    s["grammar"] = dict(s.get("grammar") or {})
    s["train_seeds"] = list(range(int(s["n_seeds"]))) if s["train_seeds"] is None else list(s["train_seeds"])
    s["n_seeds"] = len(s["train_seeds"])
    s["base_columns"] = list(DEFAULT_COLUMNS if s["base_columns"] is None else s["base_columns"])
    return s


def guard_path(path: str) -> str:
    """Refuse anything under sealed/ (L11), for reading or writing."""
    real = os.path.realpath(path)
    if real == SEALED_DIR or real.startswith(SEALED_DIR + os.sep) or \
            os.path.basename(real) == "events_index.parquet":
        raise PermissionError(f"the search loop refuses sealed paths: {path}")
    return real


def make_folds(events: pd.DataFrame, cfg: dict) -> list[Fold]:
    return walk_forward_folds(events, cfg["cv"]["n_splits"], cfg["cv"]["embargo_h"])


def make_evaluator(events: pd.DataFrame, cfg: dict, folds: list[Fold] | None = None,
                   clf_params: dict | None = None) -> Evaluator:
    s = search_config(cfg)
    gate = {"legacy_threshold": cfg.get("cv", {}).get("keep_threshold", 0.005), **s["gate"]}
    gate.pop("run_placebos", None)
    n_pl = s["n_placebos"] if s["gate"].get("run_placebos", True) else 0
    return Evaluator(events, folds if folds is not None else make_folds(events, cfg), s["base_columns"],
                     clf_params or cfg["classifier"], s["train_seeds"], alpha=s["alpha"],
                     n_placebos=n_pl, placebo_min_shift=s["placebo_min_shift"],
                     placebo_seed=s["placebo_seed"], min_valid=s["min_valid"], gate=gate,
                     grammar=s["grammar"])


def plan(events: pd.DataFrame, folds: list[Fold], cfg: dict, budget: int) -> dict:
    """What a run will cost, without fitting anything."""
    s = search_config(cfg)
    per_model = s["n_seeds"] * len(folds)
    n_pl = s["n_placebos"] if s["gate"].get("run_placebos", True) else 0
    return {"n_events": len(events), "n_markets": int(events["market_ticker"].nunique()),
            "n_event_groups": int(events["event_group"].nunique()),
            "n_scored": int(sum(len(f.val) for f in folds)),
            "folds": [{"k": f.k, "n_train": len(f.train), "n_val": len(f.val)} for f in folds],
            "budget": budget, "train_seeds": s["train_seeds"], "n_placebos": n_pl,
            "fits_per_model": per_model,
            "fits_min": per_model * (1 + budget),
            "fits_per_dm_pass_extra": per_model * n_pl,
            "fits_max": per_model * (1 + budget * (1 + n_pl))}


# ── provenance ───────────────────────────────────────────────────────────────
def _git() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    try:
        return {"commit": run("rev-parse", "HEAD").strip(),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no").strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for p in ("ksearch", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "optuna", "anthropic"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(x):
    """JSON-safe and deterministic: non-finite floats -> None, numpy -> python, tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _dump(obj) -> str:
    return json.dumps(_clean(obj), sort_keys=True, allow_nan=False)


def _tokens(meta: dict, acc: dict) -> None:
    """Sum every numeric *tokens field an arm reports in Proposal.meta (or meta['usage'])."""
    for src in (meta, meta.get("usage") if isinstance(meta.get("usage"), dict) else {}):
        for k, v in src.items():
            if "tokens" in str(k) and isinstance(v, (int, float)) and not isinstance(v, bool):
                acc[k] = acc.get(k, 0) + v


# ── run directory ────────────────────────────────────────────────────────────
def prepare_run_dir(run_dir: str, force: bool = False) -> str:
    real = guard_path(run_dir)
    if os.path.exists(real) and os.listdir(real):
        if not force:
            raise FileExistsError(f"run dir {run_dir} exists; pass force=True (--force) to overwrite")
        if not os.path.exists(os.path.join(real, "run.json")):
            raise FileExistsError(f"{run_dir} is not a run dir (no run.json); refusing to delete it")
        shutil.rmtree(real)
    os.makedirs(os.path.join(real, "oof"), exist_ok=True)
    return real


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(_clean(obj), sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


# ── the loop ─────────────────────────────────────────────────────────────────
def run_search(arm: Arm, events: pd.DataFrame, cfg: dict, budget: int, seed: int, run_dir: str, *,
               data_path: str | None = None, folds: list[Fold] | None = None,
               clf_params: dict | None = None, force: bool = False, extra: dict | None = None,
               evaluator: Evaluator | None = None) -> dict:
    """Run exactly `budget` trials of `arm` and log them to run_dir. Returns the final run.json.

    events: the dev frame (whole market histories, positional order). cfg: settings.yaml
    as a dict (cv, classifier, search). clf_params overrides cfg['classifier'] (tests).
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    if data_path is not None:
        guard_path(data_path)
    s = search_config(cfg)
    real = prepare_run_dir(run_dir, force)
    events = events.reset_index(drop=True)
    ev = evaluator or make_evaluator(events, cfg, folds, clf_params)
    run_path = os.path.join(real, "run.json")
    run = {"arm": getattr(arm, "name", type(arm).__name__), "arm_class": type(arm).__qualname__,
           "seed": seed, "budget": budget, "search": s, "cv": cfg.get("cv"),
           "classifier": ev.clf, "data": {"path": data_path,
                                          "sha256": sha256_file(data_path) if data_path else None},
           "plan": plan(events, ev.folds, cfg, budget), "git": _git(), "versions": _versions(),
           "argv": sys.argv, "start_time": _now(), "status": "running", **(extra or {})}
    _write_json(run_path, run)

    accepted: list[FeatureSpec] = []
    accepted_log: list[dict] = []
    history: list[TrialOutcome] = []
    seen: dict[str, dict] = {}       # key -> earlier record
    tokens: dict[str, float] = {}
    counts = {"accepted": 0, "rejected": 0, "failed": 0, "duplicate": 0}
    t_start = time.perf_counter()

    def write_oof(name: str) -> str:
        p = os.path.join("oof", name)
        ev.oof_frame(ev.reference(accepted)).to_parquet(os.path.join(real, p), index=False)
        return p

    try:
        ref0 = ev.reference(accepted)
        run["initial_reference"] = ref0.summary() | {"oof": write_oof("reference_a00.parquet")}
        _write_json(run_path, run)
        with open(os.path.join(real, "trials.jsonl"), "w") as log:
            for i in range(budget):
                t_trial = time.perf_counter()
                started = _now()
                state = SearchState(trial=i, budget=budget,
                                    accepted=tuple(a.to_json() for a in accepted),
                                    columns=tuple(s["base_columns"]), max_depth=s["max_depth"],
                                    history=tuple(history))
                prop = arm.propose(state)
                if not isinstance(prop, Proposal):
                    raise TypeError(f"{type(arm).__name__}.propose returned {type(prop).__name__}, not Proposal")
                meta = dict(prop.meta) if isinstance(prop.meta, dict) else {"meta": prop.meta}
                calls = meta.pop("calls", None) or []
                _tokens(meta, tokens)
                rec = {"trial": i, "proposed": prop.spec_json, "rationale": prop.rationale,
                       "arm_meta": meta, "n_calls": len(calls), "spec": None, "key": None, "formula": None,
                       "gate": None, "ref_fold_f1": None, "cand_fold_f1": None, "cand": None,
                       "placebo_gains": None, "duplicate_of": None, "fits": 0,
                       "n_accepted_before": len(accepted),
                       "reference": [a.key for a in accepted], "started_at": started}
                gain = p = df1 = None
                try:
                    spec = parse(prop.spec_json, columns=s["base_columns"], max_depth=s["max_depth"],
                                 **s["grammar"])
                except PARSE_ERRORS as e:
                    spec = None
                    status, reason = "failed", f"invalid spec: {type(e).__name__}: {e}"
                if spec is not None:
                    rec.update(spec=spec.canonical().to_json(), key=spec.key, formula=spec.formula())
                    if spec.key in seen:
                        prev = seen[spec.key]
                        status = "duplicate"
                        reason = (f"duplicate of trial {prev['trial']} ({prev['status']}: {prev['reason']}); "
                                  f"numbers are from that trial, against its reference")
                        rec["duplicate_of"] = prev["trial"]
                        g = prev["gate"] or {}
                        gain, p, df1 = g.get("mean_loss_gain"), g.get("dm_p"), g.get("f1_delta")
                    else:
                        try:
                            te: TrialEval | None = ev.trial(accepted, spec)
                        except (DegenerateFeature, SpecError) as e:
                            te = None
                            status, reason = "failed", f"{'degenerate' if isinstance(e, DegenerateFeature) else 'invalid spec'}: {e}"
                        if te is not None:
                            g = te.gate.to_dict()
                            status = "accepted" if te.gate.accept else "rejected"
                            reason = te.gate.reason
                            gain, p, df1 = g["mean_loss_gain"], g["dm_p"], g["f1_delta"]
                            rec.update(gate=g, ref_fold_f1=te.ref.fold_f1, cand_fold_f1=te.cand.fold_f1,
                                       cand=te.cand.summary(), placebo_gains=te.placebo_gains,
                                       fits=te.fits)
                            if te.gate.accept:
                                ev.promote(accepted, spec, te)
                                accepted.append(spec)
                                oof = write_oof(f"reference_a{len(accepted):02d}.parquet")
                                accepted_log.append({"trial": i, "key": spec.key, "formula": spec.formula(),
                                                     "spec": rec["spec"], "gain": gain, "dm_p": p,
                                                     "reference_after": te.cand.summary(), "oof": oof})
                        seen[spec.key] = {"trial": i, "status": status, "reason": reason, "gate": rec["gate"]}
                rec.update(status=status, reason=reason)
                outcome = TrialOutcome(trial=i, spec_json=prop.spec_json, formula=rec["formula"],
                                       status=status, reason=reason, gain=gain, p_value=p, delta_f1=df1,
                                       rationale=prop.rationale or "")
                counts[status] += 1
                arm.observe(outcome)
                history.append(outcome)
                rec["wall_s"] = round(time.perf_counter() - t_trial, 3)
                if calls:  # LLM request/response records, compatible with llm.read_transcript
                    with open(os.path.join(real, "transcript.jsonl"), "a") as tf:
                        for c in calls:
                            tf.write(json.dumps(c, sort_keys=True) + "\n")
                log.write(_dump(rec) + "\n")
                log.flush()
    except BaseException as e:
        run.update(status="interrupted" if isinstance(e, KeyboardInterrupt) else "crashed",
                   end_time=_now(), error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(),
                   trials_completed=len(history), totals=counts, accepted=accepted_log,
                   api_tokens=tokens or None, n_fits=ev.n_fits)
        _write_json(run_path, run)
        raise

    final = ev.reference(accepted)
    tested = [(h.trial, h.p_value) for h in history if h.status in ("accepted", "rejected")]
    q = benjamini_hochberg([pv for _, pv in tested]) if tested else np.array([])
    run.update(status="complete", end_time=_now(), wall_s=round(time.perf_counter() - t_start, 1),
               trials_completed=len(history), totals=counts, accepted=accepted_log,
               final_reference=final.summary() | {"oof": write_oof("final.parquet"),
                                                   "features": [a.formula() for a in accepted]},
               bh={"note": "Benjamini-Hochberg over the DM p-values of every evaluated trial (accepted or "
                           "rejected); reporting only, each p is against the reference of its time",
                   "trials": [t for t, _ in tested], "dm_p": [pv for _, pv in tested],
                   "q": [float(v) for v in q],
                   "discoveries": [t for (t, _), v in zip(tested, q) if v <= s["alpha"]]},
               api_tokens=tokens or None, n_fits=ev.n_fits)
    _write_json(run_path, run)
    return run
