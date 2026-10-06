"""Search loop and trial harness, on a small synthetic frame with a planted signal.

The label is driven by roll(oi_change_24h, mean, w=3): the mean of a market's
previous three values of an i.i.d. column, which no base column carries, so the
reference model cannot see it and only the planted spec (or a near relative)
can pass the gate. Every other column is independent noise.
"""

import copy
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from ksearch.data.manifest import REPO_ROOT
from ksearch.search import DEFAULT_COLUMNS, col, evaluate, node
from ksearch.search.arms.base import Arm, Proposal
from ksearch.search.loop import TIMING_FIELDS, guard_path, make_evaluator, make_folds, run_search
from ksearch.search.trial import cutoff_split

HOUR, DAY = 3600, 86400
T0 = 1_700_006_400  # a UTC midnight
PLANTED = node("roll", col("oi_change_24h"), stat="mean", w=3)
NOISE = [node("diff", col("spread"), k=1), node("roll", col("vol_24h"), stat="std", w=6),
         node("square", col("ret_1h")), node("mul", col("range_24h"), col("log_volume_24h"))]
CLF = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2, "subsample": 0.8,
       "colsample_bytree": 0.8, "min_child_weight": 1, "seed": 0}
CFG = {"cv": {"n_splits": 3, "embargo_h": 4, "keep_threshold": 0.005}, "classifier": CLF,
       "search": {"budget": 5, "n_seeds": 2, "n_placebos": 19, "placebo_min_shift": 6,
                  "gate": {"n_perm": 999, "n_boot": 300}}}


def synthetic(n_groups=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        start = T0 + int(g * 2.5) * DAY
        for m in range(2):
            slots = np.flatnonzero(rng.random(12 * 6) < 0.75)
            for s in slots:
                t0 = start + int(s) * 4 * HOUR
                rows.append({"market_ticker": f"M{g:02d}-{m}", "event_group": f"G{g:02d}",
                             "series": f"S{g % 5}", "t0": t0, "t_end": t0 + 4 * HOUR})
    ev = pd.DataFrame(rows)
    for c in DEFAULT_COLUMNS:
        ev[c] = rng.standard_normal(len(ev))
    ev["prev_label"] = rng.choice([-1.0, 0.0, 1.0, np.nan], len(ev))
    z = evaluate(PLANTED, ev)
    score = np.where(np.isnan(z), rng.standard_normal(len(ev)), z / np.sqrt(1 / 3)) \
        + 0.4 * rng.standard_normal(len(ev))
    ev["label"] = np.where(score > 1.0, "UP", np.where(score < -1.0, "DOWN", "FLAT"))
    return ev.sort_values(["t0", "market_ticker"], kind="mergesort").reset_index(drop=True)


@pytest.fixture(scope="module")
def ev():
    return synthetic()


class ListArm(Arm):
    """Proposes a fixed list, cycling; records what it observed."""
    name = "stub"

    def __init__(self, seed, proposals):
        super().__init__(seed)
        self.proposals, self.seen = list(proposals), []

    def propose(self, state):
        p = self.proposals[state.trial % len(self.proposals)]
        return p if isinstance(p, Proposal) else Proposal(p, rationale=f"r{state.trial}", meta={"i": state.trial})

    def observe(self, outcome):
        self.seen.append(outcome)


def trials(run_dir):
    with open(os.path.join(run_dir, "trials.jsonl")) as f:
        return [json.loads(l) for l in f]


def run(ev, proposals, tmp_path, name="run", budget=None, cfg=CFG, **kw):
    arm = ListArm(0, proposals)
    out = str(tmp_path / name)
    res = run_search(arm, ev, cfg, budget or len(proposals), 0, out, **kw)
    return arm, out, res


# ── gate on the planted signal ───────────────────────────────────────────────
@pytest.fixture(scope="module")
def planted_run(ev, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("planted")
    return run(ev, [NOISE[0].to_json(), PLANTED.to_json(), NOISE[1].to_json(), NOISE[2].to_json(),
                    NOISE[3].to_json()], tmp)


def test_planted_accepted_noise_rejected(planted_run):
    arm, out, res = planted_run
    t = trials(out)
    assert [r["status"] for r in t] == ["rejected", "accepted", "rejected", "rejected", "rejected"]
    g = t[1]["gate"]
    assert g["dm_p"] < 1e-4 and g["mean_loss_gain"] > 0.05            # far from the alpha line
    assert g["n_placebos"] == 19 and g["mean_loss_gain"] > 5 * g["placebo_threshold"]
    assert all(r["gate"]["dm_p"] > 0.05 for r in t if r["status"] == "rejected")
    assert res["accepted"][0]["key"] == PLANTED.key and res["totals"]["accepted"] == 1
    # the outcome fed back carries the gate's numbers and the rationale
    o = arm.seen[1]
    assert o.status == "accepted" and o.gain == pytest.approx(g["mean_loss_gain"]) and o.rationale == "r1"


def test_run_dir_has_every_promised_file(planted_run):
    _, out, res = planted_run
    assert sorted(os.listdir(out)) == ["oof", "run.json", "trials.jsonl"]
    assert sorted(os.listdir(os.path.join(out, "oof"))) == ["final.parquet", "reference_a00.parquet",
                                                           "reference_a01.parquet"]
    r = json.load(open(os.path.join(out, "run.json")))
    for k in ("arm", "seed", "budget", "search", "data", "git", "versions", "start_time", "end_time",
              "accepted", "final_reference", "bh", "totals", "api_tokens", "plan"):
        assert k in r, k
    assert r["status"] == "complete" and r["totals"] == {"accepted": 1, "rejected": 4, "failed": 0,
                                                         "duplicate": 0}
    assert len(r["bh"]["q"]) == 5 and r["bh"]["discoveries"] == [1]
    oof = pd.read_parquet(os.path.join(out, "oof", "final.parquet"))
    a1 = pd.read_parquet(os.path.join(out, "oof", "reference_a01.parquet"))
    pd.testing.assert_frame_equal(oof, a1)
    assert np.allclose(oof[["p_DOWN", "p_FLAT", "p_UP"]].sum(axis=1), 1)
    rec = trials(out)[1]
    for k in ("trial", "status", "spec", "formula", "key", "rationale", "arm_meta", "gate",
              "ref_fold_f1", "cand_fold_f1", "wall_s", "placebo_gains"):
        assert k in rec, k
    assert len(rec["ref_fold_f1"]) == len(rec["cand_fold_f1"]) == 3


# ── budget, failures, duplicates ─────────────────────────────────────────────
def test_budget_exact_with_failed_and_duplicate_trials(ev, tmp_path):
    props = ['{"op": "roll", "args": [', {"op": "nope", "args": [{"col": "mid"}]},
             {"op": "lag", "args": [{"col": "label"}], "params": {"k": 1}},  # forbidden column
             NOISE[0].to_json(),
             {"op": "diff", "params": {"k": 1}, "args": [{"col": "spread"}]},  # same key, other spelling
             node("hour_of_day").to_json(),                                    # constant on a 4h grid? no: 6 values
             node("n_prev_events").to_json() | {"op": "n_prev_events"},
             ["not", "a", "node"], 42]
    arm, out, res = run(ev, props, tmp_path, budget=11)
    t = trials(out)
    assert len(t) == 11 and len(arm.seen) == 11 and res["trials_completed"] == 11
    st = [r["status"] for r in t]
    assert st[:5] == ["failed", "failed", "failed", "rejected", "duplicate"]
    assert st[7:9] == ["failed", "failed"]
    assert st[9] == "failed" and st[10] == "failed"  # cycle repeats the two invalid ones
    assert "not valid JSON" in t[0]["reason"] and "unknown operator 'nope'" in t[1]["reason"]
    assert t[4]["duplicate_of"] == 3 and t[4]["gate"] is None and t[4]["fits"] == 0
    assert arm.seen[4].gain == arm.seen[3].gain and arm.seen[4].p_value == arm.seen[3].p_value
    assert sum(res["totals"].values()) == 11


def test_degenerate_column_is_a_failed_trial(ev, tmp_path):
    const = node("sign", node("abs", node("add", col("mid"), col("mid"))))  # 1 everywhere
    arm, out, res = run(ev, [const.to_json()], tmp_path, budget=1)
    r = trials(out)[0]
    assert r["status"] == "failed" and r["reason"].startswith("degenerate: fold 1: constant")
    assert r["fits"] == 0


def test_arm_exception_stops_the_run_and_keeps_the_log(ev, tmp_path):
    class Outage(ListArm):
        def propose(self, state):
            if state.trial == 2:
                raise ConnectionError("API down after retries")
            return super().propose(state)
    out = str(tmp_path / "crash")
    with pytest.raises(ConnectionError):
        run_search(Outage(0, [NOISE[0].to_json(), NOISE[1].to_json()]), ev, CFG, 5, 0, out)
    r = json.load(open(os.path.join(out, "run.json")))
    assert r["status"] == "crashed" and r["trials_completed"] == 2 and "API down" in r["error"]
    assert [x["trial"] for x in trials(out)] == [0, 1]


def test_llm_calls_go_to_a_transcript_not_the_trial_log(ev, tmp_path):
    calls = [{"trial": 0, "attempt": a, "request": {"x": a}, "response": {"text": "t"}} for a in range(2)]
    p = Proposal("not json", rationale="why", meta={"usage": {"input_tokens": 10, "output_tokens": 3},
                                                   "calls": calls, "valid": False})
    _, out, res = run(ev, [p], tmp_path, budget=2)
    t = trials(out)
    assert "calls" not in t[0]["arm_meta"] and t[0]["n_calls"] == 2
    lines = [json.loads(l) for l in open(os.path.join(out, "transcript.jsonl"))]
    assert lines == calls + calls
    assert res["api_tokens"] == {"input_tokens": 20, "output_tokens": 6}


# ── caching ──────────────────────────────────────────────────────────────────
def test_reference_cache_reused_and_replaced_on_acceptance(ev):
    folds = make_folds(ev, CFG)
    e = make_evaluator(ev, CFG, folds, CLF)
    per_model = 2 * len(folds)
    r0 = e.reference([])
    assert e.n_fits == per_model
    assert e.reference([]) is r0 and e.n_fits == per_model
    t = e.trial([], NOISE[0])
    assert t.fits == per_model and e.n_fits == 2 * per_model        # reference not refit
    t = e.trial([], PLANTED)
    assert t.gate.accept and t.fits == per_model * 20               # candidate + 19 placebos
    e.promote([], PLANTED, t)
    n = e.n_fits
    r1 = e.reference([PLANTED])
    assert r1 is t.cand and e.n_fits == n                           # new reference, no refit
    assert e.trial([PLANTED], NOISE[1]).ref is r1
    # the promoted reference equals a fresh fit of base + planted
    fresh = make_evaluator(ev, CFG, folds, CLF).reference([PLANTED])
    np.testing.assert_array_equal(fresh.proba, r1.proba)
    assert fresh.fold_f1 == r1.fold_f1


def test_fold_fitted_candidates_fit_on_training_rows_only(ev):
    folds = make_folds(ev, CFG)
    e = make_evaluator(ev, CFG, folds, CLF)
    z = node("zscore", col("mid"))
    cols = e.columns(z)
    for f, c in zip(folds, cols):
        tr = ev["mid"].to_numpy()[f.train]
        np.testing.assert_allclose(c, (ev["mid"] - tr.mean()) / tr.std())
    # changing a row outside the training fold leaves that fold's values unchanged
    ev2 = ev.copy()
    later = np.setdiff1d(np.arange(len(ev)), folds[0].train)
    ev2.loc[later, "mid"] = 100.0
    c2 = make_evaluator(ev2, CFG, folds, CLF).columns(z)[0]
    np.testing.assert_array_equal(c2[folds[0].train], cols[0][folds[0].train])


# ── determinism, overwrite, sealed ───────────────────────────────────────────
def test_same_seed_and_arm_give_identical_logs(ev, tmp_path):
    props = [NOISE[0].to_json(), "{bad", PLANTED.to_json(), NOISE[0].to_json()]
    tiny = CLF | {"n_estimators": 10, "max_depth": 2}  # placebos included; small trees keep it fast
    _, a, _ = run(ev, props, tmp_path, "a", clf_params=tiny)
    _, b, _ = run(ev, props, tmp_path, "b", clf_params=tiny)
    assert [r["status"] for r in trials(a)] == ["rejected", "failed", "accepted", "duplicate"]

    def strip(path):
        out = []
        for r in trials(path):
            for k in TIMING_FIELDS:
                r.pop(k)
            out.append(json.dumps(r, sort_keys=True))
        return out
    assert strip(a) == strip(b)
    raw_a = open(os.path.join(a, "trials.jsonl")).read()
    assert all(f in raw_a for f in TIMING_FIELDS)
    pd.testing.assert_frame_equal(pd.read_parquet(os.path.join(a, "oof", "final.parquet")),
                                  pd.read_parquet(os.path.join(b, "oof", "final.parquet")))


def test_refuses_overwrite_unless_forced(ev, tmp_path):
    _, out, _ = run(ev, [NOISE[2].to_json()], tmp_path, "x")
    with pytest.raises(FileExistsError):
        run(ev, [NOISE[2].to_json()], tmp_path, "x")
    run(ev, [NOISE[3].to_json()], tmp_path, "x", force=True)
    assert trials(out)[0]["key"] == NOISE[3].key
    other = tmp_path / "notarun"
    other.mkdir()
    (other / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not a run dir"):
        run(ev, [NOISE[2].to_json()], tmp_path, "notarun", force=True)


def test_never_touches_sealed(ev, tmp_path):
    sealed = os.path.join(REPO_ROOT, "sealed")
    with pytest.raises(PermissionError):
        guard_path(os.path.join(sealed, "x.parquet"))
    with pytest.raises(PermissionError):
        run_search(ListArm(0, [NOISE[0].to_json()]), ev, CFG, 1, 0, os.path.join(sealed, "run"))
    with pytest.raises(PermissionError):
        run(ev, [NOISE[0].to_json()], tmp_path, data_path=os.path.join(sealed, "x.parquet"))
    opened = []

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(args[0]))
    sys.addaudithook(hook)  # cannot be removed; it only appends to a list
    fast = copy.deepcopy(CFG)
    fast["search"]["n_placebos"] = 0
    run(ev, [NOISE[0].to_json(), PLANTED.to_json()], tmp_path, "audit", budget=2, cfg=fast)
    hits = [p for p in opened if os.path.realpath(p).startswith(os.path.realpath(sealed))]
    assert hits == [] and any(p.endswith("trials.jsonl") for p in opened)


# ── knowledge-cutoff helper ──────────────────────────────────────────────────
def test_cutoff_split():
    rng = np.random.default_rng(0)
    n = 4000
    t0 = T0 + np.sort(rng.integers(0, 200 * DAY, n))
    y = rng.choice(["DOWN", "FLAT", "UP"], n, p=[.15, .7, .15]).astype(object)
    ref = np.tile([0.15, 0.7, 0.15], (n, 1))
    cut = T0 + 100 * DAY
    before = t0 < cut
    good = np.where(y[:, None] == np.array(["DOWN", "FLAT", "UP"]), 0.8, 0.1)
    cand = np.where(before[:, None], good, ref)          # helps only before the cutoff
    groups = np.array([f"G{i // 20}" for i in range(n)], dtype=object)
    res = cutoff_split(y, ref, cand, t0, cut, groups)
    assert res["before"]["n"] == before.sum() and res["after"]["n"] == n - before.sum()
    assert res["before"]["gain"] > 0.5 and res["after"]["gain"] == pytest.approx(0.0)
    assert res["before"]["dm"]["dm_p"] < 1e-6 and res["after"]["dm"]["dm_p"] == 1.0
    # a timestamp string gives the same split; no events on one side gives n = 0
    iso = pd.Timestamp(cut, unit="s").isoformat()
    assert cutoff_split(y, ref, cand, t0, iso)["before"]["n"] == res["before"]["n"]
    empty = cutoff_split(y, ref, cand, t0, int(t0.max()) + 1)
    assert empty["after"] == {"n": 0, "gain": None, "dm": None} and empty["before"]["dm"] is None


# ── CLI dry run ──────────────────────────────────────────────────────────────
@pytest.fixture
def cli(monkeypatch):
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    mod = importlib.import_module("run_search")
    yield mod
    sys.path.remove(os.path.join(REPO_ROOT, "scripts"))


def test_dry_run_writes_and_fits_nothing(cli, ev, tmp_path, monkeypatch, capsys):
    data = tmp_path / "data" / "dev.parquet"
    data.parent.mkdir()
    ev.to_parquet(data, index=False)
    cfg = tmp_path / "settings.yaml"
    import yaml
    cfg.write_text(yaml.safe_dump(copy.deepcopy(CFG)))

    class Stub(Arm):
        name = "random"

        def __init__(self, seed, **grammar):
            super().__init__(seed)
            self.grammar = grammar

        def propose(self, state):
            return Proposal([PLANTED.to_json(), {"op": "nope"}, NOISE[0].to_json()][state.trial])
    fake = type(sys)("fake_random")
    Stub.__module__ = fake.__name__
    fake.Stub = Stub
    monkeypatch.setitem(sys.modules, cli.ARM_MODULES["random"], fake)
    import ksearch.eval.cv as cvmod

    def no_fit(*a, **k):
        raise AssertionError("dry run fitted a model")
    monkeypatch.setattr(cvmod, "fit_xgb", no_fit)
    monkeypatch.setattr("ksearch.search.trial.fit_xgb", no_fit)
    out = tmp_path / "runs" / "r"
    before = sorted(p for p in tmp_path.rglob("*"))
    assert cli.main(["--arm", "random", "--seed", "0", "--budget", "5", "--data", str(data),
                     "--config", str(cfg), "--out", str(out), "--dry-run"]) == 0
    assert sorted(p for p in tmp_path.rglob("*")) == before and not out.exists()
    text = capsys.readouterr().out
    assert "DRY RUN" in text and f"events {len(ev)}" in text and "OK  roll(oi_change_24h" in text
    assert "INVALID" in text and "XGBoost fits: 6 per model; 36 if" in text


def test_cli_refuses_sealed_and_missing_arm(cli, monkeypatch):
    with pytest.raises(SystemExit, match="refuses sealed"):
        cli.main(["--arm", "random", "--seed", "0", "--data",
                  os.path.join(REPO_ROOT, "sealed", "x.parquet"), "--dry-run"])
    monkeypatch.setitem(cli.ARM_MODULES, "random", "ksearch.search.arms.does_not_exist")
    with pytest.raises(SystemExit, match="not available yet"):
        cli.make_arm("random", 0, {})


def test_evaluator_refuses_post_t0_base_columns(ev):
    """The reference model is fitted on base_columns before any trial runs: a post-t0
    column there would leak into every verdict, so the evaluator refuses it up front."""
    from ksearch.search import SpecError
    bad = copy.deepcopy(CFG)
    for c in ("dmid", "label", "market_ticker"):
        bad["search"]["base_columns"] = ["mid", c]
        e2 = ev.assign(dmid=0.0)
        with pytest.raises(SpecError):
            make_evaluator(e2, bad, make_folds(e2, bad), CLF)
