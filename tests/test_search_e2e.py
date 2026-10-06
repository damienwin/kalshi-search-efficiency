"""End to end: run_search for all five arms on the synthetic planted-signal frame.

Same frame and planted signal as test_search_loop.py; a tiny XGBoost, 3 folds,
2 training seeds and a 6-trial budget keep it fast (~20 s). The LLM arms use
FakeClient (never the API) and are then replayed from their own transcript.
"""

import copy
import json
import os
import re

import numpy as np
import pandas as pd
import pytest
from optuna.trial import TrialState

from ksearch.search import col, node
from ksearch.search.arms.llm import MODES, FakeClient, LLMArm, ReplayClient
from ksearch.search.arms.random_ import RandomArm
from ksearch.search.arms.tpe import TPEArm
from ksearch.search.loop import TIMING_FIELDS, make_folds, run_search, search_config
from tests.test_search_loop import NOISE, PLANTED, synthetic, trials

BUDGET = 6
CLF = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0,
       "colsample_bytree": 1.0, "min_child_weight": 1, "seed": 0}
CFG = {"cv": {"n_splits": 3, "embargo_h": 4, "keep_threshold": 0.005}, "classifier": CLF,
       "search": {"budget": BUDGET, "n_seeds": 2, "n_placebos": 19, "placebo_min_shift": 6,
                  "gate": {"n_perm": 199, "n_boot": 100}}}
BAD = {"op": "frobnicate", "args": [{"col": "mid"}]}
CONST = node("sign", node("abs", node("add", col("mid"), col("mid"))))  # 1 on every row: degenerate
MUL_AB = node("mul", col("range_24h"), col("log_volume_24h"))
MUL_BA = {"op": "mul", "args": [{"col": "log_volume_24h"}, {"col": "range_24h"}]}  # same key, other order


@pytest.fixture(scope="module")
def ev():
    return synthetic()


def grammar_kw():
    return search_config(CFG)["grammar"]  # what scripts/run_search.py passes to every arm


def strip(path):
    out = []
    for r in trials(path):
        for k in TIMING_FIELDS:
            r.pop(k)
        out.append(r)
    return out


def check_complete(out, res, llm=False):
    """Exactly BUDGET trials, every promised file, fit accounting adds up."""
    t = trials(out)
    assert [r["trial"] for r in t] == list(range(BUDGET))
    assert res["status"] == "complete" and res["trials_completed"] == BUDGET
    assert sum(res["totals"].values()) == BUDGET
    for s, n in res["totals"].items():
        assert n == sum(r["status"] == s for r in t)
    want = {"oof", "run.json", "trials.jsonl"} | ({"transcript.jsonl"} if llm else set())
    assert set(os.listdir(out)) == want
    n_acc = res["totals"]["accepted"]
    assert sorted(os.listdir(os.path.join(out, "oof"))) == \
        ["final.parquet"] + [f"reference_a{i:02d}.parquet" for i in range(n_acc + 1)]
    assert json.load(open(os.path.join(out, "run.json"))) == json.loads(json.dumps(res))
    per_model = 2 * 3
    assert res["n_fits"] == per_model + sum(r["fits"] for r in t) <= res["plan"]["fits_max"]
    for r in t:
        if r["status"] in ("failed", "duplicate"):
            assert r["fits"] == 0 and r["gate"] is None
        else:
            n_pl = len(r["placebo_gains"] or [])
            assert r["fits"] == per_model * (1 + n_pl) and n_pl in (0, 19)
            assert (n_pl == 19) == (r["gate"]["dm_p"] < 0.05)
    assert [a["trial"] for a in res["accepted"]] == [r["trial"] for r in t if r["status"] == "accepted"]
    assert res["bh"]["trials"] == [r["trial"] for r in t if r["status"] in ("accepted", "rejected")]
    return t


# ── random and TPE ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("cls", [RandomArm, TPEArm], ids=["random", "tpe"])
def test_classic_arms_complete_and_rerun_identically(cls, ev, tmp_path):
    folds = make_folds(ev, CFG)
    kw = {"n_startup_trials": 3} if cls is TPEArm else {}
    runs = {}
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        out = str(tmp_path / name)
        res = run_search(cls(seed, **kw, **grammar_kw()), ev, CFG, BUDGET, seed, out, folds=folds)
        runs[name] = (check_complete(out, res), out)
    assert strip(runs["a"][1]) == strip(runs["b"][1])
    pd.testing.assert_frame_equal(pd.read_parquet(os.path.join(runs["a"][1], "oof", "final.parquet")),
                                  pd.read_parquet(os.path.join(runs["b"][1], "oof", "final.parquet")))
    assert [r["key"] for r in runs["a"][0]] != [r["key"] for r in runs["c"][0]]


class ExhaustedTPE(TPEArm):
    """TPE whose uniform fallback finds nothing new (a nearly exhausted space), so a repeat
    TPE draws is proposed and the loop logs it as a duplicate: the real duplicate path."""

    def _unseen_random(self):
        return None


def test_tpe_observe_contract_through_the_real_loop(ev, tmp_path):
    folds = make_folds(ev, CFG)
    cfg = copy.deepcopy(CFG)
    # a tiny space (2 columns, depth 1) so TPE repeats itself, and min_valid = the smallest
    # training fold so any NaN on fold 1's training rows (market operators) is degenerate
    cfg["search"].update(base_columns=["mid", "spread"], max_depth=1,
                         min_valid=min(len(f.train) for f in folds))
    arm = ExhaustedTPE(2, n_startup_trials=3, max_redraws=0, **grammar_kw())
    out = str(tmp_path / "tpe")
    budget = 10
    res = run_search(arm, ev, cfg, budget, 2, out, folds=folds)
    t = trials(out)
    assert len(t) == budget and res["trials_completed"] == budget
    st = [r["status"] for r in t]
    assert "failed" in st and "duplicate" in st and "rejected" in st
    assert arm.pending is None
    by_number = {tr.number: tr for tr in arm.study.get_trials(deepcopy=False)}
    assert all(tr.state in (TrialState.COMPLETE, TrialState.PRUNED) for tr in by_number.values())
    told = set()
    for r in t:
        tr = by_number[r["arm_meta"]["optuna_trial"]]
        told.add(tr.number)
        if r["status"] in ("accepted", "rejected"):
            assert tr.state == TrialState.COMPLETE and tr.value == r["gate"]["mean_loss_gain"]
        else:
            assert tr.state == TrialState.PRUNED and tr.value is None
        if r["status"] == "duplicate":
            assert r["arm_meta"]["fallback"] == "duplicate"
    # every other optuna trial is a redraw, told PRUNED, and costs no budget
    extra = set(by_number) - told
    assert len(extra) == sum(r["arm_meta"]["redraws"] for r in t)
    assert all(by_number[n].state == TrialState.PRUNED for n in extra)


# ── the three LLM arms, recorded and replayed ────────────────────────────────
def answer(spec, why="a mechanism"):
    return {"rationale": why, "spec": json.dumps(spec)}


def script(mode):
    """Six trials: valid; invalid then repaired; a commutative pair (duplicate);
    invalid twice (max_repairs=1, so a failed trial); a degenerate column."""
    second = PLANTED if mode == "full" else NOISE[1]
    return [answer(NOISE[0].to_json()),
            answer(BAD, "a made-up operator"), answer(second.to_json(), "repaired"),
            answer(MUL_AB.to_json()), answer(MUL_BA),
            answer(BAD), "not json at all",
            answer(CONST.to_json())]


EXPECTED = ["rejected", None, "rejected", "duplicate", "failed", "failed"]


@pytest.fixture(scope="module")
def llm_runs(ev, tmp_path_factory):
    folds = make_folds(ev, CFG)
    out = {}
    for mode in MODES:
        tmp = tmp_path_factory.mktemp(mode)
        arm = LLMArm(0, mode=mode, client=FakeClient(script(mode)), max_repairs=1, **grammar_kw())
        res = run_search(arm, ev, CFG, BUDGET, 0, str(tmp / "rec"), folds=folds)
        rep = LLMArm(0, mode=mode, client=ReplayClient.from_jsonl(tmp / "rec" / "transcript.jsonl"),
                     max_repairs=1, **grammar_kw())
        res_rep = run_search(rep, ev, CFG, BUDGET, 0, str(tmp / "rep"), folds=folds)
        out[mode] = {"arm": arm, "res": res, "dir": str(tmp / "rec"), "rep_res": res_rep,
                     "rep_dir": str(tmp / "rep"), "rep_arm": rep}
    return out


@pytest.mark.parametrize("mode", MODES)
def test_llm_arm_completes_with_failed_duplicate_and_repaired_trials(llm_runs, mode):
    r = llm_runs[mode]
    t = check_complete(r["dir"], r["res"], llm=True)
    want = list(EXPECTED)
    want[1] = "accepted" if mode == "full" else "rejected"
    assert [x["status"] for x in t] == want
    assert t[1]["arm_meta"]["repairs"] == 1 and t[1]["arm_meta"]["valid"]
    assert t[3]["duplicate_of"] == 2
    assert t[4]["reason"].startswith("invalid spec") and not t[4]["arm_meta"]["valid"]
    assert t[5]["reason"].startswith("degenerate: fold 1: constant")
    assert [x["n_calls"] for x in t] == [1, 2, 1, 1, 2, 1]
    lines = open(os.path.join(r["dir"], "transcript.jsonl")).read().splitlines()
    assert len(lines) == sum(x["n_calls"] for x in t) == len(r["arm"].transcript)
    assert not r["arm"].client.responses  # every scripted answer used, none left over
    if mode == "full":
        assert r["res"]["accepted"][0]["key"] == PLANTED.key


@pytest.mark.parametrize("mode", MODES)
def test_llm_transcript_replays_to_identical_trials(llm_runs, mode):
    r = llm_runs[mode]
    assert r["rep_res"]["status"] == "complete"
    assert strip(r["rep_dir"]) == strip(r["dir"])
    assert open(os.path.join(r["rep_dir"], "transcript.jsonl")).read() == \
        open(os.path.join(r["dir"], "transcript.jsonl")).read()
    assert r["rep_arm"].client.i == len(r["rep_arm"].client.entries)


def test_llm_modes_share_system_prompt_and_judging(llm_runs):
    """Same system prompt and parameters in every mode; the same candidate gets the
    same verdict whichever arm proposed it; prompts carry no tickers, groups or times."""
    reqs = {m: llm_runs[m]["arm"].client.requests for m in MODES}
    systems = {r["system"] for m in MODES for r in reqs[m]}
    assert len(systems) == 1 and "placebo" in systems.pop()
    params = {json.dumps({k: v for k, v in r.items() if k not in ("system", "messages")}, sort_keys=True)
              for m in MODES for r in reqs[m]}
    assert len(params) == 1
    gates = [trials(llm_runs[m]["dir"])[0]["gate"] for m in MODES]
    assert gates[0] == gates[1] == gates[2]
    ident = re.compile(r"\bM\d{2}-\d\b|\bG\d{2}\b|\b1[67]\d{8}\b")
    for m in MODES:
        for r in reqs[m]:
            text = r["system"] + "\n" + r["messages"][0]["content"]
            assert not ident.search(text), (m, ident.search(text))
    # what feedback each mode got on its last trial
    last = {m: reqs[m][-1]["messages"][0]["content"] for m in MODES}
    assert "accepted" in last["full"] and "duplicate" in last["full"]
    assert "rejected" not in last["blind"] and "Current feature set" not in last["blind"]
    assert "frobnicate" not in last["no_history"] and "Current feature set" in last["no_history"]


def test_llm_run_without_scripted_answers_crashes_cleanly(ev, tmp_path):
    """An exhausted client (an outage) stops the run: never padded with failed trials."""
    arm = LLMArm(0, mode="full", client=FakeClient([answer(NOISE[0].to_json())]), max_repairs=1,
                 **grammar_kw())
    out = str(tmp_path / "x")
    with pytest.raises(RuntimeError, match="ran out"):
        run_search(arm, ev, CFG, BUDGET, 0, out, folds=make_folds(ev, CFG))
    r = json.load(open(os.path.join(out, "run.json")))
    assert r["status"] == "crashed" and r["trials_completed"] == 1
    assert np.array_equal([x["trial"] for x in trials(out)], [0])
