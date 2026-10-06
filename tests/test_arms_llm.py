"""LLM arms, offline only: what each C3 condition sees, the repair loop, and
record/replay. No test touches the network; the Anthropic client is never built."""

import json
import re
from dataclasses import replace

import pytest

from ksearch.search import parse
from ksearch.search.arms.base import Proposal, SearchState, TrialOutcome
from ksearch.search.arms.llm import (
    MODES, AnthropicClient, FakeClient, LLMArm, LLMConfigError, ReplayClient, ReplayMismatch,
    read_transcript, scrub, system_prompt, user_prompt, write_transcript,
)
from ksearch.search.grammar import DEFAULT_COLUMNS, DEFAULT_MAX_DEPTH

ACC1 = {"op": "ratio", "args": [{"op": "diff", "args": [{"col": "mid"}], "params": {"k": 1}},
                                {"op": "roll", "args": [{"col": "mid"}], "params": {"stat": "std", "w": 6}}]}
REJ = {"op": "mul", "args": [{"col": "spread"}, {"col": "vol_24h"}]}
DUP = {"op": "mul", "args": [{"col": "vol_24h"}, {"col": "spread"}]}
BAD = {"op": "frobnicate", "args": [{"col": "mid"}]}
VALID = [
    {"op": "roll_z", "args": [{"col": "ret_4h"}], "params": {"w": 12}},
    {"op": "gate_pos", "args": [{"col": "ret_1h"}, {"col": "oi_change_24h"}]},
    {"op": "ewm", "args": [{"op": "abs", "args": [{"col": "ret_1h"}]}], "params": {"halflife": 2}},
    {"op": "series_target", "params": {"stat": "move", "m": 10}},
    {"op": "sub", "args": [{"col": "rel_spread"}, {"op": "lag", "args": [{"col": "rel_spread"}], "params": {"k": 3}}]},
]
GAIN_ACC, P_ACC, GAIN_REJ, P_REJ = 0.004817, 0.0371, -0.002963, 0.6842
REJ_REASON = "not significant at alpha"
FAIL_REASON = "unknown operator 'frobnicate'"
DATE_TICKER = "echoes KXINX-26SEP18H1600-T7225 around 2025-03-14 and March 14 in 2024"


def outcome(trial, spec, status, reason, gain=None, p=None):
    try:
        formula = parse(spec).formula()
    except Exception:
        formula = None
    return TrialOutcome(trial=trial, spec_json=spec, formula=formula, status=status, reason=reason,
                        gain=gain, p_value=p, delta_f1=None if gain is None else 0.0123)


HISTORY = (
    outcome(0, ACC1, "accepted", "p below alpha", GAIN_ACC, P_ACC),
    outcome(1, REJ, "rejected", REJ_REASON, GAIN_REJ, P_REJ),
    outcome(2, BAD, "failed", FAIL_REASON),
    outcome(3, DUP, "duplicate", "same canonical key as trial 2"),
)
RATIONALES = {0: "momentum scaled by recent noise", 1: "illiquid and volatile markets jump",
              2: "made-up operator " + DATE_TICKER, 3: "same idea again"}


def state(history=HISTORY, accepted=(ACC1,)):
    return SearchState(trial=len(history), budget=200, accepted=tuple(accepted),
                       columns=tuple(DEFAULT_COLUMNS), max_depth=DEFAULT_MAX_DEPTH, history=tuple(history))


def answer(spec, rationale="a mechanism"):
    return {"rationale": rationale, "spec": json.dumps(spec)}


def arm(mode="full", responses=(), **kw):
    kw.setdefault("settings_path", "/nonexistent/settings.yaml")  # module defaults, no repo config
    return LLMArm(seed=0, mode=mode, client=FakeClient(list(responses)), **kw)


# ── what each mode sees ──────────────────────────────────────────────────────
def test_system_prompt_and_params_identical_across_modes():
    s = state()
    reqs = []
    for m in MODES:
        a = arm(m, [answer(VALID[0])])
        a.propose(s)
        reqs.append(a.client.requests[0])
    for r in reqs[1:]:
        assert r["system"] == reqs[0]["system"]
        assert {k: v for k, v in r.items() if k != "messages"} == {k: v for k, v in reqs[0].items() if k != "messages"}


def test_full_sees_everything():
    u = user_prompt("full", state(), RATIONALES)
    for needle in ("+0.0048", "0.037", "-0.0030", "0.684", "accepted", "rejected", "failed", "duplicate",
                   REJ_REASON, FAIL_REASON, parse(REJ).formula(), parse(ACC1).formula(),
                   "unparsed", "frobnicate", RATIONALES[1]):
        assert needle in u, needle


def test_blind_sees_own_proposals_but_no_feedback():
    u = user_prompt("blind", state(), RATIONALES)
    assert parse(ACC1).formula() in u and parse(REJ).formula() in u and "frobnicate" in u
    assert RATIONALES[1] in u
    for word in ("accepted", "rejected", "failed", "duplicate", "gain", "p-value",
                 REJ_REASON, FAIL_REASON, "Current feature set"):
        assert not re.search(rf"\b{re.escape(word)}\b", u), word
    full_text = system_prompt(state()) + u
    for num in ("0.0048", "0.004817", "4817", "0.037", "0.0371", "0.0030", "0.00296", "2963",
                "0.684", "6842", "0.0123"):
        assert num not in full_text, num


def test_blind_is_invariant_to_every_outcome_field():
    """Same proposals, different evaluations and accepted sets -> byte-identical blind request."""
    flipped = tuple(replace(o, status="accepted", reason="r", gain=0.5, p_value=0.001) for o in HISTORY)
    a, b = arm("blind", [answer(VALID[0])]), arm("blind", [answer(VALID[0])])
    a.rationales, b.rationales = dict(RATIONALES), dict(RATIONALES)
    a.propose(state(HISTORY, accepted=(ACC1,)))
    b.propose(state(flipped, accepted=(ACC1, REJ, DUP)))
    assert a.client.requests == b.client.requests


def test_no_history_sees_accepted_set_only():
    u = user_prompt("no_history", state(), RATIONALES)
    assert parse(ACC1).formula() in u
    for needle in (parse(REJ).formula(), parse(DUP).formula(), "frobnicate", "rejected", "failed",
                   "duplicate", "0.0048", "0.037", REJ_REASON, RATIONALES[0], RATIONALES[1]):
        assert needle not in u, needle
    # invariant to the history given the accepted set
    assert u == user_prompt("no_history", replace(state(HISTORY[:1]), trial=len(HISTORY)), {})


def test_empty_history_prompts():
    s = state(history=(), accepted=())
    assert "(none yet)" in user_prompt("full", s)
    assert "earlier proposals" not in user_prompt("blind", s)


def test_history_cap_rule_does_not_leak_into_blind():
    hist = tuple(outcome(i, VALID[i % 5], "accepted" if i == 0 else "rejected", "r", 0.001, 0.5)
                 for i in range(10))
    full = user_prompt("full", state(hist), {}, history_cap=3)
    blind = user_prompt("blind", state(hist), {}, history_cap=3)
    assert "  t1 | accepted" in full and "(7 earlier non-accepted trials not shown)" in full
    assert "  t1 |" not in blind and "(7 earlier proposals not shown)" in blind
    assert blind == user_prompt("blind", state(tuple(replace(o, status="rejected") for o in hist)), {},
                                history_cap=3)


# ── C4: no tickers, dates or years in any prompt ─────────────────────────────
TICKER = re.compile(r"\b(?:KX[A-Z0-9]+|[A-Z]{2,}-\d{2}[A-Z]{3}\d{2}\w*)\b")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|"
                  r"Nov|Dec)[a-z]*\.? \d{1,2}\b")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@pytest.mark.parametrize("mode", MODES)
def test_no_tickers_dates_or_years_in_prompts(mode):
    hist = HISTORY[:2] + (replace(HISTORY[2], reason="failed on " + DATE_TICKER),) + HISTORY[3:]
    a = arm(mode, [answer(VALID[0])])
    a.rationales = dict(RATIONALES)
    a.propose(state(hist))
    req = a.client.requests[0]
    text = req["system"] + "\n" + req["messages"][0]["content"]
    for pat in (TICKER, DATE, YEAR):
        assert not pat.search(text), (pat.pattern, pat.search(text))


def test_scrub():
    out = scrub(DATE_TICKER)
    assert "KXINX" not in out and "2025" not in out and "2024" not in out and "March 14" not in out


# ── output, repair, give-up ──────────────────────────────────────────────────
def test_valid_fixtures_parse():
    for spec in VALID:
        parse(json.loads(json.dumps(spec)))


@pytest.mark.parametrize("mode", MODES)
def test_valid_answer_returns_parseable_spec(mode):
    a = arm(mode, [answer(v) for v in VALID])
    for i, v in enumerate(VALID):
        p = a.propose(state(history=(), accepted=()) if i == 0 else state())
        assert isinstance(p, Proposal) and p.meta["valid"] and p.meta["repairs"] == 0
        assert parse(p.spec_json).key == parse(v).key and p.rationale == "a mechanism"
    assert a.name == f"llm_{mode}"


def test_repair_then_valid():
    a = arm("full", [answer(BAD, "bad"), answer(VALID[1], "good")])
    p = a.propose(state())
    assert p.meta["valid"] and p.meta["repairs"] == 1 and len(p.meta["calls"]) == 2
    assert p.spec_json == VALID[1] and p.rationale == "good"
    second = a.client.requests[1]["messages"][0]["content"]
    assert "unknown operator 'frobnicate'" in second and "Attempt 1:" in second
    assert second.startswith(a.client.requests[0]["messages"][0]["content"])
    assert "unknown operator" in p.meta["errors"][0]


def test_unreadable_answers_are_repaired_too():
    a = arm("blind", ["not json", {"rationale": "x", "spec": "{bad"}, answer(VALID[2])])
    p = a.propose(state())
    assert p.meta["valid"] and p.meta["repairs"] == 2
    assert "not a JSON object" in p.meta["errors"][0] and '"spec" is not valid JSON' in p.meta["errors"][1]


def test_gives_up_after_max_repairs_and_returns_last_raw():
    last = {"op": "roll", "args": [{"col": "mid"}], "params": {"stat": "median", "w": 6}}
    a = arm("full", [answer(BAD), answer(BAD), answer(last, "last")], max_repairs=2)
    p = a.propose(state())
    assert not p.meta["valid"] and p.meta["repairs"] == 2 and len(a.client.requests) == 3
    assert p.spec_json == last and p.rationale == "last"
    with pytest.raises(Exception):
        parse(p.spec_json)  # the loop will record a failed trial


def test_meta_records_usage_and_latency():
    from ksearch.search.arms.llm import Completion
    c = Completion(text=json.dumps(answer(VALID[0])), usage={"input_tokens": 1200, "output_tokens": 300},
                   stop_reason="end_turn", model="claude-opus-5-5", latency_s=2.5)
    p = arm("full", [c]).propose(state())
    assert p.meta["usage"] == {"input_tokens": 1200, "output_tokens": 300}
    assert p.meta["latency_s"] == 2.5 and p.meta["served_models"] == ["claude-opus-5-5"]
    assert p.meta["calls"][0]["request"]["model"] == "claude-opus-5-5"


# ── record / replay ──────────────────────────────────────────────────────────
def _run(a, n=3):
    """A tiny stand-in loop: accept every valid proposal."""
    hist, acc, props = [], [], []
    for t in range(n):
        s = SearchState(trial=t, budget=n, accepted=tuple(acc), columns=tuple(DEFAULT_COLUMNS),
                        max_depth=DEFAULT_MAX_DEPTH, history=tuple(hist))
        p = a.propose(s)
        props.append(p)
        ok = p.meta["valid"]
        hist.append(TrialOutcome(t, p.spec_json, parse(p.spec_json).formula() if ok else None,
                                 "accepted" if ok else "failed", "r", 0.001 if ok else None, 0.01 if ok else None))
        if ok:
            acc.append(p.spec_json)
    return props


def test_transcript_round_trip_and_mismatch(tmp_path):
    rec = arm("full", [answer(VALID[0]), answer(BAD), answer(VALID[1]), answer(VALID[2])])
    props = _run(rec)
    path = tmp_path / "t.jsonl"
    write_transcript(path, rec.transcript)
    assert len(read_transcript(path)) == 4

    rep = LLMArm(seed=0, mode="full", client=ReplayClient.from_jsonl(path), settings_path="/nonexistent")
    assert [(p.spec_json, p.rationale, p.meta) for p in _run(rep)] == \
           [(p.spec_json, p.rationale, json.loads(json.dumps(p.meta, sort_keys=True))) for p in props]

    other = LLMArm(seed=0, mode="blind", client=ReplayClient.from_jsonl(path), settings_path="/nonexistent")
    with pytest.raises(ReplayMismatch, match=r"call 0 .*messages\[0\]\.content"):
        _run(other)

    short = LLMArm(seed=0, mode="full", client=ReplayClient(read_transcript(path)[:2]),
                   settings_path="/nonexistent")
    with pytest.raises(ReplayMismatch, match="only 2 calls"):
        _run(short)


# ── configuration and the real client, without network ───────────────────────
def test_settings_override(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("search:\n  llm:\n    model: claude-sonnet-5-5\n    max_repairs: 1\n"
                   "labels:\n  horizon_h: 4\n  move_threshold: 0.05\n")
    a = LLMArm(seed=1, mode="blind", client=FakeClient([]), settings_path=cfg)
    assert a.model == "claude-sonnet-5-5" and a.max_repairs == 1 and a.effort == "high"
    assert "more than 0.05" in a.build_prompt(state())[0]
    b = LLMArm(seed=1, client=FakeClient([]), settings_path=cfg, model="m", max_repairs=0)
    assert b.model == "m" and b.max_repairs == 0
    with pytest.raises(ValueError):
        LLMArm(seed=0, mode="half", client=FakeClient([]))


def test_defaults_and_request_shape():
    a = arm()
    p = a.request_params()
    assert p["model"] == "claude-opus-5-5" and p["output_config"]["effort"] == "high"
    assert p["output_config"]["format"]["type"] == "json_schema" and "temperature" not in p


def test_anthropic_client_is_lazy_and_missing_credentials_is_clear(monkeypatch):
    import anthropic

    def no_creds(*a, **k):
        raise anthropic.CredentialsError("no credentials found")

    monkeypatch.setattr(anthropic, "Anthropic", no_creds)
    c = AnthropicClient()  # constructing needs nothing
    a = LLMArm(seed=0, client=c, settings_path="/nonexistent")
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        a.propose(state())


# ── the "how a proposal is judged" paragraph matches the loop ────────────────
def test_system_prompt_states_the_real_acceptance_rule(tmp_path):
    """The prompt names the placebo requirement and the exact degenerate rule, from the
    same settings the loop reads, identically in every mode."""
    from ksearch.search.arms.llm import JUDGE_DEFAULTS
    from ksearch.search.loop import DEFAULTS as LOOP_DEFAULTS
    assert {k: LOOP_DEFAULTS[k] for k in ("alpha", "n_placebos", "placebo_min_shift", "min_valid")} == \
        {k: JUDGE_DEFAULTS[k] for k in ("alpha", "n_placebos", "placebo_min_shift", "min_valid")}
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("search:\n  alpha: 0.01\n  n_placebos: 39\n  placebo_min_shift: 3\n  min_valid: 250\n")
    systems = {LLMArm(seed=0, mode=m, client=FakeClient([]), settings_path=cfg).build_prompt(state())[0]
               for m in MODES}
    assert len(systems) == 1
    text = systems.pop()
    for needle in ("level 0.01", "39 placebo copies", "all but the 1 largest", "at least 3 earlier events of the same market",
                   "fewer than 250 non-missing values or only one distinct value",
                   "training rows of any cross-validation fold", "use up a trial"):
        assert needle in text, needle
    # the repo defaults: 19 placebos means beating every one of them
    default = system_prompt(state())
    assert "every one of 19 placebo copies" in default and "fewer than 100 non-missing" in default
    # placebos switched off in the gate: the prompt must not claim them
    cfg.write_text("search:\n  gate:\n    run_placebos: false\n    require_signflip: true\n")
    off = LLMArm(seed=0, client=FakeClient([]), settings_path=cfg).build_prompt(state())[0]
    assert "placebo" not in off and "sign-flip" in off


def test_absurdly_nested_answer_is_repaired_not_a_crash():
    """json.loads raises RecursionError, not JSONDecodeError, on very deep nesting; an
    answer like that must be a repairable grammar error, not an exception that stops the run."""
    deep = "[" * 100_000 + "]" * 100_000
    a = arm("full", [deep, {"rationale": "x", "spec": deep}, answer(VALID[0])])
    p = a.propose(state())
    assert p.meta["valid"] and p.meta["repairs"] == 2
    assert "not a JSON object" in p.meta["errors"][0] and "not valid JSON" in p.meta["errors"][1]
    with pytest.raises(Exception, match="not valid JSON"):
        parse(deep)  # the loop's parse: a SpecError, so a failed trial
    from ksearch.search import SpecError
    with pytest.raises(SpecError):
        parse(deep)
