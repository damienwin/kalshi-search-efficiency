"""The two non-LLM control arms against a fake loop (no CV).

The loop here does what the real one will: build a SearchState, call
propose, parse the JSON under the default grammar, score it with a synthetic
objective, and feed the outcome back through observe.
"""

import math

import numpy as np
import pytest

from ksearch.search import DEFAULT_COLUMNS, DEFAULT_MAX_DEPTH, FeatureSpec, parse, random_spec
from ksearch.search.arms import Arm, SearchState, TrialOutcome
from ksearch.search.arms.random_ import RandomArm
from ksearch.search.arms.tpe import TPEArm

COLS = tuple(DEFAULT_COLUMNS)


def uses_ret_4h(spec: FeatureSpec) -> bool:
    return "ret_4h" in spec.columns()


def reward_ret_4h(spec: FeatureSpec, trial: int) -> TrialOutcome | None:
    """Gain 0.01 when the spec reads ret_4h, else 0, plus a little seeded noise."""
    noise = np.random.default_rng([trial, 7]).normal(0, 1e-3)
    gain = (0.01 if uses_ret_4h(spec) else 0.0) + noise
    return TrialOutcome(trial, spec.to_json(), spec.formula(), "rejected", "", gain=gain, p_value=0.5)


def run(arm: Arm, n: int, score=reward_ret_4h, columns=COLS, max_depth=DEFAULT_MAX_DEPTH):
    """Fake loop: propose, parse, score, observe. Returns parsed specs and the arms' proposals."""
    history, specs, proposals = [], [], []
    for i in range(n):
        state = SearchState(i, n, (), columns, max_depth, tuple(history))
        p = arm.propose(state)
        spec = parse(p.spec_json, list(columns), max_depth)
        out = score(spec, i)
        history.append(out)
        arm.observe(out)
        specs.append(spec)
        proposals.append(p)
    return specs, proposals


def state(i: int = 0, n: int = 10) -> SearchState:
    return SearchState(i, n, (), COLS, DEFAULT_MAX_DEPTH, ())


ARMS = [RandomArm, TPEArm]


@pytest.mark.parametrize("cls", ARMS)
def test_proposals_parse_and_are_distinct(cls):
    specs, proposals = run(cls(0), 40)
    assert all(isinstance(p.spec_json, dict) for p in proposals)
    assert len({s.key for s in specs}) == len(specs)


@pytest.mark.parametrize("cls", ARMS)
def test_deterministic_for_a_seed_and_seeds_differ(cls):
    a, _ = run(cls(3), 25)
    b, _ = run(cls(3), 25)
    c, _ = run(cls(4), 25)
    assert [s.key for s in a] == [s.key for s in b]
    assert [s.key for s in a] != [s.key for s in c]


def test_random_arm_is_random_spec_without_repeats():
    """Same stream as random_spec on the same seed, repeats skipped, nothing else."""
    specs, proposals = run(RandomArm(11), 60)
    rng, expected, seen = np.random.default_rng(11), [], set()
    while len(expected) < 60:
        s = random_spec(rng)
        if s.key not in seen:
            seen.add(s.key)
            expected.append(s.key)
    assert [s.key for s in specs] == expected
    assert specs[0].key == random_spec(np.random.default_rng(11)).key
    assert any(p.meta["redraws"] for p in proposals)  # the dedup actually fired at this length


def test_random_arm_ignores_outcomes():
    def failed(spec, trial):
        return TrialOutcome(trial, spec.to_json(), spec.formula(), "failed", "degenerate")

    def accepted(spec, trial):
        return TrialOutcome(trial, spec.to_json(), spec.formula(), "accepted", "", gain=1.0, p_value=0.0)

    runs = [run(RandomArm(5), 30, score)[0] for score in (reward_ret_4h, failed, accepted)]
    assert len({tuple(s.key for s in r) for r in runs}) == 1


def test_random_arm_falls_back_to_a_repeat_when_space_is_exhausted():
    arm = RandomArm(0, max_redraws=20)
    tiny = SearchState(0, 100, (), ("mid",), 1, ())
    keys = [parse(arm.propose(tiny).spec_json, ["mid"], 1).key for _ in range(80)]
    assert len(set(keys)) < 80
    assert arm.propose(tiny).meta["fallback"] == "duplicate"


def test_tpe_uses_feedback():
    """After startup, TPE picks the rewarded column far more often than random search."""
    n, startup = 60, 10
    tpe = [np.mean([uses_ret_4h(s) for s in run(TPEArm(seed, n_startup_trials=startup), n)[0][startup:]])
           for seed in range(3)]
    rnd = [np.mean([uses_ret_4h(s) for s in run(RandomArm(seed), n)[0][startup:]]) for seed in range(3)]
    assert np.mean(tpe) > 2 * np.mean(rnd)
    assert min(tpe) > max(rnd)


def test_tpe_startup_matches_grammar_and_meta():
    _, proposals = run(TPEArm(0, n_startup_trials=5), 8)
    assert [p.meta["startup"] for p in proposals] == [True] * 5 + [False] * 3
    assert [p.meta["optuna_trial"] for p in proposals] == sorted(p.meta["optuna_trial"] for p in proposals)


@pytest.mark.parametrize("status", ["failed", "duplicate", "rejected-no-gain", "rejected-nan"])
def test_tpe_survives_failed_and_duplicate_trials(status):
    def score(spec, trial):
        if trial % 2:
            return reward_ret_4h(spec, trial)
        st, gain = {"failed": ("failed", None), "duplicate": ("duplicate", None),
                    "rejected-no-gain": ("rejected", None), "rejected-nan": ("rejected", math.nan)}[status]
        return TrialOutcome(trial, spec.to_json(), spec.formula(), st, "x", gain=gain)

    specs, _ = run(TPEArm(1, n_startup_trials=4), 20, score)
    assert len({s.key for s in specs}) == 20


def test_tpe_all_failed_still_proposes():
    def failed(spec, trial):
        return TrialOutcome(trial, spec.to_json(), spec.formula(), "failed", "constant")

    specs, _ = run(TPEArm(2, n_startup_trials=3), 15, failed)
    assert len({s.key for s in specs}) == 15


def test_tpe_dedup_falls_back_when_space_is_tiny():
    """In a nearly exhausted space TPE's redraws hit repeats; fallback still yields valid specs."""
    arm = TPEArm(0, n_startup_trials=2, max_redraws=2)
    specs, proposals = run(arm, 90, columns=("mid",), max_depth=1)  # 70 distinct specs (space_size)
    fallbacks = [p.meta["fallback"] for p in proposals]
    assert "random" in fallbacks and fallbacks[-1] == "duplicate"
    assert all(s.key == p.meta["key"] for p, s in zip(proposals, specs))
    first_repeat = next(i for i, f in enumerate(fallbacks) if f == "duplicate")
    assert len({s.key for s in specs[:first_repeat]}) == first_repeat


def test_tpe_restart_on_accept():
    def accept_every_5th(spec, trial):
        st = "accepted" if trial % 5 == 4 else "rejected"
        return TrialOutcome(trial, spec.to_json(), spec.formula(), st, "", gain=0.01 * (trial % 5))

    keep_arm, restart_arm = TPEArm(0, n_startup_trials=3), TPEArm(0, n_startup_trials=3, restart_on_accept=True)
    keep, _ = run(keep_arm, 15, accept_every_5th)
    restart, proposals = run(restart_arm, 15, accept_every_5th)
    assert keep_arm.epoch == 0 and restart_arm.epoch == 3
    assert [p.meta["epoch"] for p in proposals] == [0] * 5 + [1] * 5 + [2] * 5
    assert len({s.key for s in restart}) == 15  # dedup spans epochs


def test_tpe_misuse_raises():
    arm = TPEArm(0)
    with pytest.raises(RuntimeError, match="no pending proposal"):
        arm.observe(TrialOutcome(0, None, None, "failed", "x"))
    p = arm.propose(state(0))
    with pytest.raises(RuntimeError, match="still pending"):
        arm.propose(state(1))
    with pytest.raises(ValueError, match="outcome is for trial 5"):
        arm.observe(TrialOutcome(5, p.spec_json, None, "failed", "x"))
    with pytest.raises(ValueError, match="not the spec this arm proposed"):
        arm.observe(TrialOutcome(0, {"col": "mid"}, None, "failed", "x"))
    with pytest.raises(ValueError, match="unknown status"):
        arm.observe(TrialOutcome(0, p.spec_json, None, "kept", "x"))
    arm.observe(TrialOutcome(0, p.spec_json, None, "rejected", "", gain=0.0))  # still pending, so this works
    with pytest.raises(ValueError, match="search space changed"):
        arm.propose(SearchState(1, 10, (), COLS[:3], DEFAULT_MAX_DEPTH, ()))


def test_grammar_kwargs_are_forwarded():
    with pytest.raises(TypeError):
        RandomArm(0, not_a_grammar_option=True).propose(state())
    with pytest.raises(TypeError):
        TPEArm(0, not_a_grammar_option=True).propose(state())
