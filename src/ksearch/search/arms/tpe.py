"""Bayesian search: optuna's TPE over the grammar's decisions, via ask/tell.

`propose` asks a trial and walks `build_spec` with `trial.suggest_categorical`
for every decision, so TPE searches the same conditional categorical space
(`flat_space`) that `random_spec` samples uniformly; during the startup phase
optuna's RandomSampler draws each decision uniformly, i.e. exactly random_spec.

Objective: maximise the gate's `gain` (class-balanced log-loss improvement
over the current reference). It is the effect size the gate tests, and the
quantity a feature search wants. The DM p-value would rank nearly the same
way but saturates near 0 and 1, losing resolution exactly among the clearly
bad and clearly good candidates. Single-objective TPE uses only the ranks of
the told values (the top gamma(n) = ceil(n/10) trials form the "good"
density), so gain's scale and outliers do not matter.

Failed trials (invalid or degenerate spec), loop-reported duplicates, and any
outcome without a finite gain are told as TrialState.PRUNED with no
intermediate value. TPE ranks such trials below every completed trial
(optuna/samplers/_tpe/sampler.py `_split_trials`), so they inform the "bad"
density and TPE learns to avoid regions that yield constant or all-NaN
features, which cost budget. Telling FAIL instead would make TPE ignore them
and keep proposing degenerate regions; telling a made-up penalty value would
be rank-equivalent but invents a number.

Duplicates. TPE on a categorical space exploits hard: once the good density
is a handful of trials, the best of n_ei_candidates draws is very often the
best spec itself. The arm never proposes a spec it already proposed (by
`.key`); a repeat it draws is told PRUNED at once, unproposed and at no
budget cost, and it asks again. PRUNED, not FAIL: re-evaluating a known spec
is worth nothing, and telling TPE so moves it to the repeat's neighbours
(a tabu effect). With FAIL the sampler ignores the repeat and draws it
again: on the synthetic test objective (one rewarded column, 150 trials,
4 seeds) FAIL hit the 10-redraw limit on 56-101 of 150 proposals, i.e. TPE
degraded towards the random fallback, while PRUNED needed 26-74 redraws in
total and never fell back. The cost is a few phantom "bad" observations,
about one per three proposals. After `max_redraws` repeats the arm proposes
an unseen uniform draw, enqueued as the trial's parameters so TPE records it,
and if even that fails (a nearly exhausted toy space) the repeat itself
(meta["fallback"] = "random" / "duplicate").

Non-stationarity: after an acceptance the reference model gains a feature,
so earlier gains were measured against a different baseline. Default: keep
all history. The reference changes by one feature at a time, so what TPE has
learnt about which columns and operators carry signal or produce degenerate
values stays mostly valid; the distortion is local to the accepted feature's
neighbourhood (which now looks better than it is), the exact spec cannot be
re-proposed (dedup), its neighbours get told their new, smaller gains as they
are tried, and optuna's default weights already ramp down older trials in
the "bad" density once it holds more than 25. Restarting on every acceptance
would re-run `n_startup_trials` random trials each time, pushing TPE towards
the random arm precisely when search is succeeding. `restart_on_accept=True`
gives the alternative (a fresh study, same seed stream, only trials since the
last acceptance) for a sensitivity check.

Conditional space: `build_spec` asks only the names on the path it takes, so
the parameter set varies per trial. Independent TPE (multivariate=False,
the original algorithm) handles that natively: each decision's good and bad
densities are fitted on the trials that asked it. The multivariate options
do not help here. Plain multivariate models jointly only the intersection of
all trials' parameters, which is the root operator alone. group=True
(experimental) partitions parameters into sets that co-occur in every trial
that has any of them; in this tree that never links a parent's operator to
its children's choices, and after 40 trials it formed accidental groups
(e.g. gate_q's params with n01.col). It hit the rewarded column no more often
and was ~8x slower, growing with the trial count. consider_endpoints and the
other Parzen options act on numeric parameters only and are deprecated in
optuna 5, so they keep their defaults, as does n_ei_candidates (24): tuning
the control on a toy objective would be its own bias. constant_liar is off
because trials are strictly sequential. n_startup_trials defaults to 10: 5-20%
of a 50-200 trial budget, enough to seed the good/bad split; the Parzen
prior keeps unseen choices reachable afterwards.

Determinism: the sampler and the fallback RNG are seeded from `seed` (and the
epoch, after a restart); storage is in memory, so the same seed and the same
sequence of outcomes give the same proposals.
"""

import math

import numpy as np
import optuna
from optuna.trial import Trial, TrialState

from ksearch.search import FeatureSpec, build_spec, random_spec, spec_to_flat
from ksearch.search.arms.base import Arm, Proposal, SearchState, TrialOutcome

optuna.logging.set_verbosity(optuna.logging.WARNING)

STATUSES = {"accepted", "rejected", "failed", "duplicate"}


class TPEArm(Arm):
    """TPE over flat_space; one optuna trial per proposal, told in observe()."""

    name = "tpe"

    def __init__(self, seed: int, n_startup_trials: int = 10, restart_on_accept: bool = False,
                 max_redraws: int = 10, n_ei_candidates: int = 24, **grammar):
        super().__init__(seed)
        if n_startup_trials < 0 or max_redraws < 0:
            raise ValueError("n_startup_trials and max_redraws must be >= 0")
        self.n_startup_trials = n_startup_trials
        self.restart_on_accept = restart_on_accept
        self.max_redraws = max_redraws
        self.n_ei_candidates = n_ei_candidates
        self.grammar = grammar          # forwarded to build_spec/random_spec/spec_to_flat
        self.fallback_rng = np.random.default_rng([seed, 1])
        self.seen: set[str] = set()
        self.space: tuple[int, tuple[str, ...]] | None = None   # (max_depth, columns), fixed per run
        self.pending: tuple[Trial, int, Proposal] | None = None  # (trial, state.trial, proposal)
        self.epoch = 0
        self.study = self._new_study()

    def _new_study(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials, n_ei_candidates=self.n_ei_candidates,
            multivariate=False, constant_liar=False,
            seed=int(np.random.SeedSequence([self.seed, self.epoch]).generate_state(1)[0]))
        return optuna.create_study(direction="maximize", sampler=sampler)

    def _build(self, trial: Trial) -> FeatureSpec:
        max_depth, columns = self.space
        return build_spec(lambda n, c: trial.suggest_categorical(n, list(c)), max_depth, list(columns),
                          **self.grammar)

    def _unseen_random(self) -> FeatureSpec | None:
        max_depth, columns = self.space
        for _ in range(self.max_redraws + 1):
            spec = random_spec(self.fallback_rng, max_depth, list(columns), **self.grammar)
            if spec.key not in self.seen:
                return spec
        return None

    def propose(self, state: SearchState) -> Proposal:
        if self.pending is not None:
            raise RuntimeError(f"TPEArm.propose: trial {self.pending[1]} is still pending; "
                               "call observe() with its outcome first")
        space = (state.max_depth, tuple(state.columns))
        if self.space is None:
            self.space = space
        elif space != self.space:
            raise ValueError(f"TPEArm.propose: search space changed mid-run from "
                             f"(max_depth, columns) = {self.space} to {space}")
        fallback = None
        trial = self.study.ask()
        spec = self._build(trial)
        redraws = 0
        while spec.key in self.seen and redraws < self.max_redraws:
            self.study.tell(trial, state=TrialState.PRUNED)  # a repeat: worthless now
            trial, redraws = self.study.ask(), redraws + 1
            spec = self._build(trial)
        max_depth, columns = self.space
        if spec.key in self.seen:
            alt = self._unseen_random()
            if alt is None:
                fallback = "duplicate"
            else:
                self.study.tell(trial, state=TrialState.PRUNED)  # a repeat: worthless now
                self.study.enqueue_trial(spec_to_flat(alt, max_depth, list(columns), **self.grammar))
                trial, fallback = self.study.ask(), "random"
                spec = self._build(trial)
                assert spec.key == alt.key, "enqueued parameters did not rebuild the fallback spec"
        self.seen.add(spec.key)
        startup = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED)
                      for t in self.study.get_trials(deepcopy=False)) < self.n_startup_trials
        proposal = Proposal(spec.to_json(), meta={
            "optuna_trial": trial.number, "epoch": self.epoch, "startup": startup,
            "redraws": redraws, "fallback": fallback, "key": spec.key})
        self.pending = (trial, state.trial, proposal)
        return proposal

    def observe(self, outcome: TrialOutcome) -> None:
        if self.pending is None:
            raise RuntimeError("TPEArm.observe: no pending proposal; call propose() first")
        trial, index, proposal = self.pending
        if outcome.trial != index:
            raise ValueError(f"TPEArm.observe: outcome is for trial {outcome.trial}, "
                             f"but the pending proposal is trial {index}")
        if outcome.spec_json is not None and outcome.spec_json != proposal.spec_json:
            raise ValueError("TPEArm.observe: outcome.spec_json is not the spec this arm proposed")
        if outcome.status not in STATUSES:
            raise ValueError(f"TPEArm.observe: unknown status {outcome.status!r}; expected one of {sorted(STATUSES)}")
        gain = outcome.gain
        if outcome.status in ("accepted", "rejected") and gain is not None and math.isfinite(gain):
            self.study.tell(trial, float(gain))
        else:
            self.study.tell(trial, state=TrialState.PRUNED)
        self.pending = None
        if outcome.status == "accepted" and self.restart_on_accept:
            self.epoch += 1
            self.study = self._new_study()
