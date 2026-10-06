"""Random search: uniform draws from the shared grammar, blind to every outcome.

Each proposal is `random_spec(rng, state.max_depth, state.columns)`, the
grammar's own uniform-over-decisions sampler, so this arm and the startup
phase of TPE search exactly the space the LLM arms are validated against.

Deduplication. A draw whose canonical `.key` this arm already proposed in the
run is redrawn (up to `max_redraws` times). That makes the arm sample without
replacement: the distribution of each proposal is `random_spec` conditioned on
not repeating an earlier one, i.e. renormalised over the unseen specs. It does
change the distribution, but only where it matters: the space is huge, yet a
few shallow specs carry real mass (each zero-arity root operator is ~1/28 of
all draws), so without it a 200-trial run re-spends several trials on
`hour_of_day()`. Repeating a spec can never pass the gate twice and the LLM
arms see their history and are told not to repeat themselves, so dedup makes
the control fairer, not stronger in a way the others lack. Outcomes are still
ignored: dedup reads only the arm's own proposals.

Fallback: if every redraw is a repeat (only possible in a nearly exhausted
toy space), the last draw is proposed anyway with meta["fallback"] =
"duplicate"; the loop then logs it as a duplicate trial and the budget is
spent, which is the honest outcome for a search that has run out of space.
"""

import numpy as np

from ksearch.search import random_spec
from ksearch.search.arms.base import Arm, Proposal, SearchState


class RandomArm(Arm):
    """Uniform grammar draws seeded from `seed` only; observe() is the base no-op."""

    name = "random"

    def __init__(self, seed: int, max_redraws: int = 100, **grammar):
        super().__init__(seed)
        if max_redraws < 0:
            raise ValueError("max_redraws must be >= 0")
        self.max_redraws = max_redraws
        self.grammar = grammar          # forwarded to random_spec (e.g. include_cross)
        self.rng = np.random.default_rng(seed)
        self.seen: set[str] = set()

    def propose(self, state: SearchState) -> Proposal:
        for redraws in range(self.max_redraws + 1):
            spec = random_spec(self.rng, state.max_depth, list(state.columns), **self.grammar)
            if spec.key not in self.seen:
                break
        fallback = "duplicate" if spec.key in self.seen else None
        self.seen.add(spec.key)
        return Proposal(spec.to_json(), meta={"redraws": redraws, "fallback": fallback, "key": spec.key})
