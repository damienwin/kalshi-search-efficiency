"""The contract between the search loop and the arms.

The loop owns the budget, evaluation and the gate. An arm only proposes: it
returns raw spec JSON, and the loop parses it, so an invalid proposal (an LLM
emitting an unknown operator, say) is a failed trial that still costs budget.
After every trial the loop calls observe() with the outcome; what an arm is
allowed to see of that outcome is the arm's own business (the blind LLM arm
ignores the numbers, the random arm ignores everything).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Proposal:
    spec_json: dict | str             # unparsed; the loop calls ksearch.search.parse on it
    rationale: str = ""               # the arm's stated reason (LLM arms); logged verbatim
    meta: dict = field(default_factory=dict)  # tokens, latency, raw response, sampler state, ...


@dataclass(frozen=True)
class TrialOutcome:
    trial: int                        # 0-based index in this run
    spec_json: dict | str | None      # what was proposed
    formula: str | None               # FeatureSpec.formula() if it parsed
    status: str                       # "accepted" | "rejected" | "failed" | "duplicate"
    reason: str                       # gate reason, SpecError text, degenerate reason, ...
    gain: float | None = None         # mean class-balanced log-loss gain (reference - candidate)
    p_value: float | None = None      # one-sided Diebold-Mariano p
    delta_f1: float | None = None     # pooled macro-F1 delta, reported only
    rationale: str = ""               # the proposal's rationale, so a resumed run needs no replay


@dataclass(frozen=True)
class SearchState:
    trial: int                        # index of the trial about to be proposed
    budget: int                       # total trials for this run
    accepted: tuple[dict, ...]        # spec JSON of features kept so far, in order
    columns: tuple[str, ...]          # allowed base columns
    max_depth: int
    history: tuple[TrialOutcome, ...]  # every earlier outcome, oldest first


class Arm(ABC):
    """One search strategy. Deterministic given its seed (LLM arms: given a recorded transcript)."""

    name: str = "arm"

    def __init__(self, seed: int):
        self.seed = seed

    @abstractmethod
    def propose(self, state: SearchState) -> Proposal: ...

    def observe(self, outcome: TrialOutcome) -> None:  # noqa: B027 - optional hook
        """Called once per trial, after the gate. Default: ignore."""
