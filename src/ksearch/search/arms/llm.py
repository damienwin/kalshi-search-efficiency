"""LLM search arms: Claude proposes one FeatureSpec per trial, with a mechanism.

Three conditions (research plan C3). The system prompt (task description,
operator catalog, rules, output format), model, effort, sampling settings and
output schema are byte-identical across modes; only the per-trial user message
differs, and it differs only in the information channel being ablated:

  mode        sees                                                   ablates
  full        trial index; accepted set; every earlier trial:        nothing
              formula, own rationale, status, gain, p, reason;
              status counts; operator usage of own proposals
  no_history  trial index; accepted set (formulas)                   the trial history
  blind       trial index; every earlier proposal: formula and own   all evaluation feedback
              rationale; operator usage of own proposals

Why these lines. The accepted set is itself feedback (it says which proposals
passed), so blind does not see it. Blind does see its own past proposals:
they are its own output, not information from the evaluation, and without them
it would mostly repeat itself, so full-vs-blind would measure memory rather
than adaptation. Whether an earlier proposal parsed is decided inside the arm
(the repair loop below), not by the loop, so showing an unparseable proposal's
raw JSON leaks nothing. no_history keeps the accepted set because it is the
current state of the model the next feature joins, not the record of how it
got there; it may repeat itself, and the loop counts duplicates. Every mode
sees "trial t of B"; it is the same in all three and carries no outcome.

The original Spring loop (kalshi-sentiment-predictor/automl) is followed where
it applies: one hypothesis per trial stated with a mechanism, the record of
prior attempts in the prompt so the model avoids repeats (formulas directly
here, which an LLM can compare, instead of formula hashes), a nudge toward
operator families not yet tried, one compact line per trial. Unlike there, the
arm writes no code and runs nothing: it emits a spec and a rationale, and the
loop evaluates and gates.

C4 (knowledge-cutoff contamination). The prompt describes the task, the label,
the columns and the gate in neutral terms and contains no tickers, event
names, dates or data values, so any gain that appears before the cutoff and
vanishes after it is the model's memory, not something the prompt handed it.
Text that flows back into the prompt (the model's own rationales, gate
reasons) is passed through `scrub`, which redacts year-, date- and
ticker-like tokens.

Repair and budget. A proposal is checked with `ksearch.search.parse` before it
is returned; on SpecError (or an unreadable response) the arm re-asks, up to
max_repairs times, with the error text. Repairs do NOT consume trial budget:
budget is trials (one evaluated proposal each, PLAN_1), and random/TPE emit
valid specs by construction, so charging repairs would penalise the JSON
interface rather than search quality. They are not free: every call's tokens
and latency are in Proposal.meta, so cost per trial is reported. If the last
repair still fails, the last raw proposal is returned and the loop records a
failed trial, which does cost budget. Duplicates are not repaired: only full
and blind could detect them, and the repair rule must be the same in all modes.

Reproducibility. LLM sampling is not seed-deterministic and the Messages API
has no seed parameter, so the seed is recorded but cannot be passed. The
reproducibility mechanism is the transcript: every call's request and response
(text, usage, stop reason, served model, latency) is in Proposal.meta["calls"]
and in `LLMArm.transcript`; `ReplayClient` serves a saved transcript in order
and raises if a request differs from the recording, so a search re-runs with
no API key and fails loudly if anything upstream changed.

Context growth. One line per trial (~60 tokens with the rationale capped at
160 characters), so 200 trials is ~12k tokens, far inside the context window;
history_cap (default 250, above the planned budget) never binds in the study.
If it does bind, full shows every accepted trial plus the most recent others,
blind shows the most recent proposals (a status-independent rule, so the
selection cannot leak which were accepted), and both say how many were omitted.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ksearch.search import OPS, FeatureSpec, SpecError, catalog_description, parse
from ksearch.search.arms.base import Arm, Proposal, SearchState, TrialOutcome

MODES = ("full", "no_history", "blind")
REPO_ROOT = Path(__file__).resolve().parents[4]
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"

# Defaults when settings.yaml has no search.llm section. Opus 5.5 is the current
# recommended Claude model; thinking is always on for it, effort is the only
# depth control (its default is medium, so it is set explicitly), and it
# rejects temperature, so temperature stays None unless a model accepts it.
DEFAULTS = {
    "model": "claude-opus-5-5",
    "max_tokens": 16000,      # non-streaming ceiling; thinking shares it
    "effort": "high",         # hypothesis generation is reasoning-sensitive
    "temperature": None,      # None = not sent (Opus 5.5 returns 400 on it)
    "max_repairs": 2,         # re-asks after a SpecError; 3 calls per trial at most
    "history_cap": 250,       # > planned budget of 200, so never binds in the study
}
LABEL_DEFAULTS = {"horizon_h": 4, "move_threshold": 0.03}
# How the loop judges a proposal (search: in settings.yaml), for the system prompt. Must
# equal ksearch.search.loop.DEFAULTS for these keys (tests/test_arms_llm.py checks it);
# not imported from there because the loop pulls in xgboost.
JUDGE_DEFAULTS = {"alpha": 0.05, "n_placebos": 19, "placebo_min_shift": 6, "min_valid": 100,
                  "require_signflip": False, "require_f1": False}

RATIONALE_CHARS = 160
REASON_CHARS = 100
RAW_CHARS = 120
REPAIR_ECHO_CHARS = 600

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "spec": {"type": "string"},
    },
    "required": ["rationale", "spec"],
    "additionalProperties": False,
}

COLUMN_DOCS = {
    "mid": "midpoint of the best YES bid and ask, in [0, 1]; the market's implied probability",
    "spread": "best YES ask minus best YES bid",
    "rel_spread": "spread / mid",
    "dist_from_half": "|mid - 0.5|: how far the market is from a coin flip",
    "ret_1h": "change in mid over the past 1 hour",
    "ret_4h": "change in mid over the past 4 hours",
    "ret_24h": "change in mid over the past 24 hours",
    "prev_label": "this market's previous label as -1/0/+1 (DOWN/FLAT/UP) if it had resolved by t0, else missing",
    "vol_24h": "standard deviation of hourly changes in mid over the past 24 hours",
    "range_24h": "max minus min of mid over the past 24 hours",
    "log_volume_24h": "log(1 + contracts traded in the past 24 hours)",
    "active_hours_24h": "number of the past 24 hours with any trade",
    "oi_change_24h": "change in open interest over the past 24 hours",
    "hours_since_open": "hours since the market opened",
    "hours_to_close": "hours until the market's scheduled close",
}


# ── settings ─────────────────────────────────────────────────────────────────
def load_settings(path: str | Path | None = None) -> tuple[dict, dict, dict]:
    """(search.llm merged over DEFAULTS, labels merged over LABEL_DEFAULTS, the gate
    settings the system prompt states, merged over JUDGE_DEFAULTS as the loop merges them).
    Read-only; a missing file or section falls back to the module constants."""
    path = Path(path) if path is not None else SETTINGS_PATH
    cfg: dict = {}
    if path.exists():
        import yaml
        cfg = yaml.safe_load(path.read_text()) or {}
    search = cfg.get("search") or {}
    gate = search.get("gate") or {}
    llm = {**DEFAULTS, **(search.get("llm") or {})}
    labels = {**LABEL_DEFAULTS, **{k: v for k, v in (cfg.get("labels") or {}).items() if k in LABEL_DEFAULTS}}
    judge = {**JUDGE_DEFAULTS, **{k: v for k, v in search.items() if k in JUDGE_DEFAULTS},
             **{k: v for k, v in gate.items() if k in ("require_signflip", "require_f1")}}
    if not gate.get("run_placebos", True):
        judge["n_placebos"] = 0
    return llm, labels, judge


# ── C4 guard ─────────────────────────────────────────────────────────────────
_MONTHS = ("jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|"
           "june|july|august|september|october|november|december")
SCRUB_PATTERNS = [
    re.compile(r"\b(?:KX)?[A-Z][A-Z0-9]{1,}-[A-Z0-9][A-Z0-9.-]*\b"),       # tickers: KXINX-26SEP18H1600-T7225
    re.compile(r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b"),                           # ISO dates
    re.compile(r"\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b"),                        # 3/14/25
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.I),  # March 14
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\b", re.I),     # 14 March
    re.compile(r"\b(?:19|20)\d{2}s?\b"),                                     # years
]


def scrub(text: str) -> str:
    """Redact year-, date- and ticker-like tokens from text fed back into a prompt."""
    for pat in SCRUB_PATTERNS:
        text = pat.sub("[redacted]", text)
    return text


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ── prompts (pure) ───────────────────────────────────────────────────────────
def _catalog(state: SearchState, grammar_kwargs: dict) -> str:
    accepted = inspect.signature(catalog_description).parameters
    kw = {k: v for k, v in grammar_kwargs.items() if k in accepted}
    return catalog_description(list(state.columns), state.max_depth, fmt="text", **kw)


def judging_text(judge: dict | None = None) -> str:
    """The acceptance rule as the loop applies it (loop.py, trial.py, gate.evaluate_candidate,
    grammar.degenerate_reason), stated from the same settings."""
    j = {**JUDGE_DEFAULTS, **(judge or {})}
    rules = [f"the loss reduction is significant at level {j['alpha']} under a one-sided test that allows "
             "for dependence between related markets and over time"]
    m = int(j["n_placebos"])
    if m:
        r = math.ceil(0.95 * (m + 1) - 1e-9)  # gate.placebo_threshold's order statistic
        beat = (f"the reduction obtained by every one of {m} placebo copies of the proposal" if r == m else
                f"all but the {m - r} largest of the reductions obtained by {m} placebo copies of the proposal")
        rules.append(f"the reduction is larger than {beat}; a placebo copy gives each event the proposal's "
                     f"value from at least {j['placebo_min_shift']} earlier events of the same market "
                     "(fewer in markets with few events; missing where there is no earlier event), so it "
                     "keeps the proposal's distribution but not its timing")
    if j["require_signflip"]:
        rules.append("a sign-flip permutation test of the same reduction is also significant")
    if j["require_f1"]:
        rules.append("the macro F1 score also improves significantly")
    lines = [f"It is accepted only if {'all of these hold' if len(rules) > 1 else 'this holds'}:"]
    lines += [f"{i}. {r}." for i, r in enumerate(rules, 1)]
    comm = sorted(n for n, op in OPS.items() if op.commutative)
    lines.append(f"A proposal is a failed trial if it is still outside the grammar after the corrections you "
                 f"are asked for, or if, on the training rows of any cross-validation fold, it has fewer than "
                 f"{j['min_valid']} non-missing values or only one distinct value. A proposal identical to an "
                 f"earlier one (the same tree up to the order of the arguments of {', '.join(comm)}) is "
                 "not evaluated again. Failed and repeated proposals use up a trial like any other.")
    return "\n".join(lines)


def system_prompt(state: SearchState, labels: dict | None = None,
                  grammar_kwargs: dict | None = None, judge: dict | None = None) -> str:
    """Identical in every mode: task, columns, gate, catalog, rules, output format."""
    lab = {**LABEL_DEFAULTS, **(labels or {})}
    h, thr = lab["horizon_h"], lab["move_threshold"]
    cols = "\n".join(f"  {c}: {COLUMN_DOCS.get(c, 'no description')}" for c in state.columns)
    return f"""You propose candidate features for a classifier, one per trial, as part of a feature search.

TASK
Each row is one binary prediction-market contract at one decision time t0. A YES contract pays 1 if its event happens and 0 otherwise, so its price lies in [0, 1]. Decision times sit on a {h}-hour grid, and one market's rows are at least {h} hours apart. The label is the direction of the market's mid price over the next {h} hours: UP if it rises by more than {thr}, DOWN if it falls by more than {thr}, FLAT otherwise. FLAT is the most common class. Markets come from many unrelated topics; several markets can belong to one event (for example, strikes of one ladder) and move together.

BASE COLUMNS (all known at t0; nothing after t0 is available)
{cols}

HOW A PROPOSAL IS JUDGED
The proposal is computed for every row and added as one extra column to the base columns plus the features accepted so far. A gradient-boosted tree classifier with fixed hyperparameters is refit in walk-forward cross-validation (train on the past, validate on the future), and out-of-fold class-balanced log loss is compared with and without the proposal.
{judging_text(judge)}

THE GRAMMAR (the only way to express a feature; nothing else is evaluated)
{_catalog(state, grammar_kwargs or {})}

HOW TO PROPOSE
- One hypothesis per trial, with a mechanism: why this quantity should help separate UP, DOWN and FLAT over the next {h} hours in markets like these.
- Trees already handle any monotone transform of a single column and simple thresholds on it, so slog/abs/zscore/qrank/winsor of one column alone adds little. Prefer interactions, ratios, conditional gates and a market's own history (lags, changes, rolling statistics, shocks).
- When earlier proposals are listed, do not repeat any of them and prefer operator families and columns not tried yet.

OUTPUT
Return a JSON object with two string fields:
  "rationale": one sentence: the transformation, its inputs, and the mechanism.
  "spec": the feature as a JSON expression tree in the grammar above, serialised as a string."""


def _formula_or_raw(o: TrialOutcome) -> str:
    if o.formula:
        return o.formula
    raw = o.spec_json if isinstance(o.spec_json, str) else json.dumps(o.spec_json, separators=(",", ":"))
    return "unparsed " + _clip(raw, RAW_CHARS)


def _op_usage(history: tuple[TrialOutcome, ...]) -> str:
    """Operator counts over the arm's own proposals (status-independent)."""
    ops: Counter = Counter()
    for o in history:
        try:
            spec = FeatureSpec.from_json(o.spec_json)
        except SpecError:
            continue
        ops.update(n.op for n in spec.nodes() if not n.is_col)
    return ", ".join(f"{k} x{v}" for k, v in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"


def _fmt_num(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def _capped(history: tuple[TrialOutcome, ...], cap: int, keep_accepted: bool) -> tuple[list[TrialOutcome], int]:
    if len(history) <= cap:
        return list(history), 0
    if keep_accepted:
        acc = [o for o in history if o.status == "accepted"]
        n_rest = max(cap - len(acc), 0)
        rest = [o for o in history if o.status != "accepted"][-n_rest:] if n_rest else []
        shown = sorted(acc + rest, key=lambda o: o.trial)
    else:
        shown = list(history[-cap:])
    return shown, len(history) - len(shown)


def user_prompt(mode: str, state: SearchState, rationales: dict[int, str] | None = None,
                history_cap: int = DEFAULTS["history_cap"]) -> str:
    """The per-trial message: the only part of a request that differs by mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    rationales = rationales or {}
    lines = [f"Trial {state.trial + 1} of {state.budget}."]

    if mode in ("full", "no_history"):
        lines += ["", "Current feature set: the base columns plus these accepted features, in order:"]
        acc = []
        for i, s in enumerate(state.accepted, 1):
            try:
                acc.append(f"  {i}. {FeatureSpec.from_json(s).formula()}")
            except SpecError:
                acc.append(f"  {i}. {_clip(json.dumps(s), RAW_CHARS)}")
        lines += acc or ["  (none yet)"]

    if mode == "full" and state.history:
        shown, omitted = _capped(state.history, history_cap, keep_accepted=True)
        counts = Counter(o.status for o in state.history)
        lines += ["", f"Earlier trials in this run: {len(state.history)} ("
                  + ", ".join(f"{counts[s]} {s}" for s in ("accepted", "rejected", "failed", "duplicate"))
                  + "). gain = reduction in class-balanced log loss (positive is better); "
                  "p = one-sided p-value.",
                  "Operators used so far: " + _op_usage(state.history),
                  "trial | status | gain | p | formula | your rationale | reason"]
        if omitted:
            lines.append(f"  ({omitted} earlier non-accepted trials not shown)")
        for o in shown:
            lines.append(" | ".join([
                f"  t{o.trial + 1}", o.status, _fmt_num(o.gain, "+.4f"), _fmt_num(o.p_value, ".3f"),
                _formula_or_raw(o), scrub(_clip(rationales.get(o.trial, ""), RATIONALE_CHARS)) or "-",
                scrub(_clip(o.reason, REASON_CHARS)) or "-"]))

    if mode == "blind" and state.history:
        shown, omitted = _capped(state.history, history_cap, keep_accepted=False)
        lines += ["", f"Your earlier proposals in this run: {len(state.history)}. "
                  "You are not told how they were evaluated.",
                  "Operators used so far: " + _op_usage(state.history),
                  "trial | formula | your rationale"]
        if omitted:
            lines.append(f"  ({omitted} earlier proposals not shown)")
        for o in shown:
            lines.append(" | ".join([f"  t{o.trial + 1}", _formula_or_raw(o),
                                     scrub(_clip(rationales.get(o.trial, ""), RATIONALE_CHARS)) or "-"]))

    lines += ["", "Propose the next feature."]
    return "\n".join(lines)


def repair_prompt(base: str, failures: list[tuple[str, str]]) -> str:
    """The trial's user message plus every failed attempt so far and why it failed."""
    lines = [base, "", "Your previous answer(s) for this trial were rejected by the grammar "
             "validator before any evaluation:"]
    for i, (raw, err) in enumerate(failures, 1):
        lines += [f"Attempt {i}: {_clip(raw, REPAIR_ECHO_CHARS)}", f"Error: {err}"]
    lines.append("Return a corrected answer in the same format.")
    return "\n".join(lines)


# ── clients ──────────────────────────────────────────────────────────────────
@dataclass
class Completion:
    text: str | None                  # the structured-output JSON text; None if absent (refusal, ...)
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None
    model: str | None = None          # the model that served the request
    latency_s: float = 0.0
    request_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Completion":
        return cls(**d)


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], **params: Any) -> Completion: ...


class LLMConfigError(RuntimeError):
    """No usable credentials; raised only when a real call is attempted."""


class AnthropicClient:
    """Messages API with structured output. The SDK client is built on the
    first call, so constructing this (or importing the module) needs neither
    credentials nor network."""

    def __init__(self, max_retries: int = 4, timeout: float = 600.0, fallbacks: str | list | None = None):
        self.max_retries, self.timeout, self.fallbacks = max_retries, timeout, fallbacks
        self._client = None

    def _sdk(self):
        if self._client is None:
            import anthropic
            try:
                self._client = anthropic.Anthropic(max_retries=self.max_retries, timeout=self.timeout)
            except anthropic.AnthropicError as e:
                raise LLMConfigError(
                    "no Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`, "
                    f"or replay a transcript with ReplayClient ({e})") from e
        return self._client

    def complete(self, system: str, messages: list[dict], **params: Any) -> Completion:
        import anthropic
        params = dict(params)
        extra_body = {}
        if params.get("temperature") is not None:
            extra_body["temperature"] = params.pop("temperature")
        params.pop("temperature", None)
        kwargs = dict(params, messages=messages,
                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
        sdk, create = self._sdk(), None
        if self.fallbacks is not None:
            # opt-in only: a fallback model is a different proposer (and a different cutoff, C4)
            beta = ("server-side-fallback-2026-07-01" if isinstance(self.fallbacks, str)
                    else "server-side-fallback-2026-06-01")
            kwargs.update(betas=[beta], fallbacks=self.fallbacks)
            create = sdk.beta.messages.create
        else:
            create = sdk.messages.create
        t = time.perf_counter()
        try:
            resp = create(**kwargs, extra_body=extra_body or None)
        except anthropic.AuthenticationError as e:
            raise LLMConfigError(f"Anthropic credentials rejected: {e}") from e
        latency = time.perf_counter() - t
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        u = resp.usage
        usage = {k: getattr(u, k, None) or 0 for k in
                 ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        return Completion(text=text, usage=usage, stop_reason=resp.stop_reason, model=resp.model,
                          latency_s=round(latency, 3), request_id=getattr(resp, "_request_id", None))


class FakeClient:
    """Offline client for tests: serves scripted responses in order and keeps every request.
    A response is a Completion, a dict (sent as its JSON text) or a str (sent verbatim)."""

    def __init__(self, responses: list):
        self.responses, self.requests = list(responses), []

    def complete(self, system: str, messages: list[dict], **params: Any) -> Completion:
        self.requests.append({"system": system, "messages": messages, **params})
        if not self.responses:
            raise RuntimeError("FakeClient ran out of scripted responses")
        r = self.responses.pop(0)
        if isinstance(r, Completion):
            return r
        text = json.dumps(r) if isinstance(r, dict) else r
        return Completion(text=text, usage={"input_tokens": 0, "output_tokens": 0},
                          stop_reason="end_turn", model="fake")


class ReplayMismatch(RuntimeError):
    """The replayed run asked for something the recording does not contain."""


def _first_diff(a: Any, b: Any, path: str = "") -> str | None:
    if type(a) is not type(b):
        return path or "<root>"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}"
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path} (length {len(a)} vs {len(b)})"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else (path or "<root>")


def _norm(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True))


class ReplayClient:
    """Serves a recorded transcript in order, with no API access. Raises
    ReplayMismatch if a request differs from the recording or the recording runs out."""

    def __init__(self, transcript: list[dict]):
        self.entries, self.i = list(transcript), 0

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ReplayClient":
        return cls(read_transcript(path))

    def complete(self, system: str, messages: list[dict], **params: Any) -> Completion:
        if self.i >= len(self.entries):
            raise ReplayMismatch(f"call {self.i}: the recording has only {len(self.entries)} calls")
        e = self.entries[self.i]
        got, want = _norm({"system": system, "messages": messages, **params}), _norm(e["request"])
        where = _first_diff(got, want)
        if where is not None:
            raise ReplayMismatch(f"call {self.i} (trial {e.get('trial')}, attempt {e.get('attempt')}): "
                                 f"request differs from the recording at {where}")
        self.i += 1
        return Completion.from_dict(e["response"])


def write_transcript(path: str | Path, transcript: list[dict]) -> None:
    with open(path, "w") as f:
        for e in transcript:
            f.write(json.dumps(e, sort_keys=True) + "\n")


def read_transcript(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── the arm ──────────────────────────────────────────────────────────────────
def _read_answer(c: Completion) -> tuple[str, Any, str]:
    """(rationale, spec, raw) from a completion; raises SpecError with a repairable message."""
    raw = c.text or ""
    if c.text is None:
        raise SpecError(f"the response had no answer text (stop_reason={c.stop_reason})")
    if c.stop_reason == "max_tokens":
        raise SpecError("the answer was cut off at the token limit; give a shorter answer")
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as e:  # RecursionError: absurdly deep nesting
        raise SpecError(f"the answer is not a JSON object: {type(e).__name__}: {str(e)[:200]}") from None
    if not isinstance(obj, dict) or "spec" not in obj:
        raise SpecError('the answer must be an object with string fields "rationale" and "spec"')
    spec = obj["spec"]
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except (json.JSONDecodeError, RecursionError) as e:
            raise SpecError(f'"spec" is not valid JSON: {type(e).__name__}: {str(e)[:200]}') from None
    return str(obj.get("rationale", "")), spec, raw


class LLMArm(Arm):
    """Claude as the proposer, in one of the three C3 conditions (see module docstring)."""

    def __init__(self, seed: int, mode: str = "full", client: LLMClient | None = None,
                 model: str | None = None, max_tokens: int | None = None, effort: str | None = None,
                 temperature: float | None = None, max_repairs: int | None = None,
                 history_cap: int | None = None, settings_path: str | Path | None = None,
                 **grammar_kwargs: Any):
        super().__init__(seed)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        cfg, self.labels, self.judge = load_settings(settings_path)
        pick = lambda v, k: cfg[k] if v is None else v  # noqa: E731
        self.mode, self.name = mode, f"llm_{mode}"
        self.model = pick(model, "model")
        self.max_tokens = int(pick(max_tokens, "max_tokens"))
        self.effort = pick(effort, "effort")
        self.temperature = pick(temperature, "temperature")
        self.max_repairs = int(pick(max_repairs, "max_repairs"))
        self.history_cap = int(pick(history_cap, "history_cap"))
        self.grammar_kwargs = grammar_kwargs   # forwarded to parse/catalog (e.g. include_cross)
        self.client: LLMClient = client if client is not None else AnthropicClient()
        self.transcript: list[dict] = []
        self.rationales: dict[int, str] = {}   # own rationale by trial index

    def request_params(self) -> dict:
        """Everything sent besides system and messages; identical across modes."""
        p: dict = {"model": self.model, "max_tokens": self.max_tokens,
                   "thinking": {"type": "adaptive"},
                   "output_config": {"effort": self.effort,
                                     "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}}
        if self.temperature is not None:
            p["temperature"] = self.temperature
        return p

    def build_prompt(self, state: SearchState) -> tuple[str, str]:
        """(system, user) for the first attempt of this trial."""
        return (system_prompt(state, self.labels, self.grammar_kwargs, self.judge),
                user_prompt(self.mode, state, self.rationales, self.history_cap))

    def propose(self, state: SearchState) -> Proposal:
        system, user = self.build_prompt(state)
        params = self.request_params()
        failures: list[tuple[str, str]] = []
        calls: list[dict] = []
        rationale, spec_out, valid = "", "", False
        for attempt in range(self.max_repairs + 1):
            rationale = ""
            content = user if not failures else repair_prompt(user, failures)
            messages = [{"role": "user", "content": content}]
            c = self.client.complete(system, messages, **params)
            entry = {"arm": self.name, "seed": self.seed, "trial": state.trial, "attempt": attempt,
                     "request": _norm({"system": system, "messages": messages, **params}),
                     "response": c.to_dict()}
            self.transcript.append(entry)
            calls.append(entry)
            try:
                rationale, spec, raw = _read_answer(c)
                spec_out = spec
                parse(spec, columns=list(state.columns), max_depth=state.max_depth, **self.grammar_kwargs)
                valid = True
                break
            except SpecError as e:
                spec_out = "" if c.text is None else _raw_spec(c.text)
                failures.append((c.text or "", str(e)))
        self.rationales[state.trial] = rationale
        usage = Counter()
        for e in calls:
            usage.update({k: v or 0 for k, v in e["response"]["usage"].items()})
        meta = {"arm": self.name, "mode": self.mode, "model": self.model, "seed": self.seed,
                "valid": valid, "repairs": len(calls) - 1, "errors": [err for _, err in failures],
                "usage": dict(usage), "latency_s": round(sum(e["response"]["latency_s"] for e in calls), 3),
                "served_models": sorted({e["response"]["model"] or "" for e in calls}), "calls": calls}
        return Proposal(spec_json=spec_out, rationale=rationale, meta=meta)


def _raw_spec(text: str) -> Any:
    """Best raw spec from an answer that failed: the "spec" field if the answer
    is a JSON object (decoded if it is a JSON string), else the whole text."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return text
    if isinstance(obj, dict) and "spec" in obj:
        spec = obj["spec"]
        if isinstance(spec, str):
            try:
                return json.loads(spec)
            except (json.JSONDecodeError, RecursionError):
                return spec
        return spec
    return text
