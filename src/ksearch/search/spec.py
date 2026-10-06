"""FeatureSpec: a candidate feature as a typed expression tree, never as code.

Every search arm (random, TPE, LLM) emits one of these and nothing else. A
node is either a base column or an operator from the registry OPS with child
specs and parameters drawn from that operator's finite choice sets. The
registry is filled by grammar.py and is the only thing validation, sampling,
hashing, and the LLM prompt read, so the arms cannot drift onto different
spaces.

JSON form (plain dicts, lossless):
    {"col": "mid"}
    {"op": "roll", "args": [{"col": "mid"}], "params": {"stat": "mean", "w": 6}}
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

COL = "col"          # the leaf pseudo-operator
JSON_NESTING_CAP = 64  # from_json refuses deeper input before it can exhaust the stack


class SpecError(ValueError):
    """A spec that is malformed or outside the grammar. The message names the
    offending node and the allowed alternatives, so a proposer can repair it."""


@dataclass(frozen=True)
class Op:
    """Schema and implementation of one operator.

    kind: row | time | market | cross | fold. It states which rows the operator
    may read, and the interpreter dispatches on it; fold operators are fitted
    on the training fold. params maps each parameter to its finite, ordered
    choice set.
    """
    name: str
    kind: str
    arity: int
    params: dict[str, tuple]
    doc: str
    fn: Callable = field(repr=False, compare=False)
    commutative: bool = False
    infix: str | None = None


OPS: dict[str, Op] = {}


@dataclass(frozen=True)
class FeatureSpec:
    op: str
    args: tuple["FeatureSpec", ...] = ()
    params: tuple[tuple[str, object], ...] = ()  # sorted (name, value) pairs

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))
        p = self.params.items() if isinstance(self.params, dict) else self.params
        object.__setattr__(self, "params", tuple(sorted(((str(k), v) for k, v in p), key=lambda kv: kv[0])))

    # ── structure ────────────────────────────────────────────────────────────
    @property
    def is_col(self) -> bool:
        return self.op == COL

    @property
    def column(self) -> str:
        return dict(self.params)["name"]

    @property
    def depth(self) -> int:
        """Operator nodes on the longest root-to-leaf path; a bare column is 0."""
        return 0 if self.is_col else 1 + max((a.depth for a in self.args), default=0)

    @property
    def size(self) -> int:
        """All nodes, columns included."""
        return 1 + sum(a.size for a in self.args)

    def nodes(self):
        yield self
        for a in self.args:
            yield from a.nodes()

    def columns(self) -> set[str]:
        return {n.column for n in self.nodes() if n.is_col}

    @property
    def needs_fit(self) -> bool:
        """True if any node is fitted on the training fold (evaluate then needs fit_index)."""
        return any(OPS[n.op].kind == "fold" for n in self.nodes() if n.op in OPS)

    # ── JSON ─────────────────────────────────────────────────────────────────
    def to_json(self) -> dict:
        if self.is_col:
            return {"col": self.column}
        out: dict = {"op": self.op}
        if self.args:
            out["args"] = [a.to_json() for a in self.args]
        if self.params:
            out["params"] = dict(self.params)
        return out

    @classmethod
    def from_json(cls, obj, _nest: int = 0) -> "FeatureSpec":
        """Structure only; call grammar.validate (or grammar.parse) to check it against the grammar."""
        if isinstance(obj, str) and _nest == 0:
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, RecursionError) as e:  # RecursionError: absurdly deep nesting
                raise SpecError(f"spec is not valid JSON: {type(e).__name__}: {str(e)[:200]}") from None
        if _nest > JSON_NESTING_CAP:
            raise SpecError(f"spec is nested deeper than {JSON_NESTING_CAP} levels")
        if not isinstance(obj, dict):
            raise SpecError(f"a node must be an object like {{\"col\": ...}} or {{\"op\": ...}}, "
                            f"got {type(obj).__name__}: {str(obj)[:60]!r}")
        if "col" in obj:
            if set(obj) != {"col"} or not isinstance(obj["col"], str):
                raise SpecError(f"a column leaf must be exactly {{\"col\": \"<name>\"}}, got {obj!r}")
            return col(obj["col"])
        if not isinstance(obj.get("op"), str):
            raise SpecError(f"a node needs a string \"op\" (or \"col\" for a leaf), got keys {sorted(obj)}")
        extra = set(obj) - {"op", "args", "params"}
        if extra:
            raise SpecError(f"node {obj['op']!r} has unknown keys {sorted(extra)}; "
                            "allowed keys are op, args, params")
        args, params = obj.get("args", []), obj.get("params", {})
        if not isinstance(args, list):
            raise SpecError(f"{obj['op']!r}: \"args\" must be a list of nodes")
        if not isinstance(params, dict):
            raise SpecError(f"{obj['op']!r}: \"params\" must be an object of name: value")
        return cls(obj["op"], tuple(cls.from_json(a, _nest + 1) for a in args), params)

    # ── identity ─────────────────────────────────────────────────────────────
    def canonical(self) -> "FeatureSpec":
        """Same feature, one spelling: the arguments of commutative operators
        are sorted. Purely structural; a*a and square(a) stay distinct."""
        if self.is_col:
            return self
        args = tuple(a.canonical() for a in self.args)
        if self.op in OPS and OPS[self.op].commutative:
            args = tuple(sorted(args, key=_dumps))
        return FeatureSpec(self.op, args, self.params)

    @property
    def key(self) -> str:
        """Stable 16-hex id of the canonical form, equal across processes and sessions."""
        return hashlib.sha256(_dumps(self.canonical()).encode()).hexdigest()[:16]

    def formula(self) -> str:
        if self.is_col:
            return self.column
        op = OPS.get(self.op)
        parts = [a.formula() for a in self.args]
        if op is not None and op.infix and len(parts) == 2:
            return f"({parts[0]} {op.infix} {parts[1]})"
        parts += [f"{k}={v}" for k, v in self.params]
        return f"{self.op}({', '.join(parts)})"

    def __str__(self) -> str:
        return self.formula()


def _dumps(spec: FeatureSpec) -> str:
    return json.dumps(spec.to_json(), sort_keys=True, separators=(",", ":"))


def col(name: str) -> FeatureSpec:
    return FeatureSpec(COL, (), (("name", name),))


def node(op: str, *args: FeatureSpec, **params) -> FeatureSpec:
    """Shorthand constructor: node("roll", col("mid"), stat="mean", w=6)."""
    return FeatureSpec(op, args, tuple(params.items()))


def check_spec(spec: FeatureSpec, columns: list[str], max_depth: int | None,
               max_nodes: int | None = None, ops: tuple[str, ...] | None = None,
               excluded: dict[str, str] | None = None) -> FeatureSpec:
    """Raise SpecError unless spec lies in the grammar over `columns`.

    max_depth None skips the size limits (the interpreter uses that; it still
    refuses unknown operators, columns and parameter values). max_nodes
    defaults to the node count of a full binary tree of max_depth, so by
    default depth is the only limit that binds. ops restricts the operators
    (default: all of OPS); excluded gives the reason for a registered
    operator that ops leaves out.
    """
    ops = tuple(OPS) if ops is None else tuple(ops)
    if not isinstance(spec, FeatureSpec):
        raise SpecError(f"expected a FeatureSpec, got {type(spec).__name__}")
    allowed = list(columns)
    for n in spec.nodes():
        if n.is_col:
            if n.args or [k for k, _ in n.params] != ["name"]:
                raise SpecError("a column leaf takes exactly one parameter, name, and no args")
            if n.column not in allowed:
                raise SpecError(f"unknown column {n.column!r}; allowed columns: {', '.join(allowed)}")
            continue
        op = OPS.get(n.op) if n.op in ops else None
        if op is None:
            why = (excluded or {}).get(n.op)
            if why:
                raise SpecError(f"operator {n.op!r} is not in this search space: {why}")
            raise SpecError(f"unknown operator {n.op!r}; allowed operators: {', '.join(ops)}")
        if len(n.args) != op.arity:
            raise SpecError(f"{n.op} takes {op.arity} args, got {len(n.args)}")
        if any(not isinstance(a, FeatureSpec) for a in n.args):
            raise SpecError(f"{n.op}: every arg must be a FeatureSpec")
        given = dict(n.params)
        if len(given) != len(n.params) or set(given) != set(op.params):
            want = ", ".join(f"{k} in {list(v)}" for k, v in op.params.items()) or "no parameters"
            raise SpecError(f"{n.op} takes {want}; got {sorted(k for k, _ in n.params)}")
        for k, v in given.items():
            # type(v) is checked so that True, 6.0 or "6" cannot stand in for 6
            if not any(type(v) is type(c) and v == c for c in op.params[k]):
                raise SpecError(f"{n.op}: {k}={v!r} is not allowed; {k} must be one of {list(op.params[k])}")
    if max_depth is not None:
        if spec.is_col:
            raise SpecError("a bare column is already a base feature; the root must be an operator")
        if spec.depth > max_depth:
            raise SpecError(f"spec has depth {spec.depth}, the limit is {max_depth} nested operators")
        limit = max_nodes if max_nodes is not None else 2 ** (max_depth + 1) - 1
        if spec.size > limit:
            raise SpecError(f"spec has {spec.size} nodes, the limit is {limit}")
    return spec
