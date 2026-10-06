"""The shared feature grammar: operator catalog, interpreter, and sampling.

Every arm draws from the set of FeatureSpecs this module accepts, so the
catalog below is the search space of the study. No proposed code ever runs:
a spec is checked against the registry and evaluated by `evaluate`.

What each kind of operator may read, enforced by how the interpreter calls it:

  row     its own row's child values, nothing else
  time    the row's t0
  market  the same market's events with strictly earlier t0 (through
          _Frame.lagged, which cannot look forward) plus the row's own values.
          "w events" means the w previous rows of that market, not w*4 hours:
          events are irregular because stale quotes are dropped
  cross   rows with the same t0 in the same event group / series / everywhere.
          OPT-IN ONLY (include_cross=True): peers exist at t0 only if their
          t0+4h quote passed the label filters, which is a look-ahead
  fold    statistics of the rows in fit_index (the training fold), then the
          row's own values. Only series_target reads labels, and for a row at
          t0 it reads only training labels whose window had closed
          (t_end <= t0), so a training row never sees its own label or any
          label still unresolved at its t0

Every node's output is float with NaN where undefined; non-finite values are
turned into NaN after every node, so inf cannot propagate.

Sampling. `build_spec(suggest)` walks the tree top-down and asks `suggest(name,
choices)` for every decision; the names are fixed and finite (`flat_space`):

  n<path>.kind          "col" | "op"      (not asked at the root, which is
                                           always an operator, nor at the
                                           depth limit, which is always a column)
  n<path>.col           a base column
  n<path>.op            an operator name
  n<path>.<op>.<param>  one of that operator's choices for <param>

over the operators of space_ops(include_cross), where <path> is the string of child indices from the root ("" root, "0" its
first child, "01" that child's second child). `random_spec` is build_spec
with every suggestion drawn uniformly, and a TPE study passes
`lambda name, choices: trial.suggest_categorical(name, choices)`, so both
cover exactly the specs `validate` accepts at that depth (`spec_to_flat` is
the inverse map).
"""

import json
import math
from itertools import product

import numpy as np
import pandas as pd

from ksearch.data.labels import LABEL_VALUES
from ksearch.features.price import PRICE_FEATURES, PROVISIONAL_FEATURES
from ksearch.search.spec import COL, OPS, FeatureSpec, Op, SpecError, check_spec, col, node

HOUR = 3600
DEFAULT_COLUMNS = [c for c in PRICE_FEATURES if c not in PROVISIONAL_FEATURES]
DEFAULT_MAX_DEPTH = 3

# Known only after t0. No column list may contain them, whoever asks.
FORBIDDEN_COLUMNS = {"label", "dmid", "mid_end", "spread_end", "quote_age_end_h", "t_end"}
# Identifiers and the raw clock: read by the interpreter, never a feature themselves.
RESERVED_COLUMNS = {"market_ticker", "event_group", "series", "t0"}

LABEL_SIGN = {k: float(v) for k, v in LABEL_VALUES.items()}  # DOWN/FLAT/UP -> -1/0/+1
EPS = 1e-9

KINDS = {
    "row": "uses only the row's own values",
    "time": "uses only the row's t0 (UTC)",
    "market": "uses the same market's EARLIER events (strictly earlier t0) and the row's own values; "
              "windows count that market's previous events, which are at least 4h apart",
    "cross": "compares the row with other markets at the same t0",
    "fold": "statistics are fitted on the training fold only, then applied to every row",
}


def check_columns(columns: list[str] | None) -> list[str]:
    """The allowed base columns, refusing anything post-t0 or structural."""
    columns = list(DEFAULT_COLUMNS if columns is None else columns)
    for c in columns:
        if not isinstance(c, str):
            raise SpecError(f"column names must be strings, got {c!r}")
        if c in FORBIDDEN_COLUMNS or c.endswith("_end"):
            raise SpecError(f"column {c!r} is known only after t0 and can never be a base column")
        if c in RESERVED_COLUMNS:
            raise SpecError(f"column {c!r} is an identifier/clock column, not a feature")
    if len(set(columns)) != len(columns) or not columns:
        raise SpecError("the base column list must be non-empty and free of duplicates")
    return columns


# Opt-in only. Which markets have an event at t0 depends on their t0 + 4h
# quote passing the label filters, so the set of same-t0 peers is partly
# future information. Kept for an explicit ablation, never in the default space.
CROSS_EXCLUDED_REASON = ("cross-sectional operators are excluded from the default space because "
                         "which markets have an event at t0 depends on their t0+4h quote passing the "
                         "label filters (a look-ahead); opt in with include_cross=True")


def space_ops(include_cross: bool = False) -> tuple[str, ...]:
    """Operator names in the search space, in registry order."""
    return tuple(n for n, op in OPS.items() if include_cross or op.kind != "cross")


def _excluded(include_cross: bool) -> dict[str, str]:
    return {} if include_cross else {n: CROSS_EXCLUDED_REASON for n, op in OPS.items() if op.kind == "cross"}


def validate(spec: FeatureSpec, columns: list[str] | None = None,
             max_depth: int | None = DEFAULT_MAX_DEPTH, max_nodes: int | None = None,
             include_cross: bool = False) -> FeatureSpec:
    """Raise SpecError unless spec is in the grammar; returns spec unchanged."""
    return check_spec(spec, check_columns(columns), max_depth, max_nodes,
                      space_ops(include_cross), _excluded(include_cross))


def parse(obj, columns: list[str] | None = None, max_depth: int | None = DEFAULT_MAX_DEPTH,
          max_nodes: int | None = None, include_cross: bool = False) -> FeatureSpec:
    """JSON (dict or string) -> validated FeatureSpec. The one entry point for LLM output."""
    return validate(FeatureSpec.from_json(obj), columns, max_depth, max_nodes, include_cross)


def degenerate_reason(values: np.ndarray, fit_index=None, min_valid: int = 100) -> str | None:
    """Why an evaluated feature cannot help, or None. The harness counts a
    non-None result as a failed trial.

    Checked on the training rows when fit_index is given (a feature the model
    cannot split on in training is useless whatever it does on validation),
    else on all rows. min_valid = 100 non-NaN rows by default: below that a
    training fold of thousands of events gives XGBoost almost nothing to split on.
    """
    v = np.asarray(values, dtype=float)
    where = " on the training rows" if fit_index is not None else ""
    if fit_index is not None:
        v = v[np.asarray(fit_index)]
    v = v[~np.isnan(v)]
    if not len(v):
        return f"all values are NaN{where}"
    if len(v) < min_valid:
        return f"only {len(v)} non-NaN values{where}, fewer than min_valid={min_valid}"
    if v.min() == v.max():
        return f"constant ({v[0]:g}){where}"
    return None


# ── the rows an operator may know about ──────────────────────────────────────
class _Frame:
    """Row structure for one evaluate() call. Operators get child values as
    arrays and reach other rows only through the methods here."""

    def __init__(self, events: pd.DataFrame, fit_index):
        self.events, self.n = events, len(events)
        self.t0 = self.ints("t0")
        market, _ = pd.factorize(self.column("market_ticker"))
        if (market < 0).any():
            raise ValueError("market_ticker has missing values")
        # per-market time order, whatever order the caller's rows are in
        self.order = np.lexsort((self.t0, market))
        self.market = market[self.order]
        t = self.t0[self.order]
        new = np.ones(self.n, dtype=bool)
        new[1:] = self.market[1:] != self.market[:-1]
        if (~new[1:] & (t[1:] == t[:-1])).any():
            # "strictly earlier" is undefined between two rows of one market at one t0
            raise ValueError("events has duplicate (market_ticker, t0) rows")
        idx = np.arange(self.n)
        self.pos = idx - np.maximum.accumulate(np.where(new, idx, 0))  # earlier events in the market
        self.inv = np.empty(self.n, dtype=np.int64)
        self.inv[self.order] = idx
        self.fit = None
        if fit_index is not None:
            fit_index = np.asarray(fit_index)
            if fit_index.dtype == bool or not np.issubdtype(fit_index.dtype, np.integer):
                raise ValueError("fit_index must be positional row indices (like Fold.train)")
            if len(fit_index) == 0 or fit_index.min() < 0 or fit_index.max() >= self.n:
                raise ValueError("fit_index is empty or out of range for this events frame")
            self.fit = np.zeros(self.n, dtype=bool)
            self.fit[fit_index] = True
        self._codes: dict[str, np.ndarray] = {}

    def column(self, name: str) -> pd.Series:
        if name not in self.events.columns:
            raise ValueError(f"events has no column {name!r}")
        return self.events[name]

    def ints(self, name: str) -> np.ndarray:
        return self.column(name).to_numpy(dtype=np.int64)

    def lagged(self, x: np.ndarray, k: int) -> np.ndarray:
        """x from the same market k events earlier (x in market order); NaN if there is none.
        k >= 1 always, so this can only look back."""
        assert k >= 1
        out = np.full(self.n, np.nan)
        ok = np.flatnonzero(self.pos >= k)
        out[ok] = x[ok - k]
        return out

    def lags(self, x: np.ndarray, w: int) -> np.ndarray:
        """(n, w) matrix of the w previous events' values, most recent first."""
        return np.stack([self.lagged(x, k) for k in range(1, w + 1)], axis=1)

    def codes(self, key: str) -> np.ndarray:
        """Integer codes of a grouping key: series, event_group, hour, or all (one group)."""
        if key not in self._codes:
            if key == "all":
                c = np.zeros(self.n, dtype=np.int64)
            elif key == "hour":
                c = (self.t0 // HOUR) % 24
            else:
                c, _ = pd.factorize(self.column(key))
                if (c < 0).any():
                    raise ValueError(f"{key} has missing values")
            self._codes[key] = c.astype(np.int64)
        return self._codes[key]

    def same_t0(self, scope: str) -> np.ndarray:
        """Group id of the rows sharing this row's t0 within its scope."""
        name = f"{scope}@t0"
        if name not in self._codes:
            t, _ = pd.factorize(self.t0)
            self._codes[name] = pd.factorize(self.codes(scope) * (t.max(initial=0) + 1) + t)[0]
        return self._codes[name]


# ── registry ─────────────────────────────────────────────────────────────────
def _register(kind, arity=1, commutative=False, infix=None, **params):
    def deco(fn):
        name = fn.__name__.lstrip("_")
        assert name not in OPS and name != COL and not name[0].isdigit()
        assert all(len(set(v)) == len(v) > 0 for v in params.values())
        OPS[name] = Op(name, kind, arity, {k: tuple(v) for k, v in params.items()},
                       " ".join(fn.__doc__.split()), fn, commutative, infix)
        return fn
    return deco


LAGS = (1, 2, 3, 6)
WINDOWS = (3, 6, 12, 24)
SCOPES = ("event_group", "series", "all")


# row-wise ---------------------------------------------------------------------
@_register("row")
def _slog(x):
    """sign(x) * log(1 + |x|): compresses heavy tails, keeps the sign."""
    return np.sign(x) * np.log1p(np.abs(x))


@_register("row")
def _square(x):
    """x squared."""
    return x * x


@_register("row")
def _abs(x):
    """|x|."""
    return np.abs(x)


@_register("row")
def _sign(x):
    """-1, 0 or 1."""
    return np.sign(x)


@_register("row")
def _logit(x):
    """log(p / (1 - p)) with p = x clipped to [0.01, 0.99]; NaN unless 0 <= x <= 1
    (meant for probabilities such as mid)."""
    p = np.clip(x, 0.01, 0.99)
    return np.where((x >= 0) & (x <= 1), np.log(p / (1 - p)), np.nan)


@_register("row", 2, commutative=True, infix="+")
def _add(a, b):
    """a + b."""
    return a + b


@_register("row", 2, infix="-")
def _sub(a, b):
    """a - b."""
    return a - b


@_register("row", 2, commutative=True, infix="*")
def _mul(a, b):
    """a * b."""
    return a * b


@_register("row", 2, infix="/")
def _ratio(a, b):
    """a / b; NaN where |b| < 1e-9."""
    return np.where(np.abs(b) >= EPS, a / np.where(np.abs(b) >= EPS, b, 1.0), np.nan)


@_register("row", 2, commutative=True)
def _min(a, b):
    """Smaller of a and b (NaN if either is NaN)."""
    return np.minimum(a, b)


@_register("row", 2, commutative=True)
def _max(a, b):
    """Larger of a and b (NaN if either is NaN)."""
    return np.maximum(a, b)


def _gate(x, cond, on):
    return np.where(np.isnan(cond), np.nan, np.where(on, x, 0.0))


@_register("row", 2)
def _gate_pos(x, cond):
    """a where b > 0, else 0 (NaN where b is NaN)."""
    return _gate(x, cond, cond > 0)


# time -------------------------------------------------------------------------
@_register("time", 0)
def _hour_of_day(t0):
    """UTC hour of t0 (events sit on a 4h grid: 0, 4, ..., 20)."""
    return (t0 // HOUR) % 24


@_register("time", 0)
def _day_of_week(t0):
    """UTC day of week of t0, Monday = 0."""
    return (t0 // (24 * HOUR) + 3) % 7  # 1970-01-01 was a Thursday


# per-market, causal -------------------------------------------------------------
@_register("market", k=LAGS)
def _lag(F, x, k):
    """x at this market's k-th previous event."""
    return F.lagged(x, k)


@_register("market", k=LAGS)
def _diff(F, x, k):
    """x minus x at this market's k-th previous event."""
    return x - F.lagged(x, k)


def _roll_stats(F, x, w):
    m = F.lags(x, w)
    ok = ~np.isnan(m)
    cnt = ok.sum(1)
    mean = np.where(ok, m, 0.0).sum(1) / np.where(cnt > 0, cnt, np.nan)
    ss = (np.where(ok, m - mean[:, None], 0.0) ** 2).sum(1)
    std = np.sqrt(ss / np.where(cnt > 1, cnt - 1, np.nan))
    return m, ok, cnt, mean, std


@_register("market", stat=("mean", "std", "max", "min"), w=WINDOWS)
def _roll(F, x, stat, w):
    """Statistic of x over this market's previous w events, current event excluded.
    mean/max/min need 1 earlier value, std (sample) needs 2."""
    m, ok, cnt, mean, std = _roll_stats(F, x, w)
    if stat == "mean":
        return mean
    if stat == "std":
        return std
    ext = np.where(ok, m, -np.inf).max(1) if stat == "max" else np.where(ok, m, np.inf).min(1)
    return np.where(cnt > 0, ext, np.nan)


@_register("market", w=WINDOWS[1:])
def _roll_z(F, x, w):
    """Shock: (x - mean) / std of this market's previous w events; needs 3 earlier
    values and a non-constant history, else NaN."""
    _, _, cnt, mean, std = _roll_stats(F, x, w)
    return np.where((cnt >= 3) & (std > EPS), (x - mean) / np.where(std > EPS, std, 1.0), np.nan)


@_register("market", halflife=(1, 2, 4, 8))
def _ewm(F, x, halflife):
    """Exponentially weighted mean of x over all this market's previous events
    (current excluded); the weight halves every `halflife` events."""
    prev = pd.Series(F.lagged(x, 1))
    return prev.groupby(F.market, sort=False).ewm(halflife=halflife).mean().to_numpy()


@_register("market", 0)
def _hours_since_prev_event(F):
    """Hours since this market's previous event (4 when quotes were continuous, larger after a gap)."""
    t = F.t0[F.order].astype(float)
    return (t - F.lagged(t, 1)) / HOUR


@_register("market", 0)
def _n_prev_events(F):
    """Number of earlier events of this market."""
    return F.pos


# cross-sectional, same t0 -------------------------------------------------------
def _group_count_mean(g, x):
    ok = ~np.isnan(x)
    cnt = np.bincount(g[ok], minlength=g.max(initial=-1) + 1)
    total = np.bincount(g[ok], weights=x[ok], minlength=len(cnt))
    return cnt, total / np.where(cnt > 0, cnt, np.nan)


@_register("cross", scope=SCOPES)
def _xs_rank(g, x):
    """Rank of x among the rows at the same t0 in the same event_group (strike ladder),
    series, or all markets, scaled to (0, 1); NaN if the row has no peer."""
    cnt, _ = _group_count_mean(g, x)
    rank = pd.Series(x).groupby(g).rank(method="average").to_numpy()
    return np.where(cnt[g] >= 2, (rank - 0.5) / np.maximum(cnt[g], 1), np.nan)


@_register("cross", scope=SCOPES)
def _xs_dev(g, x):
    """x minus the mean of x over the rows at the same t0 in the same scope; NaN if no peer."""
    cnt, mean = _group_count_mean(g, x)
    return np.where(cnt[g] >= 2, x - mean[g], np.nan)


# fitted on the training fold ----------------------------------------------------
def _fit_values(F, x):
    v = x[F.fit]
    return v[~np.isnan(v)]


@_register("fold")
def _zscore(F, x):
    """(x - mean) / std with mean and std from the training fold."""
    v = _fit_values(F, x)
    if len(v) < 2 or v.std() < EPS:
        return np.full(F.n, np.nan)
    return (x - v.mean()) / v.std()


@_register("fold")
def _qrank(F, x):
    """Quantile of x within the training fold's distribution, in [0, 1]."""
    v = np.sort(_fit_values(F, x))
    if not len(v):
        return np.full(F.n, np.nan)
    q = (np.searchsorted(v, x, "left") + np.searchsorted(v, x, "right")) / (2 * len(v))
    return np.where(np.isnan(x), np.nan, q)


@_register("fold", q=(0.01, 0.05))
def _winsor(F, x, q):
    """x clipped to the training fold's [q, 1 - q] quantiles."""
    v = _fit_values(F, x)
    if not len(v):
        return np.full(F.n, np.nan)
    return np.clip(x, *np.quantile(v, [q, 1 - q]))


@_register("fold", by=("series", "hour"))
def _group_dev(F, x, by):
    """x minus the training-fold mean of x for the row's series (or UTC hour);
    NaN for a series the training fold never saw."""
    g = F.codes(by)
    sel = F.fit & ~np.isnan(x)
    cnt = np.bincount(g[sel], minlength=g.max(initial=-1) + 1)
    mean = np.bincount(g[sel], weights=x[sel], minlength=len(cnt)) / np.where(cnt > 0, cnt, np.nan)
    return x - mean[g]


@_register("fold", 2, q=(0.25, 0.5, 0.75, 0.9), side=("above", "below"))
def _gate_q(F, x, cond, q, side):
    """a where b is above (side=above) or at/below (side=below) the training fold's
    q-quantile of b, else 0 (NaN where b is NaN)."""
    v = _fit_values(F, cond)
    if not len(v):
        return np.full(F.n, np.nan)
    thr = np.quantile(v, q)
    return _gate(x, cond, cond > thr if side == "above" else cond <= thr)


@_register("fold", 0)
def _series_freq(F):
    """Share of training-fold events that belong to the row's series (0 if unseen)."""
    g = F.codes("series")
    return (np.bincount(g[F.fit], minlength=g.max(initial=-1) + 1) / F.fit.sum())[g]


@_register("fold", 0, stat=("up", "down", "move", "signed"), m=(10, 50))
def _series_target(F, stat, m):
    """Past outcome rate of the row's series: share of UP / DOWN / non-FLAT labels (or mean
    of +1/0/-1) among training-fold events of that series that had already resolved by
    this row's t0, shrunk toward the same rate over all series with prior weight m."""
    t_end = F.ints("t_end")
    fit = np.flatnonzero(F.fit)
    if (t_end[fit] <= F.t0[fit]).any():
        raise ValueError("series_target needs t_end > t0 on every training row")
    # only training labels are ever read
    sign = F.column("label").iloc[fit].map(LABEL_SIGN).to_numpy(dtype=float)
    if np.isnan(sign).any():
        raise ValueError(f"series_target needs every training label in {list(LABEL_SIGN)}")
    y = {"up": sign > 0, "down": sign < 0, "move": sign != 0, "signed": sign}[stat].astype(float)
    base = min(F.t0.min(), t_end.min())
    span = max(F.t0.max(), t_end.max()) - base + 2

    def resolved(g):
        """Per row: sum of y and count over training rows in its group with t_end <= its t0."""
        key = g[fit] * span + (t_end[fit] - base)
        o = np.argsort(key, kind="stable")
        cum = np.concatenate([[0.0], np.cumsum(y[o])])
        hi = np.searchsorted(key[o], g * span + (F.t0 - base), "right")
        lo = np.searchsorted(key[o], g * span, "left")
        return cum[hi] - cum[lo], hi - lo

    all_sum, all_n = resolved(F.codes("all"))
    own_sum, own_n = resolved(F.codes("series"))
    prior = all_sum / np.where(all_n > 0, all_n, np.nan)
    return (own_sum + m * prior) / (own_n + m)


# ── interpreter ──────────────────────────────────────────────────────────────
def _apply(op: Op, F: _Frame, xs: list[np.ndarray], p: dict) -> np.ndarray:
    if op.kind == "row":
        return op.fn(*xs, **p)
    if op.kind == "time":
        return op.fn(F.t0, **p)
    if op.kind == "market":
        return np.asarray(op.fn(F, *[x[F.order] for x in xs], **p), dtype=float)[F.inv]
    if op.kind == "cross":
        return op.fn(F.same_t0(p["scope"]), *xs)
    if F.fit is None:
        raise ValueError(f"{op.name} is fitted on the training fold: evaluate() needs fit_index")
    return op.fn(F, *xs, **p)


def evaluate(spec: FeatureSpec, events: pd.DataFrame, fit_index=None,
             columns: list[str] | None = None, include_cross: bool = False) -> np.ndarray:
    """Feature values for every row of events, in the caller's row order.

    events must hold whole market histories (pass the full dev frame, then
    index the result by fold): per-market and cross-sectional operators read
    other rows of the frame, never anything outside it. fit_index is the
    training fold as positional indices (Fold.train); it is required iff
    spec.needs_fit, and only those rows are used to fit.

    Raises SpecError for a spec outside the grammar and ValueError for a frame
    problem (missing column, duplicate (market_ticker, t0), missing fit_index).
    NaN inputs never raise.
    """
    columns = check_columns(columns)
    check_spec(spec, columns, None, None, space_ops(include_cross), _excluded(include_cross))
    F = _Frame(events, fit_index)
    memo: dict[FeatureSpec, np.ndarray] = {}

    def run(s: FeatureSpec) -> np.ndarray:
        if s not in memo:
            if s.is_col:
                out = pd.to_numeric(F.column(s.column), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            else:
                xs = [run(a) for a in s.args]
                with np.errstate(all="ignore"):
                    out = _apply(OPS[s.op], F, xs, dict(s.params))
            out = np.array(out, dtype=float)
            out[~np.isfinite(out)] = np.nan
            out.flags.writeable = False
            memo[s] = out
        return memo[s]

    return run(spec).copy()


# ── sampling ─────────────────────────────────────────────────────────────────
def build_spec(suggest, max_depth: int = DEFAULT_MAX_DEPTH,
               columns: list[str] | None = None, include_cross: bool = False) -> FeatureSpec:
    """Build a spec from suggest(name, choices) -> one element of choices.

    The root is an operator. Every other node first chooses "col" or "op",
    except at the depth limit where it is a column. See the module docstring
    for the names. Total: any suggest that returns members of `choices`
    yields a spec that validate() accepts at this max_depth.
    """
    columns = tuple(check_columns(columns))
    names = space_ops(include_cross)
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    def grow(path: str, depth_left: int) -> FeatureSpec:
        pre = f"n{path}"
        kind = "op" if not path else "col" if depth_left == 0 else suggest(f"{pre}.kind", (COL, "op"))
        if kind == COL:
            return col(suggest(f"{pre}.col", columns))
        op = OPS[suggest(f"{pre}.op", names)]
        params = {k: suggest(f"{pre}.{op.name}.{k}", v) for k, v in op.params.items()}
        return node(op.name, *[grow(path + str(i), depth_left - 1) for i in range(op.arity)], **params)

    return grow("", max_depth)


def random_spec(rng: np.random.Generator, max_depth: int = DEFAULT_MAX_DEPTH,
                columns: list[str] | None = None, include_cross: bool = False) -> FeatureSpec:
    """One draw from the grammar: build_spec with every decision uniform over its choices.

    So: the root's operator is uniform over space_ops(include_cross); each parameter is
    uniform over its choice set; each child is, with probability 1/2 each, a
    uniformly drawn base column or another operator drawn the same way, until
    max_depth nested operators, below which children are columns. This is
    uniform over the grammar's decisions, not over distinct specs: shallow
    specs are far likelier than any one deep spec, and the two spellings of a
    commutative pair double its mass.
    """
    return build_spec(lambda name, choices: choices[int(rng.integers(len(choices)))], max_depth, columns,
                      include_cross)


def flat_space(max_depth: int = DEFAULT_MAX_DEPTH, columns: list[str] | None = None,
               include_cross: bool = False) -> dict[str, tuple]:
    """Every name build_spec can ask for, with its choices: the fixed-size
    categorical space a TPE optimiser searches (conditionally, define-by-run)."""
    columns = tuple(check_columns(columns))
    names = space_ops(include_cross)
    ops = [OPS[n] for n in names]
    width = max(op.arity for op in ops)
    space: dict[str, tuple] = {}
    for depth in range(max_depth + 1):
        for path in map("".join, product(map(str, range(width)), repeat=depth)):
            pre = f"n{path}"
            if 0 < depth < max_depth:
                space[f"{pre}.kind"] = (COL, "op")
            if depth > 0:
                space[f"{pre}.col"] = columns
            if depth < max_depth:
                space[f"{pre}.op"] = names
                space.update({f"{pre}.{op.name}.{k}": v for op in ops for k, v in op.params.items()})
    return space


def spec_to_flat(spec: FeatureSpec, max_depth: int = DEFAULT_MAX_DEPTH,
                 columns: list[str] | None = None, include_cross: bool = False) -> dict[str, object]:
    """The assignment under which build_spec returns exactly this spec
    (to seed a TPE study with known specs, or to show an LLM proposal lies in the space)."""
    validate(spec, columns, max_depth, include_cross=include_cross)
    out: dict[str, object] = {}

    def walk(s: FeatureSpec, path: str, depth_left: int):
        pre = f"n{path}"
        if path and depth_left > 0:
            out[f"{pre}.kind"] = COL if s.is_col else "op"
        if s.is_col:
            out[f"{pre}.col"] = s.column
            return
        out[f"{pre}.op"] = s.op
        out.update({f"{pre}.{s.op}.{k}": v for k, v in s.params})
        for i, a in enumerate(s.args):
            walk(a, path + str(i), depth_left - 1)

    walk(spec, "", max_depth)
    return out


def space_size(max_depth: int = DEFAULT_MAX_DEPTH, columns: list[str] | None = None,
               include_cross: bool = False) -> int:
    """Number of distinct canonical specs with an operator root and depth <= max_depth."""
    n_cols = len(check_columns(columns))
    n = n_cols
    for _ in range(max_depth):
        ops = 0
        for op in map(OPS.get, space_ops(include_cross)):
            trees = n * (n + 1) // 2 if op.commutative else n ** op.arity
            ops += trees * math.prod(len(v) for v in op.params.values())
        n = n_cols + ops
    return n - n_cols


# ── description for prompts ──────────────────────────────────────────────────
def catalog(columns: list[str] | None = None, max_depth: int = DEFAULT_MAX_DEPTH,
            include_cross: bool = False) -> dict:
    """The grammar as plain data, generated from the registry."""
    columns = check_columns(columns)
    ops = [OPS[n] for n in space_ops(include_cross)]
    c = columns[0]
    example = node("ratio", node("diff", col(c), k=1), node("roll", col(c), stat="std", w=6))
    return {
        "format": {"column": {"col": "<column>"},
                   "operator": {"op": "<name>", "args": ["<node>", "..."], "params": {"<param>": "<value>"}}},
        "limits": {"root": "must be an operator", "max_depth": max_depth,
                   "max_nodes": 2 ** (max_depth + 1) - 1},
        "columns": columns,
        "kinds": {k: v for k, v in KINDS.items() if any(op.kind == k for op in ops)},
        "operators": [{"op": op.name, "kind": op.kind, "n_args": op.arity,
                       "params": {k: list(v) for k, v in op.params.items()}, "doc": op.doc}
                      for op in ops],
        "example": {"formula": example.formula(), "json": example.to_json()},
    }


def catalog_description(columns: list[str] | None = None, max_depth: int = DEFAULT_MAX_DEPTH,
                        fmt: str = "text", include_cross: bool = False) -> str:
    """The grammar for an LLM prompt: fmt "text" (compact) or "json"."""
    cat = catalog(columns, max_depth, include_cross)
    if fmt == "json":
        return json.dumps(cat, indent=1)
    lim = cat["limits"]
    lines = [
        "A feature is a JSON expression tree.",
        '  column leaf:   {"col": "<column>"}',
        '  operator node: {"op": "<name>", "args": [<node>, ...], "params": {"<param>": <value>}}',
        f"Rules: the root is an operator; at most {lim['max_depth']} nested operators and "
        f"{lim['max_nodes']} nodes; give every listed parameter exactly one of its listed values; "
        "omit args/params when an operator has none. Missing or undefined values are NaN.",
        "", "Columns: " + ", ".join(cat["columns"]), "",
        "Operators, as name(args; param in {choices}):",
    ]
    for kind, meaning in cat["kinds"].items():
        lines.append(f"[{kind}] {meaning}")
        for op in map(OPS.get, space_ops(include_cross)):
            if op.kind == kind:
                sig = ("", "x", "a, b")[op.arity]
                par = ", ".join(f"{k} in {{{', '.join(map(str, v))}}}" for k, v in op.params.items())
                sig = "; ".join(s for s in (sig, par) if s)
                lines.append(f"  {op.name}({sig}): {op.doc}")
    lines += ["", f"Example: {cat['example']['formula']}", json.dumps(cat["example"]["json"])]
    return "\n".join(lines)
