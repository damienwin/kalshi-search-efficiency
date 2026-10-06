"""The shared feature grammar: spec validity, identity, and leakage by construction.

The leak tests are parametrised over the registry, so a newly added operator
is tested without anyone writing a test for it.
"""

import json
import time
from functools import partial
from itertools import product

import numpy as np
import pandas as pd
import pytest

from ksearch.search import (
    DEFAULT_COLUMNS, DEFAULT_MAX_DEPTH, OPS, FeatureSpec, SpecError, build_spec, catalog,
    catalog_description, check_columns, col, degenerate_reason, flat_space, node, parse,
    random_spec, space_ops, space_size, spec_to_flat, validate,
)
from ksearch.search import evaluate as evaluate_default
from ksearch.search.spec import Op

# The leak, NaN and order tests cover every registered operator, including the
# opt-in cross-sectional ones, so they evaluate with include_cross=True.
evaluate = partial(evaluate_default, include_cross=True)
CROSS = {n for n, op in OPS.items() if op.kind == "cross"}

HOUR = 3600
A, B = "mid", "ret_4h"


def synthetic_events(seed: int = 0, n_groups: int = 24) -> pd.DataFrame:
    """Strike ladders on a 4h grid with gaps, NaNs, several series; rows shuffled."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        series = f"S{g % 4}"
        start = 1_700_000_000 - 1_700_000_000 % (4 * HOUR) + int(rng.integers(0, 30)) * 4 * HOUR
        grid = start + 4 * HOUR * np.arange(int(rng.integers(8, 40)))
        for s in range(int(rng.integers(1, 5))):
            for t0 in grid[rng.random(len(grid)) > 0.25]:
                rows.append({"market_ticker": f"{series}-G{g}-T{s}", "event_group": f"{series}-G{g}",
                             "series": series, "t0": int(t0), "t_end": int(t0) + 4 * HOUR,
                             "label": rng.choice(["DOWN", "FLAT", "UP"], p=[0.15, 0.7, 0.15])})
    ev = pd.DataFrame(rows)
    n = len(ev)
    for c in DEFAULT_COLUMNS:
        ev[c] = rng.normal(0, 1, n) * rng.choice([1e-3, 1, 1e3])
    ev["mid"] = rng.uniform(0.01, 0.99, n)
    ev["prev_label"] = rng.choice([-1.0, 0.0, 1.0], n)
    ev["ret_4h"] = np.where(rng.random(n) < 0.3, 0.0, ev["ret_4h"])  # exact zeros, as in the data
    for c in DEFAULT_COLUMNS:
        ev.loc[rng.random(n) < 0.1, c] = np.nan
    return ev.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="module")
def ev():
    return synthetic_events()


def all_param_specs():
    """Every operator with every parameter combination, over two base columns."""
    for name, op in OPS.items():
        for values in product(*op.params.values()):
            yield node(name, *[col(A), col(B)][:op.arity], **dict(zip(op.params, values)))


OP_SPECS = list(all_param_specs())
IDS = [s.formula() for s in OP_SPECS]


def early_fit(ev) -> tuple[int, np.ndarray, np.ndarray]:
    T = int(np.median(ev.t0.unique()))
    early = (ev.t0 <= T).to_numpy()
    return T, early, np.flatnonzero(early)


def assert_causal(spec: FeatureSpec, ev: pd.DataFrame, seed: int = 1):
    """L1 for features: rows after T, their values, labels and very existence,
    cannot move any feature value at or before T (fit on rows at or before T)."""
    rng = np.random.default_rng(seed)
    T, early, fit = early_fit(ev)
    base = evaluate(spec, ev, fit)
    late = ~early
    mutated = ev.copy()
    for c in DEFAULT_COLUMNS:
        mutated.loc[late, c] = np.where(rng.random(late.sum()) < 0.2, np.nan,
                                        rng.normal(5, 100, late.sum()))
    mutated.loc[late, "label"] = rng.choice(["DOWN", "FLAT", "UP"], late.sum())
    np.testing.assert_allclose(evaluate(spec, mutated, fit)[early], base[early], rtol=1e-12, atol=0)
    truncated = ev[early].reset_index(drop=True)
    np.testing.assert_allclose(evaluate(spec, truncated, np.arange(len(truncated))), base[early],
                               rtol=1e-12, atol=0)


# ── spec: JSON, validation, identity ─────────────────────────────────────────
def test_json_round_trip_is_lossless():
    rng = np.random.default_rng(0)
    for _ in range(500):
        s = random_spec(rng)
        d = s.to_json()
        assert FeatureSpec.from_json(d) == s
        assert FeatureSpec.from_json(json.dumps(d)) == s
        assert parse(json.loads(json.dumps(d))) == s
        assert FeatureSpec.from_json(d).key == s.key


def test_formula_is_readable():
    s = node("ratio", node("diff", col("mid"), k=1), node("roll", col("mid"), stat="std", w=6))
    assert s.formula() == "(diff(mid, k=1) / roll(mid, stat=std, w=6))"
    assert node("hour_of_day").formula() == "hour_of_day()"


BAD = [
    ({"op": "nope", "args": [{"col": "mid"}]}, "unknown operator"),
    ({"op": "abs", "args": [{"col": "label"}]}, "unknown column"),
    ({"op": "abs", "args": [{"col": "dmid"}]}, "unknown column"),
    ({"op": "abs", "args": [{"col": "mid"}, {"col": "spread"}]}, "takes 1 args"),
    ({"op": "mul", "args": [{"col": "mid"}]}, "takes 2 args"),
    ({"op": "lag", "args": [{"col": "mid"}]}, "takes k in"),
    ({"op": "lag", "args": [{"col": "mid"}], "params": {"k": 1, "w": 3}}, "takes k in"),
    ({"op": "lag", "args": [{"col": "mid"}], "params": {"k": 5}}, "must be one of [1, 2, 3, 6]"),
    ({"op": "lag", "args": [{"col": "mid"}], "params": {"k": 1.0}}, "is not allowed"),
    ({"op": "lag", "args": [{"col": "mid"}], "params": {"k": "1"}}, "is not allowed"),
    ({"op": "lag", "args": [{"col": "mid"}], "params": {"k": True}}, "is not allowed"),
    ({"op": "roll", "args": [{"col": "mid"}], "params": {"stat": "median", "w": 3}}, "stat must be one of"),
    ({"op": "abs", "args": [{"col": "mid"}], "params": {"k": 1}}, "takes no parameters"),
    ({"op": "abs", "args": {"col": "mid"}}, "must be a list"),
    ({"op": "abs", "args": [{"col": "mid"}], "code": "import os"}, "unknown keys"),
    ({"col": "mid"}, "root must be an operator"),
    ({"col": "mid", "op": "abs"}, "exactly"),
    ({"args": []}, "needs a string"),
    (["mid"], "must be an object"),
    ("{not json", "not valid JSON"),
    ({"op": "abs", "args": [{"op": "abs", "args": [{"op": "abs", "args": [
        {"op": "abs", "args": [{"col": "mid"}]}]}]}]}, "depth 4"),
]


@pytest.mark.parametrize("bad,msg", BAD, ids=[m for _, m in BAD])
def test_validation_rejects_with_actionable_message(bad, msg):
    with pytest.raises(SpecError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        parse(bad)


def test_size_limit_and_nesting_cap():
    wide = node("add", node("add", col("mid"), col("spread")), node("add", col("mid"), col("spread")))
    validate(wide)
    with pytest.raises(SpecError, match="nodes"):
        validate(wide, max_nodes=5)
    deep: dict = {"col": "mid"}
    for _ in range(500):
        deep = {"op": "abs", "args": [deep]}
    with pytest.raises(SpecError, match="nested deeper"):
        parse(deep)


@pytest.mark.parametrize("bad", ["label", "dmid", "mid_end", "spread_end", "quote_age_end_h",
                                 "t_end", "t0", "market_ticker", "series", "event_group", "foo_end"])
def test_post_t0_columns_can_never_be_allowed(bad, ev):
    with pytest.raises(SpecError):
        check_columns(DEFAULT_COLUMNS + [bad])
    with pytest.raises(SpecError):
        evaluate(node("abs", col(bad)), ev, columns=DEFAULT_COLUMNS + [bad])


def test_default_columns_exclude_provisional():
    assert "hours_to_close" not in DEFAULT_COLUMNS and "mid" in DEFAULT_COLUMNS
    validate(node("abs", col("hours_to_close")), columns=DEFAULT_COLUMNS + ["hours_to_close"])
    with pytest.raises(SpecError):
        validate(node("abs", col("hours_to_close")))


def test_canonical_hash_ignores_commutative_order_only():
    a, b, c = col("mid"), col("spread"), col("ret_1h")
    assert node("mul", a, b).key == node("mul", b, a).key
    assert node("max", node("add", a, b), c).key == node("max", c, node("add", b, a)).key
    assert node("sub", a, b).key != node("sub", b, a).key
    assert node("ratio", a, b).key != node("ratio", b, a).key
    assert node("gate_pos", a, b).key != node("gate_pos", b, a).key
    assert node("roll", a, stat="mean", w=3).key == FeatureSpec("roll", (a,), {"w": 3, "stat": "mean"}).key
    assert node("roll", a, stat="mean", w=3).key != node("roll", a, stat="mean", w=6).key
    # stable across processes and sessions, not just within one run
    assert node("mul", a, b).key == "37179b7021e27ef6"


# ── interpreter: numerics ────────────────────────────────────────────────────
@pytest.mark.parametrize("spec", OP_SPECS, ids=IDS)
def test_no_inf_and_no_raise_on_nasty_inputs(spec, ev):
    nasty = ev.copy()
    vals = np.array([np.nan, np.inf, -np.inf, 0.0, 1e300, -1e300, 1e-300, 1.0])
    for c in DEFAULT_COLUMNS:
        nasty[c] = vals[np.arange(len(nasty)) % len(vals)]
    nasty.loc[nasty.index[:200], A] = np.nan  # a fully missing stretch
    for frame in (ev, nasty):
        out = evaluate(spec, frame, np.arange(len(frame) // 2))
        assert out.dtype == float and out.shape == (len(frame),)
        assert not np.isinf(out).any()


def test_random_specs_are_finite_or_nan(ev):
    rng = np.random.default_rng(3)
    fit = np.arange(len(ev) // 2)
    for _ in range(300):
        out = evaluate(random_spec(rng), ev, fit)
        assert not np.isinf(out).any()


def test_all_nan_column_gives_nan_not_error(ev):
    e = ev.copy()
    e[A] = np.nan
    for spec in OP_SPECS:
        evaluate(spec, e, np.arange(100))


def test_rolling_and_ewm_match_a_naive_per_market_loop(ev):
    got_roll = evaluate(node("roll", col(A), stat="std", w=6), ev)
    got_z = evaluate(node("roll_z", col(A), w=6), ev)
    got_ewm = evaluate(node("ewm", col(A), halflife=2), ev)
    got_lag = evaluate(node("lag", col(A), k=3), ev)
    got_gap = evaluate(node("hours_since_prev_event"), ev)
    for _, m in ev.groupby("market_ticker"):
        m = m.sort_values("t0")
        x = m[A].to_numpy()
        for j, i in enumerate(m.index):
            prev = x[:j]
            w6 = prev[-6:][~np.isnan(prev[-6:])]
            exp_std = np.std(w6, ddof=1) if len(w6) >= 2 else np.nan
            np.testing.assert_allclose(got_roll[i], exp_std, rtol=1e-9)
            exp_z = (x[j] - w6.mean()) / exp_std if len(w6) >= 3 and exp_std > 1e-9 else np.nan
            np.testing.assert_allclose(got_z[i], exp_z, rtol=1e-9)
            seen = np.flatnonzero(~np.isnan(prev))
            wts = 0.5 ** ((j - 1 - seen) / 2)  # halflife 2, aged by events, gaps included
            exp_ewm = np.sum(wts * prev[seen]) / np.sum(wts) if len(seen) else np.nan
            np.testing.assert_allclose(got_ewm[i], exp_ewm, rtol=1e-9)
            np.testing.assert_allclose(got_lag[i], x[j - 3] if j >= 3 else np.nan)
            exp_gap = (m.t0.iloc[j] - m.t0.iloc[j - 1]) / HOUR if j else np.nan
            np.testing.assert_allclose(got_gap[i], exp_gap)


def test_cross_section_uses_same_t0_peers_only(ev):
    got = evaluate(node("xs_dev", col(A), scope="event_group"), ev)
    rank = evaluate(node("xs_rank", col(A), scope="all"), ev)
    for _, g in ev.groupby(["event_group", "t0"]):
        x = g[A].dropna()
        if len(x) >= 2:
            np.testing.assert_allclose(got[x.index], x - x.mean(), rtol=1e-9)
        else:
            assert np.isnan(got[g.index]).all()
    assert np.nanmin(rank) > 0 and np.nanmax(rank) < 1


def test_duplicate_market_t0_refused(ev):
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(node("lag", col(A), k=1), pd.concat([ev, ev.iloc[:1]], ignore_index=True))


# ── leakage, for every operator ──────────────────────────────────────────────
@pytest.mark.parametrize("spec", OP_SPECS, ids=IDS)
def test_later_rows_cannot_change_earlier_features(spec, ev):
    assert_causal(spec, ev)


def test_random_compositions_are_causal(ev):
    rng = np.random.default_rng(7)
    for _ in range(150):
        assert_causal(random_spec(rng, include_cross=True), ev)


@pytest.mark.parametrize("spec", OP_SPECS, ids=IDS)
def test_row_order_does_not_matter(spec, ev):
    fit = np.flatnonzero(ev.t0 <= ev.t0.median())
    base = evaluate(spec, ev, fit)
    perm = np.random.default_rng(5).permutation(len(ev))
    shuffled = ev.iloc[perm].reset_index(drop=True)
    inv = np.argsort(perm)
    np.testing.assert_allclose(evaluate(spec, shuffled, inv[fit])[inv], base, rtol=1e-12, atol=0)


def _with_op(monkeypatch, op: Op):
    monkeypatch.setitem(OPS, op.name, op)
    return node(op.name, col(A))


def test_leak_check_has_power_against_a_look_ahead(ev, monkeypatch):
    """Mutation check: a market operator that peeks one event ahead must fail assert_causal."""
    def peek(F, x):
        out = np.full(F.n, np.nan)
        out[:-1] = x[1:]
        return np.where(np.r_[F.market[1:] == F.market[:-1], False], out, np.nan)
    with pytest.raises(AssertionError):
        assert_causal(_with_op(monkeypatch, Op("peek", "market", 1, {}, "", peek)), ev)


def test_leak_check_has_power_against_fitting_on_all_rows(ev, monkeypatch):
    def zscore_everything(F, x):
        return (x - np.nanmean(x)) / np.nanstd(x)
    with pytest.raises(AssertionError):
        assert_causal(_with_op(monkeypatch, Op("zall", "fold", 1, {}, "", zscore_everything)), ev)


# ── fold-fitted transforms ───────────────────────────────────────────────────
FOLD_SPECS = [s for s in OP_SPECS if OPS[s.op].kind == "fold"]


@pytest.mark.parametrize("spec", FOLD_SPECS, ids=[s.formula() for s in FOLD_SPECS])
def test_fold_transform_without_fit_index_raises(spec, ev):
    assert spec.needs_fit and node("abs", spec).needs_fit
    with pytest.raises(ValueError, match="fit_index"):
        evaluate(spec, ev)
    with pytest.raises(ValueError, match="fit_index"):
        evaluate(node("abs", spec), ev)


def test_needs_fit_only_for_fold_operators():
    for s in OP_SPECS:
        assert s.needs_fit == (OPS[s.op].kind == "fold")
    evaluate(node("roll", col(A), stat="mean", w=3), synthetic_events(1))  # no fit_index needed


@pytest.mark.parametrize("spec", FOLD_SPECS, ids=[s.formula() for s in FOLD_SPECS])
def test_fold_transform_never_reads_non_fit_rows(spec, ev):
    """Changing every value and label outside fit_index leaves the fit rows' features unchanged."""
    rng = np.random.default_rng(2)
    fit = np.sort(rng.choice(len(ev), len(ev) // 2, replace=False))
    other = np.setdiff1d(np.arange(len(ev)), fit)
    base = evaluate(spec, ev, fit)
    e = ev.copy()
    for c in DEFAULT_COLUMNS:
        e.loc[other, c] = rng.normal(50, 100, len(other))
    e.loc[other, "label"] = "not-a-label"
    np.testing.assert_allclose(evaluate(spec, e, fit)[fit], base[fit], rtol=1e-12, atol=0)


@pytest.mark.parametrize("spec", OP_SPECS, ids=IDS)
def test_only_training_labels_are_ever_read(spec, ev):
    fit = np.arange(len(ev) // 2)
    e = ev.copy()
    e.loc[len(ev) // 2:, "label"] = None
    np.testing.assert_allclose(evaluate(spec, e, fit), evaluate(spec, ev, fit), rtol=1e-12, atol=0)


@pytest.mark.parametrize("stat", OPS["series_target"].params["stat"])
def test_target_encoding_sees_a_label_only_after_its_window_closed(stat, ev):
    spec = node("series_target", stat=stat, m=10)
    fit = np.arange(len(ev))
    base = evaluate(spec, ev, fit)
    t0, t_end = ev.t0.to_numpy(), ev.t_end.to_numpy()
    changed_any = False
    for j in np.random.default_rng(4).choice(len(ev), 25, replace=False):
        e = ev.copy()
        e.loc[j, "label"] = {"UP": "DOWN", "DOWN": "FLAT", "FLAT": "UP"}[e.loc[j, "label"]]
        diff = ~np.isclose(evaluate(spec, e, fit), base, rtol=1e-12, atol=0, equal_nan=True)
        assert not diff[t0 < t_end[j]].any()  # its own row included
        assert not diff[j]
        changed_any |= diff.any()
    assert changed_any  # the encoding does read labels


def test_target_encoding_refuses_malformed_windows(ev):
    e = ev.copy()
    e.loc[0, "t_end"] = e.loc[0, "t0"]
    with pytest.raises(ValueError, match="t_end > t0"):
        evaluate(node("series_target", stat="up", m=10), e, np.arange(len(e)))


# ── sampling and the TPE parameterisation ────────────────────────────────────
def test_random_specs_are_valid_and_cover_the_catalog():
    rng = np.random.default_rng(0)
    seen, depths = set(), set()
    for _ in range(3000):
        s = random_spec(rng)
        validate(s)
        assert not s.is_col and s.depth <= DEFAULT_MAX_DEPTH
        seen |= {n.op for n in s.nodes()}
        depths.add(s.depth)
    assert seen == set(space_ops()) | {"col"}
    assert depths == {1, 2, 3}
    for d in (1, 2, 5):
        assert random_spec(rng, d).depth <= d


def test_random_spec_is_deterministic():
    a = [random_spec(np.random.default_rng(11)).key for _ in range(3)]
    r1, r2 = np.random.default_rng(12), np.random.default_rng(12)
    assert len(set(a)) == 1
    assert [random_spec(r1) for _ in range(200)] == [random_spec(r2) for _ in range(200)]


def test_random_spec_root_operator_is_uniform():
    rng = np.random.default_rng(1)
    roots = pd.Series([random_spec(rng).op for _ in range(30_000)]).value_counts()
    expected = 30_000 / len(space_ops())
    assert set(roots.index) == set(space_ops())
    assert (abs(roots - expected) < 5 * np.sqrt(expected)).all()


def test_tpe_names_and_choices_match_the_flat_space():
    space = flat_space()
    rng = np.random.default_rng(2)
    asked: dict[str, tuple] = {}

    def suggest(name, choices):
        assert name in space and tuple(space[name]) == tuple(choices), name
        assert asked.setdefault(name, tuple(choices)) == tuple(choices)  # optuna needs fixed choices
        return choices[int(rng.integers(len(choices)))]

    for _ in range(2000):
        validate(build_spec(suggest))
    assert set(asked) <= set(space)
    assert all(type(c) in (str, int, float) for v in space.values() for c in v)  # categorical-safe


def test_random_arm_and_tpe_arm_draw_from_the_same_space():
    """random_spec is build_spec with uniform suggestions, and every valid spec has a flat
    assignment that build_spec maps back to it (so TPE can reach everything random can)."""
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    uniform = lambda name, choices: choices[int(r2.integers(len(choices)))]
    for _ in range(300):
        assert random_spec(r1) == build_spec(uniform)
    hand = [parse(d) for d in LLM_STYLE] + [random_spec(r1) for _ in range(500)]
    for s in hand:
        flat = spec_to_flat(s)
        assert build_spec(lambda name, choices: flat[name]) == s
        assert all(v in flat_space()[k] for k, v in flat.items())


def test_space_size_matches_brute_force_at_depth_one():
    cols = ["mid", "spread", "ret_1h"]
    keys = set()
    for s in all_param_specs():
        if s.op in CROSS:
            continue
        for leaves in product(cols, repeat=OPS[s.op].arity):
            keys.add(FeatureSpec(s.op, tuple(map(col, leaves)), s.params).key)
    assert space_size(1, cols) == len(keys)
    assert space_size(3) > space_size(2) > space_size(1)


# ── the LLM-facing side ──────────────────────────────────────────────────────
# Spring families, re-expressed in the grammar (sentiment columns are not in v2's default set)
LLM_STYLE = [
    {"op": "gate_q", "args": [{"col": "ret_4h"}, {"col": "vol_24h"}], "params": {"q": 0.75, "side": "above"}},
    {"op": "mul", "args": [{"op": "square", "args": [{"col": "vol_24h"}]},
                           {"op": "slog", "args": [{"col": "log_volume_24h"}]}]},
    {"op": "roll_z", "args": [{"col": "log_volume_24h"}], "params": {"w": 12}},
    {"op": "group_dev", "args": [{"col": "active_hours_24h"}], "params": {"by": "hour"}},
    {"op": "hour_of_day"},
    {"op": "series_freq"},
    {"op": "ratio", "args": [{"op": "ewm", "args": [{"col": "ret_1h"}], "params": {"halflife": 1}},
                             {"op": "ewm", "args": [{"col": "ret_1h"}], "params": {"halflife": 8}}]},
    {"op": "diff", "args": [{"op": "roll", "args": [{"col": "mid"}], "params": {"stat": "max", "w": 6}}],
     "params": {"k": 1}},
]


def test_catalog_description_is_generated_from_the_registry():
    text = catalog_description()
    assert all(f"{name}(" in text for name in space_ops())
    assert all(c in text for c in DEFAULT_COLUMNS) and "hours_to_close" not in text
    cat = json.loads(catalog_description(fmt="json"))
    assert [o["op"] for o in cat["operators"]] == list(space_ops())
    parse(cat["example"]["json"])
    assert len(text) < 6000  # fits comfortably in a prompt


def test_hand_written_specs_parse_and_evaluate(ev):
    for d in LLM_STYLE:
        s = parse(d)
        assert not np.isinf(evaluate(s, ev, np.arange(len(ev) // 2))).any()


def test_evaluation_is_fast_on_20k_rows():
    big = pd.concat([synthetic_events(s, 60).assign(market_ticker=lambda d, s=s: d.market_ticker + f"#{s}")
                     for s in range(8)], ignore_index=True)
    assert len(big) > 15_000
    heavy = node("ratio", node("roll_z", node("ewm", col(A), halflife=4), w=24),
                 node("xs_rank", node("qrank", col(B)), scope="all"))
    t = time.perf_counter()
    evaluate(heavy, big, np.arange(len(big) // 2))
    assert time.perf_counter() - t < 1.0


# ── cross-sectional operators are opt-in ─────────────────────────────────────
def test_default_space_has_no_cross_sectional_operators(ev):
    assert CROSS and not CROSS & set(space_ops())
    assert set(space_ops(include_cross=True)) == set(OPS)
    assert set(flat_space()["n.op"]) == set(space_ops())
    assert not any(k.split(".")[1] in CROSS for k in flat_space() if k.count(".") == 2)
    text, cat = catalog_description(), catalog()
    assert not any(f"{n}(" in text for n in CROSS) and "[cross]" not in text
    assert "cross" not in cat["kinds"] and not CROSS & {o["op"] for o in cat["operators"]}
    rng = np.random.default_rng(0)
    assert not any(CROSS & {n.op for n in random_spec(rng).nodes()} for _ in range(3000))
    assert space_size(2) < space_size(2, include_cross=True)


def test_random_and_tpe_agree_with_and_without_cross():
    for include_cross in (False, True):
        space = flat_space(include_cross=include_cross)
        r1, r2 = np.random.default_rng(3), np.random.default_rng(3)

        def uniform(name, choices):
            assert tuple(space[name]) == tuple(choices)
            return choices[int(r2.integers(len(choices)))]

        seen = set()
        for _ in range(1500):
            s = random_spec(r1, include_cross=include_cross)
            assert s == build_spec(uniform, include_cross=include_cross)
            validate(s, include_cross=include_cross)
            flat = spec_to_flat(s, include_cross=include_cross)
            assert build_spec(lambda n, c: flat[n], include_cross=include_cross) == s
            seen |= {n.op for n in s.nodes()}
        assert bool(seen & CROSS) == include_cross


@pytest.mark.parametrize("name", sorted(CROSS))
def test_default_parse_rejects_cross_ops_and_says_why(name, ev):
    d = {"op": "abs", "args": [{"op": name, "args": [{"col": "mid"}], "params": {"scope": "all"}}]}
    with pytest.raises(SpecError, match="look-ahead"):
        parse(d)
    with pytest.raises(SpecError, match="include_cross=True"):
        evaluate_default(parse(d, include_cross=True), ev)
    with pytest.raises(SpecError):
        spec_to_flat(parse(d, include_cross=True))
    assert not np.isinf(evaluate(parse(d, include_cross=True), ev)).any()


# ── degenerate features ──────────────────────────────────────────────────────
def test_degenerate_reason():
    rng = np.random.default_rng(0)
    good = rng.normal(size=1000)
    assert degenerate_reason(good) is None
    assert "NaN" in degenerate_reason(np.full(1000, np.nan))
    assert "constant" in degenerate_reason(np.r_[np.full(900, 2.0), np.full(100, np.nan)])
    assert "fewer than" in degenerate_reason(np.r_[good[:50], np.full(950, np.nan)])
    assert degenerate_reason(np.r_[good[:50], np.full(950, np.nan)], min_valid=50) is None
    # judged on the training rows only when fit_index is given
    mixed = np.r_[np.zeros(500), good[:500]]
    assert degenerate_reason(mixed) is None
    assert "constant" in degenerate_reason(mixed, fit_index=np.arange(500))
    assert "training rows" in degenerate_reason(mixed, fit_index=np.arange(500))
    assert degenerate_reason(mixed, fit_index=np.arange(500, 1000)) is None
    assert "NaN" in degenerate_reason(np.r_[np.full(500, np.nan), good[:500]], fit_index=np.arange(500))


def test_degenerate_reason_flags_real_degenerate_specs(ev):
    fit = np.arange(len(ev) // 2)
    assert degenerate_reason(evaluate(node("sub", col(A), col(A)), ev, fit), fit) is not None
    assert degenerate_reason(evaluate(node("sign", col("mid")), ev, fit), fit) is not None
    assert degenerate_reason(evaluate(node("roll", col(A), stat="mean", w=3), ev, fit), fit,
                             min_valid=50) is None
