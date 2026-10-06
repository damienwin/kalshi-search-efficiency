"""Acceptance gate: exact identities, edge cases, and small seeded Monte-Carlo
checks of size and power under realistic dependence. The larger simulations
(1,000 replications on the real dev layout) are run by hand; see the gate PR notes."""

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import stats

from ksearch.eval.cv import LABELS, clustered_paired_bootstrap
from ksearch.search.gate import (EPS, _hln, balanced_log_loss, balanced_weights,
                                 benjamini_hochberg, bootstrap_f1_test, dm_test,
                                 evaluate_candidate, hac_lag, loss_differences, placebo_p,
                                 placebo_shift, placebo_threshold, sign_flip_test)

DAY = 86400
T0 = 1_700_006_400  # a UTC midnight


# ── a synthetic panel shaped like the dev set ────────────────────────────────
def panel(n_groups=120, n_days=240, seed=0):
    """Ladders with heavy-tailed lifetimes and 1-6 strikes, events on the 4h grid.
    Returns group, day, market, t0 (ints) per event."""
    rng = np.random.default_rng(seed)
    g, day, mk, t0 = [], [], [], []
    m = 0
    for k in range(n_groups):
        start = int(rng.integers(0, n_days - 1))
        life = int(min(n_days - start, 1 + rng.lognormal(1.2, 1.1)))
        for _ in range(int(rng.integers(1, 7))):
            for dd in range(start, start + life):
                for slot in rng.choice(6, int(rng.integers(1, 3)), replace=False):
                    g.append(k), day.append(dd), mk.append(m)
                    t0.append(T0 + dd * DAY + int(slot) * 4 * 3600)
            m += 1
    return tuple(map(np.asarray, (g, day, mk, t0)))


@pytest.fixture(scope="module")
def P():
    g, day, mk, t0 = panel()
    cell = np.unique(g * 10_000 + day, return_inverse=True)[1]
    return {"g": g, "day": day, "mk": mk, "t0": t0, "cell": cell, "n": len(g),
            "G": g.max() + 1, "T": day.max() + 1, "C": cell.max() + 1}


def draw(P, rng, mu=0.0, common=0.6, rho=0.6, group=0.3, cell=1.0):
    """Zero-mean loss differences (plus mu): a daily common shock shared by every
    open group, AR(1) across days; a persistent group effect; a group-day effect
    shared by strikes; heavy-tailed noise."""
    F = np.empty(P["T"])
    e = rng.standard_normal(P["T"])
    F[0] = e[0]
    for i in range(1, P["T"]):
        F[i] = rho * F[i - 1] + math.sqrt(1 - rho ** 2) * e[i]
    return (mu + common * F[P["day"]] + group * rng.standard_normal(P["G"])[P["g"]]
            + cell * rng.standard_normal(P["C"])[P["cell"]] + rng.standard_t(4, P["n"]) / math.sqrt(2))


def probs(n, rng, conc=(1, 5, 1)):
    return rng.dirichlet(conc, n)


# ── loss ─────────────────────────────────────────────────────────────────────
def test_balanced_weights_give_each_class_equal_mass():
    y = np.array(["FLAT"] * 86 + ["UP"] * 7 + ["DOWN"] * 7, dtype=object)
    w = balanced_weights(y)
    assert w.mean() == pytest.approx(1.0)
    for c in LABELS:
        assert w[y == c].sum() == pytest.approx(len(y) / 3)


def test_loss_is_finite_at_zero_and_one():
    y = np.array(["UP", "FLAT", "DOWN"], dtype=object)
    p = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    loss = balanced_log_loss(y, p)
    assert np.isfinite(loss).all()
    assert loss[0] == pytest.approx(-math.log(EPS))  # confident miss is capped
    assert loss[1] == 0.0 and loss[2] == 0.0


def test_weights_shared_so_difference_is_paired():
    rng = np.random.default_rng(0)
    y = rng.choice(LABELS, 500, p=[.1, .8, .1]).astype(object)
    a, b = probs(500, rng), probs(500, rng)
    assert np.allclose(loss_differences(y, a, b), balanced_log_loss(y, a) - balanced_log_loss(y, b))
    assert np.allclose(loss_differences(y, a, b), -loss_differences(y, b, a))


@pytest.mark.parametrize("bad", ["nan_proba", "bad_label", "nan_t0", "none_group", "shape", "unnormalised"])
def test_bad_inputs_raise(bad):
    rng = np.random.default_rng(0)
    n = 50
    y = rng.choice(LABELS, n).astype(object)
    a, b = probs(n, rng), probs(n, rng)
    grp = np.arange(n).astype(object)
    t0 = (T0 + np.arange(n) * DAY).astype(float)
    if bad == "nan_proba":
        b[3, 0] = np.nan
    elif bad == "bad_label":
        y[0] = np.nan
    elif bad == "nan_t0":
        t0[5] = np.nan
    elif bad == "none_group":
        grp[2] = None
    elif bad == "shape":
        b = b[:, :2]
    else:
        b[0] = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        evaluate_candidate(y, a, b, grp, t0)


# ── Diebold-Mariano ──────────────────────────────────────────────────────────
def test_dm_equals_one_sample_t_on_iid_data():
    """One event per day and per group, lag 0: DM * HLN(T, 0) is exactly the t statistic."""
    d = np.random.default_rng(1).standard_normal(150) + 0.1
    t0, grp = T0 + np.arange(150) * DAY, np.arange(150)
    ref = stats.ttest_1samp(d, 0, alternative="greater")
    for cluster in (False, True):
        r = dm_test(d, grp, t0, lag=0, cluster_groups=cluster)
        assert r["dm_stat"] == pytest.approx(ref.statistic, rel=1e-12)
        assert r["dm_p"] == pytest.approx(ref.pvalue, rel=1e-10)


def test_dm_equals_t_on_equal_block_sums():
    """Four events per day: DM on events equals the t test on the daily sums."""
    rng = np.random.default_rng(2)
    d = rng.standard_normal(4 * 60)
    t0 = T0 + np.repeat(np.arange(60), 4) * DAY + np.tile(np.arange(4), 60) * 4 * 3600
    r = dm_test(d, np.arange(len(d)), t0, lag=0, cluster_groups=False)
    assert r["dm_stat"] == pytest.approx(stats.ttest_1samp(d.reshape(60, 4).sum(1), 0).statistic)


def test_dm_hac_matches_hand_computation():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(80).cumsum() * 0.1 + rng.standard_normal(80)
    L = 4
    u = x - x.mean()
    lrv = u @ u + 2 * sum((1 - k / (L + 1)) * (u[k:] @ u[:-k]) for k in range(1, L + 1))
    hand = x.sum() / math.sqrt(lrv) * math.sqrt((80 + 1 - 2 * 5 + 5 * 4 / 80) / 80)
    r = dm_test(x, np.arange(80), T0 + np.arange(80) * DAY, lag=L, cluster_groups=False)
    assert r["dm_stat"] == pytest.approx(hand)
    assert _hln(80, L) == pytest.approx(math.sqrt((81 - 10 + 20 / 80) / 80))


def test_hac_lag_rule():
    assert hac_lag(800) == 14                     # two-week floor binds
    assert hac_lag(800, block_hours=4) == 84      # same calendar reach in 4h blocks
    assert hac_lag(30) == 7                       # capped at (T - 1) // 4
    assert hac_lag(1) == 0


def test_two_way_variance_never_below_one_way(P):
    """Floor at max(V_G, V_T): the two-way test is never more liberal than either one-way test."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        d = draw(P, rng)
        two = dm_test(d, P["g"], P["t0"])["dm_stat"]
        time_only = dm_test(d, P["g"], P["t0"], cluster_groups=False)["dm_stat"]
        assert abs(two) <= abs(time_only) + 1e-9


@pytest.mark.parametrize("case", ["zeros", "single_group", "few_blocks"])
def test_dm_degenerate_cases_cannot_reject(case):
    n = 200
    d = np.full(n, 0.0) if case == "zeros" else np.random.default_rng(0).standard_normal(n) + 5
    grp = np.zeros(n) if case == "single_group" else np.arange(n) % 40
    t0 = T0 + (np.arange(n) % 5) * DAY if case == "few_blocks" else T0 + np.arange(n) * DAY
    r = dm_test(d, grp, t0)
    assert r["dm_p"] == 1.0 and r["status"] != "ok"


def test_identical_predictions_are_rejected_cleanly():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.choice(LABELS, n).astype(object)
    a = probs(n, rng)
    r = evaluate_candidate(y, a, a.copy(), np.arange(n) % 50, T0 + np.arange(n) * DAY // 2)
    assert not r.accept and r.reason == "dm_degenerate_variance"
    assert r.dm_p == 1.0 and r.signflip_p == 1.0 and r.f1_p == 1.0 and r.f1_delta == 0.0


# ── Monte-Carlo: size and power (small and seeded; big runs by hand) ─────────
def test_size_under_dependent_null(P):
    """300 null replications with strike, group and cross-group time dependence.
    The gate's test must reject near 5%; the naive tests must not. Tolerance
    [0.015, 0.10]: binomial SE at 300 reps is 0.0126, and the 1,000-rep runs on the
    real layout put this test at 0.05-0.07 for moderate dependence, so the band is
    ~3.5 SE around 0.05 widened upward for that known mild liberal bias."""
    rng = np.random.default_rng(6)
    rej = {"twoway": 0, "naive": 0, "group": 0}
    for _ in range(300):
        d = draw(P, rng)
        rej["twoway"] += dm_test(d, P["g"], P["t0"])["dm_p"] < 0.05
        rej["naive"] += stats.ttest_1samp(d, 0, alternative="greater").pvalue < 0.05
        gs = np.bincount(P["g"], d)
        rej["group"] += stats.ttest_1samp(gs, 0, alternative="greater").pvalue < 0.05
    rate = {k: v / 300 for k, v in rej.items()}
    assert 0.015 <= rate["twoway"] <= 0.10, rate
    assert rate["naive"] > 0.25, rate
    assert rate["group"] > rate["twoway"] + 0.03, rate


def test_power_rises_with_effect_and_sample(P):
    rng = np.random.default_rng(8)
    quarter = P["g"] < P["G"] // 4

    def power(mu, mask):
        hits = 0
        for _ in range(80):
            d = draw(P, rng, mu=mu)
            hits += dm_test(d[mask], P["g"][mask], P["t0"][mask])["dm_p"] < 0.05
        return hits / 80

    full = np.ones(P["n"], bool)
    p_small, p_big = power(0.1, full), power(0.3, full)
    assert p_big > p_small + 0.3 and p_big > 0.7
    assert power(0.3, quarter) < p_big - 0.15


def test_sign_flip_deterministic_and_sensitive(P):
    rng = np.random.default_rng(9)
    d = draw(P, rng, mu=0.5)
    a = sign_flip_test(d, P["g"], P["t0"], n_perm=999, seed=3)
    assert a == sign_flip_test(d, P["g"], P["t0"], n_perm=999, seed=3)
    assert a["signflip_p"] == pytest.approx(1 / 1000)
    assert sign_flip_test(-d, P["g"], P["t0"], n_perm=999)["signflip_p"] > 0.5
    assert sign_flip_test(np.zeros(P["n"]), P["g"], P["t0"])["signflip_p"] == 1.0


# ── macro F1 bootstrap ───────────────────────────────────────────────────────
def test_f1_bootstrap_matches_cv_bootstrap_and_p_is_coherent():
    rng = np.random.default_rng(10)
    n = 1500
    y = rng.choice(LABELS, n, p=[.1, .8, .1]).astype(object)
    a = probs(n, rng)
    b = a.copy()
    hit = rng.random(n) < 0.3  # candidate puts mass on the truth for 30% of events
    b[hit] = 0.05
    b[hit, np.searchsorted(LABELS, y[hit].astype(str))] = 0.9
    grp = np.arange(n) % 120
    r = bootstrap_f1_test(y, a, b, grp, n_boot=500, seed=4)
    lab = np.array(LABELS, dtype=object)
    ref = clustered_paired_bootstrap(y, lab[b.argmax(1)], lab[a.argmax(1)], grp, n_boot=500, seed=4)
    assert r["f1_delta"] == pytest.approx(ref["delta"])
    assert r["f1_ci95"] == pytest.approx(ref["ci95"])
    assert r["f1_p"] < 0.01
    flipped = bootstrap_f1_test(y, b, a, grp, n_boot=500, seed=4)
    assert flipped["f1_p"] > 0.99


# ── placebo ──────────────────────────────────────────────────────────────────
def test_placebo_only_uses_earlier_values_and_breaks_alignment(P):
    vals = np.random.default_rng(11).standard_normal(P["n"])
    vals[::17] = np.nan
    out = placebo_shift(vals, P["mk"], P["t0"], np.random.default_rng(0), min_shift=2)
    sizes = np.bincount(P["mk"])
    for m in np.flatnonzero(sizes >= 4):
        i = np.flatnonzero(P["mk"] == m)
        i = i[np.argsort(P["t0"][i], kind="mergesort")]
        got, src = out[i], vals[i]
        kept = ~np.isnan(got)
        assert kept.any() and np.isnan(got).any()
        # every kept value is one the market had at an earlier t0, never a later one
        for j in np.flatnonzero(kept):
            assert got[j] in src[:j]
    assert np.corrcoef(np.nan_to_num(vals), np.nan_to_num(out))[0, 1] < 0.1


def test_placebo_seeds():
    mk = np.repeat(np.arange(30), 12)
    t0 = T0 + np.tile(np.arange(12), 30) * 4 * 3600
    v = np.arange(len(mk), dtype=float)
    a = placebo_shift(v, mk, t0, np.random.default_rng(1))
    assert np.array_equal(a, placebo_shift(v, mk, t0, np.random.default_rng(1)), equal_nan=True)
    assert not np.array_equal(a, placebo_shift(v, mk, t0, np.random.default_rng(2)), equal_nan=True)


def test_placebo_short_markets():
    # market 0 has 2 events: the later row gets the earlier value, the earlier row NaN.
    # markets 1-3 have one event each: pooled in time order, under the same rule.
    mk = np.array([0, 0, 1, 2, 3])
    t0 = np.array([0, 10, 5, 6, 7])
    v = np.array([1.0, 2.0, 10.0, 20.0, 30.0])
    out = placebo_shift(v, mk, t0, np.random.default_rng(0))
    assert np.isnan(out[0]) and out[1] == 1.0
    assert np.isnan(out[2]) and out[3] == 10.0 and out[4] == 20.0
    # a lone single-event market has nothing earlier to take: NaN
    assert np.isnan(placebo_shift(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]), np.array([0, 1, 0]),
                                  np.random.default_rng(0))[2])
    # rows passed out of time order are shifted in time order
    out = placebo_shift(np.array([3.0, 1.0, 2.0]), np.array([0, 0, 0]), np.array([2, 0, 1]),
                        np.random.default_rng(0))
    assert np.isnan(out[1]) and out[2] == 1.0 and out[0] == 2.0


def test_placebo_threshold_order_statistic():
    g = list(range(1, 20))                         # 19 placebos: threshold is the max
    assert placebo_threshold(g) == 19
    assert placebo_threshold(g[:-1]) == math.inf   # 18 cannot certify 5%
    assert placebo_threshold(list(range(1, 40))) == 38  # ceil(0.95 * 40) = 38th smallest
    assert placebo_p(19.5, g) == pytest.approx(1 / 20)
    with pytest.raises(ValueError):
        placebo_threshold([0.1, float("nan")])


# ── multiple testing ─────────────────────────────────────────────────────────
def test_benjamini_hochberg_hand_example():
    # sorted .005 .01 .03 .04 .20 -> p*5/rank .025 .025 .05 .05 .20 (already monotone)
    p = [0.01, 0.04, 0.03, 0.005, 0.20]
    assert benjamini_hochberg(p) == pytest.approx([0.025, 0.05, 0.05, 0.025, 0.20])
    # step-up: .039 * 3/2 = .0585 exceeds .04 * 3/3 = .04, so .039 is lowered to .04
    assert benjamini_hochberg([0.039, 0.04, 0.001]) == pytest.approx([0.04, 0.04, 0.003])
    assert benjamini_hochberg([0.9, 0.95]) == pytest.approx([0.95, 0.95])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, float("nan")])


# ── decision ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def strong(P):
    """A candidate that moves 40% of the truth-class mass up, against a noisy reference."""
    rng = np.random.default_rng(12)
    n = P["n"]
    y = rng.choice(LABELS, n, p=[.07, .86, .07]).astype(object)
    a = probs(n, rng, (2, 2, 2))
    b = a.copy()
    hit = rng.random(n) < 0.4
    yi = np.searchsorted(LABELS, y.astype(str))
    b[hit, yi[hit]] += 1.0
    b /= b.sum(1, keepdims=True)
    return y, a, b, P["g"], P["t0"]


def test_gate_accepts_strong_signal_and_serialises(strong):
    r = evaluate_candidate(*strong, placebo_gains=[0.0] * 19,
                           cand_fold_f1=[.45] * 5, ref_fold_f1=[.40] * 5, n_perm=499, n_boot=200)
    assert r.accept and r.reason == "accepted"
    assert r.dm_p < 1e-6 and r.mean_loss_gain > 0 and r.placebo_threshold == 0.0
    assert r.legacy["verdict"] == "KEPT"
    d = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    assert d["accept"] is True and d["n_blocks"] == r.n_blocks
    with pytest.raises(FrozenInstanceError):
        r.accept = False


def test_gate_rule_switches(strong):
    kw = dict(n_perm=199, n_boot=100)
    big = [10.0] * 19
    assert evaluate_candidate(*strong, placebo_gains=big, **kw).reason == "below_placebo"
    assert evaluate_candidate(*strong, placebo_cutoff=10.0, **kw).reason == "below_placebo"
    few = evaluate_candidate(*strong, placebo_gains=[0.0] * 5, **kw)
    assert few.reason == "too_few_placebos" and few.to_dict()["placebo_threshold"] is None
    assert evaluate_candidate(*strong, require_placebo=True, **kw).reason == "placebo_missing"
    assert evaluate_candidate(*strong, alpha=0.0, **kw).reason == "dm_not_significant"
    y, a, b, g, t0 = strong
    worse = evaluate_candidate(y, b, a, g, t0, **kw)  # candidate and reference swapped
    assert not worse.accept and worse.dm_p > 0.99
    with pytest.raises(ValueError):
        evaluate_candidate(*strong, placebo_gains=[0.0] * 19, placebo_cutoff=0.0)


def test_placebo_never_hands_a_row_a_later_value():
    """A circular shift would give a market's first rows its last rows' values: a lead,
    i.e. information from after t0. Every placebo row must come from its own t0 or earlier."""
    rng = np.random.default_rng(5)
    for n in (2, 3, 5, 12, 40):
        t0 = np.arange(n)
        for seed in range(10):
            out = placebo_shift(t0.astype(float), np.zeros(n, int), t0, np.random.default_rng(seed), 6)
            kept = ~np.isnan(out)
            assert (out[kept] < t0[kept]).all()  # values are the t0s themselves: strictly earlier
    mk = rng.integers(0, 50, 2000)
    t0 = rng.integers(0, 10_000, 2000)
    out = placebo_shift(t0.astype(float), mk, t0, rng, 6)
    kept = ~np.isnan(out)
    assert (out[kept] <= t0[kept]).all()

