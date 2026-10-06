"""Acceptance gate for a candidate feature, from out-of-fold predictions only.

A candidate is kept when adding it lowers class-balanced log loss, tested one-
sided with a Diebold-Mariano statistic whose variance is robust to the two
kinds of dependence in this data:

- within an event group (strikes of one ladder move together, and a long-lived
  ladder's feature/label relation persists for weeks), and
- across groups open at the same time (adjacent Bitcoin ladders, one data
  release moving many markets), persisting over a few days.

The week-3 design (group sums ordered by time, Newey-West on that sequence)
handles the first but not the second: on the real dev layout with a common
daily shock it rejected a true null 14-26% of the time at alpha = 0.05. A
calendar-day DM (Driscoll-Kraay) handles the second but not the first (28%
under persistent group effects). The two-way test stayed at 2-8% in every
scenario tried. The variance here is the two-way sum
(Thompson 2011): group clusters + Bartlett HAC over calendar blocks - the
group x block cells counted twice. See `dm_test` for the details and
`tests/test_gate.py` for the size checks.

Macro F1 is reported with an event-group bootstrap p-value, a sign-flip test
gives a distribution-free cross-check, and placebo features (the candidate
shifted in time within each market) give an empirical noise floor. None of
these depends on how the feature was generated.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from ksearch.eval.cv import LABELS, _group_confusions, _macro_f1_from_cm, legacy_keep_rule

EPS = 1e-6           # log loss cap: -log(1e-6) = 13.8, so one confident miss cannot dominate
HOUR = 3600
PERSISTENCE_DAYS = 14  # minimum HAC reach for common shocks, in calendar days


# ── inputs ───────────────────────────────────────────────────────────────────
def _check(y, proba_ref, proba_cand=None, *others) -> tuple:
    y = np.asarray(y, dtype=object)
    n = len(y)
    bad = set(y.tolist()) - set(LABELS)
    if bad:
        raise ValueError(f"labels must be in {LABELS}; got {sorted(map(str, bad))[:5]}")
    out = [y]
    for name, p in (("proba_ref", proba_ref), ("proba_cand", proba_cand)):
        if p is None:
            continue
        p = np.asarray(p, dtype=float)
        if p.shape != (n, 3):
            raise ValueError(f"{name} must have shape ({n}, 3) in LABELS order, got {p.shape}")
        if not np.isfinite(p).all():
            raise ValueError(f"{name} contains NaN or inf")
        if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-4):
            raise ValueError(f"{name} rows must be probabilities summing to 1")
        out.append(p)
    for o in others:
        o = np.asarray(o)
        if len(o) != n:
            raise ValueError(f"all inputs must have {n} rows, got {len(o)}")
        if o.dtype.kind == "f" and np.isnan(o).any():
            raise ValueError("event_group / t0 contain NaN")
        if o.dtype == object and any(v is None or (isinstance(v, float) and math.isnan(v)) for v in o):
            raise ValueError("event_group / t0 contain missing values")
        out.append(o)
    if n == 0:
        raise ValueError("no scored events")
    return tuple(out)


# ── loss ─────────────────────────────────────────────────────────────────────
def balanced_weights(y) -> np.ndarray:
    """n / (K * n_class) per event: every class present carries equal total weight,
    weights average 1. Fixed by the scored labels only, so the reference and the
    candidate are scored with the same weights and their per-event difference is
    a paired quantity (the weights are what sklearn's "balanced" training uses)."""
    y = np.asarray(y, dtype=object)
    cls, inv, cnt = np.unique(y, return_inverse=True, return_counts=True)
    return len(y) / (len(cls) * cnt[inv])


def balanced_log_loss(y, proba, eps: float = EPS) -> np.ndarray:
    """Per-event class-balanced log loss. Only the true-class probability enters,
    clipped below at eps (no renormalisation: clipping one entry is enough to keep
    the loss finite, and it is applied identically to both models)."""
    y, proba = _check(y, proba)
    yi = np.searchsorted(LABELS, y.astype(str))  # LABELS is sorted
    p_true = np.clip(proba[np.arange(len(y)), yi], eps, 1.0)
    return -balanced_weights(y) * np.log(p_true)


def loss_differences(y, proba_ref, proba_cand, eps: float = EPS) -> np.ndarray:
    """d = loss(reference) - loss(candidate) per event; positive favours the candidate."""
    y, proba_ref, proba_cand = _check(y, proba_ref, proba_cand)
    return balanced_log_loss(y, proba_ref, eps) - balanced_log_loss(y, proba_cand, eps)


def loss_gain(y, proba_ref, proba_cand, eps: float = EPS) -> float:
    """Mean of d = balanced log loss(reference) - balanced log loss(candidate).
    The harness uses this same number for placebo gains."""
    return float(loss_differences(y, proba_ref, proba_cand, eps).mean())


# ── Diebold-Mariano with two-way robust variance ─────────────────────────────
def hac_lag(n_blocks: int, block_hours: float = 24) -> int:
    """Bartlett truncation lag, in calendar blocks.

    max(Newey-West rule of thumb floor(4 (T/100)^(2/9)), PERSISTENCE_DAYS of blocks),
    capped at (T - 1) // 4. Label windows are 4h and never overlap, so there is no
    mechanical MA(h-1) term to cover; the dependence left after summing a day is
    persistence of common shocks. The rule of thumb gives ~6 days at T ~ 800, which
    over-rejected (13%) when daily shocks had AR(1) coefficient 0.8; two weeks
    brought that to 7.5-8% while costing almost no power elsewhere. The cap keeps the
    HLN factor positive and the Bartlett downward bias (~ lag / T) small.
    """
    if n_blocks < 2:
        return 0
    nw = int(math.floor(4 * (n_blocks / 100) ** (2 / 9)))
    floor_ = int(math.ceil(PERSISTENCE_DAYS * 24 / block_hours))
    return int(max(0, min(max(nw, floor_), (n_blocks - 1) // 4)))


def _bartlett(U: np.ndarray, lag: int) -> float:
    """sum_k w_k sum_t U_t U_{t-k} over a calendar-indexed vector (gaps are zeros)."""
    v = float(U @ U)
    for k in range(1, min(lag, len(U) - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * float(U[k:] @ U[:-k])
    return v


def _bartlett_cells(key_g: np.ndarray, key_t: np.ndarray, U: np.ndarray, lag: int) -> float:
    """Same as _bartlett, applied within each group to its (group, block) cell sums."""
    v = float(U @ U)
    if lag == 0 or len(U) < 2:
        return v
    T = int(key_t.max()) + 1
    key = key_g.astype(np.int64) * (T + 1) + key_t        # sorted ascending by construction
    for k in range(1, lag + 1):
        j = np.searchsorted(key, key - k)
        hit = (j < len(key)) & (key_t >= k)
        hit[hit] = key[j[hit]] == (key - k)[hit]
        if hit.any():
            v += 2 * (1 - k / (lag + 1)) * float(U[hit] @ U[j[hit]])
    return v


def _hln(T: int, lag: int) -> float:
    h = lag + 1
    return math.sqrt(max(T + 1 - 2 * h + h * (h - 1) / T, 0.0) / T)


def dm_test(d, event_group, t0, *, block_hours: float = 24, lag: int | None = None,
            cluster_groups: bool = True, min_blocks: int = 20, min_groups: int = 10) -> dict:
    """One-sided DM test of H0: E[d] <= 0 vs H1: E[d] > 0 (candidate has lower loss).

    Estimand. The statistic is the pooled mean of d over events, i.e. the gap in
    pooled class-balanced log loss, the number the CV reports. Summing (not
    averaging) within a group or block keeps that estimand: a mean-per-group would
    test an equal-weight-per-ladder quantity, letting a 1-event ladder count as
    much as a 1,400-event one, and is a different question. The price of summing is
    that big ladders dominate the variance; the effective-cluster df below and the
    one-way floor account for that.

    Variance of sum(d), with u = d - mean(d) (intercept-only regression residuals;
    centring at the event level keeps power when block sizes vary):
        V_G  = sum_g U_g^2                                  group clusters
        V_T  = Bartlett HAC over calendar blocks of U_t     common shocks (Driscoll-Kraay)
        V_GT = Bartlett HAC within group over its cells     counted in both, removed once
        V    = max(V_G + V_T - V_GT, V_G, V_T)
    The two-way sum is not guaranteed positive; flooring it at the larger one-way
    estimate means the test is never less conservative than either one-way test
    (in particular never less than the week-3 group design without HAC).
    cluster_groups=False gives V = V_T: a plain DM on calendar-block sums.

    Ordering. Groups are not put in a sequence at all: a ladder's events are placed
    in the calendar blocks they occur in, so overlapping groups need no tie-break,
    and a group lasting 400 days (they exist) is not squeezed into one "time".
    Block index is calendar time, so lag k means k blocks apart even across gaps.

    Small samples. DM* = DM * HLN(T, lag) with T = number of non-empty blocks and
    h = lag + 1 (HLN's factor is derived for the rectangular kernel; with Bartlett it
    is a mild, conservative-leaning approximation), referred to Student t with
    df = min(T_eff, G_eff) - 1, where T_eff, G_eff are Kish effective counts
    n^2 / sum n_k^2. With equal block sizes and cluster_groups=False, lag 0, this is
    exactly the one-sample t statistic on block sums. Too few blocks or groups,
    non-positive variance, or df < 1 return p = 1.0: the gate can then only reject.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("d must be a finite 1-D array")
    _, event_group, t0 = _check(np.full(len(d), LABELS[0], dtype=object), None, None,
                                event_group, t0)
    t0 = np.asarray(t0, dtype=np.int64)
    _, g = np.unique(event_group, return_inverse=True)
    tb = (t0 // int(block_hours * HOUR))
    tb = tb - tb.min()
    G, Tcal = int(g.max()) + 1, int(tb.max()) + 1
    n_g, n_t = np.bincount(g, minlength=G), np.bincount(tb, minlength=Tcal)
    T = int((n_t > 0).sum())
    lag = hac_lag(T, block_hours) if lag is None else int(lag)
    out = {"mean_gain": float(d.mean()), "dm_stat": float("nan"), "dm_p": 1.0, "n_events": len(d),
           "n_blocks": T, "n_groups": G, "hac_lag": lag, "df": float("nan"),
           "block_hours": block_hours, "cluster_groups": cluster_groups, "status": "ok"}
    if T < min_blocks:
        return out | {"status": "too_few_blocks"}
    if cluster_groups and G < min_groups:
        return out | {"status": "too_few_groups"}

    u = d - d.mean()
    v_t = _bartlett(np.bincount(tb, u, Tcal), lag)
    t_eff = len(d) ** 2 / float((n_t.astype(float) ** 2).sum())
    if cluster_groups:
        U_g = np.bincount(g, u, G)
        v_g = float(U_g @ U_g)
        ck, cinv = np.unique(g.astype(np.int64) * Tcal + tb, return_inverse=True)
        v_gt = _bartlett_cells(ck // Tcal, ck % Tcal, np.bincount(cinv, u, len(ck)), lag)
        v = max(v_g + v_t - v_gt, v_g, v_t)
        df = min(t_eff, len(d) ** 2 / float((n_g.astype(float) ** 2).sum())) - 1
    else:
        v, df = v_t, t_eff - 1
    out["df"] = float(df)
    if not (v > 1e-12 * max(1.0, float(d @ d))) or df < 1:
        return out | {"status": "degenerate_variance"}
    stat = d.sum() / math.sqrt(v) * _hln(T, lag)
    return out | {"dm_stat": float(stat), "dm_p": float(stats.t.sf(stat, df))}


# ── sign-flip robustness check ───────────────────────────────────────────────
def _flip_p(sums: np.ndarray, n_perm: int, rng) -> float:
    obs = sums.sum()
    if not np.any(sums):
        return 1.0
    signs = rng.integers(0, 2, size=(n_perm, len(sums)), dtype=np.int8) * 2 - 1
    perm = signs @ sums
    return float((1 + np.sum(perm >= obs - 1e-12 * abs(obs))) / (n_perm + 1))


def sign_flip_test(d, event_group, t0, *, block_hours: float = 24, lag: int | None = None,
                   n_perm: int = 4999, seed: int = 0) -> dict:
    """Paired permutation test of H0: d is symmetric about 0, without normality.

    Two flips, each exact under one dependence structure: flipping whole event
    groups (valid if groups are independent), and flipping runs of lag + 1
    consecutive calendar blocks (valid if dependence dies out within a run).
    Neither is exact under both, so the reported p is the larger of the two. The
    statistic is the sum of d; p = (1 + #{perm >= obs}) / (n_perm + 1), so p > 0
    and it is 1 when every d is zero. Deterministic for a seed.
    """
    d = np.asarray(d, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("d must be finite")
    _, g = np.unique(np.asarray(event_group), return_inverse=True)
    tb = np.asarray(t0, dtype=np.int64) // int(block_hours * HOUR)
    tb = tb - tb.min()
    T = int(np.unique(tb).size)
    lag = hac_lag(T, block_hours) if lag is None else int(lag)
    _, run = np.unique(tb // (lag + 1), return_inverse=True)
    rng = np.random.default_rng(seed)
    p_group = _flip_p(np.bincount(g, d), n_perm, rng)
    p_time = _flip_p(np.bincount(run, d), n_perm, rng)
    return {"signflip_p": max(p_group, p_time), "signflip_p_group": p_group,
            "signflip_p_time": p_time, "n_perm": n_perm}


# ── macro F1 ─────────────────────────────────────────────────────────────────
def bootstrap_f1_test(y, proba_ref, proba_cand, event_group, *, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """Macro-F1 delta (candidate - reference, argmax predictions) with an
    event-group bootstrap CI and one-sided p-value.

    Null: delta <= 0. p = (1 + #{replicates <= 0}) / (n_boot + 1): the smallest
    alpha at which the one-sided percentile interval excludes zero, so p < 0.05
    exactly when the reported lower 5% bound is above 0 and the p agrees with the
    CI printed next to it. The shift method (centre the replicates at 0) assumes
    the bootstrap distribution is a translation of the null one; macro F1 with a
    rare class is bounded and skewed, so that is not safer, and identical
    predictions would give p = 0.5 instead of 1. Draws match
    clustered_paired_bootstrap for the same seed. Groups are treated as
    independent, so this p is optimistic when concurrent groups co-move; it is
    reported, not used by the default rule.
    """
    y, proba_ref, proba_cand, groups = _check(y, proba_ref, proba_cand, event_group)
    lab = np.array(LABELS, dtype=object)
    pr, pc = lab[proba_ref.argmax(1)], lab[proba_cand.argmax(1)]
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    cm_c = _group_confusions(y, pc, inv, G).reshape(G, 9)
    cm_r = _group_confusions(y, pr, inv, G).reshape(G, 9)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(n_boot)]).astype(float)
    reps = (_macro_f1_from_cm((w @ cm_c).reshape(-1, 3, 3))
            - _macro_f1_from_cm((w @ cm_r).reshape(-1, 3, 3)))
    delta = float(_macro_f1_from_cm(cm_c.sum(0).reshape(3, 3)) - _macro_f1_from_cm(cm_r.sum(0).reshape(3, 3)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"f1_delta": delta, "f1_ci95": [float(lo), float(hi)],
            "f1_p": float((1 + np.sum(reps <= 0)) / (n_boot + 1)), "n_boot": n_boot}


# ── placebo ──────────────────────────────────────────────────────────────────
def placebo_shift(values, market, t0, rng: np.random.Generator, min_shift: int = 1) -> np.ndarray:
    """Placebo column: the candidate's values moved back in time within each market.

    Each market's rows, sorted by t0, are rotated by an offset k drawn uniformly
    from [s, max(s, n // 2)], s = min(min_shift, n // 2), and every row whose new
    value would come from a later t0 is set to NaN. So a placebo row only ever
    carries a value observed at or before its own t0 (the feature's own lag of at
    least s events): a circular shift alone would hand a market's first rows its
    last rows' values, and a lead of a short-horizon return is the label itself.
    The price is that about k / n of each market's rows are NaN. Capping k at
    n // 2 keeps that at most half. All one-event markets are pooled, ordered by
    (t0, market), and rotated together under the same rule; a lone such row
    becomes NaN. A lag can still carry signal through autocorrelation, which only
    makes placebos stronger and the threshold conservative; raise min_shift to push
    placebos further from the truth.
    """
    values = np.asarray(values)
    market = np.asarray(market)
    t0 = np.asarray(t0)
    if not (len(values) == len(market) == len(t0)):
        raise ValueError("values, market and t0 must have the same length")
    _, mk = np.unique(market, return_inverse=True)
    order = np.lexsort((t0, mk))
    out = values.astype(float)
    counts = np.bincount(mk)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    def rotate(idx):
        n = len(idx)
        if n < 2:
            out[idx] = np.nan
            return
        s = max(1, min(min_shift, n // 2))
        k = int(rng.integers(s, max(s, n // 2) + 1))
        src = np.roll(idx, k)
        out[idx] = np.where(t0[src] > t0[idx], np.nan, values[src])

    singles = []
    for m in range(len(counts)):
        idx = order[starts[m]: starts[m] + counts[m]]
        if counts[m] == 1:
            singles.append(idx[0])
        else:
            rotate(idx)
    if singles:
        singles = np.asarray(singles)
        rotate(singles[np.lexsort((mk[singles], t0[singles]))])
    return out


def placebo_threshold(gains: Sequence[float], q: float = 0.95) -> float:
    """The ceil(q (m + 1))-th smallest of m placebo gains; +inf if that exceeds m.

    With exchangeable placebo and real gains under the null, P(real > threshold)
    <= 1 - q exactly, the same as a permutation p-value (1 + #{placebo >= gain}) /
    (m + 1) <= 1 - q. Fewer than q / (1 - q) placebos (19 at q = 0.95) cannot
    certify that, so the threshold is +inf and nothing passes.
    """
    g = np.sort(np.asarray(gains, dtype=float))
    if not np.isfinite(g).all():
        raise ValueError("placebo gains must be finite")
    r = math.ceil(q * (len(g) + 1) - 1e-9)
    return float(g[r - 1]) if 1 <= r <= len(g) else float("inf")


def placebo_p(gain: float, gains: Sequence[float]) -> float:
    g = np.asarray(gains, dtype=float)
    return float((1 + np.sum(g >= gain)) / (len(g) + 1))


# ── multiple testing ─────────────────────────────────────────────────────────
def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """BH-adjusted p-values (q-values), in input order. For reporting across all
    candidates of a search run; a candidate is a discovery at FDR level a iff its
    adjusted p <= a."""
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1 or np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be a 1-D array in [0, 1] without NaN")
    m = len(p)
    if m == 0:
        return p
    o = np.argsort(p, kind="mergesort")
    adj = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    return out


# ── decision ─────────────────────────────────────────────────────────────────
def _json(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, dict):
        return {k: _json(v) for k, v in x.items()}
    return x


@dataclass(frozen=True)
class GateResult:
    accept: bool
    reason: str               # "accepted" or the first failed check
    alpha: float
    mean_loss_gain: float     # balanced log loss(ref) - (cand), per event
    dm_stat: float
    dm_p: float
    dm_status: str
    hac_lag: int
    n_blocks: int
    n_groups: int
    n_events: int
    df: float
    signflip_p: float
    f1_delta: float
    f1_p: float
    f1_ci95: tuple[float, float]
    placebo_threshold: float | None = None
    placebo_p: float | None = None
    n_placebos: int | None = None
    legacy: dict | None = None

    def to_dict(self) -> dict:
        """JSON-safe: non-finite floats (nan stat, inf threshold) become None."""
        return _json(asdict(self))


def evaluate_candidate(y, proba_ref, proba_cand, event_group, t0, *, alpha: float = 0.05,
                       placebo_gains: Sequence[float] | None = None,
                       placebo_cutoff: float | None = None,
                       require_placebo: bool = False, require_signflip: bool = False,
                       require_f1: bool = False,
                       cand_fold_f1: Sequence[float] | None = None,
                       ref_fold_f1: Sequence[float] | None = None,
                       legacy_threshold: float = 0.005, block_hours: float = 24,
                       lag: int | None = None, cluster_groups: bool = True,
                       n_perm: int = 4999, n_boot: int = 2000, seed: int = 0,
                       eps: float = EPS) -> GateResult:
    """Gate one candidate. Probabilities are (n, 3) in LABELS order, already averaged
    over training seeds; rows are the scored out-of-fold events.

    Default rule: accept iff DM p < alpha and, if placebo gains or a threshold are
    given, mean gain > placebo threshold. require_* add the sign-flip, F1 and
    placebo-presence checks. The reason is the first failing check, in the order
    dm, placebo, signflip, f1.
    """
    if placebo_gains is not None and placebo_cutoff is not None:
        raise ValueError("pass placebo_gains or placebo_cutoff, not both")
    y, proba_ref, proba_cand, event_group, t0 = _check(y, proba_ref, proba_cand, event_group, t0)
    d = loss_differences(y, proba_ref, proba_cand, eps)
    dm = dm_test(d, event_group, t0, block_hours=block_hours, lag=lag, cluster_groups=cluster_groups)
    sf = sign_flip_test(d, event_group, t0, block_hours=block_hours, lag=dm["hac_lag"],
                        n_perm=n_perm, seed=seed)
    f1 = bootstrap_f1_test(y, proba_ref, proba_cand, event_group, n_boot=n_boot, seed=seed)

    thr = pp = n_pl = None
    if placebo_gains is not None:
        n_pl = len(placebo_gains)
        thr = placebo_threshold(placebo_gains)
        pp = placebo_p(dm["mean_gain"], placebo_gains)
    elif placebo_cutoff is not None:
        thr = float(placebo_cutoff)
    legacy = (legacy_keep_rule(list(cand_fold_f1), list(ref_fold_f1), legacy_threshold)
              if cand_fold_f1 is not None and ref_fold_f1 is not None else None)

    if dm["status"] != "ok":
        reason = f"dm_{dm['status']}"
    elif not dm["dm_p"] < alpha:
        reason = "dm_not_significant"
    elif thr is None and require_placebo:
        reason = "placebo_missing"
    elif thr is not None and not math.isfinite(thr):
        reason = "too_few_placebos"
    elif thr is not None and not dm["mean_gain"] > thr:
        reason = "below_placebo"
    elif require_signflip and not sf["signflip_p"] < alpha:
        reason = "signflip_not_significant"
    elif require_f1 and not f1["f1_p"] < alpha:
        reason = "f1_not_significant"
    else:
        reason = "accepted"
    return GateResult(
        accept=reason == "accepted", reason=reason, alpha=alpha,
        mean_loss_gain=dm["mean_gain"], dm_stat=dm["dm_stat"], dm_p=dm["dm_p"],
        dm_status=dm["status"], hac_lag=dm["hac_lag"], n_blocks=dm["n_blocks"],
        n_groups=dm["n_groups"], n_events=dm["n_events"], df=dm["df"],
        signflip_p=sf["signflip_p"], f1_delta=f1["f1_delta"], f1_p=f1["f1_p"],
        f1_ci95=tuple(f1["f1_ci95"]), placebo_threshold=thr, placebo_p=pp, n_placebos=n_pl,
        legacy=legacy)
