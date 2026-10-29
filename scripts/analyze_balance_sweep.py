"""Metrics, inference and figures for the class-balance sweeps.

    python scripts/analyze_balance_sweep.py

Reads runs/balance_sweep/ (run_balance_sweep.py, run_natural_sweeps.py). Writes
results/balance_sweep/ (aggregate metrics, inference, markdown tables; no per-event
rows) and docs/reports/figures/balance_*.png.

Every comparison is two-stage against 3-class with the same training subsample, seed,
weighting and features; gains are oriented so positive favours two-stage.

  Point estimates   mean over replicates of each metric on the importance-weighted test
                    set (ksearch.eval.balance.target_weights), then the paired difference.
  Additive losses   (log loss, Brier, direction log loss, class-balanced log loss): the
                    week 3 one-sided Diebold-Mariano test (ksearch.search.gate.dm_test,
                    variance robust to event-group and same-day dependence) on the per-event
                    weighted difference averaged over replicates, both directions, Holm
                    within each family of levels. Trend across levels: the OLS slope of the
                    gain on the level is a per-event linear combination of the per-level
                    differences, so it gets the same DM test (two-sided).
  Other metrics     (macro F1, direction AUC, flips per call, ...): event-group bootstrap
                    of the replicate-mean difference, percentile 95% CI and slope CI.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ksearch.data.manifest import REPO_ROOT
from ksearch.eval.balance import (FLAT, boot_weighted_auc, composition, confusion_metrics,
                                  effective_prior, group_multiplicities, label_index, ols_contrast,
                                  per_event_losses, prior_shift, target_weights, weighted_metrics,
                                  calls)
from ksearch.eval.cv import LABELS
from ksearch.search.gate import balanced_weights, dm_test
from direction_significance import holm

RUNS = os.path.join(REPO_ROOT, "runs", "balance_sweep")
OUT = os.path.join(REPO_ROOT, "results", "balance_sweep")
FIG = os.path.join(REPO_ROOT, "docs", "reports", "figures")
N_BOOT = 1000
PRIMARY_FS = "price_sentiment"
ADDITIVE = ["logloss", "brier", "dir_logloss", "balanced_logloss", "logloss_adj"]
BOOTSTRAPPED = ["macro_f1", "balanced_acc", "mcc", "flips_per_call", "dir_auc",
                "macro_f1_nc", "flips_per_call_nc"]
LOWER_IS_BETTER = {"logloss", "brier", "dir_logloss", "balanced_logloss", "logloss_adj",
                   "flips_per_call", "flips_per_call_nc", "flip_rate", "missed_rate", "false_move_rate"}
NATURAL_FLAT = {"all events": 0.851, "news events": 0.723}

# chart tokens (dataviz reference palette, light surface)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MODEL_COLOR = {"3class": "#2a78d6", "two_stage": "#eb6834"}
WEIGHT_COLOR = {"balanced": "#6250d6", "none": "#1baf7a"}
MODEL_NAME = {"3class": "3-class", "two_stage": "Two-stage"}
WEIGHT_NAME = {"balanced": "Balanced class weights (week 4)", "none": "Unweighted"}
FS_NAME = {"price": "price", "price_sentiment": "price + sentiment"}


# ── helpers ──────────────────────────────────────────────────────────────────
def oriented(metric: str, a_3c, b_2s):
    """Gain of two-stage over 3-class, positive = two-stage better."""
    return (a_3c - b_2s) if metric in LOWER_IS_BETTER else (b_2s - a_3c)


def fold_priors(meta, design, level, rep, fold, yi):
    """Per-event training composition of the model that predicted it."""
    counts = np.asarray(meta["train_counts"][f"{design}|{level}|{rep}"], float)
    comp = counts / counts.sum(1, keepdims=True)
    return comp[fold - 1]


def event_eff_prior(train_comp_ev, model, weights):
    if weights == "none":
        return train_comp_ev
    return np.broadcast_to(effective_prior(None, model, weights), train_comp_ev.shape)


def all_metrics(yi, p, w, prior_ev, target):
    """weighted_metrics on raw probabilities and argmax calls, plus the prior-neutral call
    (_nc) and the log loss after the known-prior correction to the test composition (_adj)."""
    m = weighted_metrics(yi, p, w, calls(p))
    nc = (p / prior_ev).argmax(1)
    cm = np.zeros((3, 3))
    np.add.at(cm, (yi, nc), w)
    m.update({f"{k}_nc": float(v) for k, v in confusion_metrics(cm).items()})
    p_adj = prior_shift(p, prior_ev, target)
    loss = per_event_losses(yi, p_adj)
    m["logloss_adj"] = float(np.average(loss["logloss"], weights=w))
    return m


def per_event_loss(metric, yi, p, w, prior_ev, target):
    """(per-event loss already multiplied by its weight so mean = the pooled metric, mask)."""
    if metric == "logloss_adj":
        p = prior_shift(p, prior_ev, target)
    if metric == "balanced_logloss":
        return balanced_weights(np.array(LABELS, dtype=object)[yi]) * per_event_losses(yi, p)["logloss"], np.ones(len(yi), bool)
    key = "logloss" if metric == "logloss_adj" else metric
    loss = per_event_losses(yi, p)[key]
    mask = ~np.isnan(loss)
    return np.where(mask, loss * w, 0.0), mask


# ── controlled sweep ─────────────────────────────────────────────────────────
def controlled(meta, ev):
    yi = label_index(ev.label)
    fold = ev.fold.to_numpy()
    groups, t0 = ev.event_group.to_numpy(), ev.t0.to_numpy()
    moved = yi != FLAT
    u_val = (yi == 2).sum() / moved.sum()
    flat_levels = [f"{f:.2f}" for f in meta["flat_levels"]]
    up_levels = [f"{u:.2f}" for u in meta["up_levels"]]
    test_targets = {"natural": None, **{f: composition(float(f), u_val) for f in flat_levels}}
    up_targets = {u: composition(meta["up_design_flat"], float(u)) for u in up_levels}
    weights = {k: (np.ones(len(yi)) if t is None else target_weights(yi, fold, t)) for k, t in test_targets.items()}
    weights.update({f"up{u}": target_weights(yi, fold, t) for u, t in up_targets.items()})
    nat_target = np.bincount(yi, minlength=3) / len(yi)

    def load(design, level, fs, w, m):
        return np.load(os.path.join(RUNS, "preds", f"{design}_{level}_{fs}_{w}_{m}.npy")).astype(float)

    # cells: (design, train level, test key)
    cells = [("flat", L, T) for L in flat_levels + ["nat", "full"] for T in test_targets]
    cells += [("up", u, f"up{u}") for u in up_levels] + [("up", u, "natural") for u in up_levels]
    rows = []
    cache = {}
    for fs in meta["feature_sets"]:
        for w in meta["weights"]:
            for design, L in {(d, l) for d, l, _ in cells}:
                for m in meta["models"]:
                    cache[(design, L, fs, w, m)] = load(design, L, fs, w, m)
            for design, L, T in cells:
                tgt = nat_target if T == "natural" else (up_targets[T[2:]] if T.startswith("up") else test_targets[T])
                for r in range(meta["reps"]):
                    comp_ev = fold_priors(meta, design, L, r, fold, yi)
                    for m in meta["models"]:
                        p = cache[(design, L, fs, w, m)][r]
                        p = p / p.sum(1, keepdims=True)
                        met = all_metrics(yi, p, weights[T], event_eff_prior(comp_ev, m, w), tgt)
                        rows.append({"design": design, "features": fs, "weights": w, "model": m,
                                     "train": L, "test": T, "rep": r, **met})
            print(f"  metrics {fs} {w}", flush=True)
    reps = pd.DataFrame(rows)

    # sanity: the "full" balanced replicates 0-4 are the week 4 models
    wk4 = pd.read_parquet(os.path.join(REPO_ROOT, "results", "direction_proba_all.parquet"))
    mine = cache[("flat", "full", "price_sentiment", "balanced", "two_stage")][:5].mean(0)
    wk4 = ev[["market_ticker", "t0"]].merge(wk4, on=["market_ticker", "t0"], how="left", validate="one_to_one")
    theirs = wk4[[f"p_two_stage_price_sentiment_{l}" for l in LABELS]].to_numpy()
    check = {"rows_matched": int(np.isfinite(theirs).all(1).sum()), "rows": len(ev),
             "max_abs_diff_vs_week4": float(np.nanmax(np.abs(mine - theirs)))}

    # ── inference ────────────────────────────────────────────────────────────
    families = {
        "diagonal": [("flat", f, f) for f in flat_levels],                # train and test at the same balance
        "deploy_shift": [("flat", "full", f) for f in flat_levels],        # week 4 models, test balance varied
        "train_shift": [("flat", f, "natural") for f in flat_levels],      # train balance varied, natural test
        "up_diagonal": [("up", u, f"up{u}") for u in up_levels],
    }
    xs = {"diagonal": [float(f) for f in flat_levels], "deploy_shift": [float(f) for f in flat_levels],
          "train_shift": [float(f) for f in flat_levels],
          "up_diagonal": [abs(float(u) - 0.5) for u in up_levels]}  # trend in imbalance |u - 0.5|
    inv, M = group_multiplicities(groups, N_BOOT, seed=0)
    inf = {}
    for fs in meta["feature_sets"]:
        for w in meta["weights"]:
            for fam, fam_cells in families.items():
                key = f"{fs}|{w}|{fam}"
                res = {"cells": [], "slope": {}}
                d_cells = {met: [] for met in ADDITIVE}
                boot = {met: [] for met in BOOTSTRAPPED}
                for design, L, T in fam_cells:
                    tgt = nat_target if T == "natural" else (up_targets[T[2:]] if T.startswith("up") else test_targets[T])
                    W = weights[T]
                    cell = {"train": L, "test": T}
                    sub = reps[(reps.design == design) & (reps.features == fs) & (reps.weights == w)
                               & (reps.train == L) & (reps.test == T)]
                    for met in ADDITIVE + BOOTSTRAPPED + ["move_auc", "ll_skill", "flip_rate", "missed_rate", "false_move_rate"]:
                        a = sub[sub.model == "3class"][met].mean()
                        b = sub[sub.model == "two_stage"][met].mean()
                        cell[met] = {"3class": a, "two_stage": b, "gain": oriented(met, a, b)}
                    # additive: per-event difference averaged over replicates
                    for met in ADDITIVE:
                        dsum, mask = 0.0, None
                        for r in range(meta["reps"]):
                            comp_ev = fold_priors(meta, design, L, r, fold, yi)
                            la, mask = per_event_loss(met, yi, cache[(design, L, fs, w, "3class")][r], W,
                                                      event_eff_prior(comp_ev, "3class", w), tgt)
                            lb, _ = per_event_loss(met, yi, cache[(design, L, fs, w, "two_stage")][r], W,
                                                   event_eff_prior(comp_ev, "two_stage", w), tgt)
                            dsum = dsum + (la - lb)
                        d = dsum / meta["reps"]
                        d = d[mask] * mask.sum() / W[mask].sum() if met != "balanced_logloss" else d
                        d_cells[met].append((d, mask))
                        up = dm_test(d, groups[mask], t0[mask])
                        down = dm_test(-d, groups[mask], t0[mask])
                        cell[met].update({"dm_p_two_stage_better": up["dm_p"], "dm_p_3class_better": down["dm_p"],
                                          "dm_status": up["status"]})
                    # bootstrap: replicate-mean metric under group multiplicities
                    Wb = M[:, inv] * W[None, :]
                    vals = {met: np.zeros((2, N_BOOT)) for met in BOOTSTRAPPED}
                    for r in range(meta["reps"]):
                        comp_ev = fold_priors(meta, design, L, r, fold, yi)
                        for j, mdl in enumerate(meta["models"]):
                            p = cache[(design, L, fs, w, mdl)][r]
                            p = p / p.sum(1, keepdims=True)
                            for suffix, ci in (("", p.argmax(1)),
                                               ("_nc", (p / event_eff_prior(comp_ev, mdl, w)).argmax(1))):
                                cmb = Wb @ np.eye(9)[yi * 3 + ci]
                                cmm = confusion_metrics(cmb.reshape(-1, 3, 3))
                                for met in BOOTSTRAPPED:
                                    base = met[:-3] if met.endswith("_nc") else met
                                    if (met.endswith("_nc")) == (suffix == "_nc") and base in cmm:
                                        vals[met][j] += cmm[base] / meta["reps"]
                            if fs == PRIMARY_FS:
                                p_up = p[moved, 2] / np.maximum(p[moved, 0] + p[moved, 2], 1e-12)
                                vals["dir_auc"][j] += boot_weighted_auc(yi[moved] == 2, p_up, Wb[:, moved]) / meta["reps"]
                    for met in BOOTSTRAPPED:
                        if met == "dir_auc" and fs != PRIMARY_FS:
                            continue
                        g = oriented(met, vals[met][0], vals[met][1])
                        boot[met].append(g)
                        lo, hi = np.nanpercentile(g, [2.5, 97.5])
                        cell[met].update({"ci95": [float(lo), float(hi)],
                                          "p_boot": float(min(1.0, 2 * min(np.nanmean(g <= 0), np.nanmean(g >= 0))))})
                    res["cells"].append(cell)
                # Holm within the family
                for met in ADDITIVE:
                    for side in ("dm_p_two_stage_better", "dm_p_3class_better"):
                        adj = holm([c[met][side] for c in res["cells"]])
                        for c, a in zip(res["cells"], adj):
                            c[met][side + "_holm"] = a
                for met in BOOTSTRAPPED:
                    if boot[met]:
                        adj = holm([c[met]["p_boot"] for c in res["cells"]])
                        for c, a in zip(res["cells"], adj):
                            c[met]["p_boot_holm"] = a
                # trend across levels
                c = ols_contrast(xs[fam])
                for met in ADDITIVE:
                    mask = d_cells[met][0][1]
                    e = sum(cj * dj for cj, (dj, _) in zip(c, d_cells[met]))
                    up = dm_test(e, groups[mask], t0[mask])
                    res["slope"][met] = {"slope": float(e.mean()),
                                         "p_two_sided": float(min(1.0, 2 * min(up["dm_p"], 1 - up["dm_p"])))}
                for met in BOOTSTRAPPED:
                    if boot[met]:
                        s = sum(cj * g for cj, g in zip(c, boot[met]))
                        lo, hi = np.nanpercentile(s, [2.5, 97.5])
                        point = float(sum(cj * cell[met]["gain"] for cj, cell in zip(c, res["cells"])))
                        res["slope"][met] = {"slope": point, "ci95": [float(lo), float(hi)],
                                             "p_two_sided": float(min(1.0, 2 * min(np.nanmean(s <= 0), np.nanmean(s >= 0))))}
                inf[key] = res
                print(f"  inference {key}", flush=True)
    return reps, inf, check


# ── natural levers ───────────────────────────────────────────────────────────
def natural():
    d = os.path.join(RUNS, "natural")
    if not os.path.isdir(d):
        return pd.DataFrame(), {}
    rows, inf = [], {}
    files = [fn[:-8].split("_", 1) for fn in os.listdir(d) if fn.endswith(".parquet")]
    for lever, value in sorted(files, key=lambda lv: (lv[0], float(lv[1]))):
        fn = f"{lever}_{value}.parquet"
        ev = pd.read_parquet(os.path.join(d, fn))
        yi = label_index(ev.label)
        nat = np.bincount(yi, minlength=3) / len(yi)
        groups, t0 = ev.event_group.to_numpy(), ev.t0.to_numpy()
        inv, M = group_multiplicities(groups, N_BOOT, seed=0)
        moved = yi != FLAT
        for fs in ("price", "price_sentiment"):
            for w in ("balanced", "none"):
                P = {}
                for m in ("3class", "two_stage"):
                    p = ev[[f"p_{fs}_{w}_{m}_{l}" for l in LABELS]].to_numpy()
                    P[m] = p / p.sum(1, keepdims=True)
                    prior = np.broadcast_to(effective_prior(nat, m, w), p.shape)
                    met = all_metrics(yi, P[m], np.ones(len(yi)), prior, nat)
                    rows.append({"lever": lever, "value": value, "features": fs, "weights": w, "model": m,
                                 "n": len(yi), "n_moves": int(moved.sum()), "flat_share": float(nat[FLAT]), **met})
                res = {}
                for met in ("logloss", "dir_logloss", "balanced_logloss"):
                    la, mask = per_event_loss(met, yi, P["3class"], np.ones(len(yi)), None, None)
                    lb, _ = per_event_loss(met, yi, P["two_stage"], np.ones(len(yi)), None, None)
                    dd = (la - lb)[mask]
                    up, down = dm_test(dd, groups[mask], t0[mask]), dm_test(-dd, groups[mask], t0[mask])
                    res[met] = {"gain": float(dd.mean()), "dm_p_two_stage_better": up["dm_p"],
                                "dm_p_3class_better": down["dm_p"]}
                Wb = M[:, inv]
                g = {}
                for m in ("3class", "two_stage"):
                    cmb = Wb @ np.eye(9)[yi * 3 + P[m].argmax(1)]
                    g[m] = confusion_metrics(cmb.reshape(-1, 3, 3))
                    p_up = P[m][moved, 2] / np.maximum(P[m][moved, 0] + P[m][moved, 2], 1e-12)
                    g[m]["dir_auc"] = boot_weighted_auc(yi[moved] == 2, p_up, Wb[:, moved])
                for met in ("macro_f1", "flips_per_call", "dir_auc"):
                    gg = oriented(met, g["3class"][met], g["two_stage"][met])
                    lo, hi = np.nanpercentile(gg, [2.5, 97.5])
                    res[met] = {"ci95": [float(lo), float(hi)],
                                "p_boot": float(min(1.0, 2 * min(np.nanmean(gg <= 0), np.nanmean(gg >= 0))))}
                inf[f"{lever}|{value}|{fs}|{w}"] = res
        print(f"  natural {lever} {value}", flush=True)
    return pd.DataFrame(rows), inf


# ── figures ──────────────────────────────────────────────────────────────────
def style(ax, ylabel=None, zero=True):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color=AXIS, linewidth=1)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def natural_markers(ax):
    for name, f in NATURAL_FLAT.items():
        ax.axvline(f, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.text(f, ax.get_ylim()[1], f" {name}", color=MUTED, fontsize=7, va="top", rotation=90)


def gain_panels(inf, fam, fs, metrics, titles, path, xlabel, x_of):
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 2.9), facecolor=SURF)
    for ax, met, title in zip(axes, metrics, titles):
        for w in ("balanced", "none"):
            cells = inf[f"{fs}|{w}|{fam}"]["cells"]
            x = [x_of(c) for c in cells]
            y = [c[met]["gain"] for c in cells]
            ax.plot(x, y, color=WEIGHT_COLOR[w], linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURF, markeredgewidth=1, label=WEIGHT_NAME[w], zorder=3)
            if "ci95" in cells[0][met]:
                lo = [c[met]["ci95"][0] for c in cells]
                hi = [c[met]["ci95"][1] for c in cells]
                ax.fill_between(x, lo, hi, color=WEIGHT_COLOR[w], alpha=0.15, linewidth=0)
            else:  # DM-tested: mark levels significant after Holm (either direction)
                for xi, c in zip(x, cells):
                    if min(c[met]["dm_p_two_stage_better_holm"], c[met]["dm_p_3class_better_holm"]) < 0.05:
                        ax.plot(xi, c[met]["gain"], "o", markersize=8, markerfacecolor="none",
                                markeredgecolor=WEIGHT_COLOR[w], markeredgewidth=1.2, zorder=4)
        style(ax)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
        if fam != "up_diagonal":
            natural_markers(ax)
    axes[0].set_ylabel("Two-stage gain over 3-class\n(positive = two-stage better)", color=INK2, fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def level_panels(reps, fs, metrics, titles, path):
    """Absolute metric per model vs FLAT share (train = test), one row per weighting."""
    d = reps[(reps.design == "flat") & (reps.features == fs) & (reps.train == reps.test)]
    d = d.assign(x=d.train.astype(float))
    fig, axes = plt.subplots(2, len(metrics), figsize=(3.0 * len(metrics), 5.2), facecolor=SURF, sharex=True)
    for i, w in enumerate(("balanced", "none")):
        for j, (met, title) in enumerate(zip(metrics, titles)):
            ax = axes[i, j]
            for m in ("3class", "two_stage"):
                g = d[(d.weights == w) & (d.model == m)].groupby("x")[met]
                mu, sd = g.mean(), g.std()
                ax.plot(mu.index, mu.values, color=MODEL_COLOR[m], linewidth=2, marker="o", markersize=4,
                        markeredgecolor=SURF, label=MODEL_NAME[m], zorder=3)
                ax.fill_between(mu.index, mu - sd, mu + sd, color=MODEL_COLOR[m], alpha=0.15, linewidth=0)
            style(ax, zero=False)
            if i == 0:
                ax.set_title(title, color=INK, fontsize=9.5, loc="left")
            if j == 0:
                ax.set_ylabel(WEIGHT_NAME[w], color=INK2, fontsize=8.5)
            if i == 1:
                ax.set_xlabel("FLAT share (train = test)", color=INK2, fontsize=8.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def heatmaps(reps, fs, met, path, title):
    d = reps[(reps.design == "flat") & (reps.features == fs) & ~reps.train.isin(["nat"])
             & (reps.test != "natural")]
    trains = [t for t in sorted(d.train.unique(), key=lambda s: (s == "full", s))]
    tests = sorted(d.test.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), facecolor=SURF)
    mats = []
    for w in ("balanced", "none"):
        g = d[d.weights == w].groupby(["model", "train", "test"])[met].mean()
        mats.append(np.array([[oriented(met, g[("3class", tr, te)], g[("two_stage", tr, te)]) for te in tests]
                              for tr in trains]))
    lim = max(np.abs(m).max() for m in mats)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    for ax, w, mat in zip(axes, ("balanced", "none"), mats):
        im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(tests)), tests, fontsize=7.5, color=MUTED)
        ax.set_yticks(range(len(trains)), [t if t != "full" else "full fold" for t in trains], fontsize=7.5, color=MUTED)
        ax.set_xlabel("Test FLAT share", color=INK2, fontsize=8.5)
        ax.set_title(WEIGHT_NAME[w], color=INK, fontsize=9.5, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=6.3,
                        color=INK if abs(mat[i, j]) < 0.6 * lim else "#ffffff")
    axes[0].set_ylabel("Training FLAT share", color=INK2, fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.set_label(f"{title}: two-stage gain (blue = two-stage better)", color=INK2, fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def natural_vs_controlled(nat, inf, fs, path):
    metrics = [("logloss", "Log loss"), ("macro_f1", "Macro F1"), ("dir_auc", "Direction AUC")]
    levers = {"threshold": ("Move threshold", "s"), "activity": ("Activity filter", "^"), "articles": ("Min. articles", "D")}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), facecolor=SURF, sharex=True)
    for i, w in enumerate(("balanced", "none")):
        cells = inf[f"{fs}|{w}|diagonal"]["cells"]
        for j, (met, title) in enumerate(metrics):
            ax = axes[i, j]
            ax.plot([float(c["train"]) for c in cells], [c[met]["gain"] for c in cells], color=MUTED,
                    linewidth=2, label="Controlled resampling (train = test)", zorder=2)
            d = nat[(nat.features == fs) & (nat.weights == w)]
            for (lever, (name, mk)), color in zip(levers.items(), ("#2a78d6", "#eb6834", "#1baf7a")):
                s = d[d.lever == lever].pivot_table(index="flat_share", columns="model", values=met).sort_index()
                ax.plot(s.index, oriented(met, s["3class"], s["two_stage"]), marker=mk, color=color,
                        linewidth=1.2, markersize=5, markeredgecolor=SURF, label=name, zorder=3)
            style(ax)
            if i == 0:
                ax.set_title(title, color=INK, fontsize=9.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{WEIGHT_NAME[w]}\ntwo-stage gain", color=INK2, fontsize=8.5)
            if i == 1:
                ax.set_xlabel("FLAT share of the event set", color=INK2, fontsize=8.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=8, labelcolor=INK2,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


# ── tables ───────────────────────────────────────────────────────────────────
def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def gain_table(inf, fs, fam, metrics, x_name):
    lines = []
    for w in ("balanced", "none"):
        res = inf[f"{fs}|{w}|{fam}"]
        lines.append(f"\n*{WEIGHT_NAME[w]}*\n")
        lines.append(f"| {x_name} | " + " | ".join(n for _, n in metrics) + " |")
        lines.append("|---|" + "---|" * len(metrics))
        for c in res["cells"]:
            cols = []
            for met, _ in metrics:
                v = c[met]
                if met in ADDITIVE:
                    p_better, p_worse = v["dm_p_two_stage_better_holm"], v["dm_p_3class_better_holm"]
                    tag = f" (2s p {fmt_p(p_better)})" if p_better < 0.05 else (f" (3c p {fmt_p(p_worse)})" if p_worse < 0.05 else "")
                    cols.append(f"{v['gain']:+.4f}{tag}")
                elif "ci95" in v:
                    cols.append(f"{v['gain']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
                else:
                    cols.append(f"{v['gain']:+.3f}")
            label = c["train"] if fam != "deploy_shift" else c["test"]
            lines.append(f"| {label} | " + " | ".join(cols) + " |")
        sl = res["slope"]
        lines.append("| **slope** | " + " | ".join(
            (f"{sl[m]['slope']:+.3f} (p {fmt_p(sl[m]['p_two_sided'])})" if m in sl else "") for m, _ in metrics) + " |")
    return "\n".join(lines)


def level_table(reps, fs):
    d = reps[(reps.design == "flat") & (reps.features == fs) & (reps.train == reps.test)]
    cols = [("ll_skill", "LL skill"), ("macro_f1", "Macro F1"), ("move_auc", "Move AUC"),
            ("dir_auc", "Dir. AUC"), ("flips_per_call", "Flips/call"), ("missed_rate", "Missed moves")]
    lines = ["| Weights | Model | FLAT share | " + " | ".join(n for _, n in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    g = d.groupby(["weights", "model", "train"])
    for (w, m, tr), s in g:
        lines.append(f"| {w} | {MODEL_NAME[m]} | {tr} | " + " | ".join(
            f"{s[c].mean():.3f} ± {s[c].std():.3f}" for c, _ in cols) + " |")
    return "\n".join(lines)


def natural_table(nat, ninf, fs):
    lines = ["| Lever | Value | Events | Moves | FLAT | Weights | Log loss gain | Macro F1 gain [95% CI] | Dir. AUC gain [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    d = nat[nat.features == fs]
    for (lever, value, w), s in d.groupby(["lever", "value", "weights"], sort=False):
        a, b = s[s.model == "3class"].iloc[0], s[s.model == "two_stage"].iloc[0]
        r = ninf[f"{lever}|{value}|{fs}|{w}"]
        ll = r["logloss"]
        tag = (f" (2s p {fmt_p(ll['dm_p_two_stage_better'])})" if ll["dm_p_two_stage_better"] < 0.05 else
               (f" (3c p {fmt_p(ll['dm_p_3class_better'])})" if ll["dm_p_3class_better"] < 0.05 else ""))
        lines.append(f"| {lever} | {value} | {int(a.n):,} | {int(a.n_moves):,} | {a.flat_share:.3f} | {w} | "
                     f"{ll['gain']:+.4f}{tag} | {b.macro_f1 - a.macro_f1:+.3f} [{r['macro_f1']['ci95'][0]:+.3f}, {r['macro_f1']['ci95'][1]:+.3f}] | "
                     f"{b.dir_auc - a.dir_auc:+.3f} [{r['dir_auc']['ci95'][0]:+.3f}, {r['dir_auc']['ci95'][1]:+.3f}] |")
    return "\n".join(lines)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(RUNS, "meta.json")) as f:
        meta = json.load(f)
    ev = pd.read_parquet(os.path.join(RUNS, "events.parquet"))
    reps, inf, check = controlled(meta, ev)
    print("week 4 reproduction check:", check)
    nat, ninf = natural()
    reps.to_parquet(os.path.join(OUT, "metrics_reps.parquet"), index=False)
    if len(nat):
        nat.to_parquet(os.path.join(OUT, "metrics_natural.parquet"), index=False)
    with open(os.path.join(OUT, "inference.json"), "w") as f:
        json.dump({"check": check, "controlled": inf, "natural": ninf, "n_boot": N_BOOT,
                   "fold_train_sizes": meta["fold_train_sizes"], "reps": meta["reps"]}, f, indent=1, default=float)

    flat_x = lambda c: float(c["train"])
    for fs in meta["feature_sets"]:
        tag = "" if fs == PRIMARY_FS else "_price"
        gain_panels(inf, "diagonal", fs, ["logloss", "macro_f1", "dir_auc" if fs == PRIMARY_FS else "dir_logloss", "flips_per_call"],
                    ["Log loss", "Macro F1", "Direction AUC" if fs == PRIMARY_FS else "Direction log loss", "Flips per directional call"],
                    os.path.join(FIG, f"balance_gain_diagonal{tag}.png"), "FLAT share (train = test)", flat_x)
        gain_panels(inf, "deploy_shift", fs, ["logloss", "logloss_adj", "macro_f1", "macro_f1_nc"],
                    ["Log loss", "Log loss, prior-corrected", "Macro F1, argmax call", "Macro F1, class-neutral call"],
                    os.path.join(FIG, f"balance_gain_deploy{tag}.png"), "Test FLAT share (trained on full folds)",
                    lambda c: float(c["test"]))
        gain_panels(inf, "up_diagonal", fs, ["logloss", "dir_logloss", "macro_f1", "flips_per_call"],
                    ["Log loss", "Direction log loss", "Macro F1", "Flips per directional call"],
                    os.path.join(FIG, f"balance_gain_up{tag}.png"), "UP share of moves (train = test, FLAT 0.85)", flat_x)
        level_panels(reps, fs, ["ll_skill", "macro_f1", "move_auc", "dir_auc"],
                     ["Log-loss skill vs class rates", "Macro F1", "Move AUC", "Direction AUC"],
                     os.path.join(FIG, f"balance_levels{tag}.png"))
        heatmaps(reps, fs, "logloss", os.path.join(FIG, f"balance_heatmap_logloss{tag}.png"), "Log loss")
        heatmaps(reps, fs, "macro_f1", os.path.join(FIG, f"balance_heatmap_f1{tag}.png"), "Macro F1")
        if len(nat):
            natural_vs_controlled(nat, inf, fs, os.path.join(FIG, f"balance_natural{tag}.png"))

    with open(os.path.join(OUT, "tables.md"), "w") as f:
        for fs in meta["feature_sets"]:
            f.write(f"\n## Features: {FS_NAME[fs]}\n")
            f.write("\n### Train = test balance (diagonal)\n")
            f.write(gain_table(inf, fs, "diagonal", [("logloss", "Log loss"), ("balanced_logloss", "Balanced LL"),
                                                     ("dir_logloss", "Direction LL"), ("macro_f1", "Macro F1"),
                                                     ("macro_f1_nc", "Macro F1 (neutral call)"),
                                                     ("flips_per_call", "Flips/call")] +
                               ([("dir_auc", "Direction AUC")] if fs == PRIMARY_FS else []), "FLAT share"))
            f.write("\n\n### Week 4 models (full folds), test balance varied\n")
            f.write(gain_table(inf, fs, "deploy_shift", [("logloss", "Log loss"), ("logloss_adj", "Log loss, prior-corrected"),
                                                         ("macro_f1", "Macro F1"), ("macro_f1_nc", "Macro F1 (neutral call)")],
                               "Test FLAT share"))
            f.write("\n\n### Training balance varied, natural test\n")
            f.write(gain_table(inf, fs, "train_shift", [("logloss", "Log loss"), ("balanced_logloss", "Balanced LL"),
                                                        ("macro_f1", "Macro F1"), ("flips_per_call", "Flips/call")],
                               "Train FLAT share"))
            f.write("\n\n### UP share of moves (FLAT 0.85, train = test)\n")
            f.write(gain_table(inf, fs, "up_diagonal", [("logloss", "Log loss"), ("dir_logloss", "Direction LL"),
                                                        ("macro_f1", "Macro F1"), ("flips_per_call", "Flips/call")],
                               "UP share"))
            f.write("\n\n### Absolute levels (train = test), mean ± sd over replicates\n\n")
            f.write(level_table(reps, fs))
            if len(nat):
                f.write("\n\n### Natural levers\n\n")
                f.write(natural_table(nat, ninf, fs))
            f.write("\n")
    print(f"wrote {OUT} and {FIG}/balance_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
