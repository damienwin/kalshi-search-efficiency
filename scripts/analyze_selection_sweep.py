"""Inference and figures for run_selection_sweep.py (criterion-chosen fixed-size training sets).

    python scripts/analyze_selection_sweep.py

Tests, fixed before looking at results (price + sentiment features, both weightings;
every additive loss uses the week 3 DM test on the per-event difference averaged over
the 20 replicates, both directions, Holm within each family):

  select A  Does dropping FLATs by the criterion instead of at random change a model's
            quality? random minus criterion, per model and FLAT level, on the natural test
            set: prior-corrected log loss and direction log loss. Family: 7 levels x 2 models.
  select B  Does it change the two-stage vs 3-class gap? (gap under the criterion) minus
            (gap under random drop), per level: log loss at the training balance (the
            balance sweep's diagonal), prior-corrected log loss and direction log loss on the
            natural test set. Family: 7 levels, plus the trend across levels.
  filter C  Two-stage vs 3-class at fixed size within a filtered pool, scored on the
            validation events passing the same filter: log loss, prior-corrected log loss,
            direction log loss. Family: the lever's levels (unfiltered included), plus the
            trend in the training FLAT share. Direction AUC gap by event-group bootstrap.

A design is reported only if it works and a test that is not already known from the balance
sweep is significant after Holm (the raw log-loss gain of balanced weighting is known: it is
the class-weight effect). "Works": a selection rule must not make the models it trains
significantly worse than random drop (test A); if it does, its B results are artifacts. Writes results/balance_sweep/selection_{inference.json,tables.md} and,
for reported designs, docs/reports/figures/balance_{select,filter}.png.
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
from ksearch.eval.balance import (FLAT, boot_weighted_auc, composition, effective_prior,
                                  group_multiplicities, label_index, ols_contrast, per_event_losses,
                                  prior_shift, target_weights)
from ksearch.search.gate import dm_test
from analyze_balance_sweep import FIG, INK, INK2, MUTED, OUT, RUNS, SURF, WEIGHT_COLOR, WEIGHT_NAME, style
from direction_significance import holm

SEL = os.path.join(RUNS, "selection")
FS = "price_sentiment"
ALPHA = 0.05
N_BOOT = 1000
CRIT_COLOR = {"random": "#898781", "articles": "#2a78d6", "vol": "#eb6834"}
CRIT_NAME = {"random": "Random drop (balance sweep)", "articles": "Keep most-covered FLATs",
             "vol": "Keep most-volatile FLATs"}
FILTER_NAME = {"all": "all", "articles": "articles ≥", "vol": "vol_24h above q"}


def per_fold_comp(counts):
    c = np.asarray(counts, float)
    return c / c.sum(1, keepdims=True)


class Losses:
    """Replicate-mean per-event losses of one prediction array, by test target."""

    def __init__(self, preds, counts, reps, fold, yi, model, w):
        self.preds, self.counts, self.reps, self.fold, self.yi = preds, counts, reps, fold, yi
        self.model, self.w = model, w

    def prior(self, r):
        if self.w == "none":
            return per_fold_comp(self.counts[r])[self.fold - 1]
        return np.broadcast_to(effective_prior(None, self.model, self.w), (len(self.yi), 3))

    def mean(self, met, target=None):
        acc = 0.0
        for r in range(self.reps):
            p = self.preds[r].astype(float)
            p = p / p.sum(1, keepdims=True)
            if met == "logloss_adj":
                p = prior_shift(p, self.prior(r), target)
            acc = acc + per_event_losses(self.yi, p)["dir_logloss" if met == "dir_logloss" else "logloss"]
        return acc / self.reps


def dm_both(d, groups, t0):
    up, down = dm_test(d, groups, t0), dm_test(-d, groups, t0)
    return {"gain": float(d.mean()), "p_pos": up["dm_p"], "p_neg": down["dm_p"]}


def weighted_diff(la, lb, W, S):
    """Per-event difference on subset S under test weights W, scaled so its mean is the
    weighted mean difference; NaN losses (direction loss on FLAT events) drop out."""
    m = S & ~np.isnan(la)
    return (W * (la - lb))[m] * m.sum() / W[m].sum(), m


def holm_family(cells, keys=("p_pos", "p_neg")):
    for k in keys:
        for c, a in zip(cells, holm([c[k] for c in cells])):
            c[k + "_holm"] = a
    return cells


def sig(c):
    return min(c["p_pos_holm"], c["p_neg_holm"]) < ALPHA


# ── select ───────────────────────────────────────────────────────────────────
def select(ev, meta, base):
    yi, fold = label_index(ev.label), ev.fold.to_numpy()
    groups, t0 = ev.event_group.to_numpy(), ev.t0.to_numpy()
    nat = np.bincount(yi, minlength=3) / len(yi)
    u = nat[2] / (nat[0] + nat[2])
    S = np.ones(len(yi), bool)
    levels = [f"{x:.2f}" for x in meta["flat_levels"]]
    out = {}
    for w in meta["weights"]:
        L = {}  # (variant, level, model) -> Losses
        for lev in levels:
            for m in meta["models"]:
                rc = [base["train_counts"][f"flat|{lev}|{r}"] for r in range(meta["reps"])]
                L[("random", lev, m)] = Losses(np.load(os.path.join(RUNS, "preds", f"flat_{lev}_{FS}_{w}_{m}.npy")),
                                               rc, meta["reps"], fold, yi, m, w)
                for c in meta["criteria"]:
                    cc = [meta["train_counts"][f"select|{c}|{lev}|{r}"] for r in range(meta["reps"])]
                    L[(c, lev, m)] = Losses(np.load(os.path.join(SEL, "preds", f"select_{c}_{lev}_{w}_{m}.npy")),
                                            cc, meta["reps"], fold, yi, m, w)
        # rep-mean losses on the two test sets
        tests = {lev: (target_weights(yi, fold, composition(float(lev), u)), composition(float(lev), u))
                 for lev in levels}
        E = {}
        for (v, lev, m), obj in L.items():
            W, tgt = tests[lev]
            E[(v, lev, m, "diag_logloss")] = (obj.mean("logloss"), W)
            E[(v, lev, m, "logloss_adj")] = (obj.mean("logloss_adj", nat), np.ones(len(yi)))
            E[(v, lev, m, "dir_logloss")] = (obj.mean("dir_logloss"), np.ones(len(yi)))
        for c in meta["criteria"]:
            # A: criterion vs random for each model
            for met in ("logloss_adj", "dir_logloss"):
                cells = []
                for m in meta["models"]:
                    for lev in levels:
                        (lr, W), (lc, _) = E[("random", lev, m, met)], E[(c, lev, m, met)]
                        d, mk = weighted_diff(lr, lc, W, S)
                        cells.append({"model": m, "level": lev, **dm_both(d, groups[mk], t0[mk])})
                out[f"A|{c}|{w}|{met}"] = holm_family(cells)
            # B: change in the two-stage gap
            for met in ("diag_logloss", "logloss_adj", "dir_logloss"):
                cells, ds = [], []
                for lev in levels:
                    (r3, W), (r2, _) = E[("random", lev, "3class", met)], E[("random", lev, "two_stage", met)]
                    (c3, _), (c2, _) = E[(c, lev, "3class", met)], E[(c, lev, "two_stage", met)]
                    d, mk = weighted_diff(c3 - c2, r3 - r2, W, S)
                    gr, _ = weighted_diff(r3, r2, W, S)
                    gc, _ = weighted_diff(c3, c2, W, S)
                    ds.append((d, mk))
                    cells.append({"level": lev, "gap_random": float(gr.mean()), "gap_criterion": float(gc.mean()),
                                  **dm_both(d, groups[mk], t0[mk])})
                cells = holm_family(cells)
                co = ols_contrast([float(l) for l in levels])
                e = sum(cj * d for cj, (d, _) in zip(co, ds))
                mk = ds[0][1]
                s = dm_both(e, groups[mk], t0[mk])
                out[f"B|{c}|{w}|{met}"] = {"cells": cells, "slope": {
                    "slope": s["gain"], "p_two_sided": float(min(1, 2 * min(s["p_pos"], s["p_neg"])))}}
        print(f"  select {w}", flush=True)
    return out


# ── filter ───────────────────────────────────────────────────────────────────
def filters(ev, meta):
    yi, fold = label_index(ev.label), ev.fold.to_numpy()
    groups, t0 = ev.event_group.to_numpy(), ev.t0.to_numpy()
    vm = np.load(os.path.join(SEL, "val_masks.npy"))
    names = meta["filters"]
    inv, M = group_multiplicities(groups, N_BOOT, seed=0)
    moved = yi != FLAT
    levels = {}
    for i, k in enumerate(names):
        S = vm[i]
        cnt = np.array([meta["train_counts"][f"filter|{k}|{r}"] for r in range(meta["reps"])], float)
        levels[k] = {"mask": S, "target": np.bincount(yi[S], minlength=3) / S.sum(),
                     "train_flat": float((cnt[..., FLAT] / cnt.sum(-1)).mean()),
                     "test_flat": float((yi[S] == FLAT).mean()), "n_test": int(S.sum()),
                     "n_test_moves": int((S & moved).sum())}
    out = {"levels": {k: {kk: v for kk, v in d.items() if kk not in ("mask", "target")} | {
        "target": d["target"].tolist()} for k, d in levels.items()}}
    lever_of = lambda k: k.split("_")[0]
    families = {lv: ["all_0"] + [k for k in names if lever_of(k) == lv] for lv in ("articles", "vol")}
    for w in meta["weights"]:
        L, P = {}, {}
        for k in names:
            cnt = [meta["train_counts"][f"filter|{k}|{r}"] for r in range(meta["reps"])]
            for m in meta["models"]:
                pr = np.load(os.path.join(SEL, "preds", f"filter_{k}_{w}_{m}.npy"))
                L[(k, m)] = Losses(pr, cnt, meta["reps"], fold, yi, m, w)
                P[(k, m)] = pr
        for lv, ks in families.items():
            for met in ("logloss", "logloss_adj", "dir_logloss"):
                cells, ds = [], []
                for k in ks:
                    S, tgt = levels[k]["mask"], levels[k]["target"]
                    la = L[(k, "3class")].mean(met, tgt)
                    lb = L[(k, "two_stage")].mean(met, tgt)
                    d, mk = weighted_diff(la, lb, np.ones(len(yi)), S)
                    full = np.zeros(len(yi))
                    full[mk] = d  # on the full index, for the trend contrast across nested subsets
                    ds.append((full, mk))
                    cells.append({"filter": k, "train_flat": levels[k]["train_flat"],
                                  "test_flat": levels[k]["test_flat"],
                                  "3class": float(np.nanmean(la[mk])), "two_stage": float(np.nanmean(lb[mk])),
                                  **dm_both(d, groups[mk], t0[mk])})
                cells = holm_family(cells)
                # trend in training FLAT share; each level's mean difference is sum(d_k)/n_k
                co = ols_contrast([levels[k]["train_flat"] for k in ks])
                any_m = np.any([mk for _, mk in ds], axis=0)
                e = sum(cj * d / mk.sum() for cj, (d, mk) in zip(co, ds))[any_m] * any_m.sum()
                s = dm_both(e, groups[any_m], t0[any_m])
                out[f"C|{lv}|{w}|{met}"] = {"cells": cells, "slope": {
                    "slope": s["gain"], "p_two_sided": float(min(1, 2 * min(s["p_pos"], s["p_neg"])))}}
            # direction AUC gap, event-group bootstrap
            cells = []
            for k in ks:
                S = levels[k]["mask"] & moved
                Wb = M[:, inv][:, S]
                vals = []
                for m in meta["models"]:
                    acc = 0.0
                    for r in range(meta["reps"]):
                        p = P[(k, m)][r][S].astype(float)
                        acc = acc + boot_weighted_auc(yi[S] == 2, p[:, 2] / np.maximum(p[:, 0] + p[:, 2], 1e-12), Wb)
                    vals.append(acc / meta["reps"])
                g = vals[1] - vals[0]
                lo, hi = np.nanpercentile(g, [2.5, 97.5])
                cells.append({"filter": k, "3class": float(np.nanmean(vals[0])), "two_stage": float(np.nanmean(vals[1])),
                              "gain": float(np.nanmean(g)), "ci95": [float(lo), float(hi)],
                              "p_boot": float(min(1, 2 * min(np.nanmean(g <= 0), np.nanmean(g >= 0))))})
            for c, a in zip(cells, holm([c["p_boot"] for c in cells])):
                c["p_boot_holm"] = a
            out[f"C|{lv}|{w}|dir_auc"] = {"cells": cells}
        print(f"  filter {w}", flush=True)
    return out


# ── decision, tables, figures ────────────────────────────────────────────────
def decide(sel, fil):
    """Significant tests not already known from the balance sweep."""
    hits = {"select": [], "filter": []}
    # a selection rule that significantly degrades the models it trains (test A, prior-corrected
    # log loss, criterion worse) is a broken method: its gap changes (B) are artifacts of that
    broken = [f"{k} @ {c['level']} {c['model']}: {c['gain']:+.3f}" for k, v in sel.items()
              if k.startswith("A|") and k.endswith("logloss_adj") for c in v if c["p_neg_holm"] < ALPHA]
    if broken:
        hits["select_broken"] = broken
        sel = {}
    for key, v in sel.items():
        cells = v if isinstance(v, list) else v["cells"]
        hits["select"] += [f"{key} @ {c['level']}{' ' + c['model'] if 'model' in c else ''}: "
                           f"{c['gain']:+.4f}" for c in cells if sig(c)]
        if isinstance(v, dict) and v["slope"]["p_two_sided"] < ALPHA:
            hits["select"].append(f"{key} slope {v['slope']['slope']:+.4f} p={v['slope']['p_two_sided']:.3g}")
    for key, v in fil.items():
        if not key.startswith("C|"):
            continue
        _, lv, w, met = key.split("|")
        if met == "logloss" and w == "balanced":
            continue  # the known class-weight effect
        for c in v["cells"]:
            if met == "dir_auc":
                if c["p_boot_holm"] < ALPHA:
                    hits["filter"].append(f"{key} @ {c['filter']}: {c['gain']:+.4f}")
            elif sig(c):
                hits["filter"].append(f"{key} @ {c['filter']}: {c['gain']:+.4f}")
        if "slope" in v and v["slope"]["p_two_sided"] < ALPHA:
            hits["filter"].append(f"{key} slope {v['slope']['slope']:+.4f} p={v['slope']['p_two_sided']:.3g}")
    return hits


def fmt_p(c):
    best = "2s" if c["p_pos_holm"] <= c["p_neg_holm"] else "3c"
    p = min(c["p_pos_holm"], c["p_neg_holm"])
    return f"{p:.3f} ({best})" if p < ALPHA else f"{p:.2f}"


def tables(sel, fil, meta, hits):
    lines = ["# Criterion-chosen fixed-size training sets", "",
             "Gains: positive = two-stage better (C, B gaps) or criterion better than random (A).",
             "p: Holm-adjusted DM p, side in parentheses when < 0.05.", ""]
    for c in meta["criteria"]:
        for w in meta["weights"]:
            lines += [f"## select B — {c}, {w}", "",
                      "| FLAT | metric | gap random | gap criterion | change | p |", "|---|---|---|---|---|---|"]
            for met in ("diag_logloss", "logloss_adj", "dir_logloss"):
                v = sel[f"B|{c}|{w}|{met}"]
                for cl in v["cells"]:
                    lines.append(f"| {cl['level']} | {met} | {cl['gap_random']:+.4f} | {cl['gap_criterion']:+.4f} "
                                 f"| {cl['gain']:+.4f} | {fmt_p(cl)} |")
                lines.append(f"| trend | {met} | | | {v['slope']['slope']:+.4f} | {v['slope']['p_two_sided']:.3f} |")
            lines += ["", f"## select A — {c}, {w} (random minus criterion)", "",
                      "| model | FLAT | prior-corrected log loss | p | direction log loss | p |",
                      "|---|---|---|---|---|---|"]
            for i, cl in enumerate(sel[f"A|{c}|{w}|logloss_adj"]):
                dl = sel[f"A|{c}|{w}|dir_logloss"][i]
                lines.append(f"| {cl['model']} | {cl['level']} | {cl['gain']:+.4f} | {fmt_p(cl)} | "
                             f"{dl['gain']:+.4f} | {fmt_p(dl)} |")
            lines.append("")
    for lv in ("articles", "vol"):
        for w in meta["weights"]:
            lines += [f"## filter C — {lv}, {w}", "",
                      "| filter | train FLAT | test FLAT | log loss gain | p | prior-corr gain | p | dir log loss gain | p | dir AUC 3c / 2s |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            vs = [fil[f"C|{lv}|{w}|{m}"]["cells"] for m in ("logloss", "logloss_adj", "dir_logloss", "dir_auc")]
            for a, b, d, au in zip(*vs):
                lines.append(f"| {a['filter']} | {a['train_flat']:.3f} | {a['test_flat']:.3f} | {a['gain']:+.4f} | {fmt_p(a)} "
                             f"| {b['gain']:+.4f} | {fmt_p(b)} | {d['gain']:+.4f} | {fmt_p(d)} "
                             f"| {au['3class']:.3f} / {au['two_stage']:.3f} |")
            sl = {m: fil[f"C|{lv}|{w}|{m}"]["slope"] for m in ("logloss", "logloss_adj", "dir_logloss")}
            lines.append("| trend | | | " + " | ".join(f"{s['slope']:+.4f} | {s['p_two_sided']:.3f}" for s in sl.values())
                         + " | |")
            lines.append("")
    lines += ["## Decision", ""]
    if "select_broken" in hits:
        lines.append(f"- **select**: not reported, the criterion makes models significantly worse in "
                     f"{len(hits['select_broken'])} of {len(meta['criteria']) * len(meta['weights']) * 14} cells "
                     f"(prior-corrected log loss; worst {min(hits['select_broken'], key=lambda s: float(s.split(': ')[1]))})")
    for d, h in hits.items():
        if d == "select_broken" or (d == "select" and "select_broken" in hits):
            continue
        lines.append(f"- **{d}**: {'reported' if h else 'not reported'} ({len(h)} significant new tests)")
        lines += [f"  - {x}" for x in h]
    return "\n".join(lines) + "\n"


def fig_select(sel, meta, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), facecolor=SURF, sharey=True)
    for ax, w in zip(axes, meta["weights"]):
        for c in meta["criteria"]:
            cells = sel[f"B|{c}|{w}|diag_logloss"]["cells"]
            x = [float(cl["level"]) for cl in cells]
            if c == meta["criteria"][0]:
                ax.plot(x, [cl["gap_random"] for cl in cells], color=CRIT_COLOR["random"], linewidth=2, marker="o",
                        markersize=4, markeredgecolor=SURF, label=CRIT_NAME["random"], zorder=3)
            ax.plot(x, [cl["gap_criterion"] for cl in cells], color=CRIT_COLOR[c], linewidth=2, marker="o",
                    markersize=4, markeredgecolor=SURF, label=CRIT_NAME[c], zorder=3)
            for xi, cl in zip(x, cells):
                if sig(cl):
                    ax.plot(xi, cl["gap_criterion"], "o", markersize=8, markerfacecolor="none",
                            markeredgecolor=CRIT_COLOR[c], markeredgewidth=1.2, zorder=4)
        style(ax)
        ax.set_title(WEIGHT_NAME[w], color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("FLAT share (train = test)", color=INK2, fontsize=8.5)
    axes[0].set_ylabel("Two-stage log-loss gain\n(positive = two-stage better)", color=INK2, fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_filter(fil, meta, base_inf, path):
    mets = [("logloss", "Log loss"), ("logloss_adj", "Prior-corrected log loss"), ("dir_logloss", "Direction log loss")]
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.4), facecolor=SURF, sharex=True)
    for i, w in enumerate(meta["weights"]):
        for j, (met, title) in enumerate(mets):
            ax = axes[i, j]
            rnd = base_inf.get(f"{FS}|{w}|diagonal")
            if rnd:
                key = met if met != "logloss_adj" else "logloss_adj"
                ax.plot([float(c["train"]) for c in rnd["cells"]], [c[key]["gain"] for c in rnd["cells"]],
                        color=CRIT_COLOR["random"], linewidth=1.5, linestyle=(0, (3, 2)), label=CRIT_NAME["random"], zorder=2)
            for lv in ("articles", "vol"):
                cells = fil[f"C|{lv}|{w}|{met}"]["cells"]
                x = [c["train_flat"] for c in cells]
                ax.plot(x, [c["gain"] for c in cells], color=CRIT_COLOR[lv], linewidth=2, marker="o", markersize=4,
                        markeredgecolor=SURF, label=f"Filter: {'articles ≥ k' if lv == 'articles' else 'vol_24h quantile'}",
                        zorder=3)
                for xi, c in zip(x, cells):
                    if sig(c):
                        ax.plot(xi, c["gain"], "o", markersize=8, markerfacecolor="none",
                                markeredgecolor=CRIT_COLOR[lv], markeredgewidth=1.2, zorder=4)
            style(ax)
            if i == 0:
                ax.set_title(title, color=INK, fontsize=9.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{WEIGHT_NAME[w]}\ntwo-stage gain", color=INK2, fontsize=8.5)
            if i == 1:
                ax.set_xlabel("Training FLAT share", color=INK2, fontsize=8.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def main() -> int:
    with open(os.path.join(SEL, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(RUNS, "meta.json")) as f:
        base = json.load(f)
    ev = pd.read_parquet(os.path.join(RUNS, "events.parquet"))
    sel = select(ev, meta, base)
    fil = filters(ev, meta)
    hits = decide(sel, fil)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "selection_inference.json"), "w") as f:
        json.dump({"select": sel, "filter": fil, "decision": hits}, f, indent=1, default=float)
    with open(os.path.join(OUT, "selection_tables.md"), "w") as f:
        f.write(tables(sel, fil, meta, hits))
    base_inf = json.load(open(os.path.join(OUT, "inference.json")))
    base_inf = base_inf.get("controlled", base_inf)
    if hits["select"]:
        fig_select(sel, meta, os.path.join(FIG, "balance_select.png"))
    if hits["filter"]:
        fig_filter(fil, meta, base_inf, os.path.join(FIG, "balance_filter.png"))
    for d, h in hits.items():
        print(f"{d}: {len(h)} significant new tests")
        for x in h:
            print("   ", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
