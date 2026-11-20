"""Week 5 figures: each model's metrics as the FLAT share rises, for random downscaling
(run_balance_sweep.py, train = test) and filtered pools at a fixed size (run_selection_sweep.py).

    python scripts/week5_figures.py

Reads results/balance_sweep/metrics_reps.parquet and runs/balance_sweep/selection/. Writes
docs/reports/figures/week5_{downscale,filter}.png and results/balance_sweep/week5_metrics.md.
Price + sentiment features; mean ± SD over the 20 replicates.
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
from ksearch.eval.balance import calls, label_index, weighted_metrics
from analyze_balance_sweep import FIG, INK, INK2, MODEL_COLOR, MODEL_NAME, OUT, RUNS, SURF, style

SEL = os.path.join(RUNS, "selection")
METRICS = [("macro_f1", "Macro F1"), ("logloss", "Log loss"), ("dir_auc", "Direction AUC"),
           ("flips_per_call", "Flips per directional call")]
PLOTTED = METRICS[:3]  # flips are in the tables only
LINE = {"none": "-", "balanced": (0, (4, 2))}
WNAME = {"none": "unweighted", "balanced": "balanced weights"}
LEVER_NAME = {"articles": "Filter: articles ≥ k (k = 0, 1, 2, 5, 10)",
              "vol": "Filter: vol_24h above pool quantile (0, 0.25, 0.5, 0.75)"}


def downscale():
    d = pd.read_parquet(os.path.join(OUT, "metrics_reps.parquet"))
    d = d[(d.design == "flat") & (d.features == "price_sentiment") & (d.train == d.test)]
    d = d[~d.train.isin(["nat", "full"])].assign(flat=lambda x: x.train.astype(float))
    return d[["weights", "model", "flat", "rep"] + [m for m, _ in METRICS]]


def filtered():
    meta = json.load(open(os.path.join(SEL, "meta.json")))
    ev = pd.read_parquet(os.path.join(RUNS, "events.parquet"))
    yi = label_index(ev.label)
    vm = np.load(os.path.join(SEL, "val_masks.npy"))
    rows = []
    for i, k in enumerate(meta["filters"]):
        S = vm[i]
        flat = float((yi[S] == 1).mean())
        levers = ["articles", "vol"] if k == "all_0" else [k.split("_")[0]]
        for w in meta["weights"]:
            for m in meta["models"]:
                P = np.load(os.path.join(SEL, "preds", f"filter_{k}_{w}_{m}.npy"))
                for r in range(meta["reps"]):
                    p = P[r][S].astype(float)
                    p /= p.sum(1, keepdims=True)
                    met = weighted_metrics(yi[S], p, np.ones(S.sum()), calls(p))
                    for lv in levers:
                        rows.append({"lever": lv, "filter": k, "weights": w, "model": m, "flat": flat,
                                     "n": int(S.sum()), "rep": r, **{mm: met[mm] for mm, _ in METRICS}})
    return pd.DataFrame(rows)


def lines(ax, d, met):
    for m in ("3class", "two_stage"):
        for w in ("none", "balanced"):
            g = d[(d.model == m) & (d.weights == w)].groupby("flat")[met]
            mu, sd = g.mean(), g.std()
            ax.plot(mu.index, mu.values, color=MODEL_COLOR[m], linestyle=LINE[w], linewidth=2, marker="o",
                    markersize=3.5, markeredgecolor=SURF, label=f"{MODEL_NAME[m]}, {WNAME[w]}", zorder=3)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=MODEL_COLOR[m], alpha=0.08, linewidth=0)


def figure(panels, path, xlabel):
    """panels: list of (row title, frame)."""
    fig = plt.figure(figsize=(11, 4.0 * len(panels) + 0.4), facecolor=SURF, layout="constrained")
    subs = np.atleast_1d(fig.subfigures(len(panels), 1))
    first = None
    for sub, (title, d) in zip(subs, panels):
        sub.set_facecolor(SURF)
        axes = sub.subplots(1, len(PLOTTED))
        first = first if first is not None else axes[0]
        for ax, (met, name) in zip(axes, PLOTTED):
            lines(ax, d, met)
            style(ax, zero=False)
            ax.set_title(name, color=INK, fontsize=9.5, loc="left")
            ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
        if title:
            sub.suptitle(title, color=INK, fontsize=10, weight="bold", x=0.01, ha="left")
    h, l = first.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def table(d, keys):
    g = d.groupby(keys + ["model", "weights"])[[m for m, _ in METRICS]].agg(["mean", "std"])
    out = ["| " + " | ".join(keys) + " | model | weights | " + " | ".join(n for _, n in METRICS) + " |",
           "|" + "---|" * (len(keys) + 2 + len(METRICS))]
    for idx, row in g.iterrows():
        vals = [f"{row[(m, 'mean')]:.3f} ± {row[(m, 'std')]:.3f}" for m, _ in METRICS]
        out.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in idx)
                   + " | " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> int:
    ds, fl = downscale(), filtered()
    figure([(None, ds)], os.path.join(FIG, "week5_downscale.png"), "FLAT share (train = test)")
    figure([(LEVER_NAME["articles"], fl[fl.lever == "articles"]), (LEVER_NAME["vol"], fl[fl.lever == "vol"])],
           os.path.join(FIG, "week5_filter.png"), "FLAT share of the filtered test set")
    with open(os.path.join(OUT, "week5_metrics.md"), "w") as f:
        f.write("# Week 5 metrics (price + sentiment, mean ± SD over 20 replicates)\n\n## Random downscaling\n\n")
        f.write(table(ds, ["flat"]) + "\n\n## Filtered pools, fixed size\n\n")
        f.write(table(fl.drop_duplicates(["filter", "weights", "model", "rep"]), ["filter", "flat"]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
