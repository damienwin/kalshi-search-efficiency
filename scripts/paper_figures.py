"""Figure 1 for the paper: running gain in class-balanced log loss against trial number.

Usage: python scripts/paper_figures.py [--out paper/figures/fig1_running_gain.pdf]

For each run, the gain after trial t is the initial reference's balanced log loss
minus the reference's after t trials (the reference changes only on acceptance,
when it becomes the accepted candidate's scores). One panel per full arm: a thin
line per seed and a thick line for the mean over five seeds. LLM no-history is 0
on every seed and is stated in the caption instead of drawn. The dotted line
is the 95th percentile of the best single null feature among 100 (Table 2), a
scale reference only. Palette: categorical slots 1-3 of the dataviz reference
palette, validated for CVD separation; panels are titled, so color is not the only cue.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = [("random", "Random", "#2a78d6"), ("tpe", "TPE", "#eb6834"),
        ("llm_full", "LLM full", "#1baf7a"), ("llm_no_history", "LLM no-history", "#8a8a85")]
BUDGET = 100
NULL_BEST_OF_100_P95 = 0.0041
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#e4e3df"


def running_gain(run_dir: str) -> np.ndarray:
    with open(os.path.join(run_dir, "run.json")) as f:
        base = json.load(f)["initial_reference"]["balanced_logloss"]
    gain = np.zeros(BUDGET + 1)
    current = 0.0
    trials = {}
    with open(os.path.join(run_dir, "trials.jsonl")) as f:
        for line in f:
            t = json.loads(line)
            trials[t["trial"]] = t
    first = min(trials)  # trial numbering may start at 0 or 1
    for i in range(BUDGET):
        t = trials.get(first + i)
        if t is not None and t["status"] == "accepted":
            current = base - t["cand"]["balanced_logloss"]
        gain[i + 1] = current
    return gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "paper", "figures", "fig1_running_gain.pdf"))
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    panels = [a for a in ARMS if a[0] != "llm_no_history"]  # no-history is 0 on every seed (caption)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5, 2.1), sharey=True)
    x = np.arange(BUDGET + 1)
    finals = {}
    for ax, (key, label, color) in zip(axes, panels):
        runs = np.array([running_gain(os.path.join(REPO_ROOT, "runs", key, f"seed{s}")) for s in range(5)])
        for r in runs:
            ax.step(x, r, where="post", color=color, linewidth=0.8, alpha=0.35, zorder=2)
        mean = runs.mean(axis=0)
        ax.step(x, mean, where="post", color=color, linewidth=2.0, zorder=3)
        finals[label] = mean[-1]
        ax.axhline(NULL_BEST_OF_100_P95, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
        ax.set_title(f"{label}  (mean {mean[-1]:.4f})", fontsize=8, color=INK, loc="left")
        ax.set_xlim(0, BUDGET)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Trial")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Gain in balanced log loss")
    axes[0].set_ylim(-0.0008, 0.025)
    axes[0].text(2, NULL_BEST_OF_100_P95 + 0.0005, "null reference", color=MUTED, fontsize=6.5)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.2, top=0.88, wspace=0.25)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=400)
    print(args.out, {k: round(float(v), 4) for k, v in finals.items()})


if __name__ == "__main__":
    main()
