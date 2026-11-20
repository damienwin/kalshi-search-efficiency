"""Markdown tables for the direction-flip grid (results/cv_dir_*), current dataset.

    python scripts/summarize_direction_grid.py

Each row: one model variant and feature set, 5 training seeds. Macro F1 per seed,
log loss mean ± sd, and the direction breakdown (ksearch.eval.cv.direction_breakdown)
averaged over seeds.
"""

import glob
import json
import os
import statistics as st

from ksearch.data.manifest import REPO_ROOT

VARIANTS = [("3class", "3-class (week 3)"), ("two_stage", "Two-stage"),
            ("flippen1", "Flip-penalized loss, λ = 1")]
FEATURES = [("price", "Price"), ("price_sentiment", "Price + sentiment")]
EVENTS = [("all", "All events"), ("news", "News events: ≥ 2 articles in the 6 hours before prediction time")]


def ms(xs, d=3):
    return f"{st.mean(xs):.{d}f} ± {st.stdev(xs):.{d}f}"


def main() -> int:
    for ev, title in EVENTS:
        rows, rows2 = [], []
        n = None
        for v, vname in VARIANTS:
            for f, fname in FEATURES:
                runs = [json.load(open(p)) for p in
                        sorted(glob.glob(os.path.join(REPO_ROOT, "results", f"cv_dir_{v}_{f}_{ev}_seed[0-9].json")))]
                if not runs:
                    continue
                n = runs[0]["n_events_scored"]
                f1 = [r["pooled_macro_f1"]["model"] for r in runs]
                ll = [r["pooled_logloss_model"] for r in runs]
                rows.append(f"| {vname} | {fname} | " + " | ".join(f"{x:.3f}" for x in f1)
                            + f" | {ms(f1)} | {ms(ll)} |")
                b = [r["direction_breakdown_model"] for r in runs]
                avg = lambda k, d=0: f"{st.mean(x[k] for x in b):.{d}f}"
                rows2.append(f"| {vname} | {fname} | {avg('move_auc', 3)} | {avg('dir_auc', 3)} | "
                             f"{avg('dir_acc', 3)} | {avg('right_direction')} | {avg('flips')} | "
                             f"{avg('false_moves')} | {avg('missed_moves')} | "
                             f"{st.mean(x['flips'] / (x['flips'] + x['right_direction']) for x in b):.1%} |")
        if not rows:
            continue
        moves = runs[0]["direction_breakdown_model"]["n_moves"]
        print(f"\n**{title} ({n:,}; {moves:,} moved)**\n")
        print("| Model | Features | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Macro F1 (mean ± sd) | Log loss (mean ± sd) |")
        print("|---|---|---|---|---|---|---|---|---|")
        print("\n".join(rows))
        print(f"\n| Model | Features | Move AUC | Direction AUC | Direction accuracy | Right direction | Flips | False moves | Missed moves | Flips ÷ directional calls |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        print("\n".join(rows2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
