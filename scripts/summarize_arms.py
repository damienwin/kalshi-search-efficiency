"""Per-arm summary of finished search runs (dev only; the sealed slice is scored by confirm_sealed.py).

    python scripts/summarize_arms.py [--runs runs] [--arms random tpe llm_full ...] [--out runs/summary]

Reads runs/<arm>/seed<k>/run.json (status "complete" only) and reports, per arm:
  seeds, trials per run, accepted / failed / duplicate counts,
  dev improvement in balanced log loss (initial reference minus final reference): mean ± sd over seeds,
  sign consistency (seeds with improvement > 0), macro-F1 change, trials to first accept,
  feature-set overlap across seeds: mean pairwise Jaccard of accepted feature keys (exact
  formula match) and of the base columns the accepted features use (softer),
  fits, wall-clock, API tokens.
Arm comparisons: exact two-sided permutation test on seed-level improvements for every
pair of arms, Holm-corrected. With 5 seeds per arm the smallest attainable p is 2/252.
Dev-set numbers measure search behaviour, not out-of-sample skill.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ksearch.data.manifest import REPO_ROOT  # noqa: E402

ARMS = ["random", "tpe", "llm_full", "llm_no_history", "llm_blind"]


def columns(spec) -> set[str]:
    if isinstance(spec, dict):
        out = {spec["col"]} if "col" in spec else set()
        for a in spec.get("args", []):
            out |= columns(a)
        return out
    return set()


def jaccard_mean(sets: list[set]) -> float | None:
    """Over pairs of seeds where at least one accepted something (two empty sets say nothing)."""
    pairs = [(x, y) for x, y in itertools.combinations(sets, 2) if x | y]
    if not pairs:
        return None
    return float(np.mean([len(x & y) / len(x | y) for x, y in pairs]))


def load_runs(root: str, arm: str) -> list[dict]:
    d = os.path.join(root, arm)
    runs = []
    if not os.path.isdir(d):
        return runs
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name, "run.json")
        if name.startswith("seed") and os.path.exists(p):
            r = json.load(open(p))
            if r.get("status") == "complete":
                runs.append(r)
    return runs


def seed_row(r: dict) -> dict:
    ini, fin = r["initial_reference"], r["final_reference"]
    acc = r["accepted"]
    tok = r.get("api_tokens") or {}
    return {"seed": r["seed"], "trials": r["trials_completed"], **r["totals"],
            "improvement": ini["balanced_logloss"] - fin["balanced_logloss"],
            "f1_change": fin["pooled_macro_f1"] - ini["pooled_macro_f1"],
            "first_accept": acc[0]["trial"] if acc else None,
            "keys": {a["key"] for a in acc}, "cols": set().union(*[columns(a["spec"]) for a in acc]) if acc else set(),
            "formulas": [a["formula"] for a in acc], "n_fits": r.get("n_fits"), "wall_s": r.get("wall_s"),
            "tokens": sum(v or 0 for v in tok.values()) if isinstance(tok, dict) else None,
            "git_dirty": (r.get("git") or {}).get("dirty")}


def arm_summary(rows: list[dict]) -> dict:
    imp = np.array([x["improvement"] for x in rows])
    f1 = np.array([x["f1_change"] for x in rows])
    sd = (lambda a: float(a.std(ddof=1)) if len(a) > 1 else None)
    fa = [x["first_accept"] for x in rows if x["first_accept"] is not None]
    return {"seeds": [x["seed"] for x in rows], "n": len(rows),
            "trials_mean": float(np.mean([x["trials"] for x in rows])),
            "accepted_mean": float(np.mean([x["accepted"] for x in rows])),
            "accepted_sd": sd(np.array([x["accepted"] for x in rows], float)),
            "failed_total": int(sum(x["failed"] for x in rows)), "duplicate_total": int(sum(x["duplicate"] for x in rows)),
            "improvement_mean": float(imp.mean()), "improvement_sd": sd(imp),
            "sign_consistency": f"{int((imp > 0).sum())}/{len(imp)}",
            "f1_change_mean": float(f1.mean()), "f1_change_sd": sd(f1),
            "first_accept_median": float(np.median(fa)) if fa else None, "seeds_with_accept": len(fa),
            "overlap_keys_jaccard": jaccard_mean([x["keys"] for x in rows]),
            "overlap_cols_jaccard": jaccard_mean([x["cols"] for x in rows]),
            "fits_mean": float(np.mean([x["n_fits"] or 0 for x in rows])),
            "wall_h_total": sum(x["wall_s"] or 0 for x in rows) / 3600,
            "tokens_total": sum(x["tokens"] or 0 for x in rows),
            "any_git_dirty": any(x["git_dirty"] for x in rows)}


def perm_test(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation p for the difference in means."""
    pool, n = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        m = np.zeros(len(pool), bool)
        m[list(idx)] = True
        hits += abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-12
        tot += 1
    return hits / tot


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "summary"))
    a = ap.parse_args()
    per_seed, summ = {}, {}
    for arm in a.arms:
        rows = [seed_row(r) for r in load_runs(a.runs, arm)]
        if rows:
            per_seed[arm], summ[arm] = rows, arm_summary(rows)
    if not summ:
        print("no complete runs")
        return 1
    comps = []
    for x, y in itertools.combinations(summ, 2):
        ix = np.array([r["improvement"] for r in per_seed[x]])
        iy = np.array([r["improvement"] for r in per_seed[y]])
        if len(ix) > 1 and len(iy) > 1:
            comps.append({"a": x, "b": y, "diff": float(ix.mean() - iy.mean()), "p": perm_test(ix, iy)})
    for c, p in zip(comps, holm([c["p"] for c in comps])):
        c["p_holm"] = p

    os.makedirs(a.out, exist_ok=True)
    ser = {arm: [{k: (sorted(v) if isinstance(v, set) else v) for k, v in r.items()} for r in rows]
           for arm, rows in per_seed.items()}
    json.dump({"arms": summ, "per_seed": ser, "comparisons": comps}, open(os.path.join(a.out, "arms.json"), "w"),
              indent=2, default=str)

    f = lambda v, d=4: "–" if v is None else f"{v:.{d}f}"
    lines = ["| Arm | Seeds | Accepted/run | Failed | Dev log-loss improvement | Sign | Macro-F1 change | "
             "Overlap (keys / cols) | Fits/run | Wall (h) | Tokens |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, s in summ.items():
        lines.append(f"| {arm} | {s['n']} | {f(s['accepted_mean'], 1)} ± {f(s['accepted_sd'], 1)} | {s['failed_total']} | "
                     f"{f(s['improvement_mean'])} ± {f(s['improvement_sd'])} | {s['sign_consistency']} | "
                     f"{f(s['f1_change_mean'])} ± {f(s['f1_change_sd'])} | {f(s['overlap_keys_jaccard'], 2)} / "
                     f"{f(s['overlap_cols_jaccard'], 2)} | {f(s['fits_mean'], 0)} | {f(s['wall_h_total'], 1)} | "
                     f"{s['tokens_total'] or '–'} |")
    if comps:
        lines += ["", "| Comparison | Mean diff | p (exact perm.) | p (Holm) |", "|---|---|---|---|"]
        lines += [f"| {c['a']} − {c['b']} | {c['diff']:+.4f} | {c['p']:.3f} | {c['p_holm']:.3f} |" for c in comps]
    if any(s["any_git_dirty"] for s in summ.values()):
        lines += ["", "Note: some runs were made from a git-dirty tree."]
    md = "\n".join(lines)
    open(os.path.join(a.out, "arms.md"), "w").write(md + "\n")
    print(md)
    for arm, rows in per_seed.items():
        print(f"\n{arm}:")
        for r in rows:
            print(f"  seed {r['seed']}: +{r['improvement']:.4f}  {r['formulas'] or '(none accepted)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
