"""Placebo-count sensitivity: re-score boundary candidates with 99 placebo copies instead of 19.

    python scripts/placebo_sensitivity.py select           # list the boundary candidates
    python scripts/placebo_sensitivity.py run [--n 99]     # score them (resumable)
    python scripts/placebo_sensitivity.py summary

Boundary candidates are the full-arm trials (random, tpe, llm_full) that passed the DM
test and were closest to the 19-placebo decision:
  - rejected after losing to exactly one of its 19 placebos, and
  - accepted with a margin over its best placebo of at most MARGIN.

Each is re-evaluated by the search's own Evaluator.trial against the reference it faced
in its run (the features accepted before it, in order), with n_placebos = 99. Placebo j
uses rng (placebo_seed, candidate key, j), so the first 19 placebos are the ones the
search drew; the script checks that they reproduce the logged gains. The 99-copy verdict
uses the same rule as the gate: the candidate must exceed the ceil(0.95 * 100) = 95th
smallest placebo gain, i.e. permutation p = (1 + #{placebo >= gain}) / 100 <= 0.05.

Dev data only; no API calls. About 23 minutes per candidate (2,500 fits).
Output: runs/placebo_sensitivity/results.jsonl
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from ksearch.search import parse  # noqa: E402
from ksearch.search.gate import placebo_p, placebo_threshold  # noqa: E402
from ksearch.search.loop import make_evaluator, search_config  # noqa: E402
from run_search import load_events  # noqa: E402

ARMS = ("random", "tpe", "llm_full")
MARGIN = 0.0004
OUT = os.path.join(REPO_ROOT, "runs", "placebo_sensitivity")


def run_trials(arm: str, seed: int) -> list[dict]:
    with open(os.path.join(REPO_ROOT, "runs", arm, f"seed{seed}", "trials.jsonl")) as f:
        return [json.loads(line) for line in f]


def select() -> list[dict]:
    out = []
    for arm in ARMS:
        for seed in range(5):
            for t in run_trials(arm, seed):
                pg = t.get("placebo_gains")
                if not pg:
                    continue
                gain = t["gate"]["mean_loss_gain"]
                k = int(sum(g >= gain for g in pg))
                margin = gain - max(pg)
                if (t["status"] != "accepted" and k == 1) or (t["status"] == "accepted" and margin <= MARGIN):
                    out.append({"arm": arm, "seed": seed, "trial": t["trial"], "status19": t["status"],
                                "k19": k, "margin19": margin, "gain": gain, "formula": t["formula"]})
    return out


def accepted_before(arm: str, seed: int, trial: int, columns, max_depth, grammar) -> tuple[list, object]:
    acc, target = [], None
    for t in run_trials(arm, seed):
        if t["trial"] == trial:
            target = t
            break
        if t["status"] == "accepted":
            acc.append(parse(t["spec"], columns=columns, max_depth=max_depth, **grammar))
    if target is None:
        raise ValueError(f"{arm} seed {seed}: no trial {trial}")
    return acc, target


def run(n: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["arm"], r["seed"], r["trial"]) for r in map(json.loads, f)}
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["search"]["n_placebos"] = n
    s = search_config(cfg)
    events = load_events(os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"), "all")
    ev = make_evaluator(events, cfg)
    for c in select():
        if (c["arm"], c["seed"], c["trial"]) in done:
            continue
        t_start = time.time()
        acc, logged = accepted_before(c["arm"], c["seed"], c["trial"], ev.base, s["max_depth"], s["grammar"])
        spec = parse(logged["spec"], columns=ev.base, max_depth=s["max_depth"], **s["grammar"])
        res = ev.trial(acc, spec)
        gains = list(map(float, res.placebo_gains))
        gain = float(res.gate.mean_loss_gain) if hasattr(res.gate, "mean_loss_gain") else float(res.gate.to_dict()["mean_loss_gain"])
        first19 = np.asarray(gains[:19])
        logged19 = np.asarray(logged["placebo_gains"], dtype=float)
        row = {**c, "n": n, "gain_rescored": gain,
               "first19_match": bool(np.allclose(first19, logged19, rtol=0, atol=1e-12)),
               "first19_max_abs_diff": float(np.max(np.abs(first19 - logged19))),
               "k_n": int(sum(g >= gain for g in gains)), "p_n": placebo_p(gain, gains),
               "accept_n": bool(gain > placebo_threshold(gains)),
               "placebo_gains": gains, "wall_s": round(time.time() - t_start, 1)}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{c['arm']} s{c['seed']} t{c['trial']}: 19 -> {c['status19']} (k={c['k19']}); "
              f"{n} -> {'accept' if row['accept_n'] else 'reject'} (k={row['k_n']}, p={row['p_n']:.3f}); "
              f"first19 match={row['first19_match']}", flush=True)


def summary() -> None:
    with open(os.path.join(OUT, "results.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        print(f"{r['arm']:9s} s{r['seed']} t{r['trial']:3d}  19: {r['status19']:9s} k={r['k19']}  "
              f"{r['n']}: {'accepted' if r['accept_n'] else 'rejected':9s} k={r['k_n']:2d} p={r['p_n']:.3f}  "
              f"match={r['first19_match']}")
    flips = [r for r in rows if (r["status19"] == "accepted") != r["accept_n"]]
    print(f"{len(rows)} candidates, {len(flips)} change verdict; "
          f"all first-19 match: {all(r['first19_match'] for r in rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "run", "summary"])
    ap.add_argument("--n", type=int, default=99)
    a = ap.parse_args()
    if a.cmd == "select":
        for c in select():
            print(c)
    elif a.cmd == "run":
        run(a.n)
    else:
        summary()
