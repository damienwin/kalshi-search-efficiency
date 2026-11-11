"""Null-feature calibration: how often does the gate accept a feature that cannot help?

    python scripts/run_null_calibration.py run     [--n 1000] [--out runs/null_calibration] [--shard i/k]
    python scripts/run_null_calibration.py summary [--out runs/null_calibration] [--search-n 100]

Probes are columns with no link to the label, scored exactly like a search candidate
(same folds, training seeds, reference model, gate; src/ksearch/search/trial.py):

  noise    i.i.d. N(0, 1)
  shuffle  one of the base columns, permuted across all events inside the same 24h
           calendar block (keeps its marginal and its time-of-day mix, breaks the
           link to the label)

Per probe it logs wall-clock, mean gain, DM p, the legacy +0.005 keep rule, and the
full gate verdict (DM, then the 19-placebo threshold for DM passers, as in a search).
Appends to probes.jsonl and skips finished probe ids, so a killed run resumes.
Dev data only (refuses sealed paths, L11).

summary reports: false-accept rate of the legacy rule and of the gate (with Wilson
CIs), the 95th percentile of null gains (one candidate), and the 95th percentile of
the max of N null gains (a whole search; bootstrap over the null gains), per kind.
The percentiles are a search-internal device, never reported as significance.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cv import guard_not_sealed  # noqa: E402
from run_search import load_events  # noqa: E402

from ksearch.data.manifest import REPO_ROOT  # noqa: E402

BLOCK_S = 24 * 3600


def probe_column(i: int, ev, seed: int) -> tuple[str, str, np.ndarray]:
    """(kind, description, full-length column) for probe i; deterministic in (seed, i)."""
    rng = np.random.default_rng([seed, i])
    n = len(ev.events)
    if i % 2 == 0:
        return "noise", "N(0,1)", rng.standard_normal(n)
    col = ev.base[(i // 2) % len(ev.base)]
    x = ev.X_base[:, ev.base.index(col)].copy()
    blk = ev.events["t0"].to_numpy(dtype=np.int64) // BLOCK_S
    out = x.copy()
    order = np.argsort(blk, kind="mergesort")
    bounds = np.flatnonzero(np.diff(blk[order])) + 1
    for idx in np.split(order, bounds):
        out[idx] = x[rng.permutation(idx)]
    return "shuffle", col, out


def run_probe(ev, i: int, seed: int) -> dict:
    from ksearch.search.gate import loss_gain, placebo_shift
    kind, desc, col = probe_column(i, ev, seed)
    t = time.perf_counter()
    f0 = ev.n_fits
    ref = ev.reference([])
    cols = [col] * len(ev.folds)
    cand = ev.cv([cols])
    gate = ev._gate(ref, cand)
    n_pl, gains = 0, None
    if ev.n_placebos and gate.dm_status == "ok" and gate.dm_p < ev.alpha:
        mk, t0 = ev.events["market_ticker"].to_numpy(), ev.events["t0"].to_numpy()
        gains = []
        for j in range(ev.n_placebos):
            rng = np.random.default_rng([ev.placebo_seed, 7919, i, j])
            pc = placebo_shift(col, mk, t0, rng, ev.placebo_min_shift)
            gains.append(loss_gain(ev.y, ref.proba, ev.cv([[pc] * len(ev.folds)]).proba))
        gate = ev._gate(ref, cand, placebo_gains=gains)
        n_pl = len(gains)
    g = gate.to_dict()
    return {"id": i, "kind": kind, "desc": desc, "gain": g["mean_loss_gain"], "dm_p": g["dm_p"],
            "dm_status": g["dm_status"], "legacy": (g["legacy"] or {}).get("verdict"),
            "legacy_delta": (g["legacy"] or {}).get("delta"), "f1_delta": g["f1_delta"],
            "placebo_threshold": g["placebo_threshold"], "accept": g["accept"], "reason": g["reason"],
            "fits": ev.n_fits - f0, "n_placebos": n_pl, "wall_s": round(time.perf_counter() - t, 2)}


def cmd_run(a) -> int:
    guard_not_sealed(a.data)
    from ksearch.search.loop import guard_path, make_evaluator, make_folds, search_config
    guard_path(a.data)
    out = guard_path(a.out)
    os.makedirs(out, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    events = load_events(a.data, "all")
    folds = make_folds(events, cfg)
    ev = make_evaluator(events, cfg, folds)
    shard, nshard = (int(x) for x in a.shard.split("/"))
    path = os.path.join(out, f"probes_{shard}of{nshard}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path) if l.strip()}
    meta = {"n": a.n, "seed": a.seed, "events": len(events), "folds": len(folds),
            "train_seeds": ev.seeds, "alpha": ev.alpha, "n_placebos": ev.n_placebos,
            "base_columns": ev.base, "search_cfg": search_config(cfg)}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    todo = [i for i in range(a.n) if i % nshard == shard and i not in done]
    print(f"{len(todo)} probes to run ({len(done)} done), shard {shard}/{nshard}", flush=True)
    t_all = time.perf_counter()
    ev.reference([])  # fit the reference once; excluded from per-probe wall
    print(f"reference fitted in {time.perf_counter() - t_all:.1f}s ({ev.n_fits} fits)", flush=True)
    with open(path, "a") as log:
        for k, i in enumerate(todo):
            r = run_probe(ev, i, a.seed)
            log.write(json.dumps(r) + "\n")
            log.flush()
            if k % 10 == 0:
                print(f"  probe {i} ({r['kind']}) gain {r['gain']:+.5f} p {r['dm_p']:.3f} "
                      f"{r['wall_s']}s  [{k + 1}/{len(todo)}]", flush=True)
    return 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cmd_summary(a) -> int:
    rows = []
    for fn in sorted(os.listdir(a.out)):
        if fn.startswith("probes_") and fn.endswith(".jsonl"):
            rows += [json.loads(l) for l in open(os.path.join(a.out, fn)) if l.strip()]
    if not rows:
        print("no probes")
        return 1
    rng = np.random.default_rng(0)
    res = {}
    for kind in ("all", "noise", "shuffle"):
        r = [x for x in rows if kind == "all" or x["kind"] == kind]
        n = len(r)
        g = np.array([x["gain"] for x in r])
        legacy = sum(x["legacy"] == "KEPT" for x in r)
        gain_gt = sum(x["gain"] > 0.005 for x in r)
        dm = sum(x["dm_p"] < 0.05 for x in r)
        acc = sum(bool(x["accept"]) for x in r)
        maxn = [np.max(rng.choice(g, a.search_n, replace=True)) for _ in range(5000)]
        res[kind] = {
            "n": n, "wall_s_mean": float(np.mean([x["wall_s"] for x in r])),
            "legacy_f1_keep_rate": legacy / n, "legacy_ci": wilson(legacy, n),
            "dm_reject_rate": dm / n, "dm_ci": wilson(dm, n),
            "gate_accept_rate": acc / n, "gate_ci": wilson(acc, n),
            "gain_mean": float(g.mean()), "gain_sd": float(g.std(ddof=1)),
            "gain_p95_single": float(np.percentile(g, 95)),
            f"gain_p95_max_of_{a.search_n}": float(np.percentile(maxn, 95)),
            "frac_gain_gt_0.005": gain_gt / n,
            "p_any_gate_accept_in_search": float(1 - (1 - acc / n) ** a.search_n),
        }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    for kind, s in res.items():
        print(f"[{kind}] n={s['n']}  {s['wall_s_mean']:.1f}s/probe")
        print(f"  old rule (macro-F1 delta>0.005 & 4/5 folds): false keep {s['legacy_f1_keep_rate']:.3f} "
              f"CI {s['legacy_ci'][0]:.3f}-{s['legacy_ci'][1]:.3f}")
        print(f"  DM p<0.05: {s['dm_reject_rate']:.3f} CI {s['dm_ci'][0]:.3f}-{s['dm_ci'][1]:.3f}"
              f"   full gate accept: {s['gate_accept_rate']:.3f} CI {s['gate_ci'][0]:.3f}-{s['gate_ci'][1]:.3f}")
        print(f"  P(>=1 gate false accept in {a.search_n} trials) ~ {s['p_any_gate_accept_in_search']:.3f}")
        print(f"  loss gain: mean {s['gain_mean']:+.5f} sd {s['gain_sd']:.5f}  p95 single {s['gain_p95_single']:+.5f}"
              f"  p95 max-of-{a.search_n} {s[f'gain_p95_max_of_{a.search_n}']:+.5f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--n", type=int, default=1000, help="probes (alternating noise / shuffle)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/k: run probes with id %% k == i")
    ap.add_argument("--search-n", type=int, default=100, help="trials per search, for the max-of-N threshold")
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"))
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "settings.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "null_calibration"))
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_summary(a)


if __name__ == "__main__":
    sys.exit(main())
