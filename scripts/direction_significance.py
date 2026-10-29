"""Paired significance tests for the two-stage and flip-penalized models (week 3 standard).

    python scripts/direction_significance.py [--seeds 0 1 2 3 4]

Fits each model on the same walk-forward folds as run_cv.py, averages every event's
out-of-fold probabilities over the training seeds, saves them, then runs the
one-sided Diebold-Mariano test (ksearch.search.gate.dm_test: variance robust to
event-group and same-day dependence) on three per-event losses:

  balanced    the gate's class-balanced log loss
  flip        balanced log loss + lam * P(opposite direction)   (lam = FLIP_LAM)
  direction   log loss of UP vs DOWN on events that moved only

Two families: each variant against the 3-class model with the same features, and
price + sentiment against price within each model. Holm adjusts p within a family.
Development data only; refuses the sealed slice like run_cv.py.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

from ksearch.data.manifest import REPO_ROOT
from ksearch.eval.cv import LABELS, fit_predict_xgb
from ksearch.eval.direction_metrics import direction_loss, flip_penalized_loss
from ksearch.eval.folds import walk_forward_folds
from ksearch.features.price import PRICE_FEATURES, PROVISIONAL_FEATURES
from ksearch.features.sentiment import SENTIMENT_FEATURES
from ksearch.search.gate import loss_differences, dm_test
from run_cv import guard_not_sealed

FLIP_LAM = 1.0
MODELS = {"3class": {"kind": "3class"},
          "two_stage": {"kind": "two_stage"},
          "flippen1": {"kind": "flip_penalty", "lam": 1.0}}
NAMES = {"3class": "3-class (week 3)", "two_stage": "Two-stage",
         "flippen1": "Flip-penalized loss, λ = 1"}
METRICS = {"balanced": "Balanced log loss", "flip": "Balanced log loss + flip penalty (λ = 1)",
           "direction": "Direction log loss, moved events"}


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def load_events(data: str, news_only: bool) -> pd.DataFrame:
    ev = pd.read_parquet(data).merge(
        pd.read_parquet(os.path.join(os.path.dirname(data), "sentiment.parquet")),
        on=["market_ticker", "t0"], how="left", validate="one_to_one")
    if news_only:
        ev = ev[ev["n_articles"] >= 2]
    with open(os.path.join(os.path.dirname(data), "universe.jsonl")) as f:
        scope = {m["ticker"]: m.get("_scope") for m in map(json.loads, f)}
    ev = ev[ev["market_ticker"].map(scope) == "sample"]
    return ev.sort_values(["t0", "market_ticker"], kind="mergesort").reset_index(drop=True)


def oof_proba(events, folds, feats, params, spec, seeds) -> np.ndarray:
    p = np.zeros((len(events), 3))
    for s in seeds:
        for f in folds:
            _, pr = fit_predict_xgb(events.iloc[f.train], events.iloc[f.val], feats,
                                    {**params, "seed": s}, spec)
            p[f.val] += pr
    return p / len(seeds)


def per_event_d(metric, y, p_ref, p_cand, groups, t0):
    """(d, groups, t0) for a metric; d > 0 favours the candidate."""
    if metric == "balanced":
        return loss_differences(y, p_ref, p_cand), groups, t0
    if metric == "flip":
        return flip_penalized_loss(y, p_ref, FLIP_LAM) - flip_penalized_loss(y, p_cand, FLIP_LAM), groups, t0
    lr, moved = direction_loss(y, p_ref)
    lc, _ = direction_loss(y, p_cand)
    return lr - lc, groups[moved], t0[moved]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "direction_significance.json"))
    args = ap.parse_args()
    guard_not_sealed(args.data)
    with open(os.path.join(REPO_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    price = [c for c in PRICE_FEATURES if c not in PROVISIONAL_FEATURES]
    sets = {"price": price, "price_sentiment": price + [c for c in SENTIMENT_FEATURES if c != "event_hour"]}
    specs = MODELS

    out = {"flip_lambda": FLIP_LAM, "seeds": args.seeds, "scopes": {}}
    for scope, news in [("all", False), ("news", True)]:
        ev = load_events(args.data, news)
        folds = walk_forward_folds(ev, cfg["cv"]["n_splits"], cfg["cv"]["embargo_h"])
        scored = np.zeros(len(ev), bool)
        for f in folds:
            scored[f.val] = True
        P = {(m, fs): oof_proba(ev, folds, sets[fs], cfg["classifier"], specs[m], args.seeds)
             for m in MODELS for fs in sets}
        cols = {"market_ticker": ev.market_ticker, "event_group": ev.event_group, "t0": ev.t0, "label": ev.label}
        for (m, fs), p in P.items():
            for i, l in enumerate(LABELS):
                cols[f"p_{m}_{fs}_{l}"] = p[:, i]
        pd.DataFrame(cols)[scored].to_parquet(
            os.path.join(REPO_ROOT, "results", f"direction_proba_{scope}.parquet"), index=False)

        y = ev.label.to_numpy(dtype=object)[scored]
        groups, t0 = ev.event_group.to_numpy()[scored], ev.t0.to_numpy()[scored]

        def run(p_ref, p_cand):
            res = {}
            for metric in METRICS:
                d, g, t = per_event_d(metric, y, p_ref[scored], p_cand[scored], groups, t0)
                dm = dm_test(d, g, t)
                res[metric] = {"mean_gain": dm["mean_gain"], "p": dm["dm_p"], "status": dm["status"],
                               "n": dm["n_events"]}
            return res

        fam = {"vs_3class": {}, "sentiment": {}}
        for fs in sets:
            for m in MODELS:
                if m != "3class":
                    fam["vs_3class"][f"{m}|{fs}"] = run(P[("3class", fs)], P[(m, fs)])
        for m in MODELS:
            fam["sentiment"][m] = run(P[(m, "price")], P[(m, "price_sentiment")])
        for family in fam.values():
            for metric in METRICS:
                keys = list(family)
                adj = holm([family[k][metric]["p"] for k in keys])
                for k, a in zip(keys, adj):
                    family[k][metric]["p_holm"] = a
        out["scopes"][scope] = {"n_scored": int(scored.sum()), **fam}
        print(f"{scope}: {scored.sum()} events scored", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
