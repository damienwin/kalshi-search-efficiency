"""One feature-search run: an arm, a seed, a fixed trial budget, on the dev set.

    python scripts/run_search.py --arm random --seed 0 [--budget 100] [--scope all]
                                 [--data data/build/v2/dev.parquet] [--out runs/random/seed0]
                                 [--force] [--dry-run]

Writes runs/<arm>/seed<k>/ (runs/<arm>_<scope>/seed<k>/ for a sub-scope): run.json,
trials.jsonl, oof/*.parquet, and transcript.jsonl for an LLM arm. Refuses sealed inputs (L11) and an existing run dir
unless --force. --dry-run loads the data, builds the folds, instantiates the arm,
draws and validates a few proposals against a stub outcome (never for an LLM arm
that cannot promise not to call its API), prints the plan, and writes and fits nothing.
"""

import argparse
import importlib
import inspect
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cv import guard_not_sealed  # noqa: E402

from ksearch.data.manifest import REPO_ROOT  # noqa: E402

# arm name -> module; imported only when asked for, so this script runs before every arm exists
ARM_MODULES = {"random": "ksearch.search.arms.random_", "tpe": "ksearch.search.arms.tpe",
               "llm_full": "ksearch.search.arms.llm", "llm_no_history": "ksearch.search.arms.llm",
               "llm_blind": "ksearch.search.arms.llm"}
N_DRY_PROPOSALS = 3


def _named_param(fn, name: str) -> bool:
    """True only for an explicit parameter: arms forward **kwargs to the grammar, so a
    VAR_KEYWORD would swallow (and break on) dry_run or settings_path."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def make_arm(name: str, seed: int, search_cfg: dict, config_path: str | None = None, dry_run: bool = False):
    """Instantiate an arm by name, with the loop's grammar kwargs (search.grammar) so arm and loop
    parse identically. Lookup in the arm's module: ARMS[name], else LLMArm(mode=...) for llm_*,
    else the Arm subclass whose class attribute .name is `name`. LLM settings (model, effort, ...)
    are read by the arm itself from the same config file (search.llm). Returns
    (arm, can_propose_offline): an LLM arm can only propose offline if it takes dry_run."""
    from ksearch.search.arms.base import Arm
    modname = ARM_MODULES[name]
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name and modname.startswith(e.name):
            raise SystemExit(f"arm {name!r} is not available yet: module {modname} does not exist") from None
        raise
    kw = dict(search_cfg.get("grammar") or {})
    factory = (getattr(mod, "ARMS", None) or {}).get(name)
    if factory is None and name.startswith("llm_") and hasattr(mod, "LLMArm"):
        factory = mod.LLMArm
        kw["mode"] = name[len("llm_"):]
    if factory is None:
        found = [c for c in vars(mod).values()
                 if inspect.isclass(c) and issubclass(c, Arm) and c is not Arm and c.__module__ == mod.__name__]
        named = [c for c in found if getattr(c, "name", None) == name]
        if len(named) != 1:
            raise SystemExit(f"cannot find arm {name!r} in {modname}: no ARMS entry and the Arm "
                             f"subclasses there are named {[getattr(c, 'name', None) for c in found]}")
        factory = named[0]
    if config_path and _named_param(factory, "settings_path"):
        kw["settings_path"] = config_path
    offline = not name.startswith("llm")
    if dry_run and _named_param(factory, "dry_run"):
        kw["dry_run"], offline = True, True
    return factory(seed, **kw), offline


def load_events(path: str, scope: str) -> pd.DataFrame:
    """Dev events in the order run_cv.py uses; scope as in run_cv.py --scope."""
    guard_not_sealed(path)
    events = pd.read_parquet(path)
    if scope != "all":
        with open(os.path.join(os.path.dirname(path), "universe.jsonl")) as f:
            sc = {m["ticker"]: m.get("_scope") for m in map(json.loads, f)}
        events = events[events["market_ticker"].map(sc) == ("spring" if scope == "spring" else "sample")]
    return events.sort_values(["t0", "market_ticker"], kind="mergesort").reset_index(drop=True)


def dry_run(arm, offline: bool, events, folds, cfg, budget) -> dict:
    from ksearch.search import parse
    from ksearch.search.arms.base import SearchState, TrialOutcome
    from ksearch.search.loop import plan, search_config
    s = search_config(cfg)
    p = plan(events, folds, cfg, budget)
    p["proposals"] = []
    if not offline:
        p["proposals_skipped"] = "LLM arm without a dry_run switch: not called, so no API request is made"
        return p
    history = []
    for i in range(min(N_DRY_PROPOSALS, budget)):
        state = SearchState(i, budget, (), tuple(s["base_columns"]), s["max_depth"], tuple(history))
        prop = arm.propose(state)
        try:
            spec = parse(prop.spec_json, columns=s["base_columns"], max_depth=s["max_depth"], **s["grammar"])
            row = {"ok": True, "formula": spec.formula(), "key": spec.key, "needs_fit": spec.needs_fit}
        except (ValueError, TypeError) as e:
            spec, row = None, {"ok": False, "error": str(e)}
        p["proposals"].append(row)
        out = TrialOutcome(i, prop.spec_json, spec.formula() if spec else None,
                           "rejected" if spec else "failed", "dry-run stub outcome", 0.0, 1.0, 0.0)
        arm.observe(out)
        history.append(out)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=list(ARM_MODULES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=None, help="default: search.budget in settings.yaml")
    ap.add_argument("--scope", choices=["all", "spring", "current"], default="all")
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "build", "v2", "dev.parquet"))
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "settings.yaml"))
    ap.add_argument("--out", default=None, help="run dir (default runs/<arm>[_<scope>]/seed<k>)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run dir")
    ap.add_argument("--dry-run", action="store_true", help="plan only: no fitting, no API calls, no writes")
    args = ap.parse_args(argv)
    guard_not_sealed(args.data)

    from ksearch.search.loop import guard_path, make_folds, run_search, search_config
    guard_path(args.data)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s = search_config(cfg)
    budget = args.budget if args.budget is not None else int(s["budget"])
    out = args.out or os.path.join(REPO_ROOT, "runs", args.arm + ("" if args.scope == "all" else f"_{args.scope}"),
                                   f"seed{args.seed}")
    guard_path(out)
    events = load_events(args.data, args.scope)
    folds = make_folds(events, cfg)
    arm, offline = make_arm(args.arm, args.seed, s, args.config, dry_run=args.dry_run)

    if args.dry_run:
        p = dry_run(arm, offline, events, folds, cfg, budget)
        print(f"DRY RUN arm={args.arm} seed={args.seed} scope={args.scope} data={args.data}")
        print(f"  would write {out}{'  (EXISTS: needs --force)' if os.path.exists(out) else ''}")
        print(f"  events {p['n_events']}  markets {p['n_markets']}  event groups {p['n_event_groups']}"
              f"  scored (out-of-fold) {p['n_scored']}")
        for f in p["folds"]:
            print(f"  fold {f['k']}: train {f['n_train']:>6}  val {f['n_val']:>6}")
        print(f"  budget {p['budget']} trials  training seeds {p['train_seeds']}  placebos/DM pass {p['n_placebos']}")
        print(f"  XGBoost fits: {p['fits_per_model']} per model; {p['fits_min']} if no candidate passes DM, "
              f"+{p['fits_per_dm_pass_extra']} per DM pass, at most {p['fits_max']}")
        if "proposals_skipped" in p:
            print(f"  proposals: skipped ({p['proposals_skipped']})")
        for i, r in enumerate(p["proposals"]):
            print(f"  proposal {i}: " + (f"OK  {r['formula']}  key={r['key']}" + ("  [fold-fitted]" if r["needs_fit"] else "")
                                         if r["ok"] else f"INVALID  {r['error']}"))
        return 0

    run = run_search(arm, events, cfg, budget, args.seed, out, data_path=args.data, folds=folds,
                     force=args.force, extra={"scope": args.scope})
    t = run["totals"]
    print(f"{args.arm} seed {args.seed}: {run['trials_completed']} trials  accepted {t['accepted']}  "
          f"rejected {t['rejected']}  failed {t['failed']}  duplicate {t['duplicate']}  -> {out}")
    for a in run["accepted"]:
        print(f"  trial {a['trial']:>3}  gain {a['gain']:+.5f}  p {a['dm_p']:.4f}  {a['formula']}")
    fr = run["final_reference"]
    print(f"  final reference: macro F1 {fr['pooled_macro_f1']:.4f}  balanced log loss {fr['balanced_logloss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
