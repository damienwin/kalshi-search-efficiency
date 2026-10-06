"""One trial: evaluate a candidate FeatureSpec against the current reference.

Fairness between arms rests on this module being the only place a candidate
is scored. Every arm's candidate goes through the same folds, the same
training seeds, the same reference predictions and the same gate, and the
outcome is a pure function of (accepted specs, candidate): the trial index,
the arm and the run seed never enter (placebo rngs are seeded from the
candidate's key, not the trial).

Leakage: base columns and non-fitted operators are causal per market, so
they are evaluated once on the whole frame. A spec with a fold-fitted node is
evaluated once per fold with fit_index = fold.train, and that fold's
training and validation rows are both taken from that evaluation (L8).
Labels are read only by fit_xgb on training rows (and by series_target on
training rows whose window closed), then by the gate on scored rows.

Cost per trial with S seeds and F folds: S*F fits for the candidate; the
reference is fitted once per accepted set and cached (after an acceptance
the candidate's own predictions become the new reference, so no refit);
placebos add n_placebos*S*F fits, only for candidates that pass DM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ksearch.eval.cv import LABELS, fit_xgb, macro_f1, predict_proba
from ksearch.eval.folds import Fold
from ksearch.search import FeatureSpec, check_columns, degenerate_reason, evaluate
from ksearch.search.gate import (GateResult, balanced_log_loss, dm_test, evaluate_candidate,
                                 loss_differences, loss_gain, placebo_shift)

REQUIRED = ["market_ticker", "event_group", "series", "t0", "t_end", "label"]
LAB = np.array(LABELS, dtype=object)


class DegenerateFeature(Exception):
    """The candidate column cannot help on some fold's training rows: a failed trial."""


@dataclass
class CVResult:
    """Seed-averaged out-of-fold probabilities of one feature set, on the scored rows."""
    proba: np.ndarray              # (n_scored, 3), LABELS order
    fold_f1: list[float]           # per-fold macro F1 of the averaged probabilities
    pooled_f1: float
    balanced_logloss: float        # mean class-balanced log loss (the gate's loss)
    logloss: float                 # plain multiclass log loss

    def summary(self) -> dict:
        return {"pooled_macro_f1": self.pooled_f1, "fold_macro_f1": self.fold_f1,
                "balanced_logloss": self.balanced_logloss, "logloss": self.logloss}


@dataclass
class TrialEval:
    gate: GateResult
    cand: CVResult
    ref: CVResult
    placebo_gains: list[float] | None = None
    fits: int = 0
    columns: list[np.ndarray] = field(default_factory=list, repr=False)  # candidate values per fold


class Evaluator:
    """Owns folds, seeds, the classifier and both caches for one run.

    Columns are per fold: a list with one full-length array per fold (the same
    array object repeated for a spec without fitted nodes).
    """

    def __init__(self, events: pd.DataFrame, folds: list[Fold], base_columns: list[str],
                 clf_params: dict, train_seeds: list[int], *, alpha: float = 0.05,
                 n_placebos: int = 19, placebo_min_shift: int = 1, placebo_seed: int = 0,
                 min_valid: int = 100, gate: dict | None = None, grammar: dict | None = None):
        check_columns(base_columns)  # the reference model reads them too: nothing known only after t0
        missing = [c for c in REQUIRED + list(base_columns) if c not in events.columns]
        if missing:
            raise ValueError(f"events lacks columns {missing}")
        if not folds:
            raise ValueError("no folds")
        if 0 < n_placebos < 19:
            raise ValueError("n_placebos must be 0 (off) or >= 19: fewer can never certify the "
                             "95% placebo threshold, so every candidate would be rejected")
        if not train_seeds or len(set(train_seeds)) != len(train_seeds):
            raise ValueError("train_seeds must be non-empty and distinct")
        self.events = events.reset_index(drop=True)
        self.folds, self.base = folds, list(base_columns)
        self.clf, self.seeds = dict(clf_params), [int(s) for s in train_seeds]
        self.alpha, self.n_placebos = alpha, int(n_placebos)
        self.placebo_min_shift, self.placebo_seed = int(placebo_min_shift), int(placebo_seed)
        self.min_valid, self.gate_kw = min_valid, dict(gate or {})
        self.grammar = dict(grammar or {})  # the same kwargs the loop's parse and the arm get
        n = len(self.events)
        self.fold_of = np.full(n, -1)
        for i, f in enumerate(folds):
            if (self.fold_of[f.val] >= 0).any():
                raise ValueError("validation folds overlap")
            self.fold_of[f.val] = i
        self.scored = np.flatnonzero(self.fold_of >= 0)  # rows in some validation fold, in frame order
        ev = self.events
        self.y = ev["label"].to_numpy(dtype=object)[self.scored]
        self.groups = ev["event_group"].to_numpy(dtype=object)[self.scored]
        self.t0 = ev["t0"].to_numpy(dtype=np.int64)[self.scored]
        self.label = ev["label"].to_numpy(dtype=object)
        self.X_base = ev[self.base].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        self.n_fits = 0
        self._ref: dict[tuple[str, ...], CVResult] = {}
        self._cols: dict[str, list[np.ndarray]] = {}

    # ── columns ──────────────────────────────────────────────────────────────
    def columns(self, spec: FeatureSpec) -> list[np.ndarray]:
        """Candidate values per fold; raises DegenerateFeature if any fold's training rows are unusable."""
        if spec.needs_fit:
            cols = [evaluate(spec, self.events, fit_index=f.train, columns=self.base, **self.grammar)
                    for f in self.folds]
        else:
            cols = [evaluate(spec, self.events, columns=self.base, **self.grammar)] * len(self.folds)
        for f, v in zip(self.folds, cols):
            why = degenerate_reason(v, fit_index=f.train, min_valid=self.min_valid)
            if why:
                raise DegenerateFeature(f"fold {f.k}: {why}")
        return cols

    def accepted_columns(self, spec: FeatureSpec) -> list[np.ndarray]:
        if spec.key not in self._cols:
            self._cols[spec.key] = self.columns(spec)
        return self._cols[spec.key]

    # ── CV ───────────────────────────────────────────────────────────────────
    def cv(self, extra: list[list[np.ndarray]]) -> CVResult:
        """Base + extra columns through every fold and seed; probabilities averaged over seeds."""
        names = self.base + [f"x{j}" for j in range(len(extra))]
        n = len(self.events)
        proba = np.zeros((n, 3))
        for i, f in enumerate(self.folds):
            X = np.column_stack([self.X_base] + [c[i] for c in extra]) if extra else self.X_base
            tr = pd.DataFrame(X[f.train], columns=names)
            tr["label"] = self.label[f.train]
            va = pd.DataFrame(X[f.val], columns=names)
            for s in self.seeds:
                clf = fit_xgb(tr, names, {**self.clf, "seed": s})
                self.n_fits += 1
                proba[f.val] += predict_proba(clf, va, names)
        proba = proba[self.scored] / len(self.seeds)
        proba /= proba.sum(1, keepdims=True)
        pred = LAB[proba.argmax(1)]
        fo = self.fold_of[self.scored]
        fold_f1 = [macro_f1(self.y[fo == i], pred[fo == i]) for i in range(len(self.folds))]
        return CVResult(proba, fold_f1, macro_f1(self.y, pred),
                        float(balanced_log_loss(self.y, proba).mean()),
                        float(log_loss(self.y, proba, labels=LABELS)))

    @staticmethod
    def ref_key(accepted: list[FeatureSpec]) -> tuple[str, ...]:
        return tuple(s.key for s in accepted)

    def reference(self, accepted: list[FeatureSpec]) -> CVResult:
        """Cached per accepted set: fitted once, reused by every trial against it."""
        key = self.ref_key(accepted)
        if key not in self._ref:
            self._ref[key] = self.cv([self.accepted_columns(s) for s in accepted])
        return self._ref[key]

    def promote(self, accepted: list[FeatureSpec], spec: FeatureSpec, ev: TrialEval) -> None:
        """After an acceptance the candidate model IS the new reference (same columns in the
        same order, same seeds), so its predictions are cached instead of refitted."""
        self._cols[spec.key] = ev.columns
        self._ref[self.ref_key(accepted + [spec])] = ev.cand

    # ── one trial ────────────────────────────────────────────────────────────
    def _gate(self, ref: CVResult, cand: CVResult, **kw) -> GateResult:
        return evaluate_candidate(self.y, ref.proba, cand.proba, self.groups, self.t0,
                                  alpha=self.alpha, cand_fold_f1=cand.fold_f1,
                                  ref_fold_f1=ref.fold_f1, **self.gate_kw, **kw)

    def placebo_rng(self, spec: FeatureSpec, j: int) -> np.random.Generator:
        return np.random.default_rng([self.placebo_seed, int(spec.key, 16), j])

    def trial(self, accepted: list[FeatureSpec], spec: FeatureSpec) -> TrialEval:
        """Raises DegenerateFeature (a failed trial); anything else is a bug or a data problem."""
        fits0 = self.n_fits
        if spec.key in self.ref_key(accepted):
            raise ValueError(f"{spec.formula()} is already in the reference")
        ref = self.reference(accepted)
        acc_cols = [self.accepted_columns(s) for s in accepted]
        cols = self.columns(spec)
        cand = self.cv(acc_cols + [cols])
        gate = self._gate(ref, cand)
        gains = None
        if self.n_placebos and gate.dm_status == "ok" and gate.dm_p < self.alpha:
            # Placebos only for DM passers (the gate author's recommendation): the same
            # per-market rotation is applied to every fold's column, so a fold-fitted
            # candidate's placebo is its own fold-fitted values, moved in time.
            mk = self.events["market_ticker"].to_numpy()
            t0 = self.events["t0"].to_numpy()
            gains = []
            for j in range(self.n_placebos):
                shifted: dict[int, np.ndarray] = {}
                pcols = []
                for c in cols:
                    if id(c) not in shifted:
                        shifted[id(c)] = placebo_shift(c, mk, t0, self.placebo_rng(spec, j),
                                                       self.placebo_min_shift)
                    pcols.append(shifted[id(c)])
                gains.append(loss_gain(self.y, ref.proba, self.cv(acc_cols + [pcols]).proba))
            gate = self._gate(ref, cand, placebo_gains=gains)
        return TrialEval(gate, cand, ref, gains, self.n_fits - fits0, cols)

    def oof_frame(self, res: CVResult) -> pd.DataFrame:
        """Per-event seed-averaged OOF probabilities, for analysis without refitting."""
        ev = self.events.iloc[self.scored]
        out = pd.DataFrame({"row": self.scored, "market_ticker": ev["market_ticker"].to_numpy(),
                            "event_group": ev["event_group"].to_numpy(), "series": ev["series"].to_numpy(),
                            "t0": self.t0, "label": self.y,
                            "fold": [self.folds[i].k for i in self.fold_of[self.scored]]})
        for i, lab in enumerate(LABELS):
            out[f"p_{lab}"] = res.proba[:, i]
        return out


# ── knowledge-cutoff split (Workstream C4) ───────────────────────────────────
def _to_ts(cutoff) -> int:
    if isinstance(cutoff, (int, np.integer)):
        return int(cutoff)
    ts = pd.Timestamp(cutoff)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp())


def cutoff_split(y, proba_ref, proba_cand, t0, cutoff, event_group=None, **dm_kw) -> dict:
    """Gain of a candidate over the reference before (t0 < cutoff) and after (t0 >= cutoff)
    a proposer's knowledge cutoff. A gain that appears only before is what memorisation
    looks like. Class-balanced weights are computed within each side, so each side's gain
    is that side's own balanced loss gap. With event_group, each side also gets the gate's
    DM test (its status says when a side has too few blocks or groups). cutoff: unix
    seconds or anything pandas parses as a timestamp (naive = UTC)."""
    y = np.asarray(y, dtype=object)
    proba_ref, proba_cand = np.asarray(proba_ref, dtype=float), np.asarray(proba_cand, dtype=float)
    t0 = np.asarray(t0, dtype=np.int64)
    c = _to_ts(cutoff)
    out = {"cutoff": c}
    for side, m in (("before", t0 < c), ("after", t0 >= c)):
        n = int(m.sum())
        r = {"n": n, "gain": None, "dm": None}
        if n:
            d = loss_differences(y[m], proba_ref[m], proba_cand[m])
            r["gain"] = float(d.mean())
            if event_group is not None:
                dm = dm_test(d, np.asarray(event_group, dtype=object)[m], t0[m], **dm_kw)
                r["dm"] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                           for k, v in dm.items()}
        out[side] = r
    return out


__all__ = ["CVResult", "DegenerateFeature", "Evaluator", "TrialEval", "cutoff_split"]
