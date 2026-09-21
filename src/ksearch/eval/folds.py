"""Walk-forward CV folds grouped by event, with purge and embargo (L9, L10).

Event groups (a strike ladder such as INX-24FEB23-*) are ordered by their
first t0 and cut into n_splits + 1 contiguous blocks. Fold k validates on
block k and trains on earlier blocks, keeping only training events whose
label window closed at least `embargo` before the first validation t0.
A group therefore never sits on both sides of a split, and no training label
overlaps or abuts the validation period.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOUR = 3600


@dataclass
class Fold:
    k: int
    train: np.ndarray  # positional indices into the events frame
    val: np.ndarray


def group_blocks(events: pd.DataFrame, n_blocks: int) -> pd.Series:
    """Block number per event, from its group's rank by first t0."""
    first = events.groupby("event_group")["t0"].min()
    order = first.sort_values(kind="mergesort").index
    block_of_group = pd.Series(np.arange(len(order)) * n_blocks // len(order), index=order)
    return events["event_group"].map(block_of_group).to_numpy()


def walk_forward_folds(events: pd.DataFrame, n_splits: int, embargo_h: float) -> list[Fold]:
    blocks = group_blocks(events, n_splits + 1)
    t0 = events["t0"].to_numpy()
    t_end = events["t_end"].to_numpy()
    folds = []
    for k in range(1, n_splits + 1):
        val = np.flatnonzero(blocks == k)
        cutoff = t0[val].min() - embargo_h * HOUR
        train = np.flatnonzero((blocks < k) & (t_end <= cutoff))
        if len(train) and len(val):
            folds.append(Fold(k, train, val))
    return folds


class FoldScoped:
    """Wrap a fitted transform so it can only be fit on the fold's training rows (L8)."""

    def __init__(self, transform, train_index):
        self.transform_ = transform
        self.allowed = set(np.asarray(train_index).tolist())

    def fit(self, X: pd.DataFrame, *args, **kwargs):
        outside = set(X.index.tolist()) - self.allowed
        if outside:
            raise ValueError(f"FoldScoped: fit saw {len(outside)} rows outside the training fold")
        self.transform_.fit(X, *args, **kwargs)
        return self

    def transform(self, X):
        return self.transform_.transform(X)


def seal_split(events: pd.DataFrame, sealed_fraction: float, embargo_h: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split off the sealed test slice by time, purging dev events that run into it.

    T is the t0 quantile at 1 - sealed_fraction. Groups whose first event is at or
    after T are sealed. Every other group stays in dev, minus events whose label
    window ends later than T - embargo: those run concurrently with the sealed
    period and are dropped (returned as the purge count).
    """
    T = int(np.quantile(events["t0"], 1 - sealed_fraction))
    first = events.groupby("event_group")["t0"].transform("min")
    sealed_mask = first >= T
    dev_mask = ~sealed_mask & (events["t_end"] <= T - embargo_h * HOUR)
    purged = int((~sealed_mask & ~dev_mask).sum())
    dev = events[dev_mask].reset_index(drop=True)
    sealed = events[sealed_mask].reset_index(drop=True)
    return dev, sealed, purged
