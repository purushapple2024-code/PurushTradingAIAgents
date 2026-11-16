"""Purged, embargoed walk-forward splits.

Walk-forward: train on everything before the test window, test on the next window, roll forward.
Purging: drop training samples whose label window (t .. t_end) overlaps the test window.
Embargo: additionally drop `embargo` bars of training data right after the test window (for
the next fold's training set) so serially-correlated features cannot leak.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: int
    test_end: int  # exclusive


def walk_forward_folds(
    n: int,
    t_end: np.ndarray,
    n_folds: int = 5,
    min_train: int = 252,
    embargo: int = 5,
) -> Iterator[Fold]:
    """Yield purged walk-forward folds over positions 0..n-1.

    `t_end[i]` is the position where sample i's label was decided (>= i). Samples with t_end < 0
    (unlabelled) are never used.
    """
    valid = t_end >= 0
    test_len = (n - min_train) // n_folds
    if test_len <= 0:
        raise ValueError(f"not enough rows ({n}) for min_train={min_train} and {n_folds} folds")
    for k in range(n_folds):
        start = min_train + k * test_len
        end = n if k == n_folds - 1 else start + test_len
        test_mask = np.zeros(n, dtype=bool)
        test_mask[start:end] = True
        # training candidates: before the test window
        pos = np.arange(n)
        train_mask = (pos < start) & valid
        # purge: label window must end before the test window starts
        train_mask &= t_end < start
        # embargo applies to data after the test window (used by later folds' training sets):
        # here we also exclude the `embargo` bars immediately before the test start to be safe
        train_mask[max(0, start - embargo) : start] = False
        test_mask &= valid
        yield Fold(np.flatnonzero(train_mask), np.flatnonzero(test_mask), start, end)


def assert_no_leakage(fold: Fold, t_end: np.ndarray) -> None:
    """Raise if any training label reaches into the test window."""
    reach = t_end[fold.train_idx]
    if (reach >= fold.test_start).any():
        raise AssertionError("purge failed: a training label ends inside the test window")
    if fold.train_idx.size and fold.train_idx.max() >= fold.test_start:
        raise AssertionError("training sample inside the test window")
