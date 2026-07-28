from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class GroupedSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    group_column: str
    calibration: pd.DataFrame | None = None

    def partitions(self) -> dict[str, pd.DataFrame]:
        parts = {"train": self.train, "validation": self.validation, "test": self.test}
        if self.calibration is not None:
            parts["calibration"] = self.calibration
        return parts

    def assert_disjoint(self) -> None:
        groups = [set(part[self.group_column]) for part in self.partitions().values()]
        if any((left & right for left, right in combinations(groups, 2))):
            raise AssertionError("group leakage detected between partitions")


def grouped_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    group_column: str = "enzyme_id",
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
    calibration_size: float | None = None,
) -> GroupedSplits:
    if group_column not in frame:
        raise ValueError(f"group column {group_column!r} is missing")
    if frame[group_column].isna().any():
        raise ValueError("group values may not be missing")
    if calibration_size is not None:
        groups = frame[group_column].unique()
        if len(groups) < 4:
            raise ValueError("at least four unique groups are required")
        fractions = np.array([validation_size, test_size, calibration_size])
        if not np.isfinite(fractions).all() or (fractions <= 0).any() or fractions.sum() >= 0.8:
            raise ValueError("invalid validation/test/calibration sizes")
        counts = np.maximum(1, np.floor(len(groups) * fractions).astype(int))
        if counts.sum() >= len(groups):
            raise ValueError("split sizes must leave at least one training group")
        shuffled = np.random.default_rng(random_seed).permutation(groups)
        validation, test, calibration, train = np.split(shuffled, np.cumsum(counts))

        def subset(selected: np.ndarray) -> pd.DataFrame:
            return frame.loc[frame[group_column].isin(selected)].reset_index(drop=True)

        splits = GroupedSplits(
            train=subset(train),
            validation=subset(validation),
            test=subset(test),
            calibration=subset(calibration),
            group_column=group_column,
        )
        splits.assert_disjoint()
        return splits
    if frame[group_column].nunique() < 3:
        raise ValueError("at least three unique groups are required")
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 0.8:
        raise ValueError("invalid validation/test sizes")
    first_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_validation_indices, test_indices = next(
        first_split.split(frame, groups=frame[group_column])
    )
    train_validation = frame.iloc[train_validation_indices].copy()
    test = frame.iloc[test_indices].copy()
    relative_validation = validation_size / (1.0 - test_size)
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=relative_validation, random_state=random_seed + 1
    )
    train_indices, validation_indices = next(
        second_split.split(train_validation, groups=train_validation[group_column])
    )
    splits = GroupedSplits(
        train=train_validation.iloc[train_indices].reset_index(drop=True),
        validation=train_validation.iloc[validation_indices].reset_index(drop=True),
        test=test.reset_index(drop=True),
        group_column=group_column,
    )
    splits.assert_disjoint()
    return splits


def study_overlap(splits: GroupedSplits, study_column: str = "study_id") -> dict[str, int]:
    parts = splits.partitions()
    if any(study_column not in part for part in parts.values()):
        return {f"{left}_{right}": 0 for left, right in combinations(parts, 2)}
    studies = {name: set(part[study_column].dropna()) for name, part in parts.items()}
    return {
        f"{left}_{right}": len(studies[left] & studies[right])
        for left, right in combinations(parts, 2)
    }
