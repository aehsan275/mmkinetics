from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class GroupedSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    group_column: str

    def assert_disjoint(self) -> None:
        train_groups = set(self.train[self.group_column])
        validation_groups = set(self.validation[self.group_column])
        test_groups = set(self.test[self.group_column])
        if (
            train_groups & validation_groups
            or train_groups & test_groups
            or validation_groups & test_groups
        ):
            raise AssertionError("group leakage detected between train/validation/test")


def grouped_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    group_column: str = "enzyme_id",
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> GroupedSplits:
    if group_column not in frame:
        raise ValueError(f"group column {group_column!r} is missing")
    if frame[group_column].isna().any():
        raise ValueError("group values may not be missing")
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
    if any(study_column not in part for part in [splits.train, splits.validation, splits.test]):
        return {"train_validation": 0, "train_test": 0, "validation_test": 0}
    train = set(splits.train[study_column].dropna())
    validation = set(splits.validation[study_column].dropna())
    test = set(splits.test[study_column].dropna())
    return {
        "train_validation": len(train & validation),
        "train_test": len(train & test),
        "validation_test": len(validation & test),
    }
