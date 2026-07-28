import numpy as np
import pandas as pd
import pytest

from mm_kinetics.config import validate_config
from mm_kinetics.data.splitting import (
    grouped_train_validation_test_split,
    study_overlap,
)


def test_four_partitions_preserve_rows_and_groups_reproducibly():
    frame = pd.DataFrame({"enzyme_id": np.repeat(np.arange(100), 3), "row": np.arange(300)})
    kwargs = {"validation_size": 0.2, "test_size": 0.2, "calibration_size": 0.2, "random_seed": 18}
    first = grouped_train_validation_test_split(frame, **kwargs)
    second = grouped_train_validation_test_split(frame, **kwargs)
    first.assert_disjoint()
    assert {name: part.enzyme_id.nunique() for name, part in first.partitions().items()} == {
        "train": 40,
        "validation": 20,
        "test": 20,
        "calibration": 20,
    }
    assert sorted(pd.concat(first.partitions().values()).row) == list(range(300))
    for name, part in first.partitions().items():
        pd.testing.assert_frame_equal(part, second.partitions()[name])
    assert len(study_overlap(first)) == 6
    other = grouped_train_validation_test_split(frame, **kwargs | {"random_seed": 19})
    assert set(first.train.enzyme_id) != set(other.train.enzyme_id)


def test_small_four_way_split_keeps_each_partition_nonempty():
    frame = pd.DataFrame({"enzyme_id": list("abcd")})
    splits = grouped_train_validation_test_split(frame, calibration_size=0.2)
    assert all(len(part) == 1 for part in splits.partitions().values())
    with pytest.raises(ValueError, match="four"):
        grouped_train_validation_test_split(frame.iloc[:3], calibration_size=0.2)


@pytest.mark.parametrize("fractions", [(0.3, 0.3, 0.2), (0.2, 0.2, 0), (0.2, 0.2, np.nan)])
def test_invalid_combined_fractions(fractions):
    validation, test, calibration = fractions
    with pytest.raises(ValueError):
        grouped_train_validation_test_split(
            pd.DataFrame({"enzyme_id": range(20)}),
            validation_size=validation,
            test_size=test,
            calibration_size=calibration,
        )
    with pytest.raises(ValueError):
        validate_config(
            {
                "model": {
                    "validation_size": validation,
                    "test_size": test,
                    "calibration_size": calibration,
                }
            }
        )


def test_legacy_split_has_three_partitions():
    splits = grouped_train_validation_test_split(pd.DataFrame({"enzyme_id": range(20)}))
    assert splits.calibration is None
    assert len(splits.partitions()) == 3
    assert len(study_overlap(splits)) == 3
