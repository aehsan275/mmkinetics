import json

import joblib
import numpy as np
import pytest

from mm_kinetics.ml.bootstrap import ConformalCalibrator, interval_metrics
from mm_kinetics.pipeline import _json_safe


def test_group_scores_use_largest_residual_once_per_enzyme():
    truth = np.array([1.0, 4.0, 2.0, 8.0, 3.0, 6.0])
    groups = np.repeat(["a", "b", "c"], 2)
    grouped = ConformalCalibrator(alpha=0.5).fit(truth, np.zeros(6), groups=groups)
    assay = ConformalCalibrator(alpha=0.5).fit(truth, np.zeros(6))
    assert grouped.quantile_ == 6
    assert assay.quantile_ == 4
    repeated = ConformalCalibrator(alpha=0.5).fit(
        np.tile(truth, 2), np.zeros(12), groups=np.tile(groups, 2)
    )
    assert repeated.calibration_count_ == grouped.calibration_count_ == 3
    assert repeated.quantile_ == grouped.quantile_


@pytest.mark.parametrize("count", [1, 2, 10, 18, 19, 20])
def test_95_percent_finite_sample_threshold(count, tmp_path):
    truth = np.arange(1, count + 1, dtype=float)
    calibrator = ConformalCalibrator().fit(truth, np.zeros(count), groups=np.arange(count))
    lower, upper = calibrator.interval(np.array([0.0]))
    assert calibrator.calibration_count_ == count
    if count >= 19:
        assert calibrator.quantile_ == count
        assert np.isfinite(lower).all() and np.isfinite(upper).all()
    else:
        assert np.isneginf(lower).all() and np.isposinf(upper).all()
    path = tmp_path / "calibrator.joblib"
    joblib.dump(calibrator, path)
    assert joblib.load(path).quantile_ == calibrator.quantile_


@pytest.mark.parametrize("groups", [[1], [[1, 2]], [1, None], [1, np.nan]])
def test_calibrator_rejects_missing_or_misaligned_groups(groups):
    with pytest.raises(ValueError, match="groups"):
        ConformalCalibrator().fit(np.ones(2), np.zeros(2), groups=groups)


@pytest.mark.parametrize("alpha", [0, 1, np.nan])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=alpha).fit(np.ones(2), np.zeros(2))


def test_metrics_distinguish_assay_and_whole_enzyme_coverage():
    result = interval_metrics(
        [0, 2, 0, 0], [-1, -1, -np.inf, -1], [1, 1, np.inf, 1], groups=["a", "a", "b", "b"]
    )
    assert result["coverage"] == 0.75
    assert result["group_coverage"] == 0.5
    assert result["unbounded_fraction"] == 0.25
    assert result["mean_width"] == np.inf


@pytest.mark.parametrize(
    "lower,upper", [(np.inf, np.inf), (-np.inf, -np.inf), (np.nan, 1), (0, np.nan), (2, 1)]
)
def test_metrics_reject_invalid_bounds(lower, upper):
    with pytest.raises(ValueError):
        interval_metrics([0], [lower], [upper])


def test_one_sided_infinite_bounds_are_valid():
    result = interval_metrics([0, 1], [-np.inf, 0], [1, np.inf])
    assert result["coverage"] == 1
    assert result["unbounded_fraction"] == 1


def test_json_replaces_nonfinite_values_recursively():
    payload = {"values": [np.inf, float("nan"), np.float64(-np.inf)], "nested": {"n": np.int64(3)}}
    encoded = json.dumps(_json_safe(payload), allow_nan=False)
    assert json.loads(encoded) == {"values": [None, None, None], "nested": {"n": 3}}
