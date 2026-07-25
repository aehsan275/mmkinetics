import numpy as np
import pytest

from mm_kinetics.ml.bootstrap import ConformalCalibrator


@pytest.mark.parametrize(
    "n, alpha, expected", [(10, 0.5, 6.0), (9, 0.3, 7.0), (5, 0.5, 3.0), (2, 0.05, 2.0)]
)
def test_conformal_uses_finite_sample_residual_rank(n, alpha, expected):
    truth = np.arange(n, 0, -1, dtype=float)
    calibrator = ConformalCalibrator(alpha=alpha).fit(truth, np.zeros(n))
    lower, upper = calibrator.interval(np.array([2.0]))
    assert calibrator.quantile_ == expected
    np.testing.assert_allclose(lower, [2.0 - expected])
    np.testing.assert_allclose(upper, [2.0 + expected])
