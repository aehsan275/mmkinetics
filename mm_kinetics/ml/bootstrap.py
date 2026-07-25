from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone


class BootstrapEnsemble:
    def __init__(self, base_estimator, *, n_bootstrap: int = 100, random_seed: int = 42) -> None:
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
        self.base_estimator = base_estimator
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        self.models_: list[object] = []
        self.training_columns_: tuple[str, ...] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> BootstrapEnsemble:
        if len(X) != len(y) or len(X) < 3:
            raise ValueError("X and y must align and contain at least three rows")
        X_reset = X.reset_index(drop=True)
        y_reset = pd.Series(np.asarray(y, dtype=float))
        if not np.isfinite(y_reset).all():
            raise ValueError("y must contain only finite values")
        rng = np.random.default_rng(self.random_seed)
        indices = (rng.integers(0, len(X_reset), len(X_reset)) for _ in range(self.n_bootstrap))
        self.models_ = [
            clone(self.base_estimator).fit(X_reset.iloc[sample], y_reset.iloc[sample])
            for sample in indices
        ]
        self.training_columns_ = tuple(X_reset.columns)
        return self

    def predict_distribution(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models_ or self.training_columns_ is None:
            raise RuntimeError("BootstrapEnsemble is not fitted")
        if tuple(X.columns) != self.training_columns_:
            raise ValueError("prediction columns do not match training columns/order")
        return np.column_stack(
            [np.asarray(model.predict(X), dtype=float) for model in self.models_]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.median(self.predict_distribution(X), axis=1)


@dataclass
class ConformalCalibrator:
    alpha: float = 0.05
    quantile_: float | None = None

    def fit(
        self, y_true: np.ndarray | pd.Series, point_prediction: np.ndarray
    ) -> ConformalCalibrator:
        truth = np.asarray(y_true, dtype=float)
        prediction = np.asarray(point_prediction, dtype=float)
        if truth.shape != prediction.shape or len(truth) < 2:
            raise ValueError("calibration truth and predictions must align and contain >=2 rows")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        scores = np.abs(truth - prediction)
        probability = min(1.0, np.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores))
        self.quantile_ = float(np.quantile(scores, probability, method="higher"))
        return self

    def interval(self, point_prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.quantile_ is None:
            raise RuntimeError("ConformalCalibrator is not fitted")
        prediction = np.asarray(point_prediction, dtype=float)
        return (prediction - self.quantile_, prediction + self.quantile_)


def interval_metrics(
    y_true: np.ndarray | pd.Series, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if not truth.shape == lower_array.shape == upper_array.shape:
        raise ValueError("interval arrays must align")
    if not (
        np.all(np.isfinite(truth))
        and np.all(np.isfinite(lower_array))
        and np.all(np.isfinite(upper_array))
    ):
        raise ValueError("interval arrays must contain only finite values")
    if np.any(lower_array > upper_array):
        raise ValueError("lower interval endpoint exceeds upper endpoint")
    return {
        "coverage": float(np.mean((truth >= lower_array) & (truth <= upper_array))),
        "mean_width": float(np.mean(upper_array - lower_array)),
        "median_width": float(np.median(upper_array - lower_array)),
    }
