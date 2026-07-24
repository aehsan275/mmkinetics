from __future__ import annotations

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
