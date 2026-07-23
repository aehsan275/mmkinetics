from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_columns: list[str] | tuple[str, ...], categorical_columns: list[str] | tuple[str, ...]
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                list(numeric_columns),
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(categorical_columns),
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_model_pipelines(
    numeric_columns: list[str] | tuple[str, ...],
    categorical_columns: list[str] | tuple[str, ...],
    *,
    random_seed: int = 42,
) -> dict[str, Pipeline]:
    estimators: dict[str, RegressorMixin] = {
        "median": DummyRegressor(strategy="median"),
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=random_seed, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=random_seed
        ),
    }
    return {
        name: Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
                ("regressor", estimator),
            ]
        )
        for name, estimator in estimators.items()
    }


def regression_metrics(
    y_true: np.ndarray | pd.Series, y_prediction: np.ndarray
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_prediction, dtype=float)
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)) if len(truth) >= 2 else np.nan,
    }


def fit_and_evaluate_models(
    models: Mapping[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        fitted[name] = model.fit(X_train, y_train)
        predictions = np.asarray(fitted[name].predict(X_validation), dtype=float)
        rows.append({"model": name, **regression_metrics(y_validation, predictions)})
    return (fitted, pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True))
