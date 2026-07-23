from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_COLUMNS = {
    "km_molar",
    "kcat_per_s",
    "log10_km",
    "log10_kcat",
    "log10_efficiency",
    "true_km_molar",
    "true_kcat_per_s",
    "true_vmax_molar_per_s",
}
MEASUREMENT_COLUMNS = {
    "substrate_concentration",
    "substrate_molar",
    "observed_velocity",
    "velocity_molar_per_s",
    "measurement_sigma",
    "true_velocity_molar_per_s",
}
IDENTIFIER_COLUMNS = {
    "assay_id",
    "experiment_id",
    "enzyme_id",
    "substrate_id",
    "study_id",
    "source_row",
    "source_file",
    "source_query",
    "retrieval_date",
    "provenance",
}


@dataclass(frozen=True)
class FeatureSet:
    X: pd.DataFrame
    y: pd.Series
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    target: str


def build_feature_set(frame: pd.DataFrame, target: str) -> FeatureSet:
    if target not in {"log10_km", "log10_kcat", "log10_efficiency"}:
        raise ValueError("target must be a supported log-transformed kinetic parameter")
    if target not in frame:
        raise ValueError(f"target column {target!r} is missing")
    usable = frame[np.isfinite(pd.to_numeric(frame[target], errors="coerce"))].copy()
    if usable.empty:
        raise ValueError(f"no finite values are available for {target}")
    blocked = (
        TARGET_COLUMNS
        | MEASUREMENT_COLUMNS
        | IDENTIFIER_COLUMNS
        | {"noise_model", "source_version", "kinetic_law", "exclusion_reason"}
    )
    feature_columns = [
        column
        for column in usable.columns
        if column not in blocked and usable[column].nunique(dropna=True) > 1
    ]
    numeric = tuple(
        column for column in feature_columns if pd.api.types.is_numeric_dtype(usable[column])
    )
    categorical = tuple(column for column in feature_columns if column not in numeric)
    if not numeric and (not categorical):
        raise ValueError("no leakage-safe features remain")
    return FeatureSet(
        X=usable.loc[:, [*numeric, *categorical]].reset_index(drop=True),
        y=usable[target].astype(float).reset_index(drop=True),
        numeric_columns=numeric,
        categorical_columns=categorical,
        target=target,
    )


def apply_feature_schema(frame: pd.DataFrame, schema: FeatureSet) -> FeatureSet:
    target = schema.target
    if target not in frame:
        raise ValueError(f"target column {target!r} is missing")
    target_values = pd.to_numeric(frame[target], errors="coerce")
    usable = frame[np.isfinite(target_values)].copy()
    if usable.empty:
        raise ValueError(f"no finite values are available for {target}")
    columns = [*schema.numeric_columns, *schema.categorical_columns]
    return FeatureSet(
        X=usable.reindex(columns=columns).reset_index(drop=True),
        y=usable[target].astype(float).reset_index(drop=True),
        numeric_columns=schema.numeric_columns,
        categorical_columns=schema.categorical_columns,
        target=target,
    )
