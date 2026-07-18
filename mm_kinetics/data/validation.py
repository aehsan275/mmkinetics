from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pint

UNIT_REGISTRY = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    n_rows: int
    n_duplicate_rows: int


def _convert(value: object, unit: object, target: str, field: str) -> float:
    if pd.isna(value) or pd.isna(unit) or str(unit).strip() == "":
        raise ValueError(f"{field} value and unit are required")
    try:
        return float((float(value) * UNIT_REGISTRY(str(unit))).to(target).magnitude)
    except (ValueError, pint.errors.PintError) as error:
        raise ValueError(f"cannot convert {field}={value} {unit!s} to {target}: {error}") from error


def validate_and_normalize_observations(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, ValidationReport]:
    required = {
        "assay_id",
        "enzyme_id",
        "substrate_id",
        "substrate_concentration",
        "substrate_unit",
        "observed_velocity",
        "velocity_unit",
        "enzyme_concentration",
        "enzyme_concentration_unit",
        "temperature",
        "temperature_unit",
        "pH",
        "provenance",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        report = ValidationReport(False, (f"missing columns: {missing}",), (), len(observations), 0)
        return (observations.copy(), report)
    frame = observations.copy()
    errors: list[str] = []
    warnings: list[str] = []
    duplicate_columns = [
        "assay_id",
        "substrate_concentration",
        "substrate_unit",
        "observed_velocity",
        "velocity_unit",
    ]
    duplicate_mask = frame.duplicated(duplicate_columns, keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} rows participate in exact measurement duplicates")
    substrate_values: list[float] = []
    velocity_values: list[float] = []
    enzyme_values: list[float] = []
    temperature_values: list[float] = []
    for index, row in frame.iterrows():
        try:
            substrate_values.append(
                _convert(row.substrate_concentration, row.substrate_unit, "mole/liter", "substrate")
            )
            velocity_values.append(
                _convert(row.observed_velocity, row.velocity_unit, "mole/liter/second", "velocity")
            )
            enzyme_values.append(
                _convert(
                    row.enzyme_concentration,
                    row.enzyme_concentration_unit,
                    "mole/liter",
                    "enzyme concentration",
                )
            )
            temperature_values.append(
                _convert(row.temperature, row.temperature_unit, "kelvin", "temperature")
            )
        except ValueError as error:
            errors.append(f"row {index}: {error}")
            substrate_values.append(np.nan)
            velocity_values.append(np.nan)
            enzyme_values.append(np.nan)
            temperature_values.append(np.nan)
    frame["substrate_original_value"] = frame["substrate_concentration"]
    frame["substrate_original_unit"] = frame["substrate_unit"]
    frame["velocity_original_value"] = frame["observed_velocity"]
    frame["velocity_original_unit"] = frame["velocity_unit"]
    frame["enzyme_concentration_original_value"] = frame["enzyme_concentration"]
    frame["enzyme_concentration_original_unit"] = frame["enzyme_concentration_unit"]
    frame["temperature_original_value"] = frame["temperature"]
    frame["temperature_original_unit"] = frame["temperature_unit"]
    frame["substrate_molar"] = substrate_values
    frame["velocity_molar_per_s"] = velocity_values
    frame["enzyme_total_molar"] = enzyme_values
    frame["temperature_kelvin"] = temperature_values
    for index, row in frame.iterrows():
        for identifier in ("assay_id", "enzyme_id", "substrate_id"):
            if pd.isna(row[identifier]) or not str(row[identifier]).strip():
                errors.append(f"row {index}: {identifier} is missing")
        if np.isfinite(row.substrate_molar) and row.substrate_molar < 0:
            errors.append(f"row {index}: substrate concentration is negative")
        if np.isfinite(row.velocity_molar_per_s) and row.velocity_molar_per_s < 0:
            errors.append(f"row {index}: velocity is negative")
        if np.isfinite(row.enzyme_total_molar) and row.enzyme_total_molar <= 0:
            errors.append(f"row {index}: enzyme concentration must be positive")
        if np.isfinite(row.temperature_kelvin) and (not 250 <= row.temperature_kelvin <= 373.15):
            warnings.append(f"row {index}: temperature {row.temperature_kelvin:g} K is unusual")
        try:
            ph = float(row.pH)
        except (TypeError, ValueError):
            errors.append(f"row {index}: pH is not numeric")
        else:
            if not np.isfinite(ph):
                errors.append(f"row {index}: pH is missing or non-finite")
            elif not 0 <= ph <= 14:
                errors.append(f"row {index}: pH is outside [0, 14]")
        if pd.isna(row.provenance) or not str(row.provenance).strip():
            errors.append(f"row {index}: provenance is missing")
    return (
        frame,
        ValidationReport(not errors, tuple(errors), tuple(warnings), len(frame), duplicate_count),
    )
