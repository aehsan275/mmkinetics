import numpy as np
import pandas as pd
import pytest

from mm_kinetics.data.validation import validate_and_normalize_observations


@pytest.mark.parametrize(
    "bad_unit", ["substrate_unit", "velocity_unit", "enzyme_concentration_unit", "temperature_unit"]
)
def test_failed_conversion_keeps_rows_aligned(bad_unit):
    row = {
        "assay_id": "A1",
        "enzyme_id": "E1",
        "substrate_id": "S1",
        "substrate_concentration": 2.0,
        "substrate_unit": "millimole/liter",
        "observed_velocity": 3.0,
        "velocity_unit": "micromole/liter/second",
        "enzyme_concentration": 5.0,
        "enzyme_concentration_unit": "nanomole/liter",
        "temperature": 25.0,
        "temperature_unit": "degC",
        "pH": 7.0,
        "provenance": "fixture",
    }
    raw = pd.DataFrame([row, {**row, "assay_id": "A2", bad_unit: "invalid_unit"}], index=[4, 9])
    normalized, report = validate_and_normalize_observations(raw)
    assert not report.valid
    assert report.n_rows == 2
    assert any("row 9:" in error for error in report.errors)
    assert normalized.index.equals(raw.index)
    assert normalized.loc[4, "substrate_molar"] == pytest.approx(0.002)
    assert normalized.loc[4, "temperature_kelvin"] == pytest.approx(298.15)
    columns = [
        "substrate_molar",
        "velocity_molar_per_s",
        "enzyme_total_molar",
        "temperature_kelvin",
    ]
    assert np.isnan(normalized.loc[9, columns].to_numpy(float)).all()


def test_empty_observations_keep_normalized_columns():
    columns = [
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
    ]
    normalized, report = validate_and_normalize_observations(pd.DataFrame(columns=columns))
    assert report.valid
    assert normalized.empty
    assert "temperature_kelvin" in normalized
