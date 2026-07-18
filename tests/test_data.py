import pandas as pd

from mm_kinetics.data.validation import validate_and_normalize_observations


def test_raw_observation_unit_normalization_and_provenance():
    raw = pd.DataFrame(
        {
            "assay_id": ["A1"],
            "enzyme_id": ["E1"],
            "substrate_id": ["S1"],
            "substrate_concentration": [2.0],
            "substrate_unit": ["millimole/liter"],
            "observed_velocity": [3.0],
            "velocity_unit": ["micromole/liter/second"],
            "enzyme_concentration": [5.0],
            "enzyme_concentration_unit": ["nanomole/liter"],
            "temperature": [25.0],
            "temperature_unit": ["degC"],
            "pH": [7.0],
            "provenance": ["fixture"],
        }
    )
    normalized, report = validate_and_normalize_observations(raw)
    assert report.valid, report.errors
    assert normalized.substrate_molar.iloc[0] == 0.002
    assert normalized.velocity_molar_per_s.iloc[0] == 3e-06
    assert normalized.enzyme_total_molar.iloc[0] == 5e-09
    assert normalized.temperature_kelvin.iloc[0] == 298.15
    assert normalized.substrate_original_unit.iloc[0] == "millimole/liter"


def test_raw_observation_rejects_missing_identity_and_nonfinite_ph():
    raw = pd.DataFrame(
        {
            "assay_id": [""],
            "enzyme_id": ["E1"],
            "substrate_id": ["S1"],
            "substrate_concentration": [1.0],
            "substrate_unit": ["millimole/liter"],
            "observed_velocity": [1.0],
            "velocity_unit": ["micromole/liter/second"],
            "enzyme_concentration": [1.0],
            "enzyme_concentration_unit": ["nanomole/liter"],
            "temperature": [25.0],
            "temperature_unit": ["degC"],
            "pH": [float("nan")],
            "provenance": ["fixture"],
        }
    )
    _, report = validate_and_normalize_observations(raw)
    assert not report.valid
    assert any("assay_id is missing" in error for error in report.errors)
    assert any("pH is missing or non-finite" in error for error in report.errors)
