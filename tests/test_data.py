from pathlib import Path

import pandas as pd

from mm_kinetics.data.sabio import ingest_sabio_export
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


def test_sabio_adapter_preserves_exclusions(tmp_path: Path):
    export = tmp_path / "sabio.tsv"
    pd.DataFrame(
        {
            "Entry ID": [1, 2],
            "UniProt ID": ["P1", "P2"],
            "Substrate": ["glucose", "ATP"],
            "Km": [2.0, 1.0],
            "Km Unit": ["millimole/liter", "millimole/liter"],
            "kcat": [10.0, 5.0],
            "kcat unit": ["1/second", "1/second"],
            "Kinetic Law": ["Michaelis-Menten", "Hill equation"],
            "EC Number": ["1.1.1.1", "2.2.2.2"],
            "PubMed ID": ["11", "22"],
        }
    ).to_csv(export, sep="\t", index=False)
    result = ingest_sabio_export(export, retrieval_date="2026-08-26", query="fixture")
    assert len(result.accepted) == 1
    assert len(result.excluded) == 1
    assert result.accepted.km_molar.iloc[0] == 0.002
    assert "incompatible kinetic law" in result.excluded.exclusion_reason.iloc[0]
    assert result.provenance["query"] == "fixture"


def test_sabio_adapter_accepts_long_form_parameter_export(tmp_path: Path):
    export = tmp_path / "sabio_long.tsv"
    pd.DataFrame(
        {
            "Entry ID": [7, 8, 9],
            "UniProt ID": ["P7", "P8", "P9"],
            "Associated Species": ["glucose", "ATP", "NADH"],
            "Parameter Type": ["Km", "kcat", "Ki"],
            "Parameter Value": [3.0, 12.0, 4.0],
            "Parameter Unit": ["millimole/liter", "1/second", "micromole/liter"],
            "Kinetic Law": ["Michaelis-Menten"] * 3,
        }
    ).to_csv(export, sep="\t", index=False)
    result = ingest_sabio_export(export, retrieval_date="2026-08-26", query="long-form")
    assert len(result.accepted) == 2
    assert len(result.excluded) == 1
    assert result.accepted.log10_km.notna().sum() == 1
    assert result.accepted.log10_kcat.notna().sum() == 1
    assert "unsupported parameter type" in result.excluded.exclusion_reason.iloc[0]
