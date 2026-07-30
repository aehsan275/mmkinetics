import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mm_kinetics import pipeline
from mm_kinetics.config import validate_config
from mm_kinetics.pipeline import run_pipeline


@pytest.fixture
def smoke_config(tmp_path):
    return validate_config(
        {
            "random_seed": 12,
            "output_dir": str(tmp_path),
            "data": {"n_enzymes": 6, "n_substrates": 2, "n_concentrations": 6, "noise_scale": 0.06},
            "bootstrap": {
                "kinetic_iterations": 20,
                "ml_iterations": 10,
                "confidence_level": 0.9,
                "failure_threshold": 0.4,
            },
            "model": {"selected_model": "ridge"},
        }
    )


def test_calibration_labels_cannot_change_model_selection_or_training(tmp_path, monkeypatch):
    config = validate_config(
        {
            "output_dir": str(tmp_path / "original"),
            "data": {"n_enzymes": 10, "n_substrates": 2, "n_concentrations": 6},
            "bootstrap": {"kinetic_iterations": 20, "ml_iterations": 10},
        }
    )
    original = run_pipeline(config)
    split = pipeline.grouped_train_validation_test_split

    def change_calibration_labels(*args, **kwargs):
        parts = split(*args, **kwargs)
        parts.calibration[config["model"]["target"]] += 100
        return parts

    monkeypatch.setattr(pipeline, "grouped_train_validation_test_split", change_calibration_labels)
    config["output_dir"] = str(tmp_path / "changed")
    changed = run_pipeline(config)
    assert original["run_summary"]["selected_model"] == changed["run_summary"]["selected_model"]
    for filename, columns in [
        ("model_comparison.csv", None),
        ("ml_predictions.csv", ["true_value", "prediction"]),
    ]:
        before = pd.read_csv(tmp_path / "original" / "data" / "processed" / filename)
        after = pd.read_csv(tmp_path / "changed" / "data" / "processed" / filename)
        pd.testing.assert_frame_equal(
            before if columns is None else before[columns],
            after if columns is None else after[columns],
        )


@pytest.mark.parametrize("target", ["log10_km", "log10_kcat", "log10_efficiency"])
def test_end_to_end_pipeline_smoke(tmp_path: Path, target: str, smoke_config):
    smoke_config["model"]["target"] = target
    result = run_pipeline(smoke_config)
    assert (tmp_path / "reports" / "generated" / "metrics.json").exists()
    assert (tmp_path / "data" / "processed" / "ml_predictions.csv").exists()
    assert len(result["figures"]) == 5
    predictions = pd.read_csv(tmp_path / "data" / "processed" / "ml_predictions.csv")
    assert set(predictions.interval_status) == {"unbounded"}
    assert np.isneginf(predictions.lower_95).all()
    assert np.isposinf(predictions.upper_95).all()
    assert result["run_summary"]["calibration"]["calibration_group_count"] == 1
    assert result["predictive_interval_metrics"]["mean_width"] is None
    assert result["predictive_interval_metrics"]["group_coverage"] == 1
    assert result["predictive_interval_metrics"]["unbounded_fraction"] == 1

    def reject_nonfinite(token):
        raise AssertionError(f"nonfinite JSON token: {token}")

    stored = json.loads(
        (tmp_path / "reports" / "generated" / "metrics.json").read_text(),
        parse_constant=reject_nonfinite,
    )
    assert stored == result


def test_module_entrypoint_loads_package():
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "mm_kinetics.pipeline", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "YAML run configuration" in completed.stdout


def test_end_to_end_sabio_export_pipeline(tmp_path: Path, smoke_config):
    export = tmp_path / "sabio.tsv"
    records = []
    for enzyme_index in range(6):
        for substrate_index in range(2):
            records.append(
                {
                    "Entry ID": f"{enzyme_index}-{substrate_index}",
                    "UniProt ID": f"P{enzyme_index}",
                    "Substrate": f"S{substrate_index}",
                    "Km": 0.5 + enzyme_index + 0.2 * substrate_index,
                    "Km Unit": "millimole/liter",
                    "kcat": 5.0 + enzyme_index,
                    "kcat unit": "1/second",
                    "Kinetic Law": "Michaelis-Menten",
                    "EC Number": f"1.1.{enzyme_index}.1",
                    "PubMed ID": f"PMID{enzyme_index // 2}",
                    "Temperature": 298.15 + enzyme_index,
                    "Temperature Unit": "kelvin",
                    "pH": 6.5 + 0.2 * enzyme_index,
                    "Organism": f"organism-{enzyme_index % 3}",
                }
            )
    pd.DataFrame(records).to_csv(export, sep="\t", index=False)
    smoke_config["random_seed"] = 8
    smoke_config["output_dir"] = str(tmp_path / "run")
    smoke_config["data"]["source"] = "sabio"
    smoke_config["data"]["sabio_export_path"] = str(export)
    smoke_config["data"]["sabio_retrieval_date"] = "2026-08-26"
    smoke_config["data"]["sabio_query"] = "integration fixture"
    result = run_pipeline(smoke_config)
    assert result["run_summary"]["source"] == "SABIO-RK"
    assert result["run_summary"]["n_ml_assays"] == len(records)
    assert (tmp_path / "run" / "data" / "processed" / "sabio_accepted.csv").exists()
    assert (tmp_path / "run" / "data" / "processed" / "sabio_provenance.json").exists()
