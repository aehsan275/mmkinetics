from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path

import yaml

DEFAULTS = {
    "random_seed": 42,
    "output_dir": ".",
    "data": {
        "source": "synthetic",
        "sabio_export_path": None,
        "sabio_retrieval_date": None,
        "sabio_query": None,
        "n_enzymes": 100,
        "n_substrates": 4,
        "assays_per_pair": 1,
        "n_concentrations": 10,
        "noise_scale": 0.08,
    },
    "bootstrap": {
        "kinetic_iterations": 200,
        "ml_iterations": 40,
        "confidence_level": 0.95,
        "failure_threshold": 0.2,
    },
    "model": {
        "target": "log10_km",
        "group_column": "enzyme_id",
        "test_size": 0.2,
        "validation_size": 0.2,
        "calibration_size": 0.2,
        "selected_model": "auto",
    },
}


def validate_config(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("configuration must be a mapping")
    config = deepcopy(DEFAULTS)
    for name, value in payload.items():
        if name not in config:
            raise ValueError(f"unknown configuration field: {name}")
        if isinstance(config[name], dict):
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a mapping")
            unknown = value.keys() - config[name].keys()
            if unknown:
                raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
            config[name].update(deepcopy(value))
        else:
            config[name] = value

    def integer(value, field, minimum=None):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be an integer")
        if minimum is not None and value < minimum:
            raise ValueError(f"{field} must be at least {minimum}")

    def number(value, field, lower, upper=None, *, inclusive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not isfinite(value)):
            raise ValueError(f"{field} must be a finite number")
        valid = value >= lower if inclusive else value > lower
        if upper is not None:
            valid = valid and (value <= upper if inclusive else value < upper)
        if not valid:
            raise ValueError(f"{field} is outside its allowed range")

    integer(config["random_seed"], "random_seed")
    if not isinstance(config["output_dir"], str):
        raise TypeError("output_dir must be a string")
    data = config["data"]
    if data["source"] not in ("synthetic", "sabio"):
        raise ValueError("data.source must be synthetic or sabio")
    for field in ("sabio_export_path", "sabio_retrieval_date", "sabio_query"):
        if data[field] is not None and (not isinstance(data[field], str)):
            raise ValueError(f"data.{field} must be a string or null")
    if data["source"] == "sabio" and (not data["sabio_export_path"]):
        raise ValueError("sabio_export_path is required when data.source='sabio'")
    for field, minimum in (
        ("n_enzymes", 3),
        ("n_substrates", 1),
        ("assays_per_pair", 1),
        ("n_concentrations", 3),
    ):
        integer(data[field], f"data.{field}", minimum)
    number(data["noise_scale"], "data.noise_scale", 0)
    bootstrap = config["bootstrap"]
    integer(bootstrap["kinetic_iterations"], "bootstrap.kinetic_iterations", 20)
    integer(bootstrap["ml_iterations"], "bootstrap.ml_iterations", 10)
    number(bootstrap["confidence_level"], "bootstrap.confidence_level", 0.5, 1)
    number(bootstrap["failure_threshold"], "bootstrap.failure_threshold", 0, 1, inclusive=True)
    model = config["model"]
    if model["target"] not in ("log10_km", "log10_kcat", "log10_efficiency"):
        raise ValueError("unsupported model.target")
    if not isinstance(model["group_column"], str):
        raise TypeError("model.group_column must be a string")
    if model["selected_model"] not in (
        "auto",
        "median",
        "linear",
        "ridge",
        "random_forest",
        "gradient_boosting",
    ):
        raise ValueError("unsupported model.selected_model")
    fractions = [model[name] for name in ("validation_size", "test_size", "calibration_size")]
    for fraction in fractions:
        number(fraction, "held-out fraction", 0, 0.5)
    if sum(fractions) >= 0.8:
        raise ValueError("held-out fractions must leave at least 20% for training")
    return config


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return validate_config({} if payload is None else payload)
