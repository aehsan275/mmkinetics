import pytest

from mm_kinetics.config import DEFAULTS, load_config, validate_config


def test_partial_settings_do_not_mutate_defaults_or_input():
    supplied = {"model": {"target": "log10_kcat"}}
    first = validate_config(supplied)
    first["model"]["target"] = "log10_efficiency"
    assert supplied["model"]["target"] == "log10_kcat"
    second = validate_config({})
    assert second == DEFAULTS
    assert second["model"]["target"] == "log10_km"


@pytest.mark.parametrize(
    "payload",
    [
        {"unknown": 1},
        {"data": {"extra": "lognormal"}},
        {"bootstrap": {"extra": 1}},
        {"model": {"extra": 1}},
        {"data": {"source": "other"}},
        {"data": {"source": "sabio"}},
        {"data": {"n_enzymes": 2}},
        {"data": {"noise_scale": float("inf")}},
        {"bootstrap": {"kinetic_iterations": 19}},
        {"bootstrap": {"ml_iterations": 9}},
        {"bootstrap": {"confidence_level": 1}},
        {"bootstrap": {"failure_threshold": -0.1}},
        {"model": {"target": "other"}},
        {"model": {"selected_model": "other"}},
        {"model": {"calibration_size": 0.5}},
    ],
)
def test_invalid_configuration_is_rejected(payload):
    with pytest.raises(ValueError):
        validate_config(payload)


@pytest.mark.parametrize(
    "payload", [[], {"data": []}, {"random_seed": True}, {"data": {"n_enzymes": 6.5}}]
)
def test_invalid_configuration_types_are_rejected(payload):
    with pytest.raises(TypeError):
        validate_config(payload)


def test_yaml_settings_and_empty_document(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  target: log10_efficiency\n", encoding="utf-8")
    assert load_config(path)["model"]["target"] == "log10_efficiency"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == DEFAULTS
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(path)
