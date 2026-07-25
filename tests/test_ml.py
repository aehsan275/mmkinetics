import numpy as np

from mm_kinetics.data.splitting import grouped_train_validation_test_split
from mm_kinetics.ml.bootstrap import (
    BootstrapEnsemble,
    ConformalCalibrator,
    interval_metrics,
)
from mm_kinetics.ml.features import apply_feature_schema, build_feature_set
from mm_kinetics.ml.models import build_model_pipelines
from mm_kinetics.synthetic import assay_level_truth, generate_synthetic_dataset


def _assays():
    raw = generate_synthetic_dataset(
        n_enzymes=12, n_substrates=2, n_concentrations=5, random_seed=10
    )
    return assay_level_truth(raw)


def test_group_split_has_no_enzyme_leakage():
    splits = grouped_train_validation_test_split(
        _assays(), group_column="enzyme_id", validation_size=0.2, test_size=0.2, random_seed=1
    )
    splits.assert_disjoint()
    assert not set(splits.train.enzyme_id) & set(splits.test.enzyme_id)


def test_feature_builder_blocks_measurements_and_targets():
    assays = _assays()
    assays["substrate_molar"] = 123.0
    features = build_feature_set(assays, "log10_km")
    assert "substrate_molar" not in features.X
    assert "km_molar" not in features.X
    assert "log10_kcat" not in features.X


def test_training_feature_schema_survives_constant_held_out_partition():
    assays = _assays()
    training = build_feature_set(assays, "log10_km")
    held_out = assays.iloc[:2].copy()
    for column in training.X.columns:
        if column in held_out:
            held_out[column] = held_out[column].iloc[0]
    transformed = apply_feature_schema(held_out, training)
    assert tuple(transformed.X.columns) == tuple(training.X.columns)
    assert len(transformed.X) == 2


def test_bootstrap_ensemble_reproducibility_and_calibration():
    assays = _assays()
    features = build_feature_set(assays, "log10_km")
    base = build_model_pipelines(
        features.numeric_columns, features.categorical_columns, random_seed=3
    )["ridge"]
    first = BootstrapEnsemble(base, n_bootstrap=20, random_seed=3).fit(features.X, features.y)
    second = BootstrapEnsemble(base, n_bootstrap=20, random_seed=3).fit(features.X, features.y)
    np.testing.assert_allclose(first.predict(features.X), second.predict(features.X))
    prediction = first.predict(features.X)
    calibrator = ConformalCalibrator(alpha=0.1).fit(features.y, prediction)
    lower, upper = calibrator.interval(prediction)
    metrics = interval_metrics(features.y, lower, upper)
    assert lower.shape == prediction.shape
    assert metrics["coverage"] >= 0.8


def test_bootstrap_ensemble_accepts_array_targets_and_rejects_nonfinite_values():
    assays = _assays()
    features = build_feature_set(assays, "log10_km")
    base = build_model_pipelines(
        features.numeric_columns, features.categorical_columns, random_seed=4
    )["ridge"]
    ensemble = BootstrapEnsemble(base, n_bootstrap=5, random_seed=4)
    ensemble.fit(features.X, features.y.to_numpy())
    assert np.isfinite(ensemble.predict(features.X)).all()
    invalid = features.y.to_numpy().copy()
    invalid[0] = np.nan
    with np.testing.assert_raises_regex(ValueError, "finite"):
        ensemble.fit(features.X, invalid)
