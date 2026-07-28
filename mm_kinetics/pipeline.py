from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mm_kinetics.bootstrap.kinetic import nonparametric_bootstrap, parametric_bootstrap
from mm_kinetics.config import load_config, validate_config
from mm_kinetics.data.sabio import ingest_sabio_export
from mm_kinetics.data.splitting import (
    grouped_train_validation_test_split,
    study_overlap,
)
from mm_kinetics.evaluation.plots import (
    plot_bootstrap_distributions,
    plot_fit,
    plot_mechanistic_simulation,
    plot_model_comparison,
    plot_predictions,
)
from mm_kinetics.kinetics.fitting import (
    fit_michaelis_menten,
    fit_michaelis_menten_weighted,
)
from mm_kinetics.kinetics.simulation import simulate_mechanistic
from mm_kinetics.ml.bootstrap import (
    BootstrapEnsemble,
    ConformalCalibrator,
    interval_metrics,
)
from mm_kinetics.ml.features import apply_feature_schema, build_feature_set
from mm_kinetics.ml.models import (
    build_model_pipelines,
    fit_and_evaluate_models,
    regression_metrics,
)
from mm_kinetics.synthetic import assay_level_truth, generate_synthetic_dataset


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _prepare_output(root: Path) -> dict[str, Path]:
    directories = {
        "synthetic": root / "data" / "synthetic",
        "processed": root / "data" / "processed",
        "figures": root / "figures",
        "models": root / "models",
        "reports": root / "reports" / "generated",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def run_pipeline(config: dict) -> dict[str, object]:
    config = validate_config(config)
    output = _prepare_output(Path(config["output_dir"]).resolve())
    validation_observations = generate_synthetic_dataset(
        n_enzymes=config["data"]["n_enzymes"],
        n_substrates=config["data"]["n_substrates"],
        assays_per_pair=config["data"]["assays_per_pair"],
        n_concentrations=config["data"]["n_concentrations"],
        noise_scale=config["data"]["noise_scale"],
        random_seed=config["random_seed"],
    )
    validation_observations.to_csv(output["synthetic"] / "raw_observations.csv", index=False)
    if config["data"]["source"] == "synthetic":
        assay_table = assay_level_truth(validation_observations)
        source_summary = {"source": "synthetic", "n_excluded": 0}
    else:
        sabio = ingest_sabio_export(
            config["data"]["sabio_export_path"],
            retrieval_date=config["data"]["sabio_retrieval_date"],
            query=config["data"]["sabio_query"],
        )
        assay_table = sabio.accepted
        sabio.accepted.to_csv(output["processed"] / "sabio_accepted.csv", index=False)
        sabio.excluded.to_csv(output["processed"] / "sabio_excluded.csv", index=False)
        (output["processed"] / "sabio_provenance.json").write_text(
            json.dumps(sabio.provenance, indent=2), encoding="utf-8"
        )
        source_summary = {"source": "SABIO-RK", "n_excluded": len(sabio.excluded)}
    target = config["model"]["target"]
    if target not in assay_table.columns:
        raise ValueError(
            f"selected target {target!r} is absent from the {source_summary['source']} assay table"
        )
    assay_table = assay_table[
        np.isfinite(pd.to_numeric(assay_table[target], errors="coerce"))
    ].copy()
    if assay_table.empty:
        raise ValueError(
            f"selected target {target!r} has no finite values in the {source_summary['source']} assay table"
        )
    if assay_table[config["model"]["group_column"]].nunique() < 4:
        raise ValueError("selected target/source has fewer than four usable ML groups")
    assay_table.to_csv(output["processed"] / "assay_table.csv", index=False)
    representative_id = validation_observations.assay_id.iloc[0]
    representative = validation_observations[validation_observations.assay_id == representative_id]
    substrate = representative.substrate_molar.to_numpy(float)
    velocity = representative.velocity_molar_per_s.to_numpy(float)
    sigma = representative.measurement_sigma.to_numpy(float)
    kinetic_fit = fit_michaelis_menten(substrate, velocity)
    weighted_fit = fit_michaelis_menten_weighted(substrate, velocity, sigma)
    kinetic_bootstrap = nonparametric_bootstrap(
        substrate,
        velocity,
        n_bootstrap=config["bootstrap"]["kinetic_iterations"],
        confidence_level=config["bootstrap"]["confidence_level"],
        random_seed=config["random_seed"],
        failure_threshold=config["bootstrap"]["failure_threshold"],
    )
    parametric = parametric_bootstrap(
        substrate,
        velocity,
        n_bootstrap=config["bootstrap"]["kinetic_iterations"],
        confidence_level=config["bootstrap"]["confidence_level"],
        random_seed=config["random_seed"] + 1,
        failure_threshold=config["bootstrap"]["failure_threshold"],
    )
    kinetic_bootstrap.samples.to_csv(
        output["processed"] / "kinetic_bootstrap_samples.csv", index=False
    )
    parametric.samples.to_csv(output["processed"] / "parametric_bootstrap_samples.csv", index=False)
    fitted_assay = pd.DataFrame(
        [
            {
                "assay_id": representative_id,
                **kinetic_fit.as_dict(),
                "weighted_Km": weighted_fit.km,
                "weighted_Vmax": weighted_fit.vmax,
                "bootstrap_Km_lower": kinetic_bootstrap.summary.loc["Km", "ci_lower"],
                "bootstrap_Km_upper": kinetic_bootstrap.summary.loc["Km", "ci_upper"],
                "bootstrap_Vmax_lower": kinetic_bootstrap.summary.loc["Vmax", "ci_lower"],
                "bootstrap_Vmax_upper": kinetic_bootstrap.summary.loc["Vmax", "ci_upper"],
                "bootstrap_failure_rate": kinetic_bootstrap.failure_rate,
            }
        ]
    )
    fitted_assay.to_csv(output["processed"] / "fitted_assays.csv", index=False)
    simulation = simulate_mechanistic(
        k1=1000000.0,
        k_minus_1=100.0,
        k2=50.0,
        enzyme_total=1e-06,
        substrate_initial=0.00015,
        t=np.linspace(0, 0.25, 500),
    )
    splits = grouped_train_validation_test_split(
        assay_table,
        group_column=config["model"]["group_column"],
        validation_size=config["model"]["validation_size"],
        test_size=config["model"]["test_size"],
        calibration_size=config["model"]["calibration_size"],
        random_seed=config["random_seed"],
    )
    overlap = study_overlap(splits)
    train_features = build_feature_set(splits.train, target)
    validation_features = apply_feature_schema(splits.validation, train_features)
    calibration_features = apply_feature_schema(splits.calibration, train_features)
    test_features = apply_feature_schema(splits.test, train_features)
    models = build_model_pipelines(
        train_features.numeric_columns,
        train_features.categorical_columns,
        random_seed=config["random_seed"],
    )
    fitted_models, model_metrics = fit_and_evaluate_models(
        models, train_features.X, train_features.y, validation_features.X, validation_features.y
    )
    model_metrics.to_csv(output["processed"] / "model_comparison.csv", index=False)
    selected = (
        str(model_metrics.iloc[0]["model"])
        if config["model"]["selected_model"] == "auto"
        else config["model"]["selected_model"]
    )
    if selected not in fitted_models:
        raise ValueError(f"selected model {selected!r} is unavailable")
    ensemble = BootstrapEnsemble(
        models[selected],
        n_bootstrap=config["bootstrap"]["ml_iterations"],
        random_seed=config["random_seed"],
    ).fit(train_features.X, train_features.y)
    calibration_prediction = ensemble.predict(calibration_features.X)
    calibrator = ConformalCalibrator(alpha=1 - config["bootstrap"]["confidence_level"]).fit(
        calibration_features.y, calibration_prediction
    )
    test_prediction = ensemble.predict(test_features.X)
    lower, upper = calibrator.interval(test_prediction)
    test_metrics = regression_metrics(test_features.y, test_prediction)
    predictive_interval_metrics = interval_metrics(test_features.y, lower, upper)
    predictions = pd.DataFrame(
        {
            "true_value": test_features.y,
            "prediction": test_prediction,
            "lower_95": lower,
            "upper_95": upper,
            "interval_width": upper - lower,
            "absolute_error": np.abs(test_prediction - test_features.y.to_numpy()),
        }
    )
    predictions.to_csv(output["processed"] / "ml_predictions.csv", index=False)
    joblib.dump(ensemble, output["models"] / "bootstrap_ensemble.joblib")
    joblib.dump(calibrator, output["models"] / "conformal_calibrator.joblib")
    figure_paths = [
        plot_mechanistic_simulation(
            simulation, output["figures"] / "01_mechanistic_simulation.png"
        ),
        plot_fit(substrate, velocity, kinetic_fit, output["figures"] / "02_mm_fit.png"),
        plot_bootstrap_distributions(
            kinetic_bootstrap, output["figures"] / "03_bootstrap_parameters.png"
        ),
        plot_model_comparison(model_metrics, output["figures"] / "04_model_comparison.png"),
        plot_predictions(
            test_features.y.to_numpy(),
            test_prediction,
            lower,
            upper,
            output["figures"] / "05_predictions.png",
        ),
    ]
    run_summary = {
        **source_summary,
        "target": target,
        "n_raw_validation_observations": len(validation_observations),
        "n_ml_assays": len(assay_table),
        "partition_groups": {
            name: part[config["model"]["group_column"]].nunique()
            for name, part in splits.partitions().items()
        },
        "representative_fit_success": kinetic_fit.success,
        "nonparametric_failure_rate": kinetic_bootstrap.failure_rate,
        "parametric_failure_rate": parametric.failure_rate,
        "selected_model": selected,
        "test_metrics": test_metrics,
    }
    metrics_payload = {
        "run_summary": run_summary,
        "predictive_interval_metrics": predictive_interval_metrics,
        "study_overlap": overlap,
        "figures": [str(path) for path in figure_paths],
    }
    (output["reports"] / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=_json_default), encoding="utf-8"
    )
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit kinetic curves and train parameter-prediction models."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML run configuration")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = run_pipeline(config)
    print(json.dumps(result, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
