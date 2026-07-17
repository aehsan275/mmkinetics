from __future__ import annotations

import numpy as np
import pandas as pd

from .kinetics.equations import michaelis_menten


def _observe_velocity(true_velocity, noise_scale, rng):
    sigma = np.maximum(noise_scale * true_velocity, np.finfo(float).eps)
    observed = true_velocity * np.exp(
        rng.normal(-0.5 * noise_scale**2, noise_scale, len(true_velocity))
    )
    return (np.clip(observed, 0.0, None), sigma)


def generate_synthetic_dataset(
    *,
    n_enzymes: int = 20,
    n_substrates: int = 5,
    assays_per_pair: int = 1,
    n_concentrations: int = 10,
    noise_scale: float = 0.08,
    random_seed: int = 42,
) -> pd.DataFrame:
    if min(n_enzymes, n_substrates, assays_per_pair) < 1 or n_concentrations < 3:
        raise ValueError("dataset dimensions must be positive and need >=3 concentrations")
    if noise_scale <= 0:
        raise ValueError("noise_scale must be positive")
    rng = np.random.default_rng(random_seed)
    enzyme_activity = rng.normal(0, 0.55, n_enzymes)
    enzyme_affinity = rng.normal(0, 0.45, n_enzymes)
    enzyme_stability = rng.normal(0, 1.0, n_enzymes)
    substrate_size = rng.normal(0, 1.0, n_substrates)
    substrate_polarity = rng.normal(0, 1.0, n_substrates)
    rows: list[dict[str, object]] = []
    for enzyme_index in range(n_enzymes):
        enzyme_id = f"ENZ{enzyme_index:04d}"
        enzyme_family = f"FAM{enzyme_index % max(3, n_enzymes // 5):03d}"
        organism = ["bacterium", "fungus", "plant", "animal"][enzyme_index % 4]
        for substrate_index in range(n_substrates):
            substrate_id = f"SUB{substrate_index:03d}"
            for replicate in range(assays_per_pair):
                temperature_kelvin = float(rng.uniform(293.15, 318.15))
                ph = float(rng.uniform(5.8, 8.5))
                enzyme_total = float(10 ** rng.uniform(-8.0, -6.0))
                condition_noise = rng.normal(0, 0.05)
                log10_km = (
                    -4.0
                    + 0.48 * enzyme_affinity[enzyme_index]
                    + 0.35 * substrate_size[substrate_index]
                    + 0.1 * substrate_polarity[substrate_index]
                    + 0.08 * (ph - 7.0) ** 2
                    + 0.003 * (temperature_kelvin - 303.15)
                    + condition_noise
                )
                log10_kcat = (
                    1.4
                    + 0.62 * enzyme_activity[enzyme_index]
                    - 0.22 * substrate_polarity[substrate_index]
                    - 0.025 * (ph - 7.2) ** 2
                    + 0.008 * (temperature_kelvin - 303.15)
                    + 0.08 * enzyme_stability[enzyme_index]
                    + rng.normal(0, 0.05)
                )
                true_km = float(10**log10_km)
                true_kcat = float(10**log10_kcat)
                true_vmax = enzyme_total * true_kcat
                substrate = true_km * np.logspace(-2, 2, n_concentrations)
                true_velocity = np.asarray(michaelis_menten(substrate, true_vmax, true_km))
                observed, sigma = _observe_velocity(true_velocity, noise_scale, rng)
                assay_id = f"A{enzyme_index:04d}-{substrate_index:03d}-{replicate:02d}"
                study_id = f"SYN-STUDY-{enzyme_index // 3:03d}"
                for concentration_index in range(n_concentrations):
                    rows.append(
                        {
                            "assay_id": assay_id,
                            "experiment_id": f"{assay_id}-{concentration_index:03d}",
                            "replicate_id": replicate,
                            "enzyme_id": enzyme_id,
                            "enzyme_family": enzyme_family,
                            "substrate_id": substrate_id,
                            "study_id": study_id,
                            "organism": organism,
                            "temperature_kelvin": temperature_kelvin,
                            "pH": ph,
                            "ionic_strength_molar": 0.1,
                            "enzyme_total_molar": enzyme_total,
                            "substrate_molar": float(substrate[concentration_index]),
                            "velocity_molar_per_s": float(observed[concentration_index]),
                            "measurement_sigma": float(sigma[concentration_index]),
                            "true_velocity_molar_per_s": float(true_velocity[concentration_index]),
                            "true_km_molar": true_km,
                            "true_kcat_per_s": true_kcat,
                            "true_vmax_molar_per_s": true_vmax,
                            "enzyme_activity_descriptor": float(enzyme_activity[enzyme_index]),
                            "enzyme_affinity_descriptor": float(enzyme_affinity[enzyme_index]),
                            "enzyme_stability_descriptor": float(enzyme_stability[enzyme_index]),
                            "substrate_size_descriptor": float(substrate_size[substrate_index]),
                            "substrate_polarity_descriptor": float(
                                substrate_polarity[substrate_index]
                            ),
                            "noise_model": "lognormal",
                            "noise_scale": noise_scale,
                            "source": "synthetic",
                            "provenance": f"seed={random_seed}",
                        }
                    )
    return pd.DataFrame(rows)


def assay_level_truth(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"assay_id", "true_km_molar", "true_kcat_per_s"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"missing required synthetic columns: {sorted(missing)}")
    exclude = {
        "experiment_id",
        "replicate_id",
        "substrate_molar",
        "velocity_molar_per_s",
        "measurement_sigma",
        "true_velocity_molar_per_s",
        "true_vmax_molar_per_s",
        "noise_scale",
    }
    candidates = [column for column in observations.columns if column not in exclude]
    assays = observations[candidates].groupby("assay_id", as_index=False).first()
    assays["km_molar"] = assays.pop("true_km_molar")
    assays["kcat_per_s"] = assays.pop("true_kcat_per_s")
    assays["log10_km"] = np.log10(assays["km_molar"])
    assays["log10_kcat"] = np.log10(assays["kcat_per_s"])
    assays["log10_efficiency"] = np.log10(assays["kcat_per_s"] / assays["km_molar"])
    return assays
