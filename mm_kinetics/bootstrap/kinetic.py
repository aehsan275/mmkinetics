from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..kinetics.equations import michaelis_menten
from ..kinetics.fitting import FitResult, fit_michaelis_menten


@dataclass(frozen=True)
class BootstrapResult:
    samples: pd.DataFrame
    summary: pd.DataFrame
    original_fit: FitResult
    n_requested: int
    n_success: int
    n_failures: int
    failure_reasons: dict[str, int]
    confidence_level: float
    failure_threshold: float

    @property
    def failure_rate(self) -> float:
        return self.n_failures / self.n_requested

    @property
    def exceeds_failure_threshold(self) -> bool:
        return self.failure_rate > self.failure_threshold


def _summarize(samples: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    records: list[dict[str, float | str | int]] = []
    for parameter in ["Km", "Vmax"]:
        values = samples[parameter].to_numpy(dtype=float) if parameter in samples else np.array([])
        if len(values) == 0:
            records.append(
                {
                    "parameter": parameter,
                    "n": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                }
            )
        else:
            records.append(
                {
                    "parameter": parameter,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci_lower": float(np.quantile(values, alpha / 2)),
                    "ci_upper": float(np.quantile(values, 1 - alpha / 2)),
                }
            )
    return pd.DataFrame(records).set_index("parameter")


def _result(
    records: list[dict[str, float | int]],
    failures: Counter[str],
    original_fit: FitResult,
    n_bootstrap: int,
    confidence_level: float,
    failure_threshold: float,
) -> BootstrapResult:
    samples = pd.DataFrame(records, columns=["iteration", "Km", "Vmax"])
    return BootstrapResult(
        samples=samples,
        summary=_summarize(samples, confidence_level),
        original_fit=original_fit,
        n_requested=n_bootstrap,
        n_success=len(samples),
        n_failures=n_bootstrap - len(samples),
        failure_reasons=dict(failures),
        confidence_level=confidence_level,
        failure_threshold=failure_threshold,
    )


def nonparametric_bootstrap(
    substrate: ArrayLike,
    velocity: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    failure_threshold: float = 0.2,
) -> BootstrapResult:
    if n_bootstrap < 1 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap settings")
    s = np.asarray(substrate, dtype=float)
    v = np.asarray(velocity, dtype=float)
    original_fit = fit_michaelis_menten(s, v)
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, float | int]] = []
    failures: Counter[str] = Counter()
    for iteration in range(n_bootstrap):
        indices = rng.integers(0, len(s), len(s))
        s_boot = s[indices]
        v_boot = v[indices]
        if np.unique(s_boot).size < 3:
            failures["fewer than three unique substrate concentrations"] += 1
            continue
        try:
            fit = fit_michaelis_menten(s_boot, v_boot)
        except ValueError as error:
            failures[f"ValueError: {error}"] += 1
            continue
        if not fit.success:
            failures[fit.failure_reason or "unknown fit failure"] += 1
            continue
        records.append({"iteration": iteration, "Km": fit.km, "Vmax": fit.vmax})
    return _result(
        records, failures, original_fit, n_bootstrap, confidence_level, failure_threshold
    )


def parametric_bootstrap(
    substrate: ArrayLike,
    velocity: ArrayLike,
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    failure_threshold: float = 0.2,
) -> BootstrapResult:
    if n_bootstrap < 1 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap settings")
    s = np.asarray(substrate, dtype=float)
    v = np.asarray(velocity, dtype=float)
    original_fit = fit_michaelis_menten(s, v)
    if not original_fit.success:
        failures = Counter({original_fit.failure_reason or "original fit failed": n_bootstrap})
        return _result([], failures, original_fit, n_bootstrap, confidence_level, failure_threshold)
    predicted = np.asarray(michaelis_menten(s, original_fit.vmax, original_fit.km), dtype=float)
    residuals = v - predicted
    residual_scale = float(np.std(residuals, ddof=2))
    residual_scale = max(residual_scale, np.finfo(float).eps)
    positive_mask = (v > 0) & (predicted > 0)
    log_scale = (
        float(np.std(np.log(v[positive_mask] / predicted[positive_mask]), ddof=1))
        if np.count_nonzero(positive_mask) >= 3
        else residual_scale / max(float(np.max(predicted)), np.finfo(float).eps)
    )
    log_scale = max(log_scale, np.finfo(float).eps)
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, float | int]] = []
    failures: Counter[str] = Counter()
    for iteration in range(n_bootstrap):
        synthetic = predicted * np.exp(rng.normal(-0.5 * log_scale**2, log_scale, len(v)))
        synthetic = np.clip(synthetic, 0.0, None)
        fit = fit_michaelis_menten(s, synthetic)
        if not fit.success:
            failures[fit.failure_reason or "unknown fit failure"] += 1
            continue
        records.append({"iteration": iteration, "Km": fit.km, "Vmax": fit.vmax})
    return _result(
        records, failures, original_fit, n_bootstrap, confidence_level, failure_threshold
    )
