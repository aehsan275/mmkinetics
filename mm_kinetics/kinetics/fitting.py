from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeWarning, curve_fit

from .equations import michaelis_menten


@dataclass(frozen=True)
class FitResult:
    vmax: float
    km: float
    covariance: NDArray[np.float64]
    residuals: NDArray[np.float64]
    fitted: NDArray[np.float64]
    rmse: float
    mae: float
    r2: float
    success: bool
    failure_reason: str | None = None
    initialization: tuple[float, float] | None = None

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "Vmax": self.vmax,
            "Km": self.km,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "initialization": self.initialization,
        }
        if include_arrays:
            result.update(covariance=self.covariance, residuals=self.residuals, fitted=self.fitted)
        return result


def _failed_result(reason: str, n: int) -> FitResult:
    return FitResult(
        vmax=np.nan,
        km=np.nan,
        covariance=np.full((2, 2), np.nan),
        residuals=np.full(n, np.nan),
        fitted=np.full(n, np.nan),
        rmse=np.nan,
        mae=np.nan,
        r2=np.nan,
        success=False,
        failure_reason=reason,
    )


def _validate_fit_inputs(
    substrate: ArrayLike, velocity: ArrayLike, sigma: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    s = np.asarray(substrate, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if s.ndim != 1 or v.ndim != 1 or len(s) != len(v):
        raise ValueError("substrate and velocity must be equally sized one-dimensional arrays")
    if len(s) < 3:
        raise ValueError("at least three observations are required")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(v)):
        raise ValueError("fit inputs must be finite")
    if np.any(s < 0) or np.any(v < 0):
        raise ValueError("substrate and velocity must be nonnegative")
    if np.unique(s).size < 3:
        raise ValueError("at least three unique substrate concentrations are required")
    sigma_array = None
    if sigma is not None:
        sigma_array = np.asarray(sigma, dtype=float)
        if sigma_array.shape != v.shape or not np.all(np.isfinite(sigma_array)):
            raise ValueError("sigma must be finite and match velocity")
        if np.any(sigma_array <= 0):
            raise ValueError("sigma must be strictly positive")
    return (s, v, sigma_array)


def fit_michaelis_menten(
    substrate: ArrayLike,
    velocity: ArrayLike,
    *,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = True,
    maxfev: int = 20000,
) -> FitResult:
    s, v, sigma_array = _validate_fit_inputs(substrate, velocity, sigma)
    positive_s = s[s > 0]
    scale_v = max(float(np.max(v)), np.finfo(float).eps)
    scale_s = max(float(np.median(positive_s)), np.finfo(float).eps)
    initializations = [
        (scale_v, scale_s),
        (scale_v * 1.25, float(np.quantile(positive_s, 0.25))),
        (scale_v * 2.0, float(np.quantile(positive_s, 0.75))),
        (max(float(np.quantile(v, 0.9)), scale_v), float(np.mean(positive_s))),
    ]
    best: FitResult | None = None
    failures: list[str] = []
    lower = [0.0, np.finfo(float).eps]
    upper = [np.inf, np.inf]
    for initialization in initializations:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                parameters, covariance = curve_fit(
                    michaelis_menten,
                    s,
                    v,
                    p0=initialization,
                    bounds=(lower, upper),
                    sigma=sigma_array,
                    absolute_sigma=absolute_sigma,
                    maxfev=maxfev,
                )
            vmax, km = map(float, parameters)
            fitted = np.asarray(michaelis_menten(s, vmax, km), dtype=float)
            residuals = v - fitted
            rmse = float(np.sqrt(np.mean(residuals**2)))
            mae = float(np.mean(np.abs(residuals)))
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((v - np.mean(v)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0 if ss_res == 0 else np.nan
            candidate = FitResult(
                vmax=vmax,
                km=km,
                covariance=np.asarray(covariance, dtype=float),
                residuals=residuals,
                fitted=fitted,
                rmse=rmse,
                mae=mae,
                r2=float(r2),
                success=True,
                initialization=(float(initialization[0]), float(initialization[1])),
            )
            if best is None or candidate.rmse < best.rmse:
                best = candidate
        except (RuntimeError, ValueError, FloatingPointError, OptimizeWarning) as error:
            failures.append(f"{type(error).__name__}: {error}")
    return best if best is not None else _failed_result("; ".join(failures), len(s))


def fit_michaelis_menten_weighted(
    substrate: ArrayLike,
    velocity: ArrayLike,
    sigma: ArrayLike,
    *,
    absolute_sigma: bool = True,
    maxfev: int = 20000,
) -> FitResult:
    return fit_michaelis_menten(
        substrate, velocity, sigma=sigma, absolute_sigma=absolute_sigma, maxfev=maxfev
    )
