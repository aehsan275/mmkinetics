from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_float_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def michaelis_menten(
    substrate: ArrayLike, vmax: ArrayLike, km: ArrayLike
) -> float | NDArray[np.float64]:
    substrate_array = _as_float_array(substrate, "substrate")
    vmax_array = _as_float_array(vmax, "vmax")
    km_array = _as_float_array(km, "km")
    if np.any(substrate_array < 0):
        raise ValueError("substrate concentrations must be nonnegative")
    if np.any(vmax_array < 0):
        raise ValueError("vmax must be nonnegative")
    if np.any(km_array <= 0):
        raise ValueError("km must be strictly positive")
    result = vmax_array * substrate_array / (km_array + substrate_array)
    return float(result) if result.ndim == 0 else np.asarray(result, dtype=float)


def michaelis_menten_kcat(
    substrate: ArrayLike, enzyme_total: ArrayLike, kcat: ArrayLike, km: ArrayLike
) -> float | NDArray[np.float64]:
    enzyme_array = _as_float_array(enzyme_total, "enzyme_total")
    kcat_array = _as_float_array(kcat, "kcat")
    if np.any(enzyme_array < 0):
        raise ValueError("enzyme_total must be nonnegative")
    if np.any(kcat_array < 0):
        raise ValueError("kcat must be nonnegative")
    return michaelis_menten(substrate, enzyme_array * kcat_array, km)
