from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class MechanisticSimulation:
    time: NDArray[np.float64]
    enzyme: NDArray[np.float64]
    complex: NDArray[np.float64]
    substrate: NDArray[np.float64]
    product: NDArray[np.float64]
    enzyme_total: float
    substrate_initial: float
    success: bool
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "time": self.time,
            "E": self.enzyme,
            "ES": self.complex,
            "S": self.substrate,
            "P": self.product,
            "enzyme_total": self.enzyme_total,
            "substrate_initial": self.substrate_initial,
            "success": self.success,
            "message": self.message,
        }


def simulate_mechanistic(
    k1: float,
    k_minus_1: float,
    k2: float,
    enzyme_total: float,
    substrate_initial: float,
    t: ArrayLike,
    *,
    complex_initial: float = 0.0,
    rtol: float = 1e-09,
    atol: float = 1e-12,
) -> MechanisticSimulation:
    rates = np.asarray([k1, k_minus_1, k2], dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0) or k1 == 0:
        raise ValueError("k1 must be positive and all rate constants must be finite/nonnegative")
    if not np.isfinite(enzyme_total) or enzyme_total <= 0:
        raise ValueError("enzyme_total must be finite and positive")
    if not np.isfinite(substrate_initial) or substrate_initial < 0:
        raise ValueError("substrate_initial must be finite and nonnegative")
    if complex_initial < 0 or complex_initial > min(enzyme_total, substrate_initial):
        raise ValueError("complex_initial violates enzyme or substrate conservation")
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or len(times) < 2 or (not np.all(np.isfinite(times))):
        raise ValueError("t must be a finite one-dimensional array with at least two points")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("t must be strictly increasing and nonnegative")
    free_substrate_initial = substrate_initial - complex_initial

    def rhs(_time: float, state: NDArray[np.float64]) -> list[float]:
        es, substrate, _product = state
        enzyme = enzyme_total - es
        association = k1 * enzyme * substrate
        dissociation = k_minus_1 * es
        catalysis = k2 * es
        return [association - dissociation - catalysis, -association + dissociation, catalysis]

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        [complex_initial, free_substrate_initial, 0.0],
        t_eval=times,
        rtol=rtol,
        atol=atol,
        method="LSODA",
    )
    if not solution.success:
        raise RuntimeError(f"mechanistic integration failed: {solution.message}")
    es, substrate, product = np.asarray(solution.y, dtype=float)
    enzyme = enzyme_total - es
    tolerance = max(atol * 10, 1e-14)
    enzyme = np.where(np.abs(enzyme) < tolerance, 0.0, enzyme)
    es = np.where(np.abs(es) < tolerance, 0.0, es)
    substrate = np.where(np.abs(substrate) < tolerance, 0.0, substrate)
    product = np.where(np.abs(product) < tolerance, 0.0, product)
    if min(np.min(enzyme), np.min(es), np.min(substrate), np.min(product)) < -tolerance:
        raise RuntimeError("mechanistic integration produced materially negative concentrations")
    return MechanisticSimulation(
        time=times,
        enzyme=enzyme,
        complex=es,
        substrate=substrate,
        product=product,
        enzyme_total=float(enzyme_total),
        substrate_initial=float(substrate_initial),
        success=True,
        message=str(solution.message),
    )
