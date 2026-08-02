from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..bootstrap.kinetic import BootstrapResult
from ..kinetics.equations import michaelis_menten
from ..kinetics.fitting import FitResult
from ..kinetics.simulation import MechanisticSimulation

sns.set_theme(context="paper", style="whitegrid")


def _save(fig: plt.Figure, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_mechanistic_simulation(simulation: MechanisticSimulation, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(simulation.time, simulation.substrate, label="S")
    ax.plot(simulation.time, simulation.complex, label="ES")
    ax.plot(simulation.time, simulation.product, label="P")
    ax.plot(simulation.time, simulation.enzyme, label="E", linestyle="--")
    ax.set(
        xlabel="Time (s)", ylabel="Concentration (M)", title="Mechanistic mass-action simulation"
    )
    ax.legend(ncol=4)
    return _save(fig, path)


def plot_fit(substrate: np.ndarray, velocity: np.ndarray, fit: FitResult, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(substrate, velocity, color="#243b53", label="Observed", zorder=3)
    grid = np.logspace(
        np.log10(max(np.min(substrate[substrate > 0]), 1e-15)), np.log10(np.max(substrate)), 300
    )
    if fit.success:
        ax.plot(grid, michaelis_menten(grid, fit.vmax, fit.km), color="#d1495b", label="MM fit")
    ax.set_xscale("log")
    ax.set(
        xlabel="Substrate concentration (M)", ylabel="Velocity (M/s)", title="Michaelis-Menten fit"
    )
    ax.legend()
    return _save(fig, path)


def plot_bootstrap_distributions(result: BootstrapResult, path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    if result.samples.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No successful fits", ha="center", va="center")
        return _save(fig, path)
    sns.histplot(result.samples["Km"], kde=True, ax=axes[0], color="#2a9d8f")
    sns.histplot(result.samples["Vmax"], kde=True, ax=axes[1], color="#e9c46a")
    axes[2].scatter(result.samples["Km"], result.samples["Vmax"], s=12, alpha=0.45)
    axes[0].set_title("Bootstrap Km")
    axes[1].set_title("Bootstrap Vmax")
    axes[2].set(xlabel="Km", ylabel="Vmax", title="Parameter correlation")
    return _save(fig, path)


def plot_model_comparison(metrics: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ordered = metrics.sort_values("rmse")
    sns.barplot(data=ordered, x="rmse", y="model", ax=ax, color="#457b9d")
    ax.set(
        xlabel="Validation RMSE (log10 units)", ylabel="", title="Grouped ML baseline comparison"
    )
    return _save(fig, path)


def plot_predictions(
    y_true: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    path: str | Path,
    *,
    confidence_level: float = 0.95,
) -> Path:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    interval_label = f"Calibrated {100 * confidence_level:g}% interval"
    order = np.argsort(y_true)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(order))
    finite = np.isfinite(lower[order]) & np.isfinite(upper[order])
    if finite.any():
        ax.fill_between(
            x,
            np.where(finite, lower[order], np.nan),
            np.where(finite, upper[order], np.nan),
            alpha=0.25,
            color="#457b9d",
            label=interval_label,
        )
    if not finite.all():
        ax.text(
            0.02,
            0.98,
            f"{interval_label}: unbounded for {(~finite).sum()}/{len(order)} assays",
            transform=ax.transAxes,
            va="top",
        )
    ax.plot(x, prediction[order], "o", color="#1d3557", label="Prediction", markersize=4)
    ax.plot(x, y_true[order], "x", color="#e63946", label="Observed")
    ax.set(
        xlabel="Test assays ordered by target",
        ylabel="log10 target",
        title="Grouped test predictions",
    )
    ax.legend()
    return _save(fig, path)
