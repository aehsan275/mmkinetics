import numpy as np
import pytest

from mm_kinetics.kinetics.equations import michaelis_menten
from mm_kinetics.kinetics.fitting import (
    fit_michaelis_menten,
    fit_michaelis_menten_weighted,
)


def _known_curve(seed=4):
    rng = np.random.default_rng(seed)
    km = 0.0002
    vmax = 4e-06
    substrate = km * np.logspace(-2, 2, 14)
    truth = michaelis_menten(substrate, vmax, km)
    sigma = np.maximum(0.03 * truth, 1e-10)
    observed = np.clip(truth + rng.normal(0, sigma), 0, None)
    return (substrate, observed, sigma, km, vmax)


def test_fit_recovers_known_parameters():
    substrate, observed, sigma, km, vmax = _known_curve()
    fit = fit_michaelis_menten(substrate, observed)
    weighted = fit_michaelis_menten_weighted(substrate, observed, sigma)
    assert fit.success and weighted.success
    assert fit.km == pytest.approx(km, rel=0.2)
    assert fit.vmax == pytest.approx(vmax, rel=0.1)
    assert fit.residuals.shape == substrate.shape
    assert np.isfinite(fit.covariance).all()


def test_fit_requires_identifiable_design():
    with pytest.raises(ValueError, match="unique"):
        fit_michaelis_menten([1, 1, 1], [1, 2, 3])
