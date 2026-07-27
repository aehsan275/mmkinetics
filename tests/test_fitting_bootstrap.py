import numpy as np
import pytest

from mm_kinetics.bootstrap.kinetic import nonparametric_bootstrap, parametric_bootstrap
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


def test_bootstraps_are_reproducible_and_record_failures():
    substrate, observed, _, _, _ = _known_curve()
    first = nonparametric_bootstrap(substrate, observed, n_bootstrap=60, random_seed=7)
    second = nonparametric_bootstrap(substrate, observed, n_bootstrap=60, random_seed=7)
    assert first.n_success + first.n_failures == 60
    assert first.samples.equals(second.samples)
    assert (first.summary.ci_lower <= first.summary.ci_upper).all()
    assert np.isfinite(first.samples[["Km", "Vmax"]]).all().all()


def test_parametric_bootstrap_dimensions():
    substrate, observed, _, _, _ = _known_curve()
    result = parametric_bootstrap(substrate, observed, n_bootstrap=40, random_seed=9)
    assert result.n_requested == 40
    assert result.n_success > 30
    assert set(result.summary.index) == {"Km", "Vmax"}


def test_fit_requires_identifiable_design():
    with pytest.raises(ValueError, match="unique"):
        fit_michaelis_menten([1, 1, 1], [1, 2, 3])


@pytest.mark.parametrize("seed, successful", [(0, 0), (12, 1)])
def test_bootstrap_summary_handles_zero_or_one_success(seed, successful):
    substrate = np.array([1.0, 2.0, 3.0])
    velocity = michaelis_menten(substrate, 3.0, 2.0)
    result = nonparametric_bootstrap(substrate, velocity, n_bootstrap=1, random_seed=seed)
    assert result.n_success == successful
    assert result.n_failures == 1 - successful
    assert (result.summary["n"] == successful).all()
    if successful:
        assert (result.summary["std"] == 0).all()
        np.testing.assert_allclose(result.summary["ci_lower"], result.summary["ci_upper"])
    else:
        assert result.summary.drop(columns="n").isna().all().all()
