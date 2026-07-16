import numpy as np
import pytest

from mm_kinetics.kinetics.simulation import simulate_mechanistic


def test_mechanistic_conservation_and_nonnegativity():
    simulation = simulate_mechanistic(
        1000000.0, 100.0, 50.0, 1e-06, 0.0002, np.linspace(0, 0.2, 300)
    )
    np.testing.assert_allclose(
        simulation.enzyme + simulation.complex, 1e-06, rtol=1e-07, atol=1e-12
    )
    np.testing.assert_allclose(
        simulation.substrate + simulation.complex + simulation.product,
        0.0002,
        rtol=1e-07,
        atol=1e-11,
    )
    assert (
        min(
            simulation.enzyme.min(),
            simulation.complex.min(),
            simulation.substrate.min(),
            simulation.product.min(),
        )
        >= 0
    )


def test_invalid_time_grid_is_rejected():
    with pytest.raises(ValueError):
        simulate_mechanistic(1000000.0, 100, 50, 1e-06, 0.0001, [0, 1, 1])
