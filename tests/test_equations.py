import numpy as np
import pytest

from mm_kinetics.kinetics.equations import michaelis_menten, michaelis_menten_kcat


def test_michaelis_menten_analytical_limits():
    vmax = 8.0
    km = 2.0
    assert michaelis_menten(0.0, vmax, km) == 0.0
    assert michaelis_menten(km, vmax, km) == pytest.approx(vmax / 2)
    assert michaelis_menten(100000000.0 * km, vmax, km) == pytest.approx(vmax, rel=1e-07)
    assert michaelis_menten(1e-08 * km, vmax, km) == pytest.approx(vmax * 1e-08, rel=1e-07)


def test_broadcasting_and_kcat_conversion():
    substrate = np.array([0.0, 1.0, 2.0])
    rates = michaelis_menten(substrate, np.array([[2.0], [4.0]]), 1.0)
    assert rates.shape == (2, 3)
    expected = michaelis_menten_kcat(substrate, enzyme_total=0.5, kcat=4.0, km=1.0)
    np.testing.assert_allclose(expected, michaelis_menten(substrate, 2.0, 1.0))


@pytest.mark.parametrize("args", [([-1.0], 1.0, 1.0), ([1.0], -1.0, 1.0), ([1.0], 1.0, 0.0)])
def test_invalid_physical_domain_is_rejected(args):
    with pytest.raises(ValueError):
        michaelis_menten(*args)
