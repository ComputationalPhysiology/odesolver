import numba
import numpy as np
from odesolver import roots


def test_secant_scalar():
    f = numba.njit(lambda x: x**3 - 1)
    res = roots.secant(f, x0=0.5)
    assert np.isclose(res.root, 1.0)


def test_newton_scalar():
    f = numba.njit(lambda x: x**3 - 1)
    df = numba.njit(lambda x: 3 * x**2)

    res = roots.newton(f, x0=0.5, fprime=df)
    assert np.isclose(res.root, 1.0)
