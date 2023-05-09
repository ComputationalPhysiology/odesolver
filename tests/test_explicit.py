import numpy as np
import pytest

from odesolver.solver import solve_ivp


def errornorm(y, y_exact):
    return np.linalg.norm(y - y_exact) / np.linalg.norm(y_exact)


def max_log_error(y, y_exact):
    return np.max(np.log2(np.abs(y - y_exact)))


@pytest.mark.parametrize(
    "method, tol",
    [
        ("forward_euler", 0.08),
        ("heun", 0.0015),
        ("explicit_midpoint", 0.015),
    ],
)
def test_exponential_decay_exact(method, tol):
    f = lambda t, u: u

    u0 = 1.0
    res = solve_ivp(f, t_span=(0.0, 3.0), y0=u0, method=method)
    u_exact = u0 * np.exp(res.t)
    assert errornorm(res.y, u_exact) < tol


@pytest.mark.parametrize(
    "method, order",
    [
        ("forward_euler", 1),
        ("heun", 2),
        ("explicit_midpoint", 2),
    ],
)
def test_convergence_exponential_decay(method, order):
    f = lambda t, u: u

    u0 = 1.0
    T = 3
    N = 50

    res = solve_ivp(f, t_span=(0.0, T), y0=u0, N=N, method=method)
    prev_err = max_log_error(res.y, u0 * np.exp(res.t))

    orders = []
    for factor in range(1, 5):
        res = solve_ivp(f, t_span=(0.0, T), y0=u0, N=pow(2, factor) * N, method=method)
        err = max_log_error(res.y, u0 * np.exp(res.t))
        orders.append(prev_err - err)
        prev_err = err

    assert np.allclose(orders, order, atol=0.06)
