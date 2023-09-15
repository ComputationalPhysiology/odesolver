import numpy as np
import time

from odesolver.solver import solve_ivp


def errornorm(y, y_exact):
    return np.linalg.norm(y - y_exact) / np.linalg.norm(y_exact)


def max_log_error(y, y_exact):
    return np.max(np.log2(np.abs(y - y_exact)))


def test_exponential_decay_exact():
    f = lambda t, u: u

    u0 = 1.0
    res = solve_ivp(f, t_span=(0.0, 3.0), y0=u0, method="heun")
    u_exact = u0 * np.exp(res.t)
    assert errornorm(res.y, u_exact) < 0.08


def test_convergence_exponential_decay():
    f = lambda t, u: u

    u0 = 1.0
    T = 3
    N = 50

    res = solve_ivp(f, t_span=(0.0, T), y0=u0, N=N, method="heun")
    prev_err = max_log_error(res.y, u0 * np.exp(res.t))

    order = []
    for factor in range(1, 5):
        res = solve_ivp(f, t_span=(0.0, T), y0=u0, N=pow(2, factor) * N, method="heun")
        err = max_log_error(res.y, u0 * np.exp(res.t))
        order.append(prev_err - err)
        prev_err = err

    breakpoint()
