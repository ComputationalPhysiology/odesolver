import numpy as np
import scipy
import odesolver
import numba

import sympy as sym


def vander_pool(t, y):
    mu = 1.0
    return np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])


def vdp(t, y, mu):
    return np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])


# def test_lu_factorize():
#     np.random.seed(1)
#     A = np.random.random((4, 4))

#     lu, piv = scipy.linalg.lu_factor(A)
#     odesolver.newton.lu_factorize(A)

#     breakpoint()


def test_compute_jacobian():
    y = y0, y1 = sym.symbols("y0 y1")
    mu = sym.symbols("mu")

    J = sym.Matrix(vdp(0.0, y, mu)).jacobian(y)

    states = np.array([0.3, 0.2])
    time = 0.0

    vdp_jit = numba.njit(vdp)
    fun = numba.njit(lambda t, y: vdp_jit(t, y, 0.4))

    jac = odesolver.newton.compute_jacobian(
        fun=fun,
        states=states,
        time=time,
    )
    J_sub = J.subs({"y0": states[0], "y1": states[1], "mu": 0.4})
    assert np.allclose(jac, np.array(J_sub).astype(np.float64))


def test_compute_factorized_jacobian():
    y = y0, y1 = sym.symbols("y0 y1")
    mu = sym.symbols("mu")

    J = sym.Matrix(vdp(0.0, y, mu)).jacobian(y)

    states = np.array([0.3, 0.2])
    time = 0.0

    vdp_jit = numba.njit(vdp)
    fun = numba.njit(lambda t, y: vdp_jit(t, y, 0.4))
    jac = odesolver.newton.compute_factorized_jacobian(
        fun=fun, y=states, t=time, dt=0.1, alpha=0.1
    )
    J_sub = J.subs({"y0": states[0], "y1": states[1], "mu": 0.4})
    breakpoint()
