from typing import Callable
import numpy as np
import numba

float_2D_array = numba.types.Array(numba.float64, 2, "C")
float_1D_array = numba.types.Array(numba.float64, 1, "C")
# odefunc = numba.types.Callable()


_EPS = 1e-12


@numba.njit(numba.void(float_2D_array))
def _lu_factorize(A):
    lA = len(A)
    for k in range(0, lA - 1):
        A[k + 1 :, k] = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k + 1 :] -= np.outer(A[k + 1 :, k], A[k, k + 1 :])


@numba.njit()
def _compute_jacobian(fun, states, time, jac):
    num_states = len(states)
    _f1 = fun(time, states)

    for i in range(num_states):
        ysafe = states[i]
        max = 1e-5 if 1e-5 > abs(ysafe) else abs(ysafe)
        delta = np.sqrt(1e-15 * max)
        states[i] += delta
        _f2 = fun(time, states)

        for j in range(num_states):
            jac[j, i] = (_f2[j] - _f1[j]) / delta

        states[i] = ysafe


def compute_jacobian(
    fun: Callable[[float, np.ndarray], np.ndarray], states: np.ndarray, time: float
) -> np.ndarray:
    N = len(states)
    jac = np.zeros((N, N))
    if not numba.extending.is_jitted(fun):
        fun = numba.njit(fun)
    _compute_jacobian(fun, states, time, jac)
    return jac


@numba.njit(numba.void(float_2D_array, float_1D_array, float_1D_array))
def _forward_backward_subst(A, b, x):
    # solves Ax = b with forward backward substitution, provided that
    # A is already LU factorized
    num_states = len(b)
    x[0] = b[0]

    for i in range(num_states):
        sum = 0.0
        for j in range(i):
            sum = sum + A[i, j] * x[j]

        x[i] = b[i] - sum

    num_minus_1 = num_states - 1
    x[num_minus_1] = x[num_minus_1] / A[num_minus_1, num_minus_1]

    for i in range(num_states - 2, -1, -1):
        sum = 0
        for j in range(i + 1, num_states):
            sum = sum + A[i, j] * x[j]

        x[i] = (x[i] - sum) / A[i, i]


def forward_backward_subst(A, b):
    x = np.zeros_like(b)
    _forward_backward_subst(A, b, x)
    return x


@numba.njit
def _compute_factorized_jacobian(fun, y, t, dt, alpha, jac):
    # Let ODE compute the jacobian
    _compute_jacobian(fun=fun, states=y, time=t, jac=jac)

    # Build scaled discretization of jacobian
    scale = -dt * alpha
    jac *= scale

    # Add mass matrix
    jac += np.diag(y)

    # // Factorize the jacobian
    _lu_factorize(jac)
    # _jac_comp += 1;


def compute_factorized_jacobian(fun, y, t, dt, alpha):
    N = len(y)
    jac = np.zeros((N, N))
    if not numba.extending.is_jitted(fun):
        fun = numba.njit(fun)
    _compute_factorized_jacobian(fun, y, t, dt, alpha, jac)
    return jac


# //-----------------------------------------------------------------------------
def newton_solve(
    fun,
    z,
    prev,
    y0,
    t,
    dt,
    alpha,
    states,
    jac,
    always_recompute_jacobian=False,
    eta=1.0,
    kappa=0.1,
    relative_tolerance=1e-12,
    max_iterations=30,
    max_relative_previous_residual=0.01,
):
    num_states = len(states)
    step_ok = True
    _newton_iterations = 0
    relative_previous_residual = 1.0
    relative_residual = 1.0
    previous_residual = 1.0
    initial_residual = 1.0
    _recompute_jacobian = True
    _rejects = 0
    # residual
    _yz = np.zeros_like(z)
    _b = np.zeros_like(z)
    _dz = np.zeros_like(z)

    while eta * relative_residual >= kappa * relative_tolerance:
        for i in range(num_states):
            _yz[i] = y0[i] + z[i]

            # Evaluate ODE using local solution
            _f1 = fun(t, _yz)

        # Build rhs for linear solve
        # z = y-y0
        #  prev is a linear combination of previous stage solutions
        for i in range(num_states):
            _b[i] = -z[i] * states[i] + dt * (prev[i] + alpha * _f1[i])

        # Calculate the residual
        residual = np.linalg.norm(_b)

        # Check for relative residual convergence
        if relative_residual < relative_tolerance:
            break

        # Recompute jacobian if necessary
        if _recompute_jacobian or always_recompute_jacobian:
            _compute_factorized_jacobian(
                fun=fun, y=_yz, t=t, dt=dt, alpha=alpha, jac=jac
            )
            _recompute_jacobian = False

        # Linear solve on factorized jacobian
        _forward_backward_subst(jac, _b, _dz)

        # _Newton_Iterations == 0
        if _newton_iterations == 0:
            initial_residual = residual

            # On first iteration we need an approximation of eta. We take
            # the one from previous step and increase it slightly. This is
            # important for linear problems which only should require 1
            # iteration to converge.
            _eta = _eta if _eta > _EPS else _EPS
            _eta = pow(_eta, 0.8)

        # 2nd time around
        else:
            # How fast are we converging?
            relative_previous_residual = residual / previous_residual

            # If too slow we flag the jacobian to be recomputed
            _recompute_jacobian = (
                relative_previous_residual >= max_relative_previous_residual
            )

            # If we diverge
            if relative_previous_residual >= 1:
                # log(DBG,
                #     "Diverges       | t : %g, it : %2d, relative_previous_residual: %f, "
                #     "relative_residual: %g. Reducing time step and recompute jacobian.",
                #     t, _newton_iterations, relative_previous_residual, relative_residual);
                step_ok = False
                _rejects += 1
                _recompute_jacobian = True
                break

            scaled_relative_previous_residual = np.max(
                pow(relative_previous_residual, max_iterations - _newton_iterations),
                _EPS,
            )
            # We converge too slow
            if residual > (
                kappa
                * relative_tolerance
                * (1 - relative_previous_residual)
                / scaled_relative_previous_residual
            ):
                # log(DBG,
                #     "To slow        | t : %g, it: %2d, relative_previous_residual: "
                #     "%f, relative_residual: %g. Recomputing Jacobian.",
                #     t, _newton_iterations, relative_previous_residual, relative_residual);
                _recompute_jacobian = True

            _eta = relative_previous_residual / (1.0 - relative_previous_residual)

        # No convergence
        if _newton_iterations > max_iterations:
            # log(DBG,
            #     "Max iterations | t : %g, it: %2d, relative_previous_residual: "
            #     "%f, relative_residual: %g. Recomputing Jacobian.",
            #     t, _newton_iterations, relative_previous_residual, relative_residual);
            _recompute_jacobian = True
            _rejects += 1
            step_ok = False
            break

        # Update local increment solution
        for i in range(num_states):
            z[i] += _dz[i]

        # Update residuals
        relative_residual = residual / initial_residual
        previous_residual = residual
        _newton_iterations += 1

        # log(5,
        #     "Monitor        | t : %g, it : %2d, relative_previous_residual: %f, "
        #     "relative_residual: %g.",
        #     t, _newton_iterations, relative_previous_residual, relative_residual);
        # eta*residual is the iteration error and an estimation of the
        # local discretization error.

    # goss_debug1("Newton converged in %d iterations.", _newton_iterations);

    return step_ok
