from typing import NamedTuple, Callable
import numba
import numpy as np
import numpy.typing as npt

_ECONVERGED = 0
_ECONVERR = -1

_iter = 100
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps


class RootResults(NamedTuple):
    root: float
    function_calls: int
    iterations: int
    converged: bool


def root(
    func: Callable[[npt.ArrayLike | float], npt.ArrayLike | float],
    x0: npt.ArrayLike | float,
    fprime: Callable[[npt.ArrayLike | float], npt.ArrayLike | float] | None = None,
    args: tuple[float, ...] = (),
    tol: float = 1.48e-8,
    maxiter: int = 50,
    disp: bool = True,
) -> RootResults:
    """_summary_

    Parameters
    ----------
    func : Callable[[npt.ArrayLike  |  float], npt.ArrayLike  |  float]
        _description_
    x0 : npt.ArrayLike | float
        _description_
    fprime : Callable[[npt.ArrayLike  |  float], npt.ArrayLike  |  float] | None, optional
        _description_, by default None
    args : tuple[float, ...], optional
        _description_, by default ()
    tol : float, optional
        _description_, by default 1.48e-8
    maxiter : int, optional
        _description_, by default 50
    disp : bool, optional
        _description_, by default True

    Returns
    -------
    RootResults
        _description_
    """
    if fprime is None:
        return secant(
            func=func,
            x0=x0,
            args=args,
            tol=tol,
            maxiter=maxiter,
            disp=disp,
        )
    else:
        return newton(
            func=func,
            x0=x0,
            fprime=fprime,
            args=args,
            tol=tol,
            maxiter=maxiter,
            disp=disp,
        )


def newton(
    func,
    x0,
    fprime,
    args=(),
    tol: float = 1.48e-8,
    maxiter: int = 50,
    disp: bool = True,
):
    """
    Find a zero from the Newton-Raphson method.

    Parameters
    ----------
    func : Callable[[npt.ArrayLike  |  float], npt.ArrayLike  |  float]
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x0 : npt.ArrayLike | float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    fprime : callable
        The derivative of the function (when available and convenient).
        If you don't have the derivate, use the secant method.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value.
    maxiter : int, optional
        Maximum number of iterations, by default 50
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge, by default True

    Returns
    -------
    results : tuple

    root - Estimated location where function is zero.
    function_calls - Number of times the function was called.
    iterations - Number of iterations needed to find the root.
    converged - True if the routine converged

    """
    if not numba.extending.is_jitted(func):
        func = numba.njit(func)
    return RootResults(
        *_newton(
            func=func,
            x0=x0,
            fprime=fprime,
            args=args,
            tol=tol,
            maxiter=maxiter,
            disp=disp,
        )
    )


@numba.njit
def _newton(
    func,
    x0,
    fprime,
    args=(),
    tol: float = 1.48e-8,
    maxiter: int = 50,
    disp: bool = True,
):
    if tol <= 0:
        raise ValueError("tol is too small <= 0")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    status = _ECONVERR

    # Newton-Raphson method
    for itr in range(maxiter):
        # first evaluate fval
        fval = func(p0, *args)
        funcalls += 1
        # If fval is 0, a root has been found, then terminate
        if fval == 0:
            status = _ECONVERGED
            p = p0
            itr -= 1
            break
        fder = fprime(p0, *args)
        funcalls += 1
        # derivative is zero, not converged
        if fder == 0:
            p = p0
            break
        newton_step = fval / fder
        # Newton step
        p = p0 - newton_step
        if abs(p - p0) < tol:
            status = _ECONVERGED
            break
        p0 = p

    if disp and status == _ECONVERR:
        msg = "Failed to converge"
        raise RuntimeError(msg)

    return (p, funcalls, itr + 1, status)


def secant(
    func, x0, args=(), tol: float = 1.48e-8, maxiter: int = 50, disp: bool = True
):
    if not numba.extending.is_jitted(func):
        func = numba.njit(func)
    return RootResults(
        *_secant(func=func, x0=x0, args=args, tol=tol, maxiter=maxiter, disp=disp)
    )


# @numba.njit
def _secant(
    func, x0, args=(), tol: float = 1.48e-8, maxiter: int = 50, disp: bool = True
):
    """
    Find a zero from the secant method.

    Parameters
    ----------
    func : callable
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    tol : float, optional(default=1.48e-8)
        The allowable error of the zero value.
    maxiter : int, optional(default=50)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge.

    Returns
    -------
    results :
            root - Estimated location where function is zero.
            function_calls - Number of times the function was called.
            iterations - Number of iterations needed to find the root.
            converged - True if the routine converged
    """

    if tol <= 0:
        raise ValueError("tol is too small <= 0")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    status = _ECONVERR

    # Secant method
    if x0 >= 0:
        p1 = x0 * (1 + 1e-4) + 1e-4
    else:
        p1 = x0 * (1 + 1e-4) - 1e-4
    q0 = func(p0, *args)
    funcalls += 1
    q1 = func(p1, *args)
    funcalls += 1
    for itr in range(maxiter):
        if q1 == q0:
            p = (p1 + p0) / 2.0
            status = _ECONVERGED
            break
        else:
            p = p1 - q1 * (p1 - p0) / (q1 - q0)
        if np.abs(p - p1) < tol:
            status = _ECONVERGED
            break
        p0 = p1
        q0 = q1
        p1 = p
        q1 = func(p1, *args)
        funcalls += 1

    if disp and status == _ECONVERR:
        msg = "Failed to converge"
        raise RuntimeError(msg)

    return (p, funcalls, itr + 1, status)
