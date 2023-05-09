from typing import NamedTuple, Callable, Tuple, Literal
import numpy.typing as npt
import numpy as np
import numba

from . import explicit


class ODEResults(NamedTuple):
    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]


ExplicitMethods = Literal["forward_euler", "heun", "explicit_midpoint"]
ImplicitMethods = Literal["backward_euler"]
Methods = Literal[ExplicitMethods, ImplicitMethods]


def solve_ivp(
    fun: Callable[
        [float, npt.NDArray[np.float64] | float], npt.NDArray[np.float64] | float
    ],
    t_span: Tuple[float, float],
    y0: npt.NDArray[np.float64],
    method: Methods = "forward_euler",
    N: int = 50,
):
    t = np.zeros(N + 1)
    y = np.zeros(N + 1)
    t0, T = t_span
    dt = (T - t0) / N

    if not numba.extending.is_jitted(fun):
        fun = numba.njit(fun)

    if method == "forward_euler":
        advance = explicit.forward_euler
    elif method == "heun":
        advance = explicit.heun
    elif method == "explicit_midpoint":
        advance = explicit.explicit_midpoint
    else:
        raise NotImplementedError

    explicit.solve_explicit(
        fun=fun, y0=y0, N=N, t=t, y=y, dt=dt, t0=t0, advance=advance
    )
    return ODEResults(t=t, y=y)
