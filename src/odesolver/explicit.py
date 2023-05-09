import numba


@numba.njit
def forward_euler(f, y, t, dt):
    return y + dt * f(t, y)


@numba.njit
def heun(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t + dt, y + dt * k1)
    return y + dt / 2 * (k1 + k2)


@numba.njit
def explicit_midpoint(f, y, t, dt):
    dt2 = dt / 2.0
    k1 = f(t, y)
    k2 = f(t + dt2, y + dt2 * k1)
    return y + dt * k2


@numba.njit
def solve_explicit(fun, y0, N, t, y, dt, t0, advance):
    y[0] = y0
    t[0] = t0

    for n in range(N):
        t[n + 1] = t[n] + dt
        y[n + 1] = advance(fun, y[n], t[n], dt)
