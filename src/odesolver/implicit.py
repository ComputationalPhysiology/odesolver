# def solve_stages(self):
#         u, f, n, t = self.u, self.f, self.n, self.t
#         neq = self.neq
#         s = self.stages
#         k0 = f(t[n], u[n])
#         k0 = np.hstack([k0 for i in range(s)])

#         sol = root(self.stage_eq, k0)

#         return np.split(sol.x, s)

#     def stage_eq(self, k_all):
#         a, c = self.a, self.c
#         s, neq = self.stages, self.neq

#         u, f, n, t = self.u, self.f, self.n, self.t
#         dt = self.dt

#         res = np.zeros_like(k_all)
#         k = np.split(k_all, s)
#         for i in range(s):
#             fi = f(t[n] + c[i] * dt, u[n] + dt *
#                    sum([a[i, j] * k[j] for j in range(s)]))
#             res[i * neq:(i + 1) * neq] = k[i] - fi

#         return res


def forward(double *y, double t, double dt)
{

    assert(_ode);

    uint i;
    bool step_ok;

    const double t_end = t + dt;
    const double ldt_0 = _ldt;
    double ldt = ldt_0 > 0 ? ldt_0 : dt;
    int num_refinements = 0;

    // A way to check if we are at t_end.
    const double eps = GOSS_EPS * 1000;

    while (true)
    {

        // Use 0.0 as initial guess
        for (i = 0; i < num_states(); ++i)
            _z1[i] = 0.0;

        // Check if we should re compute the jacobian
        if (num_refinements > num_refinements_without_always_recomputing_jacobian)
            always_recompute_jacobian = true;

        // Solve for increment
        step_ok = newton_solve(_z1.data(), _prev.data(), y, t + ldt, ldt, 1.0, always_recompute_jacobian);

        // Newton step OK
        if (step_ok)
        {

            // Add increment
            for (i = 0; i < num_states(); ++i)
                y[i] += _z1[i];

            t += ldt;

            // Check if we are finished
            if (std::fabs(t - t_end) < eps)
                break;

            // If the solver has refined, we do not allow it to double its
            // timestep for another step
            if (!_justrefined)
            {
                // double time step
                const double tmp = 2.0 * ldt;
                if (ldt_0 > 0. && tmp >= ldt_0)
                {
                    ldt = ldt_0;
                }
                else
                {
                    ldt = tmp;
                    // log(DBG, "Changing dt    | t : %g, from %g to %g", t, tmp / 2, ldt);
                }
            }
            else
            {
                _justrefined = false;
            }

            // If we are passed t_end
            if ((t + ldt + GOSS_EPS) > t_end)
            {
                ldt = t_end - t;
                // log(DBG, "Changing ldt   | t : %g, to adapt for dt end: %g", t, ldt);
            }
        }
        else
        {
            ldt /= 2.0;
            if (ldt < min_dt)
            {
                // goss_error("ImplicitEuler.cpp", "Forward ImplicitEuler",
                //            "Newtons solver failed to converge as dt become smaller "
                //            "than \"min_dt\" %e",
                //            min_dt);
                std::stringstream s;
                s << "Newtons solver failed to converge as dt become smaller than 'min_dt' " << min_dt << std::endl;
                throw std::runtime_error(s.str());
            }

            // log(DBG, "Reducing dt    | t : %g, new: %g", t, ldt);
            _recompute_jacobian = true;
            _justrefined = true;
            num_refinements += 1;
        }
    }
