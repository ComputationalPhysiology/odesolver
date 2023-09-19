class ODESolver:
    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    def __init__(self, fun, t0, y0, t_bound):
        self.t_old = None
        self.t = t0
        # self._fun, self.y = check_arguments(fun, y0, support_complex)
        # TODO: check arguments
        self._fun = fun
        self.y = y0
        self.t_bound = t_bound
        self.fun = fun

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.num_points, self.num_states = self.y.shape

        self.status = "running"

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    @property
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            return np.abs(self.t - self.t_old)

    def step(self):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != "running":
            raise RuntimeError("Attempt to step on a failed or finished " "solver.")

        if self.num_states == 0 or self.num_points == 0 or self.t == self.t_bound:
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = "finished"
        else:
            t = self.t
            success, message = self._step_impl()

            if not success:
                self.status = "failed"
            else:
                self.t_old = t
                if self.direction * (self.t - self.t_bound) >= 0:
                    self.status = "finished"

        return message

    # def dense_output(self):
    #     """Compute a local interpolant over the last successful step.

    #     Returns
    #     -------
    #     sol : `DenseOutput`
    #         Local interpolant over the last successful step.
    #     """
    #     if self.t_old is None:
    #         raise RuntimeError("Dense output is available after a successful "
    #                            "step was made.")

    #     if self.n == 0 or self.t == self.t_old:
    #         # Handle corner cases of empty solver and no integration.
    #         return ConstantDenseOutput(self.t_old, self.t, self.y)
    #     else:
    #         return self._dense_output_impl()

    def _step_impl(self):
        raise NotImplementedError

    # def _dense_output_impl(self):
    #     raise NotImplementedError
