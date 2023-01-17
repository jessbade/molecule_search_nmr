import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.optimize import minimize


def matching_score(query, shifts, tau=0.05, h=1, eps=0.01, thr=10, alpha=0.05):
    mw = MatchingWrapper(**kwargs)
    mw.set_query(query)
    mw.set_shifts(shifts)
    mw.query_preprocess()
    mw.align()
    mw.optimize()
    return mw.score


class MatchingWrapper:
    """ """

    def __init__(self, **kwargs):
        _defaults = ("query", "shifts", "tau", "h", "eps", "threshold", "alpha")
        _default_value = None
        self.__dict__.update(dict.fromkeys(_defaults, _default_value))
        self.__dict__.update(**kwargs)

    def set_query(self, query):
        if not self.__dict__.get("query"):
            self.query = query

    def set_shifts(self, shifts):
        if not self.__dict__.get("shifts"):
            self.shifts = shifts
            self.n_shifts = len(self.shifts)

    def set_parameters(self, tau=0.05, h=1, eps=0.01, thr=10, alpha=0.05):
        self.__dict__.update(**kwargs)

    def query_preprocess(self):
        self.query_x_vals = self.query[:, 0]
        self.query_y_vals = self.query[:, 1]
        self.query_x_peaks = self.query_x_vals[self.query_y_vals >= self.tau]

    def align(self):
        shifts_aligned = np.array(
            [
                self.query_x_peaks[np.argmin(np.abs(self.query_x_peaks - s))]
                for s in self.shifts
            ]
        )
        shifts_aligned[0] = np.min(self.query_x_peaks)
        shifts_aligned[-1] = np.max(self.query_x_peaks)
        self.shifts_aligned = shifts_aligned

    def cos_sim(self, a, b):
        return dot(a, b) / (norm(a) * norm(b))

    def sum_sq_diff(self, a, b):
        return np.sum(np.square(a / np.sum(a) - b / np.sum(b)))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def kernel_f(self, z, m):

        gaussian = np.exp(-4 * np.log(2) * (z**2))
        lorentzian = 1 / (1 + 4 * (z**2))

        return (1 - m) * gaussian + m * lorentzian

    def estimate(self, x, mu, sigma, m):
        return np.sum(
            [
                self.kernel_f((x - mu[j] - self.shifts_aligned[j]) / sigma[j], m[j])
                for j in range(self.n_shifts)
            ],
            0,
        )

    def opt_fun(self, var):

        mu = var[: self.n_shifts]
        sigma = np.exp(var[self.n_shifts : 2 * self.n_shifts])
        m = self.sigmoid(var[2 * self.n_shifts :])

        estimated_y_vals = self.estimate(self.query_x_vals, mu, sigma, m)

        obj = (
            -self.cos_sim(self.query_y_vals, estimated_y_vals)
            + self.sum_sq_diff(self.query_y_vals, estimated_y_vals)
            + np.sum(np.square(mu))
            + np.sum(np.square(sigma))
            + np.sum(
                np.clip(
                    eps
                    + mu[:-1]
                    + self.shifts_aligned[:-1]
                    - mu[1:]
                    - self.shifts_aligned[1:],
                    0,
                    100,
                )
                ** 2
            )
        )

        return obj

    def get_spect(self, var, x_vals):
        return self.estimate(
            x_vals,
            var[: self.n_shifts],
            np.exp(var[self.n_shifts : 2 * self.n_shifts]),
            self.sigmoid(var[2 * self.n_shifts :]),
        )

    def optimize(self):
        if (
            np.max(np.abs(self.shifts_aligned - self.shifts)) > self.thr
            or np.max(
                [np.min(np.abs(self.shifts_aligned - x)) for x in self.query_x_peaks]
            )
            > self.thr
        ):

            self.score = -100

        else:

            x0 = np.concatenate(
                [
                    np.zeros(self.n_shifts),
                    np.log(self.h) * np.ones(self.n_shifts),
                    np.zeros(self.n_shifts),
                ],
                0,
            )
            opt_res = minimize(self.opt_fun, x0, method="L-BFGS-B")
            shifts_optimized = opt_res.x[: self.n_shifts] + self.shifts_aligned
            estimated_y_vals = self.get_spect(opt_res.x, self.query_x_vals)

            self.score = self.cos_sim(
                self.query_y_vals, estimated_y_vals
            ) - self.alpha * norm(shifts_optimized - self.shifts)
