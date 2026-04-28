from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from .basic import ConcentrationInterpolator, TemperatureInterpolator


@dataclass(frozen=True, eq=True)
class SoftplusFit(ConcentrationInterpolator, TemperatureInterpolator):
    """
    Fits data using a sum of softplus functions for smooth, flexible interpolation.

    The softplus function provides numerically stable, smooth approximations suitable
    for thermodynamic data. This interpolator uses multiple softplus terms to capture
    complex behavior.
    """

    n_softplus: int = 2
    """Number of softplus terms to fit."""
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    """Loss function for robust fitting."""
    max_nfev: int = 100
    """Maximum number of function evaluations."""

    def fit(self, x, y) -> Callable[[float], float]:
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        # 1. Normalize x for better conditioning
        xm = x.mean()
        xs = x.std() if x.std() > 0 else 1.0
        xn = (x - xm) / xs

        # 2. Numerically stable softplus and its derivative (sigmoid)
        def softplus(t):
            t = np.asarray(t, float)
            return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)

        def sigmoid(t):
            # Numerically stable: avoid overflow in exp for large |t|.
            t = np.asarray(t, float)
            out = np.empty_like(t)
            pos = t >= 0
            neg = ~pos
            out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
            ex = np.exp(t[neg])
            out[neg] = ex / (1.0 + ex)
            return out

        # 3. Build model with n softplus terms
        n = self.n_softplus

        def model(xn, *params):
            assert len(params) == 3 * n + 1
            out = 0.0
            for i in range(n):
                a = params[3*i]
                b = params[3*i + 1]
                c = params[3*i + 2]
                out = out + a * softplus(b * (xn + c))
            out = out + params[-1]  # vertical offset
            return out

        # 4. Smart initial parameter guesses
        amplitude = 0.5 * (np.max(y) - np.min(y))
        offset = np.median(y)
        p0 = []
        for i in range(n):
            # Spread the "knees" over range
            frac = (i + 0.5) / n
            knee = np.quantile(xn, frac)
            a_guess = amplitude / n
            b_guess = 3.0 if (i % 2 == 0) else -3.0  # alternate increasing/decreasing
            p0 += [a_guess, b_guess, -knee]
        p0 += [offset]

        # Reasonable bounds
        lb = []
        ub = []
        for i in range(n):
            # lb += [-np.inf, -20.0, -10.0]
            lb += [0.0, -20.0, -10.0]
            ub += [ np.inf,  20.0,  10.0]
        lb += [-np.inf]
        ub += [ np.inf]

        # 5. Fit, with analytical Jacobian of the residuals w.r.t. parameters.
        #
        # For a single term  a * softplus(b * (xn + c)),  letting t = b*(xn+c):
        #   d/da    = softplus(t)
        #   d/db    = a * sigmoid(t) * (xn + c)
        #   d/dc    = a * sigmoid(t) * b
        # The vertical offset contributes a column of ones.
        def resid(p):
            return model(xn, *p) - y

        def jac(p):
            J = np.empty((xn.size, 3 * n + 1), dtype=float)
            for i in range(n):
                a = p[3*i]
                b = p[3*i + 1]
                c = p[3*i + 2]
                u = xn + c
                t = b * u
                sig = sigmoid(t)
                J[:, 3*i]     = softplus(t)
                J[:, 3*i + 1] = a * sig * u
                J[:, 3*i + 2] = a * sig * b
            J[:, -1] = 1.0
            return J

        res = least_squares(
            resid, p0, jac=jac, bounds=(lb, ub),
            loss=self.loss, max_nfev=self.max_nfev,
        )
        popt = res.x

        # 6. Return a prediction function (accepts original x, handles normalization)
        def predictor(x_new):
            x_new = np.asarray(x_new, float)
            xn_new = (x_new - xm) / xs
            return model(xn_new, *popt)

        return predictor
