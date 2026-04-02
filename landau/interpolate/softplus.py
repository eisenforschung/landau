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

        # 2. Numerically stable softplus
        def softplus(t):
            t = np.asarray(t, float)
            return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)

        # 3. Build model with n softplus terms
        def model(xn, *params):
            assert len(params) == 3 * self.n_softplus + 1
            out = 0
            for i in range(self.n_softplus):
                a = params[3*i]
                b = params[3*i + 1]
                c = params[3*i + 2]
                out += a * softplus(b * (xn + c))
            out += params[-1]  # vertical offset
            return out

        # 4. Smart initial parameter guesses
        amplitude = 0.5 * (np.max(y) - np.min(y))
        offset = np.median(y)
        p0 = []
        for i in range(self.n_softplus):
            # Spread the "knees" over range
            frac = (i + 0.5) / self.n_softplus
            knee = np.quantile(xn, frac)
            a_guess = amplitude / self.n_softplus
            b_guess = 3.0 if (i % 2 == 0) else -3.0  # alternate increasing/decreasing
            p0 += [a_guess, b_guess, -knee]
        p0 += [offset]

        # Reasonable bounds
        lb = []
        ub = []
        for i in range(self.n_softplus):
            # lb += [-np.inf, -20.0, -10.0]
            lb += [0.0, -20.0, -10.0]
            ub += [ np.inf,  20.0,  10.0]
        lb += [-np.inf]
        ub += [ np.inf]

        # 5. Fit
        def resid(p):
            return model(xn, *p) - y

        res = least_squares(
            resid, p0, bounds=(lb, ub), loss=self.loss, max_nfev=self.max_nfev,
        )
        popt = res.x

        # 6. Return a prediction function (accepts original x, handles normalization)
        def predictor(x_new):
            x_new = np.asarray(x_new, float)
            xn_new = (x_new - xm) / xs
            return model(xn_new, *popt)

        return predictor
