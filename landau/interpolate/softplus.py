from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .basic import ConcentrationInterpolator, TemperatureInterpolator


def _softplus(t):
    t = np.asarray(t, float)
    return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)


def _sigmoid(t):
    # Numerically stable: avoid overflow in exp for large |t|.
    t = np.asarray(t, float)
    out = np.empty_like(t)
    pos = t >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    ex = np.exp(t[neg])
    out[neg] = ex / (1.0 + ex)
    return out


@dataclass(frozen=True, eq=True)
class SoftplusFit(ConcentrationInterpolator, TemperatureInterpolator):
    """
    Fits data using a sum of softplus functions for smooth, flexible interpolation.

    The softplus function provides numerically stable, smooth approximations suitable
    for thermodynamic data. This interpolator uses multiple softplus terms to capture
    complex behavior.

    The default :meth:`.fit` runs a local Levenberg–Marquardt / trust-region fit
    with analytical Jacobian.  For data where the local fit gets stuck in a
    near-constant local minimum (which can happen with the default ``soft_l1``
    loss when residuals at the initial guess are large), call :meth:`.global_fit`
    instead — it uses scipy's :func:`~scipy.optimize.differential_evolution`
    to find a basin and polishes with the local fit.
    """

    n_softplus: int = 2
    """Number of softplus terms to fit."""
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    """Loss function for robust fitting."""
    max_nfev: int = 100
    """Maximum number of function evaluations for the local fit."""
    f_scale: float = 1.0
    """Soft margin between inlier and outlier residuals for the robust loss
    (see :func:`scipy.optimize.least_squares`).  Larger values weaken the
    influence of the robust loss — at the limit it behaves quadratically over
    the full residual range, equivalent to ``loss='linear'``.  Has no effect
    when ``loss='linear'``."""

    def _prepare(self, x, y):
        """Set up normalization, model, Jacobian, bounds and initial guess.

        Returns ``(xn, y, model, jac, p0, lb, ub, xm, xs)``.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        xm = x.mean()
        xs = x.std() if x.std() > 0 else 1.0
        xn = (x - xm) / xs

        n = self.n_softplus

        def model(xn_, p):
            out = np.full_like(xn_, p[-1], dtype=float)
            for i in range(n):
                a = p[3 * i]
                b = p[3 * i + 1]
                c = p[3 * i + 2]
                out = out + a * _softplus(b * (xn_ + c))
            return out

        # Analytical Jacobian of residuals w.r.t. parameters.
        # For one term  a * softplus(b*(xn+c)), with t = b*(xn+c):
        #   d/da = softplus(t)
        #   d/db = a * sigmoid(t) * (xn + c)
        #   d/dc = a * sigmoid(t) * b
        # The vertical offset contributes a column of ones.
        def jac(p):
            J = np.empty((xn.size, 3 * n + 1), dtype=float)
            for i in range(n):
                a = p[3 * i]
                b = p[3 * i + 1]
                c = p[3 * i + 2]
                u = xn + c
                t = b * u
                sig = _sigmoid(t)
                J[:, 3 * i] = _softplus(t)
                J[:, 3 * i + 1] = a * sig * u
                J[:, 3 * i + 2] = a * sig * b
            J[:, -1] = 1.0
            return J

        # Initial guess.  ``a_guess`` is set to the full peak-to-peak range of
        # ``y`` rather than ``ptp/(2*n)``: starting with amplitudes that are
        # large enough to actually reach the data prevents the optimizer from
        # collapsing into the wide near-constant basin that ``soft_l1`` opens
        # up around small ``a`` (see PR #82).  The bound on ``a`` is open
        # above, so over-estimating is harmless — the optimizer trims it.
        ptp = float(np.ptp(y))
        offset = float(np.median(y))
        a_guess = ptp if ptp > 0 else 1.0
        p0 = []
        for i in range(n):
            frac = (i + 0.5) / n
            knee = float(np.quantile(xn, frac))
            b_guess = 3.0 if (i % 2 == 0) else -3.0
            p0 += [a_guess, b_guess, -knee]
        p0 += [offset]
        p0 = np.array(p0, dtype=float)

        lb, ub = [], []
        for _ in range(n):
            lb += [0.0, -20.0, -10.0]
            ub += [np.inf, 20.0, 10.0]
        lb += [-np.inf]
        ub += [np.inf]
        return xn, y, model, jac, p0, lb, ub, xm, xs

    def _least_squares(self, xn, y, model, jac, p0, lb, ub):
        def resid(p):
            return model(xn, p) - y

        return least_squares(
            resid,
            p0,
            jac=jac,
            bounds=(lb, ub),
            loss=self.loss,
            f_scale=self.f_scale,
            max_nfev=self.max_nfev,
        )

    @staticmethod
    def _make_predictor(model, popt, xm, xs):
        def predictor(x_new):
            x_new = np.asarray(x_new, float)
            xn_new = (x_new - xm) / xs
            return model(xn_new, popt)

        return predictor

    def fit(self, x, y) -> Callable[[float], float]:
        """Local nonlinear least-squares fit with analytical Jacobian."""
        xn, y, model, jac, p0, lb, ub, xm, xs = self._prepare(x, y)
        res = self._least_squares(xn, y, model, jac, p0, lb, ub)
        return self._make_predictor(model, res.x, xm, xs)

    def global_fit(
        self,
        x,
        y,
        seed: int | None = 0,
        de_maxiter: int = 200,
        de_popsize: int = 20,
        de_tol: float = 1e-8,
    ) -> Callable[[float], float]:
        """Globally explore the parameter space with differential evolution,
        then polish the best candidate with the local fit.

        Slower than :meth:`.fit` (typically by 1–2 orders of magnitude) but
        much more reliable on multimodal residual landscapes.

        Parameters
        ----------
        x, y :
            Data to fit.
        seed :
            Seed for the random state of differential evolution; pass ``None``
            for nondeterministic behaviour.
        de_maxiter :
            Maximum number of DE generations.
        de_popsize :
            DE population size multiplier (population = ``de_popsize * n_params``).
        de_tol :
            Convergence tolerance for DE.
        """
        xn, y, model, jac, p0, lb, ub, xm, xs = self._prepare(x, y)
        n = self.n_softplus

        # Differential evolution requires finite bounds.  ``a`` and the
        # vertical offset are unbounded in the local fit; tighten them here to
        # the smallest box that comfortably contains a reasonable fit.
        ptp = float(np.ptp(y)) or 1.0
        de_bounds = []
        for _ in range(n):
            de_bounds += [
                (0.0, 5.0 * ptp),  # a
                (-20.0, 20.0),     # b
                (-10.0, 10.0),     # c
            ]
        de_bounds += [(float(np.min(y)) - ptp, float(np.max(y)) + ptp)]  # offset

        def cost(p):
            r = model(xn, p) - y
            # Use the same robust loss as the local fit.
            if self.loss == "linear":
                return 0.5 * float(np.sum(r * r))
            z = (r / self.f_scale) ** 2
            scale = self.f_scale ** 2
            if self.loss == "soft_l1":
                rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
            elif self.loss == "huber":
                rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
            elif self.loss == "cauchy":
                rho = np.log1p(z)
            elif self.loss == "arctan":
                rho = np.arctan(z)
            else:
                raise ValueError(f"unknown loss {self.loss!r}")
            return 0.5 * scale * float(np.sum(rho))

        de_res = differential_evolution(
            cost,
            bounds=de_bounds,
            seed=seed,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=de_tol,
            polish=False,
        )
        # Polish with the local fit so we still get the analytical-Jacobian
        # refinement and a result that's bit-for-bit comparable to ``fit``.
        polish = self._least_squares(xn, y, model, jac, de_res.x, lb, ub)
        return self._make_predictor(model, polish.x, xm, xs)
