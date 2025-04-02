from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import scipy.optimize as so
try:
    import polyfit
except ImportError:
    polyfit = None

from scipy.constants import Boltzmann, eV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression

kB = Boltzmann / eV


def G_calphad(T, pl, *p):
    with np.errstate(divide="ignore"):
        g = T * np.log(T) * pl + sum(pi * T**i for i, pi in enumerate(p))
    if isinstance(T, np.ndarray) and T.ndim > 0:
        g[np.isclose(T, 0)] = p[0]
    elif np.isclose(T, 0.0):
        g = p[0]
    return g


Interpolation = Callable[[float], float]
"""Generic Interface for a 1D interpolation."""


class Interpolator(ABC):
    """
    This class acts as a factory for an interplation.

    Call :meth:`.fit()` to obtain a specific interpolation.

    Implementations should be hashable and immutable to allow caching in the
    thermodynamics module.
    """

    @abstractmethod
    def fit(self, x, y) -> Interpolation:
        pass


# subclasses for type hinting only; interface the same
class TemperatureInterpolator(Interpolator):
    pass


class ConcentrationInterpolator(Interpolator):
    pass


@dataclass(frozen=True, eq=True)
class PolyFit(TemperatureInterpolator, ConcentrationInterpolator):
    nparam: int | Literal["auto"]
    """Number of parameters, if "auto" fit a 10 parameter polynomial under L1 and discard parameters <1e-10, then refit."""
    regularizer_strength: float = 1e-8
    """Strength of L2-norm regularization."""
    enforce_curvature: bool = False
    """Ensure that the interpolation has negative curvature as expected for thermodynamic potentials."""

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if self.nparam == "auto":
            reg = make_pipeline(
                    PolynomialFeatures(10),
                    Lasso(self.regularizer_strength, fit_intercept=False)
            )
            reg.fit(x.reshape(-1, 1), y)
            nparam = sum(abs(reg.steps[-1][1].coef_) > 1e-10)
        else:
            nparam = self.nparam
        if not self.enforce_curvature or polyfit is None:
            if self.enforce_curvature:
                warnings.warn("enforce_curvature=True is only supported when the `polyfit` package from PyPI is installed. "
                              "Falling back to regular fitting.")
            reg = make_pipeline(
                    PolynomialFeatures(nparam - 1),
                    Ridge(self.regularizer_strength, fit_intercept=False)
            )
            reg.fit(x.reshape(-1, 1), y)
            coef = reg.steps[-1][1].coef_[::-1]
        else:
            reg = polyfit.PolynomRegressor(
                    nparam, lam=self.regularizer_strength
            ).fit(
                    x.reshape(-1, 1), y,
                    constraints={0: polyfit.Constraints(curvature="concave")}
            )
            coef = reg.coeffs_[::-1]
        return np.poly1d(coef)


@dataclass(frozen=True, eq=True)
class SGTE(TemperatureInterpolator):
    nparam: int

    def __post_init__(self):
        assert self.nparam > 1, "Must fit at least two parameters!"

    def fit(self, x, y):
        parameters, *_ = so.curve_fit(G_calphad, x, y, p0=[0] * self.nparam)
        return lambda x: G_calphad(x, *parameters)


@dataclass(frozen=True, eq=True)
class RedlichKister(ConcentrationInterpolator):
    """
    Fits the "enthalpic" part of a Redlich-Kister expansion, i.e. without the
    ideal configuration entropy.
    """

    nparam: int

    def __post_init__(self):
        assert self.nparam > 0, "Must fit at least one parameter!"

    def fit(self, c, f):
        """
        Beware: You need to manually remove the entropy if included in f.
        """
        # FIXME: assumes terminals are unique
        I = c.argsort()
        f = f[I]
        c = c[I]
        assert np.isclose(c[0], 0) and np.isclose(c[-1], 1), "Must include terminals when fitting Redlich-Kister!"
        f0 = f[0]
        df = f[-1] - f[0]
        f -= f0 + df * c
        nparam = min(self.nparam, len(c) - 2)
        rk_parameters, _ = so.curve_fit(RedlichKisterInterpolation._eval_mix, c, f, p0=np.zeros(nparam))
        return RedlichKisterInterpolation(df, f0, rk_parameters)


@dataclass(frozen=True)
class RedlichKisterInterpolation:
    df: float
    """Change in mixing "enthalpy" across composition range."""
    f0: float
    """Absolute "enthalpy" at concentration 0"""
    rk_parameters: np.ndarray[float]
    """Redlich-Kister parameters."""

    @staticmethod
    def _eval_mix(x, *L):
        pre = x * (1 - x)
        if isinstance(x, np.ndarray):
            vam = np.vander((2 * x - 1), N=len(L), increasing=True)
            return pre * np.einsum("ij,j->i", vam, L)
        else:
            return pre * sum(Li * (2 * x - 1) ** i for i, Li in enumerate(L))

    @staticmethod
    def _eval_mix_derivative(x, *L):
        pre = x * (1 - x)
        xi = 2 * x - 1
        x2 = xi**2
        ds = np.stack([(2 * k * pre - x2) * xi ** (k - 1) for k in range(len(L))])
        if len(ds.shape) == 1:
            return (L * ds).sum()
        else:
            return np.transpose(L) @ ds

    # def fit_derivative(self, c, mu, f0=0, c0=0):
    #     """
    #     Beware: You need to manually remove the entropy if included in mu.
    #     """
    #     # optimization works better if all parameters are on one scale
    #     # df tends to be eV, but *L 1-10meV
    #     # so just center on a rough guess and fit difference to it for df
    #     df_guess = np.median(mu)
    #     nparam = min(len(self.rk_parameters), len(c) - 2)
    #     (self.df, *self.rk_parameters), *_ = so.curve_fit(
    #             lambda c, df, *L: df_guess + df + self._eval_mix_derivative(c, *L),
    #             c, mu, p0=np.zeros(nparam + 1)
    #     )
    #     self.df += df_guess
    #     self.f0 = f0 - self(c0)
    #     return self

    def __call__(self, c):
        return self._eval_mix(c, *self.rk_parameters) + self.f0 + self.df * c


@dataclass(frozen=True, eq=True)
class StitchedFit(TemperatureInterpolator):
    """
    An interpolator with more control over the extrapolation regions.
    """

    interpolating: TemperatureInterpolator = SGTE(4)
    # use the interpolating fit for lower temps too, i.e. extrapolate
    low: TemperatureInterpolator | None = None
    # use a straight line (constant entropy) for higher temperatures
    upp: TemperatureInterpolator | None = PolyFit(2)

    """How many samples near the edges to use to fit the extrapolating interpolator."""
    edge: int = 10

    def fit(self, t, f):
        tmin = t.min()
        tmax = t.max()
        mid = self.interpolating.fit(t, f)
        low = None
        upp = None
        if self.low is not None:
            low = self.low.fit(t[: self.edge], f[: self.edge])
        if self.upp is not None:
            upp = self.upp.fit(t[-self.edge :], f[-self.edge :])

        def interpolation(t):
            t = np.array(t)
            f = mid(t)
            if low is not None:
                f = np.where(t < tmin, low(t), f)
            if upp is not None:
                f = np.where(t > tmax, upp(t), f)
            if f.ndim == 0:
                f = f.item()
            return f

        return interpolation
