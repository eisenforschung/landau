from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import numpy.typing as npt
import scipy.optimize as so
from scipy.interpolate import UnivariateSpline
try:
    import polyfit
except ImportError:
    polyfit = None

from scipy.constants import Boltzmann, eV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from scipy.optimize import least_squares

kB = Boltzmann / eV


def _scalarize(x):
    """Collapse a 0-d numpy result to a Python scalar; pass everything else through.

    The interpolation callables and phase methods promise a Python scalar out
    when given scalar input; broadcasting through ``np.asarray`` / ``np.vectorize``
    otherwise leaves 0-d arrays. This is the central place to undo that.
    """
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x


def G_calphad(T, pl, *p):
    T_arr = np.asarray(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = T_arr * np.log(T_arr) * pl + sum(pi * T_arr**i for i, pi in enumerate(p))
    g = np.where(np.isclose(T_arr, 0), p[0], g)
    if T_arr.ndim == 0:
        return g.item()
    return g


class Interpolation(ABC):
    """A fitted 1-D interpolation ``f(x)``.

    Subclasses must be callable. :meth:`deriv` returns ``f'`` as another
    :class:`Interpolation`; the default is a vectorised central difference,
    and subclasses with a closed form (polynomials, SGTE, Redlich-Kister)
    override it with the exact derivative.
    """

    @abstractmethod
    def __call__(self, x: npt.ArrayLike) -> np.ndarray | float:
        """Evaluate the interpolation at ``x`` (scalar or array-like)."""
        ...

    def deriv(self) -> "Interpolation":
        """Return ``f'`` as another :class:`Interpolation` (numerical by default)."""
        return NumericalDerivative(self)


class NumericalDerivative(Interpolation):
    """``f'`` by a scale-aware central difference; the generic fallback derivative.

    A single fixed-step central difference (vectorised, no Python-level loop)
    rather than :func:`scipy.differentiate.derivative`, which only exists from
    SciPy 1.15 and would lift the package's ``scipy>=1.11.2`` floor.

    Args:
        func: the interpolation to differentiate.
        step: relative step; the absolute step is ``step * max(1, |x|)``.
    """

    def __init__(self, func: Callable, step: float = 1e-6):
        self.func = func
        self.step = step

    def __call__(self, x: npt.ArrayLike) -> np.ndarray | float:
        x = np.asarray(x, dtype=float)
        h = self.step * np.maximum(1.0, np.abs(x))
        d = (self.func(x + h) - self.func(x - h)) / (2.0 * h)
        return d.item() if d.ndim == 0 else d


class _CallableInterpolation(Interpolation):
    """Adapts a plain callable (closure) into an :class:`Interpolation`.

    Inherits the numerical :meth:`Interpolation.deriv`, so interpolations built
    from closures (stitched, softplus, Whitney) gain a derivative for free.
    """

    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, x: npt.ArrayLike) -> np.ndarray | float:
        return self.func(x)


class PolynomialInterpolation(Interpolation):
    """A polynomial fit with an analytic (polynomial) derivative.

    Wraps a :class:`numpy.poly1d` so the poly1d representation (and its
    descending-power coefficient order) stays an implementation detail;
    :attr:`coefficients` exposes them in ascending power order.
    """

    def __init__(self, poly: np.poly1d):
        self.poly = poly

    @property
    def coefficients(self) -> np.ndarray:
        """Coefficients in ascending power order: ``c0 + c1*x + c2*x**2 + ...``."""
        return self.poly.coeffs[::-1]

    def __call__(self, x: npt.ArrayLike) -> np.ndarray | float:
        return self.poly(x)

    def deriv(self) -> "Interpolation":
        """The exact derivative polynomial."""
        return PolynomialInterpolation(self.poly.deriv())


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
        return PolynomialInterpolation(np.poly1d(coef))


@dataclass(frozen=True, eq=True)
class SplineFit(ConcentrationInterpolator):
    """
    Fits data with a univariate B-spline of degree ``degree``.

    Wraps :class:`scipy.interpolate.UnivariateSpline`.  ``smoothing`` selects
    between strict interpolation (``0.0``, the default — the spline passes
    through every sample) and least-squares smoothing (``> 0`` — the spline may
    miss the samples by a total squared residual of ``smoothing`` in exchange
    for a smoother curve).  ``None`` defers to scipy's own default (``s`` equal
    to the number of samples).

    Outside ``[min(x), max(x)]`` the fit continues the boundary spline segment
    (scipy's default ``ext=0``): ``degree=1`` extends linearly, while higher
    degrees follow the edge polynomial and can diverge quickly, so extrapolated
    values should be treated with care.
    """

    degree: int = 3
    """Spline degree ``k`` (3 = cubic).  Clamped to ``len(x) - 1`` so a fit with
    fewer samples than ``degree + 1`` drops to the highest degree the data
    supports."""
    smoothing: float | None = 0.0
    """Smoothing factor ``s`` for :class:`~scipy.interpolate.UnivariateSpline`.
    ``0.0`` interpolates strictly; larger values trade fidelity for smoothness;
    ``None`` uses scipy's default."""

    def fit(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        k = min(self.degree, len(x) - 1)
        spline = UnivariateSpline(x, y, k=k, s=self.smoothing)

        def interpolation(c):
            return _scalarize(spline(c))

        return interpolation


@dataclass(frozen=True, eq=True)
class SGTE(TemperatureInterpolator):
    nparam: int

    def __post_init__(self):
        assert self.nparam > 1, "Must fit at least two parameters!"

    def fit(self, x, y):
        parameters, *_ = so.curve_fit(G_calphad, x, y, p0=[0] * self.nparam)
        return SGTEInterpolation(tuple(parameters))


@dataclass(frozen=True)
class SGTEInterpolation(Interpolation):
    """``G(T) = T*ln(T)*pl + sum_i p_i*T^i`` with the analytic ``dG/dT``."""

    parameters: tuple[float, ...]

    def __call__(self, T: npt.ArrayLike) -> np.ndarray | float:
        return G_calphad(T, *self.parameters)

    def deriv(self) -> Interpolation:
        """The analytic ``dG/dT`` as a callable interpolation."""
        pl, *p = self.parameters

        def dG(T):
            T_arr = np.asarray(T, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                g = pl * (np.log(T_arr) + 1.0) + sum(i * pi * T_arr ** (i - 1) for i, pi in enumerate(p) if i >= 1)
            return g.item() if g.ndim == 0 else g

        return _CallableInterpolation(dG)


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
class RedlichKisterInterpolation(Interpolation):
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
        # k=0: algebraically simplifies (0 - xi^2)*xi^(-1) = -xi, avoids 0^(-1) at x=0.5
        terms = [-xi] + [(2 * k * pre - x2) * xi ** (k - 1) for k in range(1, len(L))]
        ds = np.stack(terms)
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

    def __call__(self, c: npt.ArrayLike) -> np.ndarray | float:
        return self._eval_mix(c, *self.rk_parameters) + self.f0 + self.df * c

    def deriv(self) -> Interpolation:
        """The analytic ``d/dc`` from :meth:`_eval_mix_derivative`."""
        # d/dc [ mix(c) + f0 + df*c ] = mix'(c) + df
        rk_parameters = self.rk_parameters
        df = self.df
        return _CallableInterpolation(
            lambda c: self._eval_mix_derivative(c, *rk_parameters) + df
        )


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

        return _CallableInterpolation(interpolation)
