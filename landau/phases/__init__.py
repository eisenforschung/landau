from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache, cache, cached_property
from typing import Iterable, Optional, ClassVar
from pyiron_snippets.deprecate import deprecate

import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.optimize as so
import scipy.spatial as ss
import scipy.special as se

import numpy as np

from ..interpolate import ConcentrationInterpolator, TemperatureInterpolator, SGTE, PolyFit, RedlichKister, SoftplusFit
from ..interpolate.basic import _scalarize, SurfaceInterpolator

from scipy.constants import Boltzmann, eV

kB = Boltzmann / eV


__all__ = [
    "Phase",
    "AbstractLinePhase",
    "LinePhase",
    "TemperatureDependentLinePhase",
    "IdealSolution",
    "RegularSolution",
    "InterpolatingPhase",
    "SlowInterpolatingPhase",
    "FastInterpolatingPhase",
    "Surface2DInterpolatingPhase",
    "AbstractPointDefect",
    "ConstantPointDefect",
    "PointDefectSublattice",
    "PointDefectedPhase",
    "AsePhase",
]


def S(c):
    return kB * (se.entr(c) + se.entr(1 - c))


@dataclass(frozen=True)
class Phase(ABC):
    """
    Represents a phase in a binary phase diagram.
    """

    name: str

    @abstractmethod
    def semigrand_potential(self, T, dmu):
        """
        Calculate the semigrand potential of the phase.
        """
        pass

    @abstractmethod
    def concentration(self, T, dmu):
        """
        Concentration of the phase at the given state.
        """
        pass

    def __repr__(self):
        return f'{type(self).__name__}("{self.name}")'

    __str__ = __repr__


@dataclass(frozen=True)
class AbstractLinePhase(Phase):
    """Base class for fixed concentration phases.

    Required overloads are :meth:`.AbstractLinePhase.line_concentration` and
    :meth:`.AbstractLinePhase.line_free_energy`.
    """

    @property
    @abstractmethod
    def line_concentration(self):
        pass

    @abstractmethod
    def line_free_energy(self, T):
        pass

    def free_energy(self, T, c):
        return self.line_free_energy(T)

    def concentration(self, T, dmu):
        return _scalarize(np.full(np.broadcast(T, dmu).shape, self.line_concentration))

    def semigrand_potential(self, T, dmu):
        f = self.line_free_energy(T)
        return f - self.line_concentration * dmu


@dataclass(frozen=True)
class LinePhase(AbstractLinePhase):
    """
    Simple phase with a fixed concentration and temperature independent entropy.
    """

    fixed_concentration: float
    line_energy: float
    line_entropy: float = 0

    @property
    def line_concentration(self):
        return self.fixed_concentration

    def line_free_energy(self, T):
        return self.line_energy - T * self.line_entropy


@dataclass(frozen=True)
class TemperatureDependentLinePhase(AbstractLinePhase):
    """ "
    Simple phase with a fixed concentration and temperature dependent free
    energy.
    """

    fixed_concentration: float
    """The fixed concentration of the phase"""
    temperatures: Iterable[float]
    """Temperatures at which the free energy of the phase has been sampled."""
    free_energies: Iterable[float]
    """Sampled free energy of the phase has been computed."""
    interpolator: TemperatureInterpolator = SGTE(3)
    """How to interpolate to arbitrary temperatures from the samples."""
    _hash: int = field(default=0, init=False)

    def __post_init__(self, *args, **kwargs):
        def to_ro_numpy(iterable):
            a = np.array(iterable)
            a.flags.writeable = False
            return a

        object.__setattr__(self, "temperatures", to_ro_numpy(self.temperatures))
        object.__setattr__(self, "free_energies", to_ro_numpy(self.free_energies))
        # precompute hash: hashing arrays every cache lookup is too expensive
        # and we any way advertise as frozen
        object.__setattr__(
            self,
            "_hash",
            hash(
                (
                    hash(self.fixed_concentration),
                    hash(self.temperatures.tobytes()),
                    hash(self.free_energies.tobytes()),
                    hash(self.interpolator),
                )
            ),
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return all(
            (
                self.fixed_concentration == other.fixed_concentration,
                np.array_equal(self.temperatures, other.temperatures),
                np.array_equal(self.free_energies, other.free_energies),
            )
        )

    @property
    @cache
    def _interpolation(self):
        return self.interpolator.fit(self.temperatures, self.free_energies)

    @property
    def line_concentration(self):
        return self.fixed_concentration

    def line_free_energy(self, T):
        return self._interpolation(T)

    def check_interpolation(self, Tl=0.9, Tu=1.1, samples=50, plot_error=False):
        """Plot the temperature interpolation against its samples to visually assess fit quality.

        Args:
            Tl (float): lower edge of the plotted range as a fraction of the minimum sampled temperature
            Tu (float): upper edge of the plotted range as a fraction of the maximum sampled temperature
            samples (int): number of points along the interpolated curve
            plot_error (bool): if True, plot only the interpolation error at the samples instead of the free energies
        """
        # try to plot about 100 points
        n = max(int(len(self.temperatures) // 100), 1)
        ts = self.temperatures[::n]
        fs = self.free_energies[::n]
        if plot_error:
            plt.scatter(ts, self.line_free_energy(ts) - fs, label=self.name)
            return
        Ts = np.linspace(np.min(self.temperatures) * Tl, np.max(self.temperatures) * Tu, samples)
        (l,) = plt.plot(Ts, self.line_free_energy(Ts), label=self.name)
        plt.scatter(ts, fs, color=l.get_color())


@deprecate("use TemperatureDependentLinePhase instead", version="2.0")
def TemperatureDepandantLinePhase(*args, **kwargs):
    return TemperatureDependentLinePhase(*args, **kwargs)


@dataclass(frozen=True, eq=True)
class IdealSolution(Phase):
    phase1: AbstractLinePhase
    phase2: AbstractLinePhase

    def __post_init__(self, *args, **kwargs):
        phase1, phase2 = sorted((self.phase1, self.phase2), key=lambda p: p.line_concentration)
        assert phase1.line_concentration == 0 and phase2.line_concentration == 1, "Must give terminal phases!"
        # bypass frozen=True for the sake of init only
        object.__setattr__(self, "phase1", phase1)
        object.__setattr__(self, "phase2", phase2)

    def semigrand_potential(self, T, dmu):
        T = np.asarray(T)
        dmu = np.asarray(dmu)
        p1 = self.phase1
        p2 = self.phase2
        f1 = p1.line_free_energy(T)
        f2 = p2.line_free_energy(T)
        df = f2 - f1
        with np.errstate(divide='ignore', over="ignore", invalid='ignore'):
            expo = -(df - dmu) / kB / T
            phi = f1 - kB * T * np.log(1 + np.exp(expo))
            I = ~np.isfinite(phi)
            if I.any():
                if phi.shape == ():
                    phi = f2 - dmu
                else:
                    phi[I] = f2 - dmu[I]
        return _scalarize(phi)

    def concentration(self, T, dmu):
        p1 = self.phase1
        p2 = self.phase2
        f1 = p1.line_free_energy(T)
        f2 = p2.line_free_energy(T)
        df = f2 - f1
        with np.errstate(divide='ignore', over='ignore'):
            return 1 / (1 + np.exp(+(df - dmu) / kB / T))


@dataclass(frozen=True, eq=True)
class RegularSolution(Phase):
    """
    A regular solution model phase that interpolates through a given set of line phases using Redlich-Kister
    polynomials.
    """

    phases: Iterable[AbstractLinePhase]
    """Line phases to interpolate, *must* include the terminals."""
    num_coeffs: int = 4
    """Number of Redlich-Kister coefficients for the mixing "enthalpy"; restricted to number of phases - 2."""
    add_entropy: bool = False
    """If False, assume that the free energies of the line phases already include configurational mixing entropy. If
    True add ideal mixing entropy."""

    def __post_init__(self, *args, **kwargs):
        # bypass frozen=True for the sake of init only
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "num_coeffs", min(len(self.phases) - 2, self.num_coeffs))
        concs = tuple(p.line_concentration for p in self.phases)
        assert 0 in concs and 1 in concs, "Must give the terminal phases!"
        left_terminals = sum(c == 0 for c in concs)
        right_terminals = sum(c == 1 for c in concs)
        assert left_terminals == 1 and right_terminals == 1, (
            "Cannot pass multiple terminal phases of the same concentration!"
        )

    @lru_cache(maxsize=250)
    def _get_interpolation(self, T):
        cc = np.array([l.line_concentration for l in self.phases])
        ff = np.array([l.line_free_energy(T) for l in self.phases], dtype=float)

        # TODO: needs better naming: If the free energies of the phase objects
        # already contain the entropy of mixing, remove it here first, before
        # we try to fit the redlich kister coeffs
        if not self.add_entropy:
            ff += T * S(cc)
        return RedlichKister(self.num_coeffs).fit(cc, ff)

    def free_energy(self, T, c):
        return self._get_interpolation(T)(c) - T * S(c)

    def excess_free_energy(self, T, c):
        cc = np.linspace(0, 1)
        ff = self.free_energy(T, cc)
        f0 = ff[0]
        f1 = ff[-1]
        return si.interp1d(cc, ff - (f0 * (1 - cc) + f1 * cc), kind="cubic")(c)

    def semigrand_potential(self, T, dmu, plot=False, raw=False):
        def get_mu_c(c):
            f = self.free_energy(T, c)

            f0 = f[0]
            f1 = f[-1]
            I = f <= c * f1 + (1 - c) * f0
            fI = f[I]
            cI = c[I]

            # system is fully demixing
            if I.sum() == 2:
                M = f1 - f0
                f12 = (f0 + f1) / 2
                return (np.array([0, 0.5, 1]), np.array([f0, f12, f1]), np.array([M - 1e-3, M, M + 1e-3]))

            hull = ss.ConvexHull(list(zip(cI, fI)))
            cH, fH = hull.points[hull.vertices].T
            Is = np.argsort(cH)
            cH = cH[Is]
            fH = fH[Is]

            M = np.gradient(fH, cH)
            return cH, fH, M

        n = 50
        c, f, M = get_mu_c(np.linspace(0, 1, n))
        limit = 5e-3
        while np.median(abs(np.diff(M))) > limit and n < 5e4:
            n *= 2
            Ms = np.linspace(M.min(), M.max(), n)
            c = si.interp1d(M, c)(Ms)
            c, f, M = get_mu_c(c)
            if plot:
                plt.subplot(121)
                plt.plot(c[:-1], np.diff(M), "v", label=n)
                plt.subplot(122)
                plt.plot(c, M, ".")
        if plot and n > 50:
            plt.subplot(121)
            plt.title("Spacing of Chemical Potential Sampling")
            plt.xlabel("c")
            plt.ylabel("np.diff(mu)")
            plt.legend(title="grid points")
            plt.subplot(122)
            plt.xlabel("c")
            plt.ylabel(r"$\Delta \mu$")
            plt.show()

        p = f - c * M
        if raw:
            return f, c, M, p

        assert np.median(abs(np.diff(M))) <= limit, "Weird"

        # schon etwas dreist, aber naja
        pi = si.interp1d(
            M,
            p,
            fill_value=np.nan,
            bounds_error=False,
            # needs to be at least quadratic, otherwise we'll see
            # jumps in the numerically calculated concentration
            kind="quadratic",
        )(dmu)
        pl = self.free_energy(T, 1) - dmu * 1
        f0 = self.free_energy(T, 0)
        if not isinstance(dmu, np.ndarray):
            if np.isnan(pi):
                pi = np.inf
            return _scalarize(min(pi, pl, f0))
        pl[pl > f0] = f0
        I = np.isnan(pi)
        pi[I] = pl[I]
        if plot:
            plt.plot(M, p, "o-", label="calculated")
            plt.plot(dmu, pi, label="extrapolated")
            plt.legend()
        return pi

    def concentration(self, T, dmu):
        if not isinstance(dmu, np.ndarray) or dmu.size == 1:
            dmus = np.linspace(-1, 1, 5) * 1e-4 + dmu
            res = self.concentration(T, dmus)[2]
            return np.array([res]) if isinstance(dmu, np.ndarray) else res
        return np.clip(-np.gradient(self.semigrand_potential(T, dmu), dmu, edge_order=2), 0, 1)

    @deprecate('Use check_concentration_interpolation instead')
    def check_interpolation(self, T=1000, samples=50):
        self.check_concentration_interpolation(T=T, samples=samples)

    def check_concentration_interpolation(
            self,
            T=1000,
            samples=50,
            plot_excess=False,
            plot_error=False,
    ):
        """Plot free energies of an interpolating phase and its underlying line
        phases to visually assess fit quality.

        Args:
            T (float): at which temperature to check interpolation
            samples (int): number of sampling points for plot
            plot_excess (bool): if True, subtract free energy at concentration range endpoints for legibility
            plot_error (bool): if True, plot only the interpolation error at the samples instead of the free energies
            """
        check_concentration_interpolation(self, self.phases, T, samples, plot_excess, (0, 1), plot_error=plot_error)


from numbers import Real


@dataclass(frozen=True, eq=True)
class InterpolatingPhase(Phase):
    """A Version of RegularSolutionPhase that does not depend on terminals.  FIXME: These two classes should be unified."""

    phases: Iterable[AbstractLinePhase]
    num_coeffs: int = None
    add_entropy: bool = False
    num_samples: int = 100
    maximum_extrapolation: float = 0

    def __post_init__(self, *args, **kwargs):
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "num_coeffs", min(len(self.phases), self.num_coeffs or np.inf))

    @lru_cache(maxsize=250)
    def _get_interpolation(self, T):
        if not isinstance(T, Real):
            raise TypeError(T)
        cc = np.array([l.line_concentration for l in self.phases])
        ff = np.array([l.line_free_energy(T) for l in self.phases])

        # TODO: needs better naming: If the free energies of the phase objects
        # already contain the entropy of mixing, remove it here first, before
        # we try to fit the redlich kister coeffs
        if not self.add_entropy:
            ff += T * S(cc)
        if cc[0] == 0 and cc[-1] == 1:
            return RedlichKister(max(1, self.num_coeffs - 2)).fit(cc, ff)
        else:
            return PolyFit(self.num_coeffs).fit(cc, ff)

    def free_energy(self, T, c):
        return np.vectorize(
            lambda T, c: self._get_interpolation(T)(c) - T * S(c),
            otypes=[float]
        )(T, c)
        # return self._get_interpolation(T)(c) - T * S(c)

    def _find_phi_c(self, T, dmu):
        """Calculate potential and concentration together.

        Formally we need to solve

        phi = min_c { f(c) - c * dmu }

        but this is too slow to solve with normal optimizers from scipy and can
        get stuck in local minima.  Instead do the brute force minimization on
        a grid (self.num_samples), then refine the gridded concentrations with
        a single step of a newton-raphson like optimization.  This makes sure
        that the output concentrations are smooth and non-degenerate.
        """
        output_shape = np.broadcast_shapes(np.shape(T), np.shape(dmu))
        T = np.atleast_1d(T)[..., np.newaxis]
        dmu = np.atleast_1d(dmu)[..., np.newaxis]

        cs = [p.line_concentration for p in self.phases]
        conc = np.linspace(
            max(0, min(cs) - self.maximum_extrapolation),
            min(1, max(cs) + self.maximum_extrapolation),
            self.num_samples
        )
        ff = self.free_energy(T, conc)
        phi = ff - conc * dmu
        I = phi.argmin(axis=-1, keepdims=True)
        phi = np.take_along_axis(phi, I, axis=-1)[..., 0]
        c = conc[I[..., 0]]

        df = np.take_along_axis(
                np.gradient(ff, conc, axis=-1, edge_order=2),
                I, axis=-1
        )
        d2f = np.take_along_axis(
                np.gradient(
                    np.gradient(ff, conc, axis=-1, edge_order=2),
                    conc, axis=-1, edge_order=2
                ),
                I, axis=-1
        )
        dc = (dmu - df) / d2f
        nc = np.clip(c + dc[..., 0], 0, 1)
        phi -= (nc-c)*(dmu-df)[..., 0]
        c = nc

        c = c.reshape(output_shape)
        phi = phi.reshape(output_shape)

        return _scalarize(phi), _scalarize(c)

    def semigrand_potential(self, T, dmu):
        return self._find_phi_c(T, dmu)[0]

    def concentration(self, T, dmu):
        return self._find_phi_c(T, dmu)[1]

    @deprecate('Use check_concentration_interpolation instead')
    def check_interpolation(self, T=1000, samples=50):
        self.check_concentration_interpolation(T=T, samples=samples)

    def check_concentration_interpolation(
            self,
            T=1000,
            samples=50,
            plot_excess=False,
            plot_error=False,
    ):
        """Plot free energies of an interpolating phase and its underlying line
        phases to visually assess fit quality.

        Args:
            T (float): at which temperature to check interpolation
            samples (int): number of sampling points for plot
            plot_excess (bool): if True, subtract free energy at concentration range endpoints for legibility
            plot_error (bool): if True, plot only the interpolation error at the samples instead of the free energies
            """
        cs = [p.line_concentration for p in self.phases]
        concentration_range = (
                max(0, min(cs) - self.maximum_extrapolation),
                min(1, max(cs) + self.maximum_extrapolation)
        )
        check_concentration_interpolation(
            self, self.phases, T, samples, plot_excess, concentration_range, plot_error=plot_error
        )

@dataclass(frozen=True, eq=True)
class SlowInterpolatingPhase(Phase):
    """
    A slower version of RegularSolutionPhase that does not depend on terminals.
    FIXME: These two classes should be unified.
    """

    phases: Iterable[AbstractLinePhase]
    add_entropy: bool = False
    maximum_extrapolation: float = 0
    concentration_range: tuple[float, float] = (0., 1.)
    interpolator: Optional[ConcentrationInterpolator] = None

    def __post_init__(self, *args, **kwargs):
        object.__setattr__(self, "phases", tuple(self.phases))

        explicit_range = self.concentration_range != (0., 1.)
        explicit_extrap = self.maximum_extrapolation != 0

        if explicit_range and explicit_extrap:
            raise ValueError("concentration_range and maximum_extrapolation are mutually exclusive")

        if not explicit_range:
            cs = [p.line_concentration for p in self.phases]
            concentration_range = (
                max(0, min(cs) - self.maximum_extrapolation),
                min(1, max(cs) + self.maximum_extrapolation)
            )
            object.__setattr__(self, "concentration_range", concentration_range)

        if not (self.concentration_range[0] == 0 and self.concentration_range[1] == 1) and isinstance(self.interpolator, RedlichKister):
            raise ValueError("RedlichKister interpolation requires terminal phases at both c=0 and c=1")

        if self.interpolator is None:
            if (self.concentration_range[0] == 0 and self.concentration_range[1] == 1):
                object.__setattr__(self, "interpolator", RedlichKister(min(5, len(self.phases))))
            else:
                object.__setattr__(self, "interpolator", PolyFit(min(4, len(self.phases))))

    @lru_cache(maxsize=2500)
    def _get_interpolation(self, T):
        if not isinstance(T, Real):
            raise TypeError(T)
        cc = np.array([l.line_concentration for l in self.phases])
        ff = np.array([l.line_free_energy(T) for l in self.phases])

        # TODO: needs better naming: If the free energies of the phase objects
        # already contain the entropy of mixing, remove it here first, before
        # we try to fit the redlich kister coeffs

        if not self.add_entropy:
            ff += T * S(cc)

        return self.interpolator.fit(cc, ff)

    def free_energy(self, T, c):
        return np.vectorize(
            lambda T, c: self._get_interpolation(T)(c) - T * S(c),
            otypes=[float]
        )(T, c)

    @lru_cache(maxsize=5000)
    def _find_phi_c_scalar(self, T, dmu):
        semi = lambda c: self.free_energy(T, c) - dmu * c if self.concentration_range[0] <= c <= self.concentration_range[1] else np.nan
        cmin, phimin, *_ = so.brute(semi, (self.concentration_range,), full_output=True)
        cmin = np.squeeze(np.clip(cmin, *self.concentration_range)).item()
        phimin = semi(cmin)
        return phimin, cmin

    def _find_phi_c(self, T, dmu):
        phi, c = np.squeeze(np.vectorize(self._find_phi_c_scalar)(T, dmu))
        return _scalarize(phi), _scalarize(c)

    def semigrand_potential(self, T, dmu):
        return self._find_phi_c(T, dmu)[0]

    def concentration(self, T, dmu):
        return self._find_phi_c(T, dmu)[1]

    @deprecate('Use check_concentration_interpolation instead')
    def check_interpolation(self, T=1000, samples=50):
        self.check_concentration_interpolation(T=T, samples=samples)

    def check_concentration_interpolation(
            self,
            T=1000,
            samples=50,
            plot_excess=False,
            plot_error=False,
    ):
        """Plot free energies of an interpolating phase and its underlying line
        phases to visually assess fit quality.

        Args:
            T (float): at which temperature to check interpolation
            samples (int): number of sampling points for plot
            plot_excess (bool): if True, subtract free energy at concentration range endpoints for legibility
            plot_error (bool): if True, plot only the interpolation error at the samples instead of the free energies
            concentration_range (tuple of float): min/max concentration range"""
        check_concentration_interpolation(
            self, self.phases, T, samples, plot_excess, self.concentration_range, plot_error=plot_error
        )


class FastInterpolatingPhase(SlowInterpolatingPhase):
    """A faster, equally accurate replacement for :class:`SlowInterpolatingPhase`.

    Computes the same quantity -- ``phi = min_c [ f(c) - c*dmu ]`` with
    ``f(c) = fe(c) - T*S(c)`` -- but vectorised over the whole ``dmu`` array
    instead of one ``scipy.optimize.brute`` call per scalar.

    For a fixed ``T`` the free-energy curve ``f(c)`` is evaluated once on a grid
    to locate the global basin (handling miscibility gaps), then the minimum is
    polished with a few Newton steps in the logit variable ``u = log(c/(1-c))``.
    The ideal-mixing entropy contributes ``-T*S'(c) = kB*T*u``, which is *linear*
    in ``u``, so the polish is uniformly well conditioned from the dilute to the
    concentrated limit -- where a plain ``c``-space Newton step is stiff. The
    polish is confined to the grid cell around the basin and the lowest of
    {Newton result, cell edges} is kept, so a minimum sitting on a range
    boundary is recovered exactly and the global basin is never abandoned.

    Reproduces the true minimum to ~1e-6; faster than the ``brute`` reference by
    two orders of magnitude on representative phases (see
    ``benchmarks/bench_fast_interpolating_phase.py``).
    """

    # solver tuning (ClassVar -> never treated as dataclass fields)
    _n_grid: ClassVar[int] = 201      # basin-locating grid resolution over the concentration range
    _n_newton: ClassVar[int] = 6      # logit-space Newton polish steps
    _fd2: ClassVar[float] = 1e-3      # wider difference step for fe'' (limits 1/h^2 round-off)

    def _solve_fixed_T(self, T: float, dmu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Minimise ``f(c) - dmu*c`` over ``c`` for one ``T`` and a whole ``dmu`` array.

        The free-energy curve ``f(c) = fe(c) - T*S(c)`` is built once on a grid;
        a single vectorised ``argmin`` over ``dmu`` against that shared curve picks
        the global basin for every ``dmu`` at once (this is where the speed comes
        from), then a bounded logit-space Newton polish sharpens each minimum.

        The Newton loop is hand-rolled rather than :func:`scipy.optimize.newton`
        because the polish must be clipped per element to the grid cell ``[cl, cr]``
        around the basin -- the confinement that stops Newton from escaping to a
        different stationary point on a non-convex landscape. ``scipy.optimize.newton``
        offers no per-element bounds, so it cannot express that guard.

        Args:
            T: temperature (scalar); fixes the free-energy curve and its fit.
            dmu: chemical-potential array (any shape).

        Returns:
            ``(phi, c)`` arrays shaped like ``dmu``: the semigrand potential and
            the minimising concentration.
        """
        a, b = self.concentration_range
        fe = self._get_interpolation(T)
        # fe.deriv() is analytic for PolyFit/RedlichKister, so the located c is
        # the exact stationary point and c = -d(phi)/d(dmu) holds to machine
        # precision; a generic interpolator falls back to a numerical derivative.
        fe_prime = fe.deriv()
        kT = kB * T

        conc = np.linspace(a, b, self._n_grid)
        f_grid = fe(conc) - T * S(conc)

        flat = np.asarray(dmu, dtype=float).ravel()

        def obj(c):
            return fe(c) - T * S(c) - flat * c

        # global basin: argmin of f(c) - dmu*c over the grid for each dmu
        idx = (f_grid[None, :] - flat[:, None] * conc[None, :]).argmin(axis=1)

        # confine the polish to the cell [conc[idx-1], conc[idx+1]]; the true
        # minimum lies within one cell of the grid argmin
        cl = conc[np.maximum(idx - 1, 0)]
        cr = conc[np.minimum(idx + 1, self._n_grid - 1)]
        with np.errstate(divide="ignore"):
            # -inf/+inf at the 0/1 ends keep the dilute/saturated tails reachable
            ua = np.where(cl <= 0.0, -np.inf, se.logit(np.clip(cl, 1e-300, 1.0)))
            ub = np.where(cr >= 1.0, np.inf, se.logit(np.clip(cr, 0.0, 1.0 - 1e-16)))

        u = np.clip(se.logit(np.clip(conc[idx], 1e-15, 1.0 - 1e-15)), ua, ub)
        h2 = self._fd2
        for _ in range(self._n_newton):
            c = se.expit(u)
            fp = fe_prime(c)
            fpp = (fe(c + h2) - 2 * fe(c) + fe(c - h2)) / (h2 * h2)
            g = fp + kT * u - flat                  # stationarity residual f'(c) = dmu
            gp = fpp * (c * (1.0 - c)) + kT          # dg/du
            u = np.clip(u - g / gp, ua, ub)
        c_newton = np.clip(se.expit(u), a, b)

        # keep the lowest objective among the polished point and the two cell
        # edges, so a minimum sitting exactly on a cell/range boundary is taken
        # as-is rather than chased past it by Newton
        cands = np.stack([c_newton, cl, cr])
        vals = np.stack([obj(c_newton), obj(cl), obj(cr)])
        best = vals.argmin(axis=0)
        c = np.take_along_axis(cands, best[None, :], axis=0)[0]
        phi = np.take_along_axis(vals, best[None, :], axis=0)[0]
        return phi.reshape(np.shape(dmu)), c.reshape(np.shape(dmu))

    @lru_cache(maxsize=512)
    def _find_phi_c_cached(
        self,
        t_shape: tuple[int, ...],
        t_bytes: bytes,
        d_shape: tuple[int, ...],
        d_bytes: bytes,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve and cache by raw ``(T, dmu)`` bytes.

        ``semigrand_potential`` and ``concentration`` are called separately by
        ``calc_phase_diagram`` with the same ``(T, dmu)``; keying the cache on
        the array bytes (arrays are unhashable) lets the second call reuse the
        first solve. A scalar ``T`` solves in one :meth:`_solve_fixed_T` call;
        an array ``T`` is grouped by unique temperature, since each distinct
        ``T`` needs its own free-energy fit.
        """
        T = np.frombuffer(t_bytes, dtype=float).reshape(t_shape)
        dmu = np.frombuffer(d_bytes, dtype=float).reshape(d_shape)
        out_shape = np.broadcast_shapes(t_shape, d_shape)
        if T.ndim == 0:
            phi, c = self._solve_fixed_T(float(T), dmu)
            phi = np.broadcast_to(phi, out_shape)
            c = np.broadcast_to(c, out_shape)
        else:
            # each distinct T needs its own fit; group the work by temperature
            Tb = np.broadcast_to(T, out_shape)
            dmub = np.broadcast_to(dmu, out_shape)
            phi = np.empty(out_shape)
            c = np.empty(out_shape)
            for uT in np.unique(Tb):
                m = Tb == uT
                p, cc = self._solve_fixed_T(float(uT), dmub[m])
                phi[m] = p
                c[m] = cc
        return np.asarray(phi), np.asarray(c)

    def _find_phi_c(
        self, T: float | np.ndarray, dmu: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Return ``(phi, c)`` for ``(T, dmu)``, scalars collapsed back to Python floats.

        Thin wrapper over the bytes-keyed cache: builds the cache key, then
        copies the result out so callers cannot mutate the cached arrays.
        """
        # asarray preserves 0-d shape; ascontiguousarray only for the byte key
        Ta = np.asarray(T, dtype=float)
        Da = np.asarray(dmu, dtype=float)
        phi, c = self._find_phi_c_cached(
            Ta.shape, np.ascontiguousarray(Ta).tobytes(),
            Da.shape, np.ascontiguousarray(Da).tobytes(),
        )
        # copy out so callers cannot mutate the cached arrays
        return _scalarize(phi.copy()), _scalarize(c.copy())


@dataclass(frozen=True, eq=True)
class Surface2DInterpolatingPhase(FastInterpolatingPhase):
    """FastInterpolatingPhase backed by a fitted 2-D free-energy surface.

    Unlike the parent's :meth:`_get_interpolation` — which fits a fresh 1-D curve
    f(c) at each temperature from the line phases' free energies — this class fits
    a single surface f(T, c) once via ``surface_interpolator.fit()`` and returns
    fixed-T slices via ``FittedSurface.slice_at(T)``.  The inherited logit-Newton
    solver and the full semigrand/concentration API from
    :class:`FastInterpolatingPhase` are reused unchanged.

    Training data: each line phase is sampled at ``num_temperature_samples`` evenly
    spaced temperatures over ``temperature_range`` (or the union of the phases' own
    sampled ranges).  The entropy-removed free energy H = f + T·S(c) is passed to
    the interpolator when ``add_entropy=False`` (the usual case with calphy data).

    Args:
        surface_interpolator: A :class:`~landau.interpolate.SurfaceInterpolator`
            that fits the 2-D surface from flat (T, c, H) arrays and returns a
            :class:`~landau.interpolate.FittedSurface`.  **Required** — there is
            no default; omitting it raises :exc:`TypeError` at construction time.
        num_temperature_samples: Number of T values sampled per line phase for
            the training set.
        temperature_range: ``(Tmin, Tmax)`` span used for training.  Should cover
            the full solve grid; defaults to the union of the line phases' own
            sampled temperature ranges.
    """

    surface_interpolator: Optional[SurfaceInterpolator] = None
    num_temperature_samples: int = 40
    temperature_range: Optional[tuple] = None

    def __post_init__(self):
        super().__post_init__()
        if self.surface_interpolator is None:
            raise TypeError(
                f"{type(self).__name__} requires a surface_interpolator keyword argument"
            )

    def _gather_training_data(self):
        n = self.num_temperature_samples
        sweeps = [np.asarray(getattr(p, "temperatures", []), float) for p in self.phases]
        have = [s for s in sweeps if s.size]
        if self.temperature_range is not None:
            glo, ghi = map(float, self.temperature_range)
        elif have:
            glo = min(float(s.min()) for s in have)
            ghi = max(float(s.max()) for s in have)
        else:
            raise ValueError(
                "need line phases with sampled temperatures or an explicit temperature_range"
            )

        tg = np.linspace(glo, ghi, n)
        TT, CC, FF = [], [], []
        for p in self.phases:
            TT.append(tg)
            CC.append(np.full(tg.shape, float(p.line_concentration)))
            FF.append(np.asarray(p.line_free_energy(tg), float))
        TT = np.concatenate(TT)
        CC = np.concatenate(CC)
        FF = np.concatenate(FF)

        if not self.add_entropy:
            FF = FF + TT * S(CC)

        return TT, CC, FF

    @cached_property
    def _fitted_surface(self):
        TT, CC, FF = self._gather_training_data()
        return self.surface_interpolator.fit(TT, CC, FF)

    @lru_cache(maxsize=512)
    def _get_interpolation(self, T):
        if not isinstance(T, Real):
            raise TypeError(T)
        return self._fitted_surface.slice_at(float(T))


def check_concentration_interpolation(
        phase: SlowInterpolatingPhase | InterpolatingPhase | RegularSolution,
        phases: list[AbstractLinePhase],
        T: float,
        samples: int,
        plot_excess: bool,
        concentration_range: tuple[float, float],
        plot_error: bool = False,
):
    """Plot free energies of an interpolating phase and its underlying line
    phases to visually assess fit quality.

    Args:
        phase (SlowInterpolatingPhase, InterpolatingPhase, RegularSolution):
            a mixing phase to check
        phases (AbstractLinePhase): list of phases that are interpolated
        T (float): at which temperature to check interpolation
        samples (int): number of sampling points for plot
        plot_excess (bool): if True, subtract free energy at concentration range endpoints for legibility
        concentration_range (tuple of float): min/max concentration range
        plot_error (bool): if True, plot only the interpolation error at the samples instead of the free energies"""

    if plot_error:
        cs, err = [], []
        for p in phases:
            cline = p.line_concentration
            line_free_energy = p.line_free_energy(T)
            # line_free_energy doesn't automatically respect add_entropy, unlike free_energy
            if phase.add_entropy:
                line_free_energy -= T * S(cline)
            cs.append(cline)
            err.append(phase.free_energy(T, cline) - line_free_energy)
        plt.scatter(cs, err, label=phase.name)
        return

    cmin, cmax = concentration_range
    x = np.linspace(cmin, cmax, samples)

    free_energy = phase.free_energy(T, x)

    if plot_excess:
        p_min = min(phases, key=lambda p: p.fixed_concentration)
        p_max = max(phases, key=lambda p: p.fixed_concentration)
        f_min = p_min.line_free_energy(T)
        f_max = p_max.line_free_energy(T)

        # line_free_energy doesn't automatically respect add_entropy, unlike free_energy
        if phase.add_entropy:
            f_min -= T * S(cmin)
            f_max -= T * S(cmax)

        free_energy -= (((cmax-x)*f_min + (x-cmin)*f_max)/(cmax-cmin))

    plt.plot(x, free_energy, label=phase.name)

    for p in phases:
        line_free_energy = p.line_free_energy(T)
        cline = p.line_concentration

        if phase.add_entropy:
            line_free_energy -= T * S(cline)

        if plot_excess:
            line_free_energy -= (((cmax-cline)*f_min + (cline-cmin)*f_max)/(cmax-cmin))

        plt.scatter(cline, line_free_energy)


# The point-defect classes now live in landau.phases.pointdefects. The pre-split
# public names are re-exported here for back-compat and removed at the 2.0 release:
# AbstractPointDefect (an ABC, kept subclassable) as a plain alias, the concrete
# classes behind a deprecation shim pointing at the new module. The classes added
# with this split -- AbstractPointDefectSublattice and
# LowTemperatureExpansionSublattice -- are *not* re-exported; import them from
# landau.phases.pointdefects.
from . import pointdefects as _pointdefects

# AbstractPointDefect predates the split, so it is re-exported unchanged and stays
# usable as a base class.
AbstractPointDefect = _pointdefects.AbstractPointDefect


@deprecate("import it from landau.phases.pointdefects instead", version="2.0")
def ConstantPointDefect(*args, **kwargs):
    return _pointdefects.ConstantPointDefect(*args, **kwargs)


@deprecate("import it from landau.phases.pointdefects instead", version="2.0")
def PointDefectSublattice(*args, **kwargs):
    return _pointdefects.PointDefectSublattice(*args, **kwargs)


@deprecate("import it from landau.phases.pointdefects instead", version="2.0")
def PointDefectedPhase(*args, **kwargs):
    return _pointdefects.PointDefectedPhase(*args, **kwargs)


from .asewrapper import AsePhase
