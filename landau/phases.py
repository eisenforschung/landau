from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache, cache
from typing import Iterable

import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.optimize as so
import scipy.spatial as ss
import scipy.special as se

import numpy as np

from .interpolate import TemperatureInterpolator, SGTE, PolyFit, RedlichKister

from scipy.constants import Boltzmann, eV

kB = Boltzmann / eV


def S(c):
    return kB * (se.entr(c) + se.entr(1 - c))


def Sprime(c):
    with np.errstate(divide="ignore"):
        s = -kB * (np.log(c / (1 - c)))
    s[np.isclose(c, 0)] = +np.inf
    s[np.isclose(c, 1)] = -np.inf
    return s


def c_from_dmu(dmu, T, e_defect):
    return 1 / (1 + np.exp(-(dmu - e_defect) / kB / T))


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
        ones = np.ones(np.broadcast(T, dmu).shape)
        if ones.shape != ():
            return self.line_concentration * ones
        else:
            return self.line_concentration

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
        if isinstance(T, np.ndarray):
            U = self.line_energy * np.ones_like(T)
        else:
            U = self.line_energy
        return U - T * self.line_entropy


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

    def check_interpolation(self, Tl=0.9, Tu=1.1, samples=50):
        Ts = np.linspace(np.min(self.temperatures) * Tl, np.max(self.temperatures) * Tu, samples)
        (l,) = plt.plot(Ts, self.line_free_energy(Ts), label=self.name)
        # try to plot about 100 points
        n = max(int(len(self.temperatures) // 100), 1)
        plt.scatter(self.temperatures[::n], self.free_energies[::n], c=l.get_color())


def TemperatureDepandantLinePhase(*args, **kwargs):
    print("TYPO ALERT!")
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
                    phi = f2 - dmu[I]
                else:
                    phi[I] = f2 - dmu[I]
        if phi.shape == ():
            phi = phi.item()
        return phi

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
        ff = np.array([l.line_free_energy(T) for l in self.phases])

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
            x = min(pi, pl, f0)
            if isinstance(x, np.ndarray):
                x = x.item()
            return x
        pl[pl > f0] = f0
        I = np.isnan(pi)
        pi[I] = pl[I]
        if plot:
            plt.plot(M, p, "o-", label="calculated")
            plt.plot(dmu, pi, label="extrapolated")
            plt.legend()
        return pi

    def concentration(self, T, dmu):
        if not isinstance(dmu, np.ndarray):
            dmus = np.linspace(-1, 1, 5) * 1e-3 + dmu
            return self.concentration(T, dmus)[2]
        return np.clip(-np.gradient(self.semigrand_potential(T, dmu), dmu, edge_order=2), 0, 1)

    def check_interpolation(self, T=1000, samples=50):
        x = np.linspace(0, 1, samples)
        plt.plot(x, self.free_energy(T, x), label=self.name)
        for p in self.phases:
            plt.scatter(p.line_concentration, p.line_free_energy(T))


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

        if c.ndim == 0:
            c = c.item()
        if phi.ndim == 0:
            phi = phi.item()
        return phi, c

    def semigrand_potential(self, T, dmu):
        return self._find_phi_c(T, dmu)[0]

    def concentration(self, T, dmu):
        return self._find_phi_c(T, dmu)[1]

    def check_interpolation(self, T=1000, samples=50):
        x = np.linspace(0, 1, samples)
        plt.plot(x, self.free_energy(T, x))
        for p in self.phases:
            plt.scatter(p.line_concentration, p.line_free_energy(T), label=self.name)


class AbstractPointDefect(ABC):
    @abstractmethod
    def excess_free_energy(self, T):
        pass

    # @property
    # @abstractmethod
    # def excess_solutes(self):
    #     pass


@dataclass(frozen=True)
class ConstantPointDefect(AbstractPointDefect):
    """
    A point defect that adds a contribution to the free energy of a host
    lattice.

    Excess energy and entropy are assumed to be
    """

    name: str
    excess_energy: float
    # [E]_N
    excess_entropy: float
    # [S]_N
    excess_solutes: float
    # # [n]_N / N = c_defect - c_reference

    # def __init__(self, name,
    #              excess_energy, excess_solutes,
    #              sublattice, sublattice_fraction,
    #              excess_entropy=0,
    #              # unused
    #              excess_concentration=None,
    # ):
    #     super().__init__()
    #     self.storage.name = name
    #     self.storage.excess_energy = excess_energy
    #     self.storage.excess_entropy = excess_entropy
    #     # self.storage.excess_concentration = excess_concentration
    #     # [n]_N
    #     self.storage.excess_solutes = excess_solutes
    #     # sublattice index; on which sublattice in the host lattice this defect lives
    #     # just to avoid accidental overlap when combining multiple point defects
    #     self.storage.sublattice = sublattice
    #     # the fraction of sites this sub lattice makes up of the host lattice
    #     self.storage.sublattice_fraction = sublattice_fraction

    def excess_free_energy(self, T):
        return self.excess_energy - T * self.excess_entropy

    # For diagnostics only; Sublattice classes only uses excess_free_energy method

    def semigrand_potential_contribution(self, T, dmu):
        fe = self.excess_free_energy(T)
        ne = self.excess_solutes
        return -kB * T * np.log(1 + np.exp(-(fe - ne * dmu) / kB / T))

    def concentration_contribution(self, T, dmu):
        ne = self.excess_solutes
        return ne * self.defect_concentration(T, dmu)

    def defect_concentration(self, T, dmu):
        # analytical derivative of the semigrand potential above
        # c = -dphi/dmu
        # c = [n]_N eta x
        # we want to return x
        fe = self.storage.excess_energy
        ne = self.storage.excess_solutes
        return 1 / (1 + np.exp(+(fe - ne * dmu) / kB / T))


@dataclass(frozen=True)
class PointDefectSublattice:
    """
    Groups together PointDefect that live on the same sublattice within a host
    structure.
    """

    name: str
    sublattice: int
    sublattice_fraction: float
    defects: list[AbstractPointDefect]

    def _get_zes(self, T, dmu):
        fes = [d.excess_free_energy(T) for d in self.defects]
        nes = [d.excess_solutes for d in self.defects]
        return np.array([np.exp(-(fe - ne * dmu) / kB / T) for fe, ne in zip(fes, nes)])

    def semigrand_potential_contribution(self, T, dmu):
        zes = self._get_zes(T, dmu).sum(axis=0)
        dphi = -kB * T * np.log(1 + zes)
        return self.sublattice_fraction * dphi

    def concentration_contribution(self, T, dmu):
        zes = self._get_zes(T, dmu)
        nes = np.array([p.excess_solutes for p in self.defects])
        eta = self.sublattice_fraction
        return eta * sum(ne * ze for ne, ze in zip(nes, zes)) / (1 + zes.sum(axis=0))


@dataclass(frozen=True)
class PointDefectedPhase(Phase):
    """
    Phase that combines any host phase and any number of point defects in it.
    """

    line_phase: AbstractLinePhase
    """Underlying phase object of the host lattice."""
    sublattices: list[PointDefectSublattice]
    """Sublattices and their point defects."""

    def __post_init__(self, *args, **kwargs):
        # TODO check unique sublattice indices on sublattice objects (or maybe not)
        pass

    def semigrand_potential(self, T, dmu):
        phi = self.line_phase.semigrand_potential(T, dmu)
        for l in self.sublattices:
            phi += l.semigrand_potential_contribution(T, dmu)
        return phi

    def concentration(self, T, dmu):
        c = self.line_phase.line_concentration
        for d in self.sublattices:
            c += d.concentration_contribution(T, dmu)
        return c
