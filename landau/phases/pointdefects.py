from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import Phase, AbstractLinePhase, kB, _scalarize


__all__ = [
    "AbstractPointDefect",
    "ConstantPointDefect",
    "AbstractPointDefectSublattice",
    "PointDefectSublattice",
    "LowTemperatureExpansionSublattice",
    "PointDefectedPhase",
]


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

    def excess_free_energy(self, T):
        return self.excess_energy - T * self.excess_entropy


@dataclass(frozen=True)
class AbstractPointDefectSublattice(ABC):
    r"""
    Groups together point defects that live on the same sublattice within a host
    structure and turns their excess potentials into the sublattice's
    contribution to the semi-grand potential and concentration.

    The per-site partition variables :math:`z_i = e^{-\beta[\phi_i]}` are shared
    machinery; concrete subclasses differ only in how those are combined, via two
    hooks evaluated on :math:`s = \sum_i z_i`:

    - :meth:`_site_log_partition` returns :math:`L(s)` with the per-site
      semi-grand potential :math:`-k_B T\, L(s)`;
    - :meth:`_site_normalization` returns the denominator :math:`D(s)` of the
      defect site fraction :math:`x_i = z_i / D(s)`.

    The two are not independent: :math:`c = -\partial\phi/\partial(\Delta\mu)`
    forces :math:`D(s) = 1 / L'(s)`, so a subclass must keep them consistent.
    """

    name: str
    sublattice: int
    sublattice_fraction: float
    defects: list[AbstractPointDefect]

    def _get_zes(self, T, dmu):
        fes = [d.excess_free_energy(T) for d in self.defects]
        nes = [d.excess_solutes for d in self.defects]
        return np.array([np.exp(-(fe - ne * dmu) / kB / T) for fe, ne in zip(fes, nes)])

    @abstractmethod
    def _site_log_partition(self, zes_sum):
        """Per-site semi-grand potential in units of :math:`-k_B T`."""

    @abstractmethod
    def _site_normalization(self, zes_sum):
        """Denominator of the defect site fraction :math:`x_i = z_i / D`."""

    def semigrand_potential_contribution(self, T, dmu):
        zes_sum = self._get_zes(T, dmu).sum(axis=0)
        dphi = -kB * T * self._site_log_partition(zes_sum)
        return self.sublattice_fraction * dphi

    def concentration_contribution(self, T, dmu):
        zes = self._get_zes(T, dmu)
        nes = np.array([p.excess_solutes for p in self.defects])
        eta = self.sublattice_fraction
        return eta * sum(ne * ze for ne, ze in zip(nes, zes)) / self._site_normalization(zes.sum(axis=0))


@dataclass(frozen=True)
class PointDefectSublattice(AbstractPointDefectSublattice):
    r"""
    Exact point defect sublattice with site competition.

    Evaluates the full semi-grand potential per site,
    :math:`-k_B T \ln(1 + \sum_i z_i)` with :math:`z_i = e^{-\beta[\phi_i]}`. The
    denominator :math:`1 + \sum_i z_i` encodes site competition (one site holds
    one defect, so the site fractions :math:`x_i = z_i / (1 + \sum_{i'} z_{i'})`
    obey :math:`\sum_i x_i \le 1`).
    """

    def _site_log_partition(self, zes_sum):
        return np.log(1 + zes_sum)

    def _site_normalization(self, zes_sum):
        return 1 + zes_sum


@dataclass(frozen=True)
class LowTemperatureExpansionSublattice(AbstractPointDefectSublattice):
    r"""
    Point defect sublattice in the low-temperature (dilute) limit.

    The leading term of the low-temperature expansion of
    :class:`PointDefectSublattice` replaces :math:`\ln(1 + \sum_i z_i)` with
    :math:`\sum_i z_i` and drops the site-competition denominator, leaving the
    defect site fractions :math:`x_i = z_i` independent and unnormalised. This is
    accurate while the defects are dilute (:math:`z_i \ll 1`, i.e. formation
    energies large compared to :math:`k_B T`) and lets each defect be treated as
    an isolated, non-interacting excitation; away from that limit the populations
    are no longer bounded by site competition and the exact form should be used.
    """

    def _site_log_partition(self, zes_sum):
        return zes_sum

    def _site_normalization(self, zes_sum):
        return 1.0


@dataclass(frozen=True)
class PointDefectedPhase(Phase):
    """
    Phase that combines any host phase and any number of point defects in it.
    """

    line_phase: AbstractLinePhase
    """Underlying phase object of the host lattice."""
    sublattices: list[AbstractPointDefectSublattice]
    """Sublattices and their point defects."""

    def __post_init__(self, *args, **kwargs):
        # TODO check unique sublattice indices on sublattice objects (or maybe not)
        pass

    def _raw_phi_c(self, T, dmu):
        """Unclamped semi-grand potential and concentration (sum of the host and
        sublattice contributions)."""
        phi = self.line_phase.semigrand_potential(T, dmu)
        c = self.line_phase.line_concentration
        for s in self.sublattices:
            phi = phi + s.semigrand_potential_contribution(T, dmu)
            c = c + s.concentration_contribution(T, dmu)
        return phi, c

    # keep exp arguments comfortably below the ~709 float64 overflow threshold
    _EXP_LIMIT = 650.0

    def _safe_dmu_bound(self, T):
        """A ``|dmu|`` large enough to straddle any saturation crossing yet small
        enough that ``exp(-(fe - n dmu)/kB/T)`` cannot overflow. At very low T the
        crossing may sit outside this bound; then ``c`` cannot be probed away from
        stoichiometry without overflow and ``_saturation_window`` reports no
        crossing (correct for the bounded exact model; the unbounded expansion is
        not used in that regime)."""
        defects = [d for s in self.sublattices for d in s.defects]
        fe_max = max((abs(d.excess_free_energy(T)) for d in defects), default=0.0)
        n_max = max((abs(d.excess_solutes) for d in defects), default=1) or 1
        return max(1e-6, (self._EXP_LIMIT * kB * T - fe_max) / n_max)

    def _saturation_window(self, T):
        """Precompute the chemical potentials where the raw concentration crosses
        0 and 1 (``-inf`` / ``+inf`` when it never does, i.e. c is bounded on that
        side). Knowing these a priori means the raw, possibly overflowing,
        expression is only ever evaluated for ``dmu`` inside the window."""
        M = self._safe_dmu_bound(T)

        def craw(m):
            with np.errstate(over="ignore", invalid="ignore"):
                return float(np.ravel(self._raw_phi_c(T, m)[1])[0])

        # c is monotonically increasing in dmu, so the crossings are unique; a
        # non-finite endpoint (overflow at very low T) reads as "no crossing".
        c_lo, c_hi = craw(-M), craw(M)
        dmu_lo = brentq(lambda m: craw(m), -M, M) if c_lo < 0.0 < c_hi else -np.inf
        dmu_hi = brentq(lambda m: craw(m) - 1.0, -M, M) if c_lo < 1.0 < c_hi else np.inf
        return dmu_lo, dmu_hi

    def _clamp_fixed_T(self, T, dmu):
        """Clamp the concentration to [0, 1] for a scalar ``T`` and continue phi
        as the tangent line past each saturation point so that
        ``c = -d(phi)/d(dmu)`` still holds (the unnormalised LTE site fractions
        would otherwise drive c out of [0, 1] and phi to -inf)."""
        dmu = np.asarray(dmu, dtype=float)
        dmu_lo, dmu_hi = self._saturation_window(T)
        phi = np.empty(dmu.shape, dtype=float)
        c = np.empty(dmu.shape, dtype=float)

        # the exact model is unclamped (window is +-inf) and the raw expression can
        # overflow at the extreme dmu that mu-range autodetection probes; those
        # points are never stable, so ignore the overflow rather than warn
        interior = (dmu >= dmu_lo) & (dmu <= dmu_hi)
        if interior.any():
            with np.errstate(over="ignore", invalid="ignore"):
                phi[interior], c[interior] = self._raw_phi_c(T, dmu[interior])

        upper = dmu > dmu_hi  # c would exceed 1: line phase at c = 1
        if upper.any():
            phi_hi = float(np.ravel(self._raw_phi_c(T, dmu_hi)[0])[0])
            phi[upper] = phi_hi - (dmu[upper] - dmu_hi)  # slope -1 => c = 1
            c[upper] = 1.0

        lower = dmu < dmu_lo  # c would fall below 0: line phase at c = 0
        if lower.any():
            phi_lo = float(np.ravel(self._raw_phi_c(T, dmu_lo)[0])[0])
            phi[lower] = phi_lo  # slope 0 => c = 0
            c[lower] = 0.0

        return phi, c

    def _phi_c(self, T, dmu):
        Tb, mub = np.broadcast_arrays(np.asarray(T, dtype=float), np.asarray(dmu, dtype=float))
        phi = np.empty(Tb.shape, dtype=float)
        c = np.empty(Tb.shape, dtype=float)
        fphi, fc, fT, fmu = phi.ravel(), c.ravel(), Tb.ravel(), mub.ravel()
        for t in np.unique(fT):
            at = fT == t
            fphi[at], fc[at] = self._clamp_fixed_T(float(t), fmu[at])
        return _scalarize(phi), _scalarize(c)

    def semigrand_potential(self, T, dmu):
        return self._phi_c(T, dmu)[0]

    def concentration(self, T, dmu):
        return self._phi_c(T, dmu)[1]
