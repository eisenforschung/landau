from dataclasses import dataclass
from functools import cache
from typing import Iterable

import numpy as np
from scipy.special import expit

from . import Phase, AbstractLinePhase, kB, _scalarize


__all__ = [
    "IntermetallicPhase",
]


def _lower_hull(c, f):
    """Indices of the lower-convex-hull vertices of the points ``(c, f)``.

    Andrew's monotone chain over points pre-sorted by ``(c, f)``: a point is a
    vertex of the lower envelope unless it lies on or above the chord between its
    neighbours. These vertices are the connection points of the piecewise-linear
    low-temperature free energy -- the stable compounds; metastable compounds
    sitting above the hull are dropped.
    """
    keep = []
    for i in range(len(c)):
        # pop while the last segment does not turn upward at the new point
        # (cross product of the last edge with the candidate edge <= 0)
        while len(keep) >= 2:
            o, a = keep[-2], keep[-1]
            cross = (c[a] - c[o]) * (f[i] - f[o]) - (f[a] - f[o]) * (c[i] - c[o])
            if cross <= 0:
                keep.pop()
            else:
                break
        keep.append(i)
    return keep


@dataclass(frozen=True, eq=True)
class IntermetallicPhase(Phase):
    r"""A chain of ideal-solution segments over the low-temperature convex hull.

    Built for intermetallics: a set of stoichiometric compounds whose free
    energy is piecewise linear at low temperature and rounds off as the
    temperature lifts configurational entropy along each segment. Like
    :class:`InterpolatingPhase` it is given line-phase-like free energies sampled
    across the concentration axis, but instead of fitting one smooth curve it
    keeps the compounds as the structure of the free-energy surface.

    **Connection points.** At ``T_low`` the lower convex hull of the sampled
    ``(c, f)`` points is taken; its vertices are the *connection points* -- the
    stable compounds (the kinks of the piecewise-linear low-T curve). Compounds
    lying above the hull are metastable and dropped. Dropping them is required
    for correctness, not cosmetics: a compound above its neighbours' tie line has
    out-of-order segment slopes and would otherwise plant a spurious stable phase
    below the true hull. The connection free energies are reused verbatim at
    every temperature, so the connection points follow whatever temperature
    dependence the line phases carry (e.g. an SGTE curve).

    **Segments.** Each segment between adjacent connection points ``c_k`` and
    ``c_{k+1}`` is an ideal solution between its two compounds, with
    ``L_k = c_{k+1} - c_k`` exchange sites per atom and a slope (the chemical
    potential at which its two endpoints are degenerate)

    .. math:: s_k = \frac{g_{k+1} - g_k}{c_{k+1} - c_k}.

    The site count grows with the segment length, so a short segment carries
    little configurational entropy.

    **Semi-grand potential.** The segments fill *independently* -- segment ``k``
    activates around its own slope ``s_k`` as the chemical potential sweeps -- so
    their contributions are summed rather than minimised. This is what keeps the
    free-energy surface smooth: there is no hard ``min`` to switch between
    segments, so no cusp at the connection points. The result is written
    directly in chemical-potential space (no concentration-space curve is built
    and re-transformed):

    .. math::

        \phi(\Delta\mu) &= g_0 - \Delta\mu\, c_0
            - k_B T \sum_k L_k \ln\!\big(1 + e^{(\Delta\mu - s_k)/k_B T}\big), \\
        c(\Delta\mu) &= c_0 + \sum_k L_k\, \sigma\!\big((\Delta\mu - s_k)/k_B T\big),

    with :math:`\sigma` the logistic function. The configurational smoothing
    scale is :math:`k_B T`, so the curve is piecewise linear at low temperature
    (each compound a stable phase over a chemical-potential window) and rounds
    off as the temperature rises. ``c = -\partial\phi/\partial(\Delta\mu)`` holds
    exactly, ``c`` stays within ``[c_0, c_N]`` (saturating at the terminal
    compounds), and a single ``[0, 1]`` segment reproduces :class:`IdealSolution`
    exactly.
    """

    phases: Iterable[AbstractLinePhase]
    """Line phases sampling the free-energy surface; the stable subset becomes
    the connection points."""
    T_low: float = 0.0
    """Temperature at which the connection points (the convex-hull vertices) are
    identified. Should be low enough that the ordered structure is intact."""

    def __post_init__(self, *args, **kwargs):
        object.__setattr__(self, "phases", tuple(self.phases))
        if len(self.phases) == 0:
            raise ValueError("IntermetallicPhase needs at least one line phase")

    @property
    @cache
    def _connections(self):
        """``(concentrations, phases)`` of the connection points, sorted by ``c``."""
        c = np.array([p.line_concentration for p in self.phases], dtype=float)
        f = np.array([p.line_free_energy(self.T_low) for p in self.phases], dtype=float)
        order = np.lexsort((f, c))  # by concentration, then free energy
        c, phases = c[order], [self.phases[i] for i in order]
        keep = _lower_hull(c, f[order])
        return c[keep], [phases[i] for i in keep]

    @property
    def connection_concentrations(self):
        """Concentrations of the stable compounds that anchor the segments."""
        return self._connections[0]

    def _solve_fixed_T(self, T, dmu):
        """``(phi, c)`` for one ``T`` over a whole ``dmu`` array.

        Sums the independent per-segment ideal-solution contributions; no
        optimiser and no concentration-space curve -- the semigrand potential is
        evaluated directly in chemical-potential space.
        """
        cs, comps = self._connections
        g = np.array([p.line_free_energy(T) for p in comps], dtype=float)
        dmu = np.asarray(dmu, dtype=float)
        flat = dmu.ravel()
        kT = kB * T

        L = np.diff(cs)              # segment lengths (empty for a single compound)
        s = np.diff(g) / L           # segment slopes
        arg = (flat[:, None] - s[None, :]) / kT
        # log(1 + exp(arg)) via logaddexp keeps the saturated tails stable
        phi = g[0] - flat * cs[0] - kT * (L[None, :] * np.logaddexp(0.0, arg)).sum(axis=1)
        c = cs[0] + (L[None, :] * expit(arg)).sum(axis=1)
        return phi.reshape(dmu.shape), c.reshape(dmu.shape)

    def _find_phi_c(self, T, dmu):
        T = np.asarray(T, dtype=float)
        dmu = np.asarray(dmu, dtype=float)
        out_shape = np.broadcast_shapes(T.shape, dmu.shape)
        if T.ndim == 0:
            phi, c = self._solve_fixed_T(float(T), dmu)
            phi = np.broadcast_to(phi, out_shape).copy()
            c = np.broadcast_to(c, out_shape).copy()
        else:
            # each distinct T reuses its own connection free energies
            Tb = np.broadcast_to(T, out_shape)
            mub = np.broadcast_to(dmu, out_shape)
            phi = np.empty(out_shape)
            c = np.empty(out_shape)
            for uT in np.unique(Tb):
                m = Tb == uT
                phi[m], c[m] = self._solve_fixed_T(float(uT), mub[m])
        return _scalarize(phi), _scalarize(c)

    def semigrand_potential(self, T, dmu):
        return self._find_phi_c(T, dmu)[0]

    def concentration(self, T, dmu):
        return self._find_phi_c(T, dmu)[1]
