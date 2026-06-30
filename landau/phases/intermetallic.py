from dataclasses import dataclass
from functools import cache
from typing import Iterable

import numpy as np
from scipy.special import expit

from . import Phase, AbstractLinePhase, kB, S, _scalarize


__all__ = [
    "IntermetallicPhase",
]


def _lower_hull(c, f):
    """Indices of the lower-convex-hull vertices of the points ``(c, f)``.

    Andrew's monotone chain over points pre-sorted by ``(c, f)``: a point is a
    vertex of the lower envelope unless it lies on or above the chord between its
    neighbours. These vertices are the connection points of the piecewise-linear
    low-temperature free energy -- the compounds of locally maximal (positive)
    curvature; metastable compounds sitting above the hull are dropped.
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
    r"""A piecewise ideal solution over the low-temperature convex hull.

    Built for intermetallics: a set of stoichiometric compounds whose free
    energy is piecewise linear at low temperature and acquires configurational
    entropy along each segment as temperature rises. Like
    :class:`InterpolatingPhase` it is given line-phase-like free energies sampled
    across the concentration axis, but instead of fitting one smooth curve it
    keeps the compounds as the structure of the free-energy surface.

    **Connection points.** At ``T_low`` the lower convex hull of the sampled
    ``(c, f)`` points is taken; its vertices are the *connection points*, i.e.
    the stable compounds (the kinks of the piecewise-linear low-T curve). The
    compounds lying above the hull are metastable and dropped. Their free
    energies are reused verbatim at every temperature, so the connection points
    follow whatever temperature dependence the line phases carry (e.g. an SGTE
    curve) and the whole structure decreases with temperature.

    **Segment entropy.** Between two adjacent connection points at ``c_a`` and
    ``c_b`` the free energy is the tie line plus a *segment ideal entropy*,
    scaled by the segment length so it vanishes at the connection points:

    .. math::

        f(c, T) = (1-x)\,f_a(T) + x\,f_b(T) - T\,(c_b - c_a)\,S(x),
        \qquad x = \frac{c - c_a}{c_b - c_a},

    with :math:`S(x) = -k_B[x\ln x + (1-x)\ln(1-x)]`. Each segment is therefore
    an ideal solution between its two compounds: the substitutable site count
    grows with the segment length, so a short segment carries little entropy and
    a single segment spanning ``[0, 1]`` reproduces :class:`IdealSolution`
    exactly.

    **Equilibrium.** The semi-grand potential minimises ``f(c) - dmu*c`` over
    ``c``. Within a segment this has the closed form of a (scaled) ideal
    solution, so no numerical optimisation is needed: each segment contributes

    .. math::

        \phi_k = f_a - \Delta\mu\, c_a
                 - k_B T\,(c_b - c_a)\,\ln\!\big(1 + e^{-(g - \Delta\mu)/k_B T}\big),
        \qquad g = \frac{f_b - f_a}{c_b - c_a},

    and the global potential is the minimum over segments. The minimising
    concentration is :math:`c = c_a + (c_b - c_a)\,\sigma((\Delta\mu - g)/k_B T)`
    with :math:`\sigma` the logistic function, so ``c = -d(phi)/d(dmu)`` holds to
    machine precision and ``c`` stays within the span of the connection points.
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

    def free_energy(self, T, c):
        """Piecewise free energy ``f(c, T)`` (tie line minus segment entropy)."""
        cs, phases = self._connections
        f = np.array([p.line_free_energy(T) for p in phases], dtype=float)
        c = np.asarray(c, dtype=float)
        if len(cs) == 1:
            return _scalarize(np.where(c == cs[0], f[0], np.nan))
        s = np.clip(np.searchsorted(cs, c, side="right") - 1, 0, len(cs) - 2)
        ca, cb, fa, fb = cs[s], cs[s + 1], f[s], f[s + 1]
        x = (c - ca) / (cb - ca)
        # clip only the entropy argument so out-of-range c extends the edge line
        fe = fa + (fb - fa) * x - T * (cb - ca) * S(np.clip(x, 0.0, 1.0))
        return _scalarize(fe)

    def _solve_fixed_T(self, T, dmu):
        """Minimise ``f(c) - dmu*c`` for one ``T`` over a whole ``dmu`` array.

        Each segment is a scaled ideal solution with an analytic minimum, so the
        solve is one vectorised evaluation per segment plus an ``argmin`` over
        segments -- no optimiser, exact to machine precision.
        """
        cs, phases = self._connections
        f = np.array([p.line_free_energy(T) for p in phases], dtype=float)
        dmu = np.asarray(dmu, dtype=float)
        flat = dmu.ravel()

        if len(cs) == 1:
            phi = f[0] - flat * cs[0]
            c = np.full_like(flat, cs[0])
            return phi.reshape(dmu.shape), c.reshape(dmu.shape)

        ca, cb, fa, fb = cs[:-1], cs[1:], f[:-1], f[1:]
        L = cb - ca
        g = (fb - fa) / L  # segment slope: dmu where the two endpoints are degenerate
        kT = kB * T

        z = (g[:, None] - flat[None, :]) / kT
        c_seg = ca[:, None] + L[:, None] * expit(-z)
        # log(1 + exp(-z)) via logaddexp keeps the saturated tails stable
        phi_seg = fa[:, None] - flat[None, :] * ca[:, None] - kT * L[:, None] * np.logaddexp(0.0, -z)

        best = phi_seg.argmin(axis=0)[None, :]
        phi = np.take_along_axis(phi_seg, best, axis=0)[0]
        c = np.take_along_axis(c_seg, best, axis=0)[0]
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
