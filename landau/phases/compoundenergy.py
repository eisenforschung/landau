r"""Binary Compound Energy Formalism (CEF) phase.

A minimal, general CEF phase for a binary A--B system with an arbitrary number
of sublattices.  The model is the one used throughout CALPHAD assessments (e.g.
Sundman, Fries & Oates, *Calphad* **22** (1998) 335 for Au--Cu):

.. math::

    G(\vec y, T) = \underbrace{\sum_\text{end-members}
        {}^0G_c(T) \prod_s p_s^{(c)}}_{G_\text{ref}}
        + \underbrace{k_B T \sum_s a_s \big(y_s \ln y_s
        + (1-y_s)\ln(1-y_s)\big)}_{G_\text{ideal}}
        + \underbrace{\sum_i {}^EG_i(\vec y, T)}_{G_\text{excess}}

where ``y_s`` is the site fraction of component B on sublattice ``s``, ``a_s``
its relative size (summing to 1), and ``p_s^{(c)} = y_s`` if end-member ``c``
carries B on sublattice ``s`` else ``1 - y_s``.  The overall B concentration is
``c(y) = sum_s a_s y_s``.

The three ingredients are **composable objects**, in the spirit of the
point-defect classes: a phase is assembled from a tuple of :class:`Sublattice`
(each an ``a_s``), a tuple of :class:`Endmember` (each a corner configuration and
its ``{}^0G_c(T)``), and a tuple of :class:`ExcessTerm` that are summed.  Each
excess term supplies its own analytic gradient where it has one
(:class:`RegularSolutionExcess`) and falls back to a finite difference otherwise
(:class:`CallableExcess`), so the solver reads exact per-term gradients rather
than one blanket finite difference over an opaque callable.

Semi-grand solve -- one minimisation, straight from site fractions.  The site
fractions are internal (order) parameters and the semi-grand potential is their
Legendre transform,

.. math::

    \phi(T, \Delta\mu) = \min_{\vec y \in [0,1]^S}\!\big[G(\vec y,T) - \Delta\mu\,c(\vec y)\big].

Setting :math:`\partial_{y_s}[\,G - \Delta\mu\,c\,] = 0` gives the self-consistent
(Bragg-Williams / BGBW) fixed point

.. math::

    y_s = \sigma\!\Big(\frac{\Delta\mu - a_s^{-1}\,\partial_{y_s}(G_\text{ref}+G_\text{excess})}{k_B T}\Big),

because the ideal term contributes :math:`k_B T\,a_s\,\mathrm{logit}(y_s)`, linear in
``logit(y_s)``.  Iterating this map (damped) drives ``y`` to the stationary point of
each basin; it is vectorised over the whole ``(T, dmu)`` grid -- a multi-``T`` grid is
one batched solve per start, not one per ``T`` -- and run from a disordered start plus
every ordered end-member corner, keeping the lowest ``phi`` per point -- so
ordered/disordered coexistence surfaces as ``c(dmu)`` jumps without any
fixed-composition inner minimisation.  ``c = -partial phi/partial dmu``
holds at the stationary point.  All energies are per atom (eV); convert CALPHAD J/mol
by dividing by 96485.33.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Callable, ClassVar

import numpy as np
import scipy.optimize as so
import scipy.special as se

from . import Phase, kB, _scalarize

__all__ = [
    "CompoundEnergyPhase",
    "Sublattice",
    "Endmember",
    "ExcessTerm",
    "RegularSolutionExcess",
    "CallableExcess",
]

_FD = 1e-6  # finite-difference step for excess gradients without a closed form


@dataclass(frozen=True)
class Sublattice:
    """One sublattice of a :class:`CompoundEnergyPhase`.

    Args:
        multiplicity: relative sublattice size ``a_s``.  The phase's
            multiplicities must sum to 1.
    """

    multiplicity: float


@dataclass(frozen=True)
class Endmember:
    """A compound-energy end-member: an ordered corner and its Gibbs energy.

    Args:
        occupation: length-``S`` configuration tuple (``0`` = A, ``1`` = B on that
            sublattice) picking one constituent per sublattice.
        energy: callable ``G(T)`` returning the end-member energy in eV/atom.
    """

    occupation: tuple[int, ...]
    energy: Callable[[float], float]


@dataclass(frozen=True)
class ExcessTerm(ABC):
    """One additive excess-energy contribution ``EG_i(y, T)`` (eV/atom).

    Subclasses implement :meth:`energy`; :meth:`gradient` defaults to a central
    finite difference in every sublattice but is overridden with the closed form
    where one exists.  ``y`` has its site fractions on the last axis and must
    broadcast over any leading axes.
    """

    @abstractmethod
    def energy(self, y, T):
        """Excess energy for ``y`` of shape ``(..., S)`` -> ``(...)`` (eV/atom)."""

    def gradient(self, y, T):
        """``d EG_i/dy_s`` for ``y`` of shape ``(D, S)`` -> ``(D, S)``.

        Central finite difference in every sublattice, evaluated in one
        :meth:`energy` call: the ``2*S`` perturbed batches are stacked on a
        leading axis rather than calling :meth:`energy` ``2*S`` times.
        """
        y = np.asarray(y, dtype=float)
        D, S = y.shape
        h = _FD
        P = np.broadcast_to(y, (2, S, D, S)).copy()
        diag = np.arange(S)
        P[0, diag, :, diag] = np.clip(y[:, diag].T + h, 1e-12, 1 - 1e-12)
        P[1, diag, :, diag] = np.clip(y[:, diag].T - h, 1e-12, 1 - 1e-12)
        ex = self.energy(P, T)  # (2, S, D)
        denom = P[0, diag, :, diag] - P[1, diag, :, diag]  # (S, D)
        return ((ex[0] - ex[1]) / denom).T


@dataclass(frozen=True)
class RegularSolutionExcess(ExcessTerm):
    """Regular-solution excess ``L(T) * sum_s y_s (1 - y_s)`` (eV/atom).

    Args:
        coefficient: callable ``L(T)`` in eV/atom.
    """

    coefficient: Callable[[float], float]

    def energy(self, y, T):
        y = np.asarray(y, dtype=float)
        return np.asarray(self.coefficient(T)) * np.sum(y * (1 - y), axis=-1)

    def gradient(self, y, T):
        y = np.asarray(y, dtype=float)
        return np.asarray(self.coefficient(T), dtype=float)[..., None] * (1 - 2 * y)


@dataclass(frozen=True)
class CallableExcess(ExcessTerm):
    """Excess energy from an arbitrary callable ``EG(y, T)`` (eV/atom).

    The escape hatch for interactions without a bundled closed form (e.g. a CEF
    reciprocal term): the gradient is the inherited finite difference.  ``y`` has
    its site fractions on the last axis and must broadcast over any leading axes.

    Args:
        function: callable ``EG(y, T)`` in eV/atom.
    """

    function: Callable[[np.ndarray, float], float]

    def energy(self, y, T):
        return self.function(y, T)


@dataclass(frozen=True)
class CompoundEnergyPhase(Phase):
    """Binary compound-energy-formalism phase (real :class:`~landau.phases.Phase`).

    Args:
        name: phase name.
        sublattices: the sublattices ``(Sublattice(a_0), ...)``; the ``a_s`` must
            sum to 1.
        endmembers: the ``2**S`` :class:`Endmember` corners (one per configuration).
        excess: additive :class:`ExcessTerm` contributions, summed.  Empty = ideal.
        orderings: which ordered end-member corners the solve is seeded from.
            ``None`` (default) uses all ``2**S`` corners, so the phase is the full
            partitioning CEF (global minimum over every ordering).  Pass a single
            corner, e.g. ``((1, 1, 0, 0),)``, to pin the phase to one ordering basin --
            useful for rendering an ordered superstructure (L1_0, L1_2, ...) as its own
            phase that competes with a disordered phase in ``calc_phase_diagram``.
        include_disordered_seed: also seed from the disordered point ``(c, ..., c)``.
            Keep ``True`` for a disordered or partitioning phase; set ``False`` on a
            basin-pinned ordered phase so it stays on its ordered branch.
    """

    sublattices: tuple[Sublattice, ...] = ()
    endmembers: tuple[Endmember, ...] = ()
    excess: tuple[ExcessTerm, ...] = ()
    orderings: tuple[tuple[int, ...], ...] | None = None
    include_disordered_seed: bool = True

    # solver tuning (ClassVar -> never treated as dataclass fields)
    _scf_max_iter: ClassVar[int] = 80   # max SCF map evaluations per start (2 per Steffensen cycle)
    _scf_tol: ClassVar[float] = 1e-9    # max|dy| convergence tolerance
    _y_margin: ClassVar[float] = 0.02   # pinned sublattices held this far from 0.5 (keeps orderings distinct)

    def __post_init__(self):
        subs = tuple(self.sublattices)
        object.__setattr__(self, "sublattices", subs)
        ems = tuple(sorted(self.endmembers, key=lambda e: tuple(e.occupation)))
        object.__setattr__(self, "endmembers", ems)
        object.__setattr__(self, "excess", tuple(self.excess))
        S = len(subs)
        a = np.array([s.multiplicity for s in subs], dtype=float)
        assert len(ems) == 2**S, f"need 2**S={2**S} end-members for {S} sublattices"
        assert abs(a.sum() - 1) < 1e-9, "sublattice multiplicities must sum to 1"
        object.__setattr__(self, "_a", a)
        # config array (1 where B sits) and the +/-1 sign per (endmember, sublattice)
        # for the analytic G_ref gradient, plus the ordered corners the solve seeds from
        cfg = np.array([e.occupation for e in ems], dtype=float)
        object.__setattr__(self, "_configs", cfg)
        object.__setattr__(self, "_sign", 2.0 * cfg - 1.0)
        seeds = product((0, 1), repeat=S) if self.orderings is None else self.orderings
        object.__setattr__(self, "_corners", [np.clip(np.array(c, dtype=float), 1e-9, 1 - 1e-9) for c in seeds])
        assert self._corners or self.include_disordered_seed, "phase has no ordering seed"
        # A basin-pinned ordered phase -- exactly one ordering and no disordered seed --
        # is confined to that ordering's sublattice symmetry (sublattices sharing the
        # same occupation stay equal, so the solve cannot drift into a different
        # ordering) *and* to that ordering's quadrant of site-fraction space: every
        # B-rich sublattice is clamped above 0.5 and every A-rich one below (by
        # ``_y_margin``), so the phase cannot relax onto the disordered diagonal or
        # into a neighbouring ordering.  This makes it a distinct ordered phase whose
        # free energy is finite and continuous everywhere -- where it is not the ground
        # state it simply loses the ``argmin`` -- rather than carrying a ``phi = +inf``
        # "absent" flag, whose discontinuity broke boundary tracing and aliased the
        # ordered domes' metastable continuations.
        pinned = self.orderings is not None and len(self._corners) == 1 and not self.include_disordered_seed
        groups = None
        y_lo = np.full(S, 1e-9)
        y_hi = np.full(S, 1 - 1e-9)
        if pinned:
            pat = np.round(self._corners[0]).astype(int)
            groups = [np.where(pat == v)[0] for v in np.unique(pat)]
            m = self._y_margin
            y_lo = np.where(pat == 1, 0.5 + m, 1e-9)
            y_hi = np.where(pat == 1, 1 - 1e-9, 0.5 - m)
        object.__setattr__(self, "_pinned", pinned)
        object.__setattr__(self, "_groups", groups)
        object.__setattr__(self, "_y_lo", y_lo)
        object.__setattr__(self, "_y_hi", y_hi)
        # warm-start cache: rounded T -> (sorted dmu, y*) from the last solve at that
        # T. Refinement probes clustered (T, dmu), so seeding the SCF from the nearest
        # previous solution converges in a few iterations instead of ~40 from a corner.
        object.__setattr__(self, "_warm", {})

    # --- model (vectorised over any leading axes of y) -----------------------

    @lru_cache(maxsize=2000)
    def _endmember_evals(self, T):
        """End-member energies at ``T`` in end-member (occupation-sorted) order."""
        return np.array([e.energy(T) for e in self.endmembers])

    def _excess_energy(self, y, T):
        """Summed excess energy ``sum_i EG_i(y, T)`` (0.0 if there are no terms)."""
        return sum((term.energy(y, T) for term in self.excess), 0.0)

    def _free_energy(self, T, y, evals):
        """Gibbs energy ``G(y, T)`` for ``y`` of shape ``(..., S)`` -> ``(...)`` (eV/atom).

        ``evals`` is ``(M,)`` for a scalar ``T`` or ``(..., M)`` for a per-element ``T``.
        """
        p = np.where(self._configs == 1, y[..., None, :], 1.0 - y[..., None, :])  # (...,M,S)
        gref = np.sum(p.prod(axis=-1) * evals, axis=-1)  # (...,)  broadcasts evals (M,) or (...,M)
        gid = -kB * T * ((se.entr(y) + se.entr(1.0 - y)) @ self._a)
        return gref + gid + self._excess_energy(y, T)

    def free_energy(self, T, y):
        """Gibbs energy at site fractions ``y`` (length ``S``) and temperature ``T`` (eV/atom)."""
        return self._free_energy(T, np.asarray(y, dtype=float), self._endmember_evals(T))

    def composition(self, y):
        """Overall B concentration ``sum_s a_s y_s``."""
        return np.asarray(y, dtype=float) @ self._a

    def _grad_non_ideal(self, T, y, evals):
        """Gradient ``d(G_ref + G_excess)/dy_s`` for ``y`` of shape ``(D, S)`` -> ``(D, S)``."""
        p = np.where(self._configs == 1, y[..., None, :], 1.0 - y[..., None, :])  # (...,M,S)
        # leave-one-out product: prod_{s'!=s} p = (prod_s p) / p_s  (p bounded away from 0)
        loo = p.prod(axis=-1, keepdims=True) / p  # (...,M,S)
        # evals is (M,) for scalar T or (...,M) for per-element T; broadcast either way
        gref = np.sum(evals[..., :, None] * (self._sign * loo), axis=-2)  # (...,S)
        if not self.excess:
            return gref
        return gref + sum(term.gradient(y, T) for term in self.excess)

    # --- direct semi-grand solve (one minimisation per start) ----------------

    def _fixed_point_step(self, ya, idx, dmu, Tarr, ev, a):
        """One application of the SCF map ``y -> sigmoid((dmu - grad/a)/kT)`` to the
        active columns ``idx``, clamped to ``[_y_lo, _y_hi]`` and (for a pinned phase)
        averaged over each ordering group."""
        field = self._grad_non_ideal(Tarr[idx], ya, ev[idx]) / a
        yn = np.clip(se.expit((dmu[idx, None] - field) / (kB * Tarr[idx, None])), self._y_lo, self._y_hi)
        if self._groups is not None:  # confine to the ordering's sublattice symmetry
            for g in self._groups:
                yn[:, g] = yn[:, g].mean(axis=1, keepdims=True)
        return yn

    def _scf(self, T, dmu, evals, y0):
        """Solve the SCF fixed point for one start, vectorised over ``(T, dmu)``.

        ``T`` is a scalar or a ``(len(dmu),)`` array (per-element temperature) and
        ``evals`` the matching ``(M,)`` or ``(len(dmu), M)`` end-member energies.
        Returns the stationary site fractions ``y`` of shape ``(len(dmu), S)``.

        The plain damped map converges only linearly and, near a spinodal (the fcc
        disordered branch between two ordered domes), has a fixed-point multiplier
        below ``-1`` and settles into a period-2 limit cycle.  This uses Steffensen's
        method -- apply the map twice, then Aitken-``Δ²`` extrapolate -- in the
        **logit variable** ``u = logit(y)``: the extrapolation is done on ``u`` (which
        is unbounded, so it can ride through the oscillation and accelerate the linear
        rate several-fold), and the ``sigmoid`` maps it straight back into ``(0, 1)``
        so ``y`` can never leave range -- the failure mode of a bare Aitken step.  For
        a pinned phase ``u`` is confined to its ordering quadrant's logit box.  Each
        cycle costs two map evaluations; columns drop from the working set as they
        converge.
        """
        a = self._a
        N = dmu.size
        Tarr = np.broadcast_to(np.asarray(T, dtype=float), (N,))
        ev = evals if evals.ndim == 2 else np.broadcast_to(evals, (N, evals.shape[-1]))
        ulo, uhi = se.logit(self._y_lo), se.logit(self._y_hi)
        y = np.clip(np.broadcast_to(y0, (N, a.size)).astype(float), self._y_lo, self._y_hi)
        active = np.arange(N)  # indices still iterating
        for _ in range(self._scf_max_iter // 2):
            y0a = y[active]
            y1 = self._fixed_point_step(y0a, active, dmu, Tarr, ev, a)
            y2 = self._fixed_point_step(y1, active, dmu, Tarr, ev, a)
            u0, u1, u2 = se.logit(y0a), se.logit(y1), se.logit(y2)
            denom = u2 - 2.0 * u1 + u0
            # Aitken extrapolation where the curvature is resolvable, else the plain
            # second iterate; clip back into the logit box before mapping to y
            small = np.abs(denom) < 1e-14
            u_acc = np.where(small, u2, u0 - (u1 - u0) ** 2 / np.where(small, 1.0, denom))
            y_new = np.clip(se.expit(np.clip(u_acc, ulo, uhi)), self._y_lo, self._y_hi)
            y[active] = y_new
            done = np.max(np.abs(y_new - y0a), axis=1) < self._scf_tol
            if done.any():
                active = active[~done]
                if active.size == 0:
                    break
        return y

    def _warm_seed(self, T, flat):
        """Interpolate the nearest cached solution onto ``flat`` as an SCF seed."""
        if not self._warm:
            return None
        Tc = min(self._warm, key=lambda t: abs(t - T))
        if abs(Tc - T) > 40.0:
            return None
        mu_c, y_c = self._warm[Tc]
        seed = np.stack([np.interp(flat, mu_c, y_c[:, s]) for s in range(y_c.shape[1])], axis=1)
        return np.clip(seed, 1e-9, 1 - 1e-9)

    def _from_seed(self, T, flat, evals, a, y0):
        """One SCF solve from ``y0``; returns the semi-grand ``(phi, c, y)``.

        A pinned phase is kept ordered by the site-fraction clamp (its ``y`` never
        leaves its ordering's quadrant), so ``phi`` is finite everywhere and the
        phase simply loses the ``argmin`` where it is not the ground state.
        """
        y = self._scf(T, flat, evals, y0)
        c = y @ a
        phi = self._free_energy(T, y, evals) - flat * c
        return phi, c, y

    def _solve_fixed_T(self, T, dmu):
        """Semi-grand ``(phi, c)`` for one ``T`` and a whole ``dmu`` array.

        Runs the SCF map from a warm start (nearest cached solution) plus the corner
        seeds and keeps, per ``dmu``, the site fractions with the lowest ``G - dmu*c``.
        A pinned phase is confined to one ordering quadrant by the site-fraction clamp,
        so its ordered corner already finds that basin; it is solved **cold from the
        corner only** so the result is a deterministic function of ``(T, dmu)``.
        """
        evals = self._endmember_evals(T)
        a = self._a
        flat = np.asarray(dmu, dtype=float).ravel()
        if self._pinned:
            starts = [self._corners[0]]
        else:
            warm = self._warm_seed(T, flat)
            corners = ([np.full(a.size, 0.5)] if self.include_disordered_seed else []) + self._corners
            starts = ([warm] if warm is not None else []) + corners

        best_phi = np.full(flat.size, np.inf)
        best_c = np.zeros(flat.size)
        best_y = None
        for y0 in starts:
            phi, c, y = self._from_seed(T, flat, evals, a, y0)
            take = phi < best_phi
            best_phi[take] = phi[take]
            best_c[take] = c[take]
            best_y = y.copy() if best_y is None else best_y
            best_y[take] = y[take]

        # cache this solve for warm-starting nearby probes (bounded ring of temperatures)
        if len(self._warm) >= 12:
            self._warm.pop(next(iter(self._warm)))
        order = np.argsort(flat)
        self._warm[round(float(T))] = (flat[order], best_y[order])
        return best_phi.reshape(np.shape(dmu)), best_c.reshape(np.shape(dmu))

    def _solve_batch(self, T, dmu):
        """Semi-grand ``(phi, c)`` for per-element ``T`` and ``dmu`` (same 1-D shape).

        The array-``T`` analogue of :meth:`_solve_fixed_T`: one batched SCF over the
        whole ``(T, dmu)`` grid from the corner seeds, vectorised in ``T`` as well so a
        multi-``T`` grid is one solve per start instead of one per unique ``T``.  Cold
        corners (no warm start), but the per-``T`` slices are cached into ``_warm`` so
        the later clustered refinement probes still warm-start off the grid.
        """
        a = self._a
        Tf = np.asarray(T, dtype=float).ravel()
        flat = np.asarray(dmu, dtype=float).ravel()
        ev = np.stack([np.broadcast_to(e.energy(Tf), Tf.shape) for e in self.endmembers], axis=-1)
        corners = ([np.full(a.size, 0.5)] if self.include_disordered_seed else []) + self._corners
        starts = [self._corners[0]] if self._pinned else corners

        best_phi = np.full(flat.size, np.inf)
        best_c = np.zeros(flat.size)
        best_y = None
        for y0 in starts:
            phi, c, y = self._from_seed(Tf, flat, ev, a, y0)
            take = phi < best_phi
            best_phi[take] = phi[take]
            best_c[take] = c[take]
            best_y = y.copy() if best_y is None else best_y
            best_y[take] = y[take]

        # cache each unique-T slice so later scalar-T refinement probes warm-start
        for uT in np.unique(Tf):
            if len(self._warm) >= 12:
                self._warm.pop(next(iter(self._warm)))
            m = Tf == uT
            order = np.argsort(flat[m])
            self._warm[round(float(uT))] = (flat[m][order], best_y[m][order])
        return best_phi.reshape(np.shape(dmu)), best_c.reshape(np.shape(dmu))

    @lru_cache(maxsize=50000)
    def _find_phi_c_cached(self, t_shape, t_bytes, d_shape, d_bytes):
        """Solve and cache by raw ``(T, dmu)`` bytes; array ``T`` solves in one batch."""
        T = np.frombuffer(t_bytes, dtype=float).reshape(t_shape)
        dmu = np.frombuffer(d_bytes, dtype=float).reshape(d_shape)
        out_shape = np.broadcast_shapes(t_shape, d_shape)
        if T.ndim == 0:
            phi, c = self._solve_fixed_T(float(T), dmu)
            return np.broadcast_to(phi, out_shape).copy(), np.broadcast_to(c, out_shape).copy()
        Tb = np.broadcast_to(T, out_shape)
        dmub = np.broadcast_to(dmu, out_shape)
        phi, c = self._solve_batch(Tb.ravel(), dmub.ravel())
        return phi.reshape(out_shape), c.reshape(out_shape)

    def _find_phi_c(self, T, dmu):
        Ta = np.asarray(T, dtype=float)
        Da = np.asarray(dmu, dtype=float)
        phi, c = self._find_phi_c_cached(
            Ta.shape, np.ascontiguousarray(Ta).tobytes(),
            Da.shape, np.ascontiguousarray(Da).tobytes(),
        )
        return _scalarize(phi.copy()), _scalarize(c.copy())

    def semigrand_potential(self, T, dmu):
        return self._find_phi_c(T, dmu)[0]

    def concentration(self, T, dmu):
        return self._find_phi_c(T, dmu)[1]

    # --- reduced free-energy curve f(c) (diagnostics; not on the hot path) ----

    def _fc_scalar(self, T, c, evals):
        """``f(c, T) = min`` over site fractions at fixed overall composition ``c``."""
        S = len(self.sublattices)
        a = self._a
        bounds = [(0.0, 1.0)] * S
        constraint = {"type": "eq", "fun": lambda y: float(a @ y) - c}
        best = np.inf
        seeds = ([np.full(S, c)] if self.include_disordered_seed else []) + self._corners
        for y0 in seeds:
            try:
                res = so.minimize(
                    lambda y: float(self._free_energy(T, np.asarray(y), evals)), y0,
                    method="SLSQP", bounds=bounds, constraints=constraint,
                )
            except ValueError:
                continue
            if res.success and np.isfinite(res.fun) and res.fun < best:
                best = res.fun
        return best

    def free_energy_c(self, T, c):
        """Reduced Gibbs energy ``f(c, T) = min_{y: c(y)=c} G(y, T)`` (eV/atom)."""
        evals = self._endmember_evals(T)
        return np.vectorize(lambda c: self._fc_scalar(T, float(c), evals))(c)
