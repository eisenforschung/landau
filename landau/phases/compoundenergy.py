r"""Binary Compound Energy Formalism (CEF) phase.

A minimal, general CEF phase for a binary A--B system with an arbitrary number
of sublattices.  The model is the one used throughout CALPHAD assessments (e.g.
Sundman, Fries & Oates, *Calphad* **22** (1998) 335 for Au--Cu):

.. math::

    G(\vec y, T) = \underbrace{\sum_\text{end-members}
        {}^0G_c(T) \prod_s p_s^{(c)}}_{G_\text{ref}}
        + \underbrace{k_B T \sum_s a_s \big(y_s \ln y_s
        + (1-y_s)\ln(1-y_s)\big)}_{G_\text{ideal}}
        + {}^EG(\vec y, T)

where ``y_s`` is the site fraction of component B on sublattice ``s``, ``a_s``
its relative size (``site_multiplicities``, summing to 1), and ``p_s^{(c)} = y_s``
if end-member ``c`` carries B on sublattice ``s`` else ``1 - y_s``.  The overall
B concentration is ``c(y) = sum_s a_s y_s``.

Semi-grand solve -- one minimisation, straight from site fractions.  The site
fractions are internal (order) parameters and the semi-grand potential is their
Legendre transform,

.. math::

    \phi(T, \Delta\mu) = \min_{\vec y \in [0,1]^S}\!\big[G(\vec y,T) - \Delta\mu\,c(\vec y)\big].

Setting :math:`\partial_{y_s}[\,G - \Delta\mu\,c\,] = 0` gives the self-consistent
(Bragg-Williams / BGBW) fixed point

.. math::

    y_s = \sigma\!\Big(\frac{\Delta\mu - a_s^{-1}\,\partial_{y_s}(G_\text{ref}+{}^EG)}{k_B T}\Big),

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

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Callable, ClassVar, Mapping

import numpy as np
import scipy.optimize as so
import scipy.special as se

from . import Phase, kB, _scalarize

__all__ = ["CompoundEnergyPhase"]


@dataclass(frozen=True)
class CompoundEnergyPhase(Phase):
    """Binary compound-energy-formalism phase (real :class:`~landau.phases.Phase`).

    Args:
        name: phase name.
        site_multiplicities: relative sublattice sizes ``a_s`` (must sum to 1).
        endmember_energies: maps each length-``S`` configuration tuple (``0`` = A,
            ``1`` = B on that sublattice) to a callable ``G(T)`` returning the
            end-member energy in eV/atom.  All ``2**S`` configurations must be given.
        excess: optional excess energy ``EG(y, T)`` in eV/atom.  ``y`` is an array
            whose last axis is the ``S`` site fractions; it must broadcast over any
            leading axes (sum reductions over ``axis=-1``, index sublattices as
            ``y[..., s]``) so the vectorised solve can evaluate it on a whole batch.
            ``None`` means ideal.
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

    site_multiplicities: tuple[float, ...] = ()
    endmember_energies: Mapping[tuple[int, ...], Callable[[float], float]] = field(default_factory=dict)
    excess: Callable[[np.ndarray, float], float] | None = None
    orderings: tuple[tuple[int, ...], ...] | None = None
    include_disordered_seed: bool = True

    # solver tuning (ClassVar -> never treated as dataclass fields)
    _scf_max_iter: ClassVar[int] = 80   # self-consistent-field iterations per start
    _scf_damp: ClassVar[float] = 0.5    # initial mixing y <- (1-d) y + d target
    _scf_damp_min: ClassVar[float] = 0.05  # floor for the per-column adaptive mixing
    _scf_tol: ClassVar[float] = 1e-9    # max|dy| convergence tolerance
    _fd: ClassVar[float] = 1e-6         # finite-difference step for the excess gradient
    _order_tol: ClassVar[float] = 0.05  # min long-range order for a pinned ordered phase to exist

    def __post_init__(self):
        a = tuple(float(x) for x in self.site_multiplicities)
        object.__setattr__(self, "site_multiplicities", a)
        object.__setattr__(self, "endmember_energies", tuple(sorted(self.endmember_energies.items())))
        S = len(a)
        assert len(self.endmember_energies) == 2**S, f"need 2**S={2**S} end-member energies for {S} sublattices"
        assert abs(sum(a) - 1) < 1e-9, "site_multiplicities must sum to 1"
        # config array (1 where B sits), the +/-1 sign per (endmember, sublattice) for
        # the analytic G_ref gradient, and the ordered corners the solve is seeded from
        cfg = np.array([c for c, _ in self.endmember_energies], dtype=float)
        object.__setattr__(self, "_configs", cfg)
        object.__setattr__(self, "_sign", 2.0 * cfg - 1.0)
        seeds = product((0, 1), repeat=S) if self.orderings is None else self.orderings
        object.__setattr__(self, "_corners", [np.clip(np.array(c, dtype=float), 1e-9, 1 - 1e-9) for c in seeds])
        assert self._corners or self.include_disordered_seed, "phase has no ordering seed"
        # A basin-pinned ordered phase -- exactly one ordering and no disordered seed --
        # is confined to that ordering's sublattice symmetry (sublattices sharing the
        # same occupation stay equal, so the solve cannot drift into a different
        # ordering) and is reported absent where it disorders.  This makes it a distinct
        # phase that competes cleanly in ``calc_phase_diagram`` instead of relaxing onto
        # -- and tying with -- the disordered or a neighbouring ordered branch.
        pinned = self.orderings is not None and len(self._corners) == 1 and not self.include_disordered_seed
        groups = None
        if pinned:
            pat = np.round(self._corners[0]).astype(int)
            groups = [np.where(pat == v)[0] for v in np.unique(pat)]
        object.__setattr__(self, "_pinned", pinned)
        object.__setattr__(self, "_groups", groups)
        # warm-start cache: rounded T -> (sorted dmu, y*) from the last solve at that
        # T. Refinement probes clustered (T, dmu), so seeding the SCF from the nearest
        # previous solution converges in a few iterations instead of ~40 from a corner.
        object.__setattr__(self, "_warm", {})

    # --- model (vectorised over any leading axes of y) -----------------------

    @lru_cache(maxsize=2000)
    def _endmember_evals(self, T):
        """End-member energies at ``T`` in end-member (config-sorted) order."""
        return np.array([g(T) for _, g in self.endmember_energies])

    def _free_energy(self, T, y, evals):
        """Gibbs energy ``G(y, T)`` for ``y`` of shape ``(..., S)`` -> ``(...)`` (eV/atom).

        ``evals`` is ``(M,)`` for a scalar ``T`` or ``(..., M)`` for a per-element ``T``.
        """
        a = np.asarray(self.site_multiplicities)
        p = np.where(self._configs == 1, y[..., None, :], 1.0 - y[..., None, :])  # (...,M,S)
        gref = np.sum(p.prod(axis=-1) * evals, axis=-1)  # (...,)  broadcasts evals (M,) or (...,M)
        gid = -kB * T * ((se.entr(y) + se.entr(1.0 - y)) @ a)
        gex = self.excess(y, T) if self.excess is not None else 0.0
        return gref + gid + gex

    def free_energy(self, T, y):
        """Gibbs energy at site fractions ``y`` (length ``S``) and temperature ``T`` (eV/atom)."""
        return self._free_energy(T, np.asarray(y, dtype=float), self._endmember_evals(T))

    def composition(self, y):
        """Overall B concentration ``sum_s a_s y_s``."""
        return np.asarray(y, dtype=float) @ np.asarray(self.site_multiplicities)

    def _grad_non_ideal(self, T, y, evals):
        """Gradient ``d(G_ref + EG)/dy_s`` for ``y`` of shape ``(..., S)`` -> ``(..., S)``."""
        p = np.where(self._configs == 1, y[..., None, :], 1.0 - y[..., None, :])  # (...,M,S)
        # leave-one-out product: prod_{s'!=s} p = (prod_s p) / p_s  (p bounded away from 0)
        loo = p.prod(axis=-1, keepdims=True) / p  # (...,M,S)
        # evals is (M,) for scalar T or (...,M) for per-element T; broadcast either way
        gref = np.sum(evals[..., :, None] * (self._sign * loo), axis=-2)  # (...,S)
        if self.excess is None:
            return gref
        # central finite difference of the excess in every sublattice, in one
        # excess call: stack the 2*S perturbed batches on a leading axis (excess
        # broadcasts over it) rather than calling excess 2*S times.
        h = self._fd
        D, S = y.shape
        P = np.broadcast_to(y, (2, S, D, S)).copy()
        diag = np.arange(S)
        P[0, diag, :, diag] = np.clip(y[:, diag].T + h, 1e-12, 1 - 1e-12)
        P[1, diag, :, diag] = np.clip(y[:, diag].T - h, 1e-12, 1 - 1e-12)
        ex = self.excess(P, T)  # (2, S, D)
        denom = P[0, diag, :, diag] - P[1, diag, :, diag]  # (S, D)
        return gref + ((ex[0] - ex[1]) / denom).T

    # --- direct semi-grand solve (one minimisation per start) ----------------

    def _scf(self, T, dmu, evals, y0):
        """Self-consistent field iteration for one start, vectorised over ``(T, dmu)``.

        ``T`` is a scalar or a ``(len(dmu),)`` array (per-element temperature) and
        ``evals`` the matching ``(M,)`` or ``(len(dmu), M)`` end-member energies.
        Columns are dropped from the working set as they converge, so the iteration
        cost shrinks toward the few ``(T, dmu)`` that keep iterating.
        Returns the stationary site fractions ``y`` of shape ``(len(dmu), S)``.

        The mixing factor is adaptive per column.  Near a spinodal the plain
        damped map has a fixed-point multiplier below ``-1`` and settles into a
        period-2 limit cycle straddling the true stationary point (the fcc
        disordered branch between two ordered domes does this), so it never meets
        the tolerance and returns a start-dependent, non-stationary value.  When a
        column's undamped residual flips sign between steps -- the signature of
        that oscillation -- its mixing is halved (down to ``_scf_damp_min``),
        which pulls the multiplier back inside the unit circle and the column
        converges.  Well-behaved columns keep the full ``_scf_damp`` and its
        faster rate.
        """
        a = np.asarray(self.site_multiplicities)
        N = dmu.size
        Tarr = np.broadcast_to(np.asarray(T, dtype=float), (N,))
        ev = evals if evals.ndim == 2 else np.broadcast_to(evals, (N, evals.shape[-1]))
        kT = kB * Tarr
        d = np.full(N, self._scf_damp)          # per-column mixing, cut on oscillation
        prev_delta = np.zeros((N, a.size))       # last undamped residual target - y
        y = np.clip(np.broadcast_to(y0, (N, a.size)).astype(float), 1e-9, 1 - 1e-9)
        active = np.arange(N)  # indices still iterating
        ya = y[active]
        for _ in range(self._scf_max_iter):
            field = self._grad_non_ideal(Tarr[active], ya, ev[active]) / a  # (n_active, S)
            target = se.expit((dmu[active, None] - field) / kT[active, None])
            delta = target - ya
            # a residual that reversed sign since the previous step is orbiting the
            # fixed point rather than approaching it -> shrink this column's mixing
            osc = np.any(delta * prev_delta[active] < 0.0, axis=1)
            d[active] = np.where(osc, np.maximum(0.5 * d[active], self._scf_damp_min), d[active])
            prev_delta[active] = delta
            y_new = np.clip(ya + d[active, None] * delta, 1e-12, 1 - 1e-12)
            if self._groups is not None:  # confine to the ordering's sublattice symmetry
                for g in self._groups:
                    y_new[:, g] = y_new[:, g].mean(axis=1, keepdims=True)
            y[active] = y_new
            done = np.max(np.abs(y_new - ya), axis=1) < self._scf_tol
            if self._groups is not None:
                # a pinned column seeded ordered that has fallen below _order_tol is
                # disordering (monotone outside the dome) -> it will be reported absent,
                # so stop refining it now rather than iterate to full convergence
                done |= self._order_parameter(y_new) < self._order_tol
            if done.any():
                keep = ~done
                active = active[keep]
                if active.size == 0:
                    break
                ya = y[active]
            else:
                ya = y_new
        return y

    def _order_parameter(self, y):
        """Long-range order of a pinned phase: the largest spread between sublattice groups."""
        means = [y[..., g].mean(axis=-1) for g in self._groups]
        op = np.zeros(y.shape[:-1])
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                op = np.maximum(op, np.abs(means[i] - means[j]))
        return op

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
        """One SCF solve from ``y0``; returns ``(phi, c, y)`` with pinned-absence applied."""
        y = self._scf(T, flat, evals, y0)
        c = y @ a
        phi = self._free_energy(T, y, evals) - flat * c
        if self._pinned:  # a pinned ordered phase is absent where it has disordered
            phi = np.where(self._order_parameter(y) < self._order_tol, np.inf, phi)
        return phi, c, y

    def _solve_fixed_T(self, T, dmu):
        """Semi-grand ``(phi, c)`` for one ``T`` and a whole ``dmu`` array.

        Runs the SCF map from a warm start (nearest cached solution) plus the corner
        seeds and keeps, per ``dmu``, the site fractions with the lowest ``G - dmu*c``.
        A pinned phase is confined to a single ordering basin, so its ordered corner
        already finds that basin; it is solved **cold from the corner only** (no warm
        start).  Trusting a warm seed for a pinned phase was start-dependent near its
        dome edge -- a warm seed drawn from a neighbour could steer the confined solve
        into a disordered result and report the phase *absent* (``phi = +inf``) where
        it is really present, which then let ``_dominated`` miss it and a Clausius-
        Clapeyron trace overstep the invariant. The cold corner is deterministic.
        """
        evals = self._endmember_evals(T)
        a = np.asarray(self.site_multiplicities)
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
        a = np.asarray(self.site_multiplicities)
        Tf = np.asarray(T, dtype=float).ravel()
        flat = np.asarray(dmu, dtype=float).ravel()
        ev = np.stack([np.broadcast_to(g(Tf), Tf.shape) for _, g in self.endmember_energies], axis=-1)
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
        S = len(self.site_multiplicities)
        a = np.asarray(self.site_multiplicities)
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
