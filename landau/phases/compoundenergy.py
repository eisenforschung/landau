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
each basin; it is vectorised over the whole ``dmu`` array (one solve per ``T``) and
run from a disordered start plus every ordered end-member corner, keeping the lowest
``phi`` per ``dmu`` -- so ordered/disordered coexistence surfaces as ``c(dmu)`` jumps
without any fixed-composition inner minimisation.  ``c = -partial phi/partial dmu``
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
    _scf_max_iter: ClassVar[int] = 200  # self-consistent-field iterations per start
    _scf_damp: ClassVar[float] = 0.5    # mixing factor y <- (1-d) y + d target
    _scf_tol: ClassVar[float] = 1e-11   # max|dy| convergence tolerance
    _fd: ClassVar[float] = 1e-6         # finite-difference step for the excess gradient

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

    # hash by name so the phase can key the per-T caches; distinct phases in one
    # diagram carry distinct names.
    def __hash__(self):
        return hash(self.name)

    # --- model (vectorised over any leading axes of y) -----------------------

    @lru_cache(maxsize=2000)
    def _endmember_evals(self, T):
        """End-member energies at ``T`` in end-member (config-sorted) order."""
        return np.array([g(T) for _, g in self.endmember_energies])

    def _free_energy(self, T, y, evals):
        """Gibbs energy ``G(y, T)`` for ``y`` of shape ``(..., S)`` -> ``(...)`` (eV/atom)."""
        a = np.asarray(self.site_multiplicities)
        p = np.where(self._configs == 1, y[..., None, :], 1.0 - y[..., None, :])  # (...,M,S)
        gref = p.prod(axis=-1) @ evals  # (...,)
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
        gref = np.einsum("m,...ms->...s", evals, self._sign * loo)
        if self.excess is None:
            return gref
        # central finite difference of the (cheap) excess in each sublattice
        h = self._fd
        gex = np.empty_like(y)
        for s in range(y.shape[-1]):
            yp = y.copy()
            ym = y.copy()
            yp[..., s] = np.clip(yp[..., s] + h, 1e-12, 1 - 1e-12)
            ym[..., s] = np.clip(ym[..., s] - h, 1e-12, 1 - 1e-12)
            gex[..., s] = (self.excess(yp, T) - self.excess(ym, T)) / (yp[..., s] - ym[..., s])
        return gref + gex

    # --- direct semi-grand solve (one minimisation per start) ----------------

    def _scf(self, T, dmu, evals, y0):
        """Self-consistent field iteration for one start, vectorised over ``dmu``.

        Returns the stationary site fractions ``y`` of shape ``(len(dmu), S)``.
        """
        a = np.asarray(self.site_multiplicities)
        kT = kB * T
        d = self._scf_damp
        y = np.clip(np.broadcast_to(y0, (dmu.size, a.size)).astype(float), 1e-9, 1 - 1e-9)
        for _ in range(self._scf_max_iter):
            field = self._grad_non_ideal(T, y, evals) / a  # (D,S)
            target = se.expit((dmu[:, None] - field) / kT)
            y_new = np.clip((1 - d) * y + d * target, 1e-12, 1 - 1e-12)
            if np.max(np.abs(y_new - y)) < self._scf_tol:
                return y_new
            y = y_new
        return y

    def _solve_fixed_T(self, T, dmu):
        """Semi-grand ``(phi, c)`` for one ``T`` and a whole ``dmu`` array.

        Runs the SCF map from the disordered start plus every ordered corner and
        keeps, per ``dmu``, the site fractions with the lowest ``G - dmu*c``.
        """
        evals = self._endmember_evals(T)
        a = np.asarray(self.site_multiplicities)
        flat = np.asarray(dmu, dtype=float).ravel()
        starts = ([np.full(a.size, 0.5)] if self.include_disordered_seed else []) + self._corners

        best_phi = np.full(flat.size, np.inf)
        best_c = np.zeros(flat.size)
        for y0 in starts:
            y = self._scf(T, flat, evals, y0)
            c = y @ a
            phi = self._free_energy(T, y, evals) - flat * c
            take = phi < best_phi
            best_phi[take] = phi[take]
            best_c[take] = c[take]
        return best_phi.reshape(np.shape(dmu)), best_c.reshape(np.shape(dmu))

    @lru_cache(maxsize=512)
    def _find_phi_c_cached(self, t_shape, t_bytes, d_shape, d_bytes):
        """Solve and cache by raw ``(T, dmu)`` bytes; group array ``T`` by unique value."""
        T = np.frombuffer(t_bytes, dtype=float).reshape(t_shape)
        dmu = np.frombuffer(d_bytes, dtype=float).reshape(d_shape)
        out_shape = np.broadcast_shapes(t_shape, d_shape)
        if T.ndim == 0:
            phi, c = self._solve_fixed_T(float(T), dmu)
            return np.broadcast_to(phi, out_shape).copy(), np.broadcast_to(c, out_shape).copy()
        Tb = np.broadcast_to(T, out_shape)
        dmub = np.broadcast_to(dmu, out_shape)
        phi = np.empty(out_shape)
        c = np.empty(out_shape)
        for uT in np.unique(Tb):
            m = Tb == uT
            phi[m], c[m] = self._solve_fixed_T(float(uT), dmub[m])
        return phi, c

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
