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

Semi-grand solve, mirroring :class:`~landau.phases.FastInterpolatingPhase`.  The
site fractions are internal (order) parameters, and

.. math::

    \phi(T, \Delta\mu) = \min_{\vec y \in [0,1]^S}\!\big[G(\vec y,T) - \Delta\mu\,c(\vec y)\big]
        = \min_{c \in [0,1]}\!\big[f(c,T) - \Delta\mu\,c\big],

so the ordering (inner) minimisation is separated from the semi-grand (outer)
transform: for each ``T`` the reduced curve ``f(c,T) = min_{y: c(y)=c} G(y,T)`` is
built once on a concentration grid, then a single vectorised ``argmin`` over the
whole ``dmu`` array picks the global basin (handling the ordered/disordered
common tangents), refined to sub-grid accuracy by a parabolic step.  All energies
are per atom (eV) to match the rest of ``landau``; convert CALPHAD J/mol
parameters by dividing by 96485.33.
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
        excess: optional excess energy ``EG(y, T)`` in eV/atom, where ``y`` is the
            length-``S`` array of B site fractions.  ``None`` means ideal.
    """

    site_multiplicities: tuple[float, ...] = ()
    endmember_energies: Mapping[tuple[int, ...], Callable[[float], float]] = field(default_factory=dict)
    excess: Callable[[np.ndarray, float], float] | None = None

    # solver tuning (ClassVar -> never treated as dataclass fields)
    _n_grid: ClassVar[int] = 101  # concentration grid for the f(c) reduction

    def __post_init__(self):
        a = tuple(float(x) for x in self.site_multiplicities)
        object.__setattr__(self, "site_multiplicities", a)
        object.__setattr__(self, "endmember_energies", tuple(sorted(self.endmember_energies.items())))
        S = len(a)
        assert len(self.endmember_energies) == 2**S, f"need 2**S={2**S} end-member energies for {S} sublattices"
        assert abs(sum(a) - 1) < 1e-9, "site_multiplicities must sum to 1"
        # config array (1 where B sits) and the ordered end-member corners, cached
        object.__setattr__(self, "_configs", np.array([cfg for cfg, _ in self.endmember_energies], dtype=float))
        object.__setattr__(
            self, "_corners", [np.clip(np.array(cfg, dtype=float), 1e-9, 1 - 1e-9) for cfg in product((0, 1), repeat=S)]
        )

    # hash by name so the phase can key the per-T caches; distinct phases in one
    # diagram carry distinct names.
    def __hash__(self):
        return hash(self.name)

    # --- model ---------------------------------------------------------------

    @lru_cache(maxsize=2000)
    def _endmember_evals(self, T):
        """End-member energies at ``T`` in end-member (config-sorted) order."""
        return np.array([g(T) for _, g in self.endmember_energies])

    def _free_energy(self, T, y, evals):
        """Gibbs energy from site fractions with pre-evaluated end-member energies."""
        a = np.asarray(self.site_multiplicities)
        prob = np.prod(np.where(self._configs == 1, y, 1.0 - y), axis=1)  # (M,)
        gref = float(evals @ prob)
        # ideal mixing: +kB*T*sum a_s (y ln y + (1-y)ln(1-y)) = -kB*T*sum a_s (entr y + entr 1-y)
        gid = -kB * T * float(np.dot(a, se.entr(y) + se.entr(1.0 - y)))
        gex = float(self.excess(y, T)) if self.excess is not None else 0.0
        return gref + gid + gex

    def free_energy(self, T, y):
        """Gibbs energy at site fractions ``y`` (length ``S``) and temperature ``T`` (eV/atom)."""
        return self._free_energy(T, np.asarray(y, dtype=float), self._endmember_evals(T))

    def composition(self, y):
        """Overall B concentration ``sum_s a_s y_s``."""
        return float(np.dot(self.site_multiplicities, np.asarray(y, dtype=float)))

    # --- ordering (inner) minimisation: reduce to f(c) -----------------------

    def _fc_scalar(self, T, c, evals):
        """``f(c, T) = min`` over site fractions at fixed overall composition ``c``.

        Multi-start SLSQP with the composition equality constraint, seeded from
        the disordered point plus every ordered end-member corner so both ordered
        and disordered basins are sampled; an infeasible corner seed is pulled onto
        the constraint by SLSQP while staying in its basin.
        """
        S = len(self.site_multiplicities)
        a = np.asarray(self.site_multiplicities)
        bounds = [(0.0, 1.0)] * S
        constraint = {"type": "eq", "fun": lambda y: float(a @ y) - c}
        best = np.inf
        for y0 in [np.full(S, c), *self._corners]:
            try:
                res = so.minimize(
                    lambda y: self._free_energy(T, y, evals), y0,
                    method="SLSQP", bounds=bounds, constraints=constraint,
                )
            except ValueError:
                continue
            if res.success and np.isfinite(res.fun) and res.fun < best:
                best = res.fun
        return best

    @lru_cache(maxsize=2000)
    def _fc_grid(self, T):
        """The reduced free-energy curve ``f(c, T)`` on the concentration grid."""
        evals = self._endmember_evals(T)
        conc = np.linspace(0.0, 1.0, self._n_grid)
        f = np.array([self._fc_scalar(T, float(c), evals) for c in conc])
        return conc, f

    def free_energy_c(self, T, c):
        """Reduced Gibbs energy ``f(c, T)`` at overall composition ``c`` (eV/atom)."""
        evals = self._endmember_evals(T)
        return np.vectorize(lambda c: self._fc_scalar(T, float(c), evals))(c)

    # --- semi-grand (outer) transform: min over c of f(c) - dmu*c ------------

    def _solve_fixed_T(self, T, dmu):
        """Minimise ``f(c) - dmu*c`` over ``c`` for one ``T`` and a whole ``dmu`` array."""
        conc, f_grid = self._fc_grid(T)
        n = self._n_grid
        flat = np.asarray(dmu, dtype=float).ravel()

        val = f_grid[None, :] - flat[:, None] * conc[None, :]  # (D, n)
        idx = val.argmin(axis=1)

        # parabolic sub-grid refinement inside the winning cell [i-1, i+1]
        i0 = np.clip(idx, 1, n - 2)
        rows = np.arange(flat.size)
        vm, v0, vp = val[rows, i0 - 1], val[rows, i0], val[rows, i0 + 1]
        denom = vm - 2 * v0 + vp
        shift = np.where(denom > 0, 0.5 * (vm - vp) / np.where(denom > 0, denom, 1.0), 0.0)
        shift = np.clip(shift, -1.0, 1.0)
        dc = conc[1] - conc[0]
        c = np.clip(conc[i0] + shift * dc, 0.0, 1.0)
        phi = v0 - 0.25 * (vm - vp) * shift  # parabola vertex value

        return phi.reshape(np.shape(dmu)), c.reshape(np.shape(dmu))

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
