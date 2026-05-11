"""
Clausius-Clapeyron coexistence tracing: comparison against brute-force isothermal refinement.

Phase model
-----------
Sub-regular (2-parameter Redlich-Kister) solution:

    f(T, c) = c*(1-c)*(L0 + L1*(2c-1)) - T * S_mix(c)

With L1 ≠ 0 the miscibility gap is visibly asymmetric (critical concentration c_c ≠ 0.5).
Both semigrand_potential and concentration are evaluated analytically (global minimisation of
f - mu*c via root-finding on df/dc = mu), so there are no stencil artefacts.  If this demo
were ported to the actual RegularSolution class, concentration() uses a 5-point finite-
difference stencil; the stencil step should be reduced to ≤ 0.1 meV to avoid smearing near
the kink.

Requirements
------------
    pip install numpy scipy matplotlib          # plot dependencies
    pip install -e /path/to/landau             # for Phase base class

Usage
-----
    python scripts/cc_demo.py
    → writes scripts/cc_demo.png
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.constants import Boltzmann, eV
import scipy.optimize as so

from landau.phases import Phase

kB = Boltzmann / eV   # eV / K


# ── Entropy helpers ───────────────────────────────────────────────────────────

def _S_mix(c):
    c = np.asarray(c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return kB * (np.where(c > 0, -c * np.log(c), 0.0)
                   + np.where(c < 1, -(1 - c) * np.log(1 - c), 0.0))


def _dS_dc(c):
    # d/dc [S_mix(c)] = kB * ln((1-c)/c)
    with np.errstate(divide="ignore"):
        return kB * np.log((1.0 - c) / c)


# ── Sub-regular solution phase ────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalyticSubRegularSolution(Phase):
    """Sub-regular (2-parameter Redlich-Kister) free energy.

    f(T, c) = c*(1-c)*(L0 + L1*(2c-1)) - T * S_mix(c)

    With L1 ≠ 0 the c ↔ (1-c) symmetry is broken:
      - the spinodal and binodal are asymmetric in c
      - the critical concentration shifts away from 0.5
      - the coexistence line mu*(T) is nonzero even at T→0

    semigrand_potential and concentration are both analytic (global minimiser
    of f - mu*c found by root-finding on df/dc = mu), so there are no
    finite-difference stencil artefacts near the kink.
    """
    L0: float   # eV — symmetric Redlich-Kister coefficient
    L1: float   # eV — asymmetric Redlich-Kister coefficient

    def _f(self, T, c):
        return c * (1 - c) * (self.L0 + self.L1 * (2 * c - 1)) - T * _S_mix(c)

    def _df_dc(self, T, c):
        """∂f/∂c = chemical potential of component 1."""
        # d/dc [c(1-c)(L0 + L1(2c-1))] = L0(1-2c) + L1(-6c²+6c-1)
        return (self.L0 * (1 - 2 * c)
                + self.L1 * (-6 * c**2 + 6 * c - 1)
                - T * _dS_dc(c))

    def _stable_c(self, T, mu):
        """Global argmin_c [f(T,c) - mu*c] via root-finding on df/dc = mu."""
        c_grid = np.linspace(1e-10, 1 - 1e-10, 20000)
        g = self._df_dc(T, c_grid) - mu
        sign_changes = np.where(np.diff(np.sign(g)))[0]
        roots = []
        for i in sign_changes:
            try:
                r = so.brentq(lambda c: self._df_dc(T, c) - mu,
                              c_grid[i], c_grid[i + 1], xtol=1e-14)
                roots.append(r)
            except ValueError:
                pass
        if not roots:
            raise ValueError(f"No root at T={T:.2f} K, mu={mu:.5f} eV")
        if len(roots) == 1:
            return roots[0]
        # Multiple roots → global minimum of g(c) = f(c) - mu*c
        gvals = [self._f(T, r) - mu * r for r in roots]
        return roots[int(np.argmin(gvals))]

    def semigrand_potential(self, T, mu):
        if np.isscalar(mu):
            c = self._stable_c(T, mu)
            return float(self._f(T, c) - mu * c)
        return np.array([self.semigrand_potential(T, m) for m in np.asarray(mu)])

    def concentration(self, T, mu):
        if np.isscalar(mu):
            return float(self._stable_c(T, mu))
        return np.array([self._stable_c(T, m) for m in np.asarray(mu)])


# ── Phase parameters ──────────────────────────────────────────────────────────
#
# L1 = 0.04 eV (40 % of L0) gives a clearly asymmetric gap.
# Analytic estimate of T_c: maximise T_spinodal(c) = [2L0 + L1(12c-6)]*c(1-c)/kB
# over c → c_c ≈ 0.68, T_c ≈ 723 K.
#
L0, L1 = 0.10, 0.04   # eV
phase = AnalyticSubRegularSolution(name="subregular", L0=L0, L1=L1)


# ── Find T_c analytically ─────────────────────────────────────────────────────


def _Tc_spinodal(n_c=20000):
    """Compute T_c = max_c T_spinodal(c) = [2L0+L1(12c-6)]*c*(1-c)/kB.

    For the sub-regular solution, ∂²f/∂c² = 0 when T = T_spinodal(c).
    T_c is the maximum of T_spinodal over c, which is where the spinodal
    (and the binodal) closes.  This gives T_c and the critical concentration c_c.
    """
    c_arr = np.linspace(1e-4, 1 - 1e-4, n_c)
    T_sp = (2 * L0 + L1 * (12 * c_arr - 6)) * c_arr * (1 - c_arr) / kB
    idx = int(np.argmax(T_sp))
    return float(T_sp[idx]), float(c_arr[idx])


print("Computing T_c from spinodal condition ...")
T_c, c_c = _Tc_spinodal()
print(f"  T_c = {T_c:.1f} K,  critical concentration c_c = {c_c:.3f}")


# ── Core algorithms ───────────────────────────────────────────────────────────

def refine_isothermal_jump(phase_left, phase_right, T, mu_lo, mu_hi,
                           tol=1e-9, c_thresh_hint=None):
    """Return (mu_star, c_left, c_right).

    Case A — two different phases:
        Bisect on  Φ_left(μ) − Φ_right(μ).  No numerical derivatives.

    Case B — same phase (miscibility gap):
        Bisection accelerated by the tangent-intersection probe (secant step).
        Branch membership determined by .concentration() — never by dΦ/dμ.

        c_thresh_hint: optional float giving the branch discriminant.  If None,
            the midpoint of the bracket concentrations is used.  Pass the midpoint
            of the *coexistence* concentrations (c_left + c_right)/2 from the
            previous step so the discriminant tracks c_c — essential for asymmetric
            phases where both c_left and c_right may be > 0.5 near T_c.
    """
    if phase_left is not phase_right:
        lo, hi = mu_lo, mu_hi
        while hi - lo > tol:
            mid   = (lo + hi) / 2.0
            delta = (phase_left.semigrand_potential(T, mid)
                     - phase_right.semigrand_potential(T, mid))
            lo, hi = (mid, hi) if delta < 0 else (lo, mid)
        mu_star = (lo + hi) / 2.0
        return (mu_star,
                phase_left.concentration(T, mu_star),
                phase_right.concentration(T, mu_star))

    # Case B — single phase / miscibility gap
    ph = phase_left
    phi_lo = ph.semigrand_potential(T, mu_lo)
    phi_hi = ph.semigrand_potential(T, mu_hi)
    c_lo   = ph.concentration(T, mu_lo)
    c_hi   = ph.concentration(T, mu_hi)

    if abs(c_hi - c_lo) < 1e-10:
        raise ValueError(
            f"No concentration jump in [{mu_lo:.4f}, {mu_hi:.4f}] at T={T:.2f} K")

    if c_thresh_hint is not None:
        c_thresh = float(c_thresh_hint)
        # Validate: discriminant must lie strictly between c_lo and c_hi
        if not (min(c_lo, c_hi) < c_thresh < max(c_lo, c_hi)):
            # Hint is outside bracket — fall back to midpoint
            c_thresh = (c_lo + c_hi) / 2.0
    else:
        c_thresh = (c_lo + c_hi) / 2.0

    lo, hi = mu_lo, mu_hi
    while hi - lo > tol:
        denom = c_hi - c_lo
        if abs(denom) > 1e-15:
            mu_trial = (phi_lo - phi_hi + c_hi * hi - c_lo * lo) / denom
        else:
            mu_trial = (lo + hi) / 2.0

        if not (lo < mu_trial < hi):
            mu_trial = (lo + hi) / 2.0

        phi_t = ph.semigrand_potential(T, mu_trial)
        c_t   = ph.concentration(T, mu_trial)

        if c_t < c_thresh:
            lo, phi_lo, c_lo = mu_trial, phi_t, c_t
        else:
            hi, phi_hi, c_hi = mu_trial, phi_t, c_t

    mu_star = (lo + hi) / 2.0
    return mu_star, ph.concentration(T, lo), ph.concentration(T, hi)


def trace_coexistence_line(phase_left, phase_right,
                           T_start, mu_lo_init, mu_hi_init,
                           T_end, delta_c_min=5e-4, dT_max=50.0):
    """Clausius-Clapeyron predictor-corrector coexistence tracer.

    Predictor: linear extrapolation of μ*(T) using the finite-difference CC slope
               dμ*/dT from consecutive converged points.  No explicit entropy needed.
    Corrector: refine_isothermal_jump centred on the prediction.
    Step size: dT ∝ (delta_c)^2 so that near T_c (where delta_c → 0) the steps
               shrink automatically, keeping the predictor accurate.
    Stopping:  when delta_c = c_right − c_left < delta_c_min.

    Returns list of dicts: T, mu_star, c_left, c_right, mu_predicted, predictor_err.
    """
    sign_dT = +1 if T_end > T_start else -1

    mu_star, c_left, c_right = refine_isothermal_jump(
        phase_left, phase_right, T_start, mu_lo_init, mu_hi_init)
    delta_c    = c_right - c_left
    half_width = (mu_hi_init - mu_lo_init) / 2.0
    # Track the branch discriminant so it follows c_c near T_c.
    c_thresh = (c_left + c_right) / 2.0
    results    = [dict(T=T_start, mu_star=mu_star, c_left=c_left, c_right=c_right,
                       mu_predicted=mu_star, predictor_err=0.0)]

    if delta_c < delta_c_min:
        return results

    # Bootstrap: one small step to seed the CC slope dμ*/dT.
    # Using two refine_isothermal_jump calls instead of a numerical ∂Φ/∂T.
    dT_boot = sign_dT * min(0.5, abs(T_end - T_start) * 0.002)
    T_boot  = T_start + dT_boot
    mu_b, c_lb, c_rb = refine_isothermal_jump(
        phase_left, phase_right, T_boot,
        mu_star - half_width, mu_star + half_width,
        c_thresh_hint=c_thresh)
    c_thresh = (c_lb + c_rb) / 2.0
    dmu_dT = (mu_b - mu_star) / dT_boot
    results.append(dict(T=T_boot, mu_star=mu_b, c_left=c_lb, c_right=c_rb,
                        mu_predicted=mu_b, predictor_err=0.0))
    T, mu_star, c_left, c_right = T_boot, mu_b, c_lb, c_rb
    delta_c = c_right - c_left

    while sign_dT * (T_end - T) > 1e-12 and delta_c >= delta_c_min:
        # Adaptive step: near T_c where delta_c ~ sqrt(T_c-T),
        # dT ∝ delta_c^2 keeps |Δ(delta_c)| roughly constant per step.
        dT_adapt = dT_max * (delta_c ** 2)
        dT = sign_dT * min(dT_max, max(0.1, dT_adapt))
        if sign_dT * (T + dT - T_end) > 0:
            dT = T_end - T

        T_next       = T + dT
        mu_predicted = mu_star + dmu_dT * dT

        # Bracket centred on prediction with a generous safety margin
        bracket = max(half_width * 0.5, abs(dmu_dT * dT) * 2.0)
        mu_next, c_ln, c_rn = refine_isothermal_jump(
            phase_left, phase_right, T_next,
            mu_predicted - bracket, mu_predicted + bracket,
            c_thresh_hint=c_thresh)   # pass tracked discriminant

        predictor_err = abs(mu_next - mu_predicted)
        dmu_dT  = (mu_next - mu_star) / dT
        delta_c = c_rn - c_ln
        c_thresh = (c_ln + c_rn) / 2.0   # update discriminant

        results.append(dict(T=T_next, mu_star=mu_next, c_left=c_ln, c_right=c_rn,
                            mu_predicted=mu_predicted, predictor_err=predictor_err))
        T, mu_star, c_left, c_right = T_next, mu_next, c_ln, c_rn

    return results


# ── Reference: high-density brute-force with tight tolerance ─────────────────

def dense_reference(ph, T_array, mu_lo, mu_hi, tol=1e-11, delta_c_min=5e-4):
    """Independent high-precision call at every T — serves as ground truth.

    Maintains a tracked branch discriminant c_thresh that follows the coexistence
    concentrations as they converge to c_c near T_c.
    """
    results = []
    c_thresh = None   # will be set after the first successful refinement
    for T in T_array:
        try:
            c_lo = ph.concentration(T, mu_lo)
            c_hi = ph.concentration(T, mu_hi)
            if c_hi - c_lo < delta_c_min:
                break
            mu_star, c_left, c_right = refine_isothermal_jump(
                ph, ph, T, mu_lo, mu_hi, tol=tol, c_thresh_hint=c_thresh)
            if c_right - c_left < delta_c_min:
                break
            c_thresh = (c_left + c_right) / 2.0   # track discriminant
            results.append(dict(T=T, mu_star=mu_star, c_left=c_left, c_right=c_right))
        except Exception:
            break
    return results


def brute_force_coexistence(ph, T_array, mu_lo, mu_hi, tol=1e-9, delta_c_min=5e-4):
    """Standard-tolerance brute-force at each T in T_array."""
    return dense_reference(ph, T_array, mu_lo, mu_hi, tol=tol, delta_c_min=delta_c_min)


# ── Setup run parameters ──────────────────────────────────────────────────────

T_low  = 200.0
T_high = T_c - 0.05          # trace within 0.05 K of the critical temperature
DELTA_C_MIN = 5e-5            # stop when gap is this narrow (resolves near T_c with 20k-pt grid)
mu_lo_wide, mu_hi_wide = -0.25, 0.25

print(f"\nT range: {T_low:.0f} – {T_high:.1f} K  (up to T_c − 1 K)")
print(f"delta_c stopping threshold: {DELTA_C_MIN}")

# ── 1) Dense reference (300 points, tol = 1e-11) ─────────────────────────────
print("\nRunning dense reference (300 points, tol=1e-11) ...")
t0 = time.perf_counter()
T_ref    = np.linspace(T_low, T_high, 300)
ref_data = dense_reference(phase, T_ref, mu_lo_wide, mu_hi_wide,
                            tol=1e-11, delta_c_min=DELTA_C_MIN)
t_ref = time.perf_counter() - t0
print(f"  → {len(ref_data)} points in {t_ref:.2f} s  ({1e3*t_ref/max(len(ref_data),1):.1f} ms/pt)")

# ── 2) Brute-force (100 points, tol = 1e-9) ──────────────────────────────────
print("Running brute-force (100 calls, tol=1e-9) ...")
t0 = time.perf_counter()
T_bf    = np.linspace(T_low, T_high, 100)
bf_data = brute_force_coexistence(phase, T_bf, mu_lo_wide, mu_hi_wide,
                                   tol=1e-9, delta_c_min=DELTA_C_MIN)
t_bf = time.perf_counter() - t0
print(f"  → {len(bf_data)} points in {t_bf:.2f} s  ({1e3*t_bf/max(len(bf_data),1):.1f} ms/pt)")

# ── 3) CC tracer ──────────────────────────────────────────────────────────────
print("Running CC tracer (adaptive steps, tol=1e-9) ...")
t0 = time.perf_counter()
cc_data = trace_coexistence_line(
    phase, phase, T_low, mu_lo_wide, mu_hi_wide,
    T_end=T_high, delta_c_min=DELTA_C_MIN, dT_max=50.0)
t_cc = time.perf_counter() - t0
print(f"  → {len(cc_data)} steps in {t_cc:.2f} s  ({1e3*t_cc/max(len(cc_data),1):.1f} ms/step)")


# ── Unpack results ────────────────────────────────────────────────────────────

def unpack(data, key):
    return np.array([r[key] for r in data])

ref_T  = unpack(ref_data, "T");   ref_mu = unpack(ref_data, "mu_star")
ref_cL = unpack(ref_data, "c_left");  ref_cR = unpack(ref_data, "c_right")

bf_T  = unpack(bf_data, "T");    bf_mu = unpack(bf_data, "mu_star")
bf_cL = unpack(bf_data, "c_left");   bf_cR = unpack(bf_data, "c_right")

cc_T    = unpack(cc_data, "T");      cc_mu   = unpack(cc_data, "mu_star")
cc_cL   = unpack(cc_data, "c_left"); cc_cR   = unpack(cc_data, "c_right")
cc_pred = unpack(cc_data, "mu_predicted"); cc_perr = unpack(cc_data, "predictor_err")


# ── Accuracy metrics ──────────────────────────────────────────────────────────

# CC vs brute-force
T_common = bf_T[(bf_T >= cc_T.min()) & (bf_T <= cc_T.max())]
cc_on_bf  = np.interp(T_common, cc_T, cc_mu)
bf_on_bf  = np.interp(T_common, bf_T, bf_mu)
err_cc_bf = np.abs(cc_on_bf - bf_on_bf)

# CC and BF vs dense reference — evaluate at each method's own T points
# (interpolates the reference onto the sparser grid, avoiding aliasing artefacts)
ref_on_cc  = np.interp(cc_T, ref_T, ref_mu)
ref_on_bf  = np.interp(bf_T, ref_T, ref_mu)
err_cc_ref = np.abs(cc_mu - ref_on_cc)
err_bf_ref = np.abs(bf_mu - ref_on_bf)

# Predictor errors (skip bootstrap points 0 and 1)
perr = cc_perr[2:]
cc_T_perr = cc_T[2:]


# ── Print summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*62}")
print(f"{'Method':<22} {'N pts':>6} {'Wall time':>10} {'ms/pt':>9}")
print(f"{'-'*62}")
print(f"{'Dense reference':<22} {len(ref_data):>6} {t_ref:>9.2f}s {1e3*t_ref/max(len(ref_data),1):>8.1f}")
print(f"{'Brute-force':<22} {len(bf_data):>6} {t_bf:>9.2f}s {1e3*t_bf/max(len(bf_data),1):>8.1f}")
print(f"{'CC tracer':<22} {len(cc_data):>6} {t_cc:>9.2f}s {1e3*t_cc/max(len(cc_data),1):>8.1f}")
print(f"{'='*62}")
speedup = (t_bf / t_cc) if t_cc > 0 else float("inf")
print(f"CC tracer speedup vs brute-force: {speedup:.1f}×")

print(f"\nAccuracy (vs dense reference at tol=1e-11):")
print(f"  CC tracer  — median {np.median(err_cc_ref):.2e} eV,  max {np.max(err_cc_ref):.2e} eV")
print(f"  Brute-force — median {np.median(err_bf_ref):.2e} eV,  max {np.max(err_bf_ref):.2e} eV")
print(f"  Max |CC − BF| = {err_cc_bf.max():.2e} eV")

print(f"\nPredictor errors (before corrector step, n={len(perr)}):")
if len(perr) > 0:
    print(f"  Median {np.median(perr):.2e} eV,  max {perr.max():.2e} eV")
    frac_big = (perr > 1e-4).mean()
    print(f"  Steps with predictor err > 1e-4 eV: {frac_big:.0%}")
    print(f"  → Corrector {'IS needed' if frac_big > 0.1 else 'is NOT needed'} for this system at these step sizes")


# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    rf"Sub-regular solution  ($L_0={L0}$ eV, $L_1={L1}$ eV,  asymmetric gap,"
    rf"  $T_c\approx{T_c:.0f}$ K)",
    fontsize=13,
)

# ── Panel 1: coexistence line μ*(T) ──────────────────────────────────────────
ax = axes[0, 0]
ax.plot(ref_T, ref_mu * 1e3, "k-",  lw=1.5, zorder=4,
        label=f"Dense reference (300 pts, tol=1e-11)")
ax.plot(bf_T,  bf_mu  * 1e3, "b--", lw=1.5,
        label=f"Brute-force ({len(bf_T)} calls, {t_bf:.1f} s)")
ax.plot(cc_T,  cc_mu  * 1e3, "ro-", ms=5, lw=1.2,
        label=f"CC tracer ({len(cc_T)} steps, {t_cc:.1f} s)")
if len(cc_T) > 2:
    ax.plot(cc_T_perr, cc_pred[2:] * 1e3, "g+", ms=7, alpha=0.7,
            label="CC predictor only (before corrector)")
ax.axvline(T_c, ls=":", color="gray", lw=0.8, label=f"$T_c$ ≈ {T_c:.0f} K")
ax.set_xlabel("T  (K)")
ax.set_ylabel(r"$\mu^*$  (meV)")
ax.set_title(r"Coexistence line $\mu^*(T)$")
ax.legend(fontsize=8)

# ── Panel 2: binodal (c–T diagram, shows asymmetry) ──────────────────────────
ax = axes[0, 1]
if len(ref_cL):
    ax.fill_betweenx(ref_T, ref_cL, ref_cR, alpha=0.12, color="gray")
    ax.plot(ref_cL, ref_T, "k-",  lw=1.5, label="Dense reference")
    ax.plot(ref_cR, ref_T, "k-",  lw=1.5)
ax.plot(bf_cL,  bf_T,  "b--", lw=1.5, label="Brute-force")
ax.plot(bf_cR,  bf_T,  "b--", lw=1.5)
ax.plot(cc_cL,  cc_T,  "ro-", ms=4, lw=1.2, label="CC tracer")
ax.plot(cc_cR,  cc_T,  "ro-", ms=4, lw=1.2)
ax.axhline(T_c, ls=":", color="gray", lw=0.8)
ax.axvline(0.5, ls="--", color="lightgray", lw=0.6,
           label="c = 0.5  (symmetric axis)")
ax.plot(c_c, T_c, "k*", ms=12, zorder=5, label=f"critical point  c_c={c_c:.3f}")
ax.set_xlabel("Concentration  $c$")
ax.set_ylabel("T  (K)")
ax.set_title("Binodal  (asymmetric gap, critical c ≠ 0.5)")
ax.legend(fontsize=8, loc="lower center")

# ── Panel 3: accuracy vs dense reference ─────────────────────────────────────
ax = axes[1, 0]
ax.semilogy(cc_T, err_cc_ref + 1e-16, "r-", lw=1.5,
            label=r"$|\mu^*_\mathrm{CC} - \mu^*_\mathrm{ref}|$  (at CC T points)")
ax.semilogy(bf_T, err_bf_ref + 1e-16, "b--", lw=1.2,
            label=r"$|\mu^*_\mathrm{BF} - \mu^*_\mathrm{ref}|$  (at BF T points)")
if len(T_common):
    ax.semilogy(T_common, err_cc_bf + 1e-16, "g.-", lw=1.0, alpha=0.8,
                label=r"$|\mu^*_\mathrm{CC} - \mu^*_\mathrm{BF}|$")
ax.axhline(1e-9, ls="--", color="gray", lw=0.8, label="bisection tol (1e-9 eV)")
ax.axhline(1e-11, ls=":", color="gray", lw=0.8, label="ref tol (1e-11 eV)")
ax.set_xlabel("T  (K)")
ax.set_ylabel(r"$|\Delta\mu^*|$  (eV)")
ax.set_title("Accuracy vs dense reference")
ax.legend(fontsize=8)
ax.set_ylim(bottom=1e-14)

# ── Panel 4: predictor error — is the corrector necessary? ───────────────────
ax = axes[1, 1]
if len(perr):
    ax.semilogy(cc_T_perr, perr + 1e-16, "m.-", lw=1.3,
                label=r"$|\mu^*_\mathrm{corrected} - \mu^*_\mathrm{predicted}|$")
ax.axhline(1e-9, ls="--", color="gray", lw=0.8, label="bisection tol (1e-9 eV)")
ax.axhline(1e-4, ls="--", color="orange", lw=0.8, label="1e-4 eV visual threshold")
ax.set_xlabel("T  (K)")
ax.set_ylabel("Predictor error  (eV)")
ax.set_title("Corrector necessity")
ax.legend(fontsize=8)
ax.set_ylim(bottom=1e-14)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cc_demo.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
