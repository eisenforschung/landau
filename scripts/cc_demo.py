"""
Comparison of coexistence tracing methods for a demixing regular solution.

Two methods:
  1. Brute-force: independent isothermal refinement at each T (wide bracket, many calls)
  2. Clausius-Clapeyron tracer: predictor-corrector with finite-difference CC slope

Both use the same refine_isothermal_jump as the inner solver.
The 'corrector necessity' panel shows the predictor error before the corrector step.

Uses an analytic regular-solution phase with sharp kinks so that concentration()
gives exact coexistence values.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.constants import Boltzmann, eV
import scipy.optimize as so
from landau.phases import Phase

kB = Boltzmann / eV  # eV/K

# Helper: configurational entropy density and its derivative
def _S(c):
    c = np.asarray(c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = kB * (np.where(c > 0, -c * np.log(c), 0) + np.where(c < 1, -(1-c) * np.log(1-c), 0))
    return s

def _dS_dc(c):
    with np.errstate(divide="ignore"):
        return kB * np.log((1 - c) / c)


# ── Analytic regular solution phase ─────────────────────────────────────────
@dataclass(frozen=True)
class AnalyticRegularSolution(Phase):
    """f(T, c) = omega*c*(1-c) + delta_f*c - T*S(c)

    semigrand_potential = min_c [f - mu*c] with exact branch selection.
    concentration       = argmin_c [f - mu*c], with sharp jump at mu*.
    """
    omega: float    # mixing energy (eV)
    delta_f: float  # asymmetry, shifts mu* (eV)

    def _f(self, T, c):
        return self.omega * c * (1 - c) + self.delta_f * c - T * _S(c)

    def _df(self, T, c):
        """df/dc"""
        return self.omega * (1 - 2*c) + self.delta_f - T * _dS_dc(c)

    def _stable_concentration(self, T, dmu):
        """Find c(s) that minimise f(c) - dmu*c.  Returns the stable (global min) c."""
        # df/dc = dmu has 1 or 3 solutions in (0, 1)
        c_arr = np.linspace(1e-10, 1 - 1e-10, 2000)
        res   = self._df(T, c_arr) - dmu
        sign_changes = np.where(np.diff(np.sign(res)))[0]
        roots = []
        for i in sign_changes:
            try:
                root = so.brentq(lambda c: self._df(T, c) - dmu, c_arr[i], c_arr[i+1])
                roots.append(root)
            except ValueError:
                pass
        if not roots:
            return 0.5   # fallback (shouldn't happen in normal range)
        if len(roots) == 1:
            return roots[0]
        # Multiple roots → pick global minimum of f(c) - dmu*c
        g_vals = [self._f(T, r) - dmu * r for r in roots]
        return roots[int(np.argmin(g_vals))]

    def semigrand_potential(self, T, dmu):
        if np.isscalar(dmu):
            c = self._stable_concentration(T, dmu)
            return float(self._f(T, c) - dmu * c)
        return np.array([self.semigrand_potential(T, d) for d in np.asarray(dmu)])

    def concentration(self, T, dmu):
        if np.isscalar(dmu):
            return float(self._stable_concentration(T, dmu))
        return np.array([self._stable_concentration(T, d) for d in np.asarray(dmu)])


# ── Phase setup ──────────────────────────────────────────────────────────────
# omega = 0.1 eV → Tc ≈ 580 K;  delta_f = 0.015 eV makes dmu*(T) vary
omega   = 0.1
delta_f = 0.015
T_c     = omega / (2.0 * kB)   # ≈ 580 K (exact for symmetric system)
phase   = AnalyticRegularSolution(name="demix", omega=omega, delta_f=delta_f)

print(f"Tc ≈ {T_c:.1f} K")


# ── refine_isothermal_jump ───────────────────────────────────────────────────

def refine_isothermal_jump(phase_left, phase_right, T, mu_lo, mu_hi, tol=1e-9):
    """Return (mu_star, c_left, c_right).

    Case A – two different phases: bisect on Phi_left − Phi_right.
    Case B – same phase (miscibility gap): bisection accelerated by tangent-
             intersection probe; branch discriminated by .concentration().
             c_left / c_right read from the converged bracket endpoints.
    """
    if phase_left is not phase_right:
        lo, hi = mu_lo, mu_hi
        while hi - lo > tol:
            mid   = (lo + hi) / 2.0
            delta = (phase_left.semigrand_potential(T, mid)
                     - phase_right.semigrand_potential(T, mid))
            lo, hi = (mid, hi) if delta < 0 else (lo, mid)
        mu_star = (lo + hi) / 2.0
        return mu_star, phase_left.concentration(T, mu_star), phase_right.concentration(T, mu_star)

    # Case B – single phase miscibility gap
    ph = phase_left
    phi_lo = ph.semigrand_potential(T, mu_lo)
    phi_hi = ph.semigrand_potential(T, mu_hi)
    c_lo   = ph.concentration(T, mu_lo)
    c_hi   = ph.concentration(T, mu_hi)

    if abs(c_hi - c_lo) < 1e-10:
        raise ValueError(f"No concentration jump in bracket [{mu_lo}, {mu_hi}] at T={T}")

    c_thresh = (c_lo + c_hi) / 2.0   # fixed branch discriminator

    lo, hi = mu_lo, mu_hi
    while hi - lo > tol:
        denom    = c_hi - c_lo
        mu_trial = ((phi_lo - phi_hi + c_hi * hi - c_lo * lo) / denom
                    if abs(denom) > 1e-15 else (lo + hi) / 2.0)
        if not (lo < mu_trial < hi):
            mu_trial = (lo + hi) / 2.0

        phi_t = ph.semigrand_potential(T, mu_trial)
        c_t   = ph.concentration(T, mu_trial)

        if c_t < c_thresh:
            lo, phi_lo, c_lo = mu_trial, phi_t, c_t
        else:
            hi, phi_hi, c_hi = mu_trial, phi_t, c_t

    mu_star = (lo + hi) / 2.0
    # c_left / c_right from bracket endpoints (sharp-kink phase: exact branch values)
    return mu_star, ph.concentration(T, lo), ph.concentration(T, hi)


# ── trace_coexistence_line ───────────────────────────────────────────────────

def trace_coexistence_line(phase_left, phase_right,
                           T_start, mu_lo_init, mu_hi_init,
                           T_end, delta_c_min=5e-3, dT_max=50.0):
    """Predictor-corrector coexistence tracer.

    Returns list of dicts: T, mu_star, c_left, c_right, mu_predicted, predictor_err.
    """
    sign_dT = +1 if T_end > T_start else -1

    mu_star, c_left, c_right = refine_isothermal_jump(
        phase_left, phase_right, T_start, mu_lo_init, mu_hi_init)
    delta_c    = c_right - c_left
    half_width = (mu_hi_init - mu_lo_init) / 2.0
    results    = [dict(T=T_start, mu_star=mu_star, c_left=c_left, c_right=c_right,
                       mu_predicted=mu_star, predictor_err=0.0)]

    if delta_c < delta_c_min:
        return results

    # Bootstrap: one small step to seed the CC slope dmu*/dT
    dT_boot   = sign_dT * min(1.0, abs(T_end - T_start) * 0.01)
    T_boot    = T_start + dT_boot
    mu_b, c_lb, c_rb = refine_isothermal_jump(
        phase_left, phase_right, T_boot,
        mu_star - half_width, mu_star + half_width)
    dmu_dT = (mu_b - mu_star) / dT_boot
    results.append(dict(T=T_boot, mu_star=mu_b, c_left=c_lb, c_right=c_rb,
                        mu_predicted=mu_b, predictor_err=0.0))
    T, mu_star, c_left, c_right = T_boot, mu_b, c_lb, c_rb
    delta_c = c_right - c_left

    while sign_dT * (T_end - T) > 1e-12 and delta_c >= delta_c_min:
        # dT proportional to delta_c^2 targets constant |d(delta_c)| per step
        # (near Tc: delta_c ~ sqrt(Tc-T), so d(delta_c)/dT ~ 1/(2*sqrt), dT ~ delta_c^2)
        dT_adaptive = dT_max * delta_c ** 2
        dT = sign_dT * min(dT_max, max(0.2, dT_adaptive))
        if sign_dT * (T + dT - T_end) > 0:
            dT = T_end - T

        T_next       = T + dT
        mu_predicted = mu_star + dmu_dT * dT
        bracket      = half_width + abs(dmu_dT * dT) * 1.5

        mu_next, c_ln, c_rn = refine_isothermal_jump(
            phase_left, phase_right, T_next,
            mu_predicted - bracket, mu_predicted + bracket)

        predictor_err = abs(mu_next - mu_predicted)
        dmu_dT  = (mu_next - mu_star) / dT
        delta_c = c_rn - c_ln

        results.append(dict(T=T_next, mu_star=mu_next, c_left=c_ln, c_right=c_rn,
                            mu_predicted=mu_predicted, predictor_err=predictor_err))
        T, mu_star, c_left, c_right = T_next, mu_next, c_ln, c_rn

    return results


# ── Brute-force reference: independent isothermal calls ──────────────────────

def brute_force_coexistence(phase, T_array, mu_lo_wide, mu_hi_wide, delta_c_min=5e-3):
    """Refine miscibility-gap coexistence independently at each T."""
    results = []
    for T in T_array:
        c_lo = phase.concentration(T, mu_lo_wide)
        c_hi = phase.concentration(T, mu_hi_wide)
        if c_hi - c_lo < delta_c_min:
            break
        try:
            mu_star, c_left, c_right = refine_isothermal_jump(
                phase, phase, T, mu_lo_wide, mu_hi_wide)
        except Exception:
            break
        if c_right - c_left < delta_c_min:
            break
        results.append(dict(T=T, mu_star=mu_star, c_left=c_left, c_right=c_right))
    return results


# ── Analytical coexistence for reference ────────────────────────────────────
def analytic_coexistence(omega, delta_f, T_array):
    """Find common tangent coexistence by continuation from near Tc downward.

    The equations for the common tangent construction are:
      (1)  df/dc(c_L) = df/dc(c_R)          (equal slope)
      (2)  f(c_R) - f(c_L) = mu* * (c_R - c_L)  (equal Phi at mu*)
    where mu* = df/dc(c_L).
    """
    results = []

    def f(T, c):
        return omega * c * (1-c) + delta_f * c - T * _S(c)

    def df(T, c):
        return omega * (1 - 2*c) + delta_f + T * kB * np.log(c / (1 - c))

    def equations(x, T):
        c_L, c_R = x
        if not (1e-8 < c_L < c_R < 1-1e-8):
            return [1e6, 1e6]
        mu_L = df(T, c_L)
        mu_R = df(T, c_R)
        f_diff = f(T, c_R) - f(T, c_L)
        return [mu_L - mu_R, f_diff - mu_L * (c_R - c_L)]

    # Start from T = 0.95 * T_c where the gap is visible but well-conditioned
    T_seed = 0.95 * T_c
    dc_init = 0.2    # generous initial offset from 0.5
    x0 = [0.5 - dc_init, 0.5 + dc_init]
    try:
        sol = so.fsolve(equations, x0, args=(T_seed,), full_output=True)
        c_L, c_R = sol[0]
        if c_L <= 1e-8 or c_R >= 1-1e-8 or c_R - c_L < 1e-3:
            return results
        x_prev = [c_L, c_R]
    except Exception:
        return results

    # Continuation downward and upward from T_seed
    T_sorted = sorted(T_array)

    # Downward from T_seed
    down_results = []
    xp = list(x_prev)
    for T in sorted([t for t in T_sorted if t <= T_seed], reverse=True):
        try:
            sol = so.fsolve(equations, xp, args=(T,), full_output=True)
            c_L, c_R = sol[0]
            if c_L < 1e-8 or c_R > 1-1e-8 or c_R - c_L < 1e-3:
                break
            xp = [c_L, c_R]
            mu_star = df(T, c_L)
            down_results.append(dict(T=T, mu_star=mu_star, c_left=c_L, c_right=c_R))
        except Exception:
            break

    # Upward from T_seed
    up_results = []
    xp = list(x_prev)
    for T in sorted([t for t in T_sorted if t > T_seed]):
        try:
            sol = so.fsolve(equations, xp, args=(T,), full_output=True)
            c_L, c_R = sol[0]
            if c_L < 1e-8 or c_R > 1-1e-8 or c_R - c_L < 1e-3:
                break
            xp = [c_L, c_R]
            mu_star = df(T, c_L)
            up_results.append(dict(T=T, mu_star=mu_star, c_left=c_L, c_right=c_R))
        except Exception:
            break

    return sorted(down_results + up_results, key=lambda r: r["T"])


# ── Run ──────────────────────────────────────────────────────────────────────
T_low  = 200.0
T_high = T_c * 0.98

# Wide bracket: well outside the jump at all T
# At T_low, the jump is very sharp so use a bracket 10x wider than needed
mu_lo_wide, mu_hi_wide = -0.05, 0.12

# Verify bracket works at both ends of T range
c_lo_check_low  = phase.concentration(T_low,  mu_lo_wide)
c_hi_check_low  = phase.concentration(T_low,  mu_hi_wide)
c_lo_check_high = phase.concentration(T_high, mu_lo_wide)
c_hi_check_high = phase.concentration(T_high, mu_hi_wide)
print(f"Wide bracket [{mu_lo_wide}, {mu_hi_wide}] eV")
print(f"  T={T_low:.0f}: c = [{c_lo_check_low:.3f}, {c_hi_check_low:.3f}]")
print(f"  T={T_high:.0f}: c = [{c_lo_check_high:.3f}, {c_hi_check_high:.3f}]")

# 1) Brute-force (80 independent calls)
print("\nRunning brute-force...")
T_bf    = np.linspace(T_low, T_high, 80)
bf_data = brute_force_coexistence(phase, T_bf, mu_lo_wide, mu_hi_wide)
print(f"Brute-force: {len(bf_data)} successful calls")

# 2) CC tracer
print("Running CC tracer...")
cc_data = trace_coexistence_line(phase, phase, T_low, mu_lo_wide, mu_hi_wide,
                                 T_end=T_high, delta_c_min=5e-3, dT_max=50.0)
print(f"CC tracer: {len(cc_data)} steps (including bootstrap)")

# 3) Analytic reference
print("Running analytic reference...")
T_ref     = np.linspace(T_low, T_high, 200)
ref_data  = analytic_coexistence(omega, delta_f, T_ref)
print(f"Analytic: {len(ref_data)} points")


# ── Unpack ───────────────────────────────────────────────────────────────────
bf_T  = np.array([r["T"]        for r in bf_data])
bf_mu = np.array([r["mu_star"]  for r in bf_data])
bf_cL = np.array([r["c_left"]   for r in bf_data])
bf_cR = np.array([r["c_right"]  for r in bf_data])

cc_T    = np.array([r["T"]             for r in cc_data])
cc_mu   = np.array([r["mu_star"]       for r in cc_data])
cc_cL   = np.array([r["c_left"]        for r in cc_data])
cc_cR   = np.array([r["c_right"]       for r in cc_data])
cc_pred = np.array([r["mu_predicted"]  for r in cc_data])
cc_perr = np.array([r["predictor_err"] for r in cc_data])

ref_T  = np.array([r["T"]       for r in ref_data])
ref_mu = np.array([r["mu_star"] for r in ref_data])
ref_cL = np.array([r["c_left"]  for r in ref_data])
ref_cR = np.array([r["c_right"] for r in ref_data])

# Error: |mu*_CC - mu*_BF|  (interpolate CC onto BF T-grid)
mask = (bf_T >= cc_T.min()) & (bf_T <= cc_T.max())
cc_on_bf = np.interp(bf_T[mask], cc_T, cc_mu)
abs_err_cc_bf = np.abs(cc_on_bf - bf_mu[mask])

# Error vs analytic
cc_on_ref = np.interp(ref_T, cc_T, cc_mu, left=np.nan, right=np.nan)
bf_on_ref = np.interp(ref_T, bf_T, bf_mu, left=np.nan, right=np.nan)
cc_err_analytic = np.abs(cc_on_ref - ref_mu)
bf_err_analytic = np.abs(bf_on_ref - ref_mu)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n--- Summary ---")
print(f"Max  |μ*_CC − μ*_BF|     = {abs_err_cc_bf.max():.2e} eV")
print(f"Mean |μ*_CC − μ*_BF|     = {abs_err_cc_bf.mean():.2e} eV")
perr = cc_perr[2:]
print(f"\nPredictor errors (before corrector):")
print(f"  Median = {np.median(perr):.2e} eV")
print(f"  Max    = {perr.max():.2e} eV")
frac_big = (perr > 1e-4).mean()
print(f"  Steps with predictor err > 1e-4 eV: {frac_big:.0%}")
print(f"  → corrector {'IS' if frac_big > 0.1 else 'is NOT'} necessary at this step size")


# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    rf"Analytic regular solution  ($\omega$ = {omega} eV, $\delta f$ = {delta_f} eV,"
    rf"  $T_c \approx$ {T_c:.0f} K)",
    fontsize=13,
)

# Panel 1: coexistence line in (T, dmu) space
ax = axes[0, 0]
ax.plot(ref_T,  ref_mu * 1e3,  "k-",  lw=1.5, label="Analytic (reference)", zorder=3)
ax.plot(bf_T,   bf_mu  * 1e3,  "b--", lw=1.5, label=f"Brute-force ({len(bf_T)} iso. calls)")
ax.plot(cc_T,   cc_mu  * 1e3,  "ro-", ms=5, lw=1.2,
        label=f"CC tracer ({len(cc_T)} steps)")
ax.plot(cc_T[2:], cc_pred[2:] * 1e3, "g+", ms=7, alpha=0.8, label="Predictor only")
ax.axvline(T_c, ls=":", color="gray", lw=0.8)
ax.set_xlabel("T  (K)")
ax.set_ylabel(r"$\Delta\mu^*$  (meV)")
ax.set_title(r"Coexistence line in $(T,\,\Delta\mu^*)$ space")
ax.legend(fontsize=8)

# Panel 2: binodal
ax = axes[0, 1]
ax.fill_betweenx(ref_T, ref_cL, ref_cR, alpha=0.12, color="gray", label="two-phase region")
ax.plot(ref_cL, ref_T, "k-",  lw=1.5, label="Analytic")
ax.plot(ref_cR, ref_T, "k-",  lw=1.5)
ax.plot(bf_cL,  bf_T,  "b--", lw=1.5, label="Brute-force")
ax.plot(bf_cR,  bf_T,  "b--", lw=1.5)
ax.plot(cc_cL,  cc_T,  "ro-", ms=4, lw=1.2, label="CC tracer")
ax.plot(cc_cR,  cc_T,  "ro-", ms=4, lw=1.2)
ax.axhline(T_c, ls=":", color="gray", lw=0.8)
ax.set_xlabel("Concentration c")
ax.set_ylabel("T  (K)")
ax.set_title("Binodal")
ax.legend(fontsize=8, loc="lower center")

# Panel 3: absolute agreement between methods
ax = axes[1, 0]
ax.semilogy(bf_T[mask], abs_err_cc_bf + 1e-16, "g.-", lw=1.2,
            label=r"$|\mu^*_\mathrm{CC} - \mu^*_\mathrm{BF}|$")
ok = ~np.isnan(cc_err_analytic)
ax.semilogy(ref_T[ok], cc_err_analytic[ok] + 1e-16, "r-", lw=1.2,
            label=r"$|\mu^*_\mathrm{CC} - \mu^*_\mathrm{analytic}|$")
ok2 = ~np.isnan(bf_err_analytic)
ax.semilogy(ref_T[ok2], bf_err_analytic[ok2] + 1e-16, "b--", lw=1.2,
            label=r"$|\mu^*_\mathrm{BF} - \mu^*_\mathrm{analytic}|$")
ax.axhline(1e-9, ls="--", color="gray", lw=0.8, label="bisection tol (10⁻⁹ eV)")
ax.set_xlabel("T  (K)")
ax.set_ylabel(r"$|\Delta\mu^*|$ error  (eV)")
ax.set_title("Accuracy relative to analytic solution")
ax.legend(fontsize=8)
ax.set_ylim(bottom=1e-12)

# Panel 4: predictor error (how necessary is the corrector?)
ax = axes[1, 1]
ax.semilogy(cc_T[2:], cc_perr[2:] + 1e-16, "m.-", lw=1.3,
            label=r"$|\mu^*_\mathrm{corrected} - \mu^*_\mathrm{predicted}|$  (corrector change)")
ax.axhline(1e-9, ls="--", color="gray", lw=0.8, label="bisection tol")
ax.axhline(1e-4, ls="--", color="orange", lw=0.8, label="1e-4 eV visual threshold")
ax.set_xlabel("T  (K)")
ax.set_ylabel("Predictor error  (eV)")
ax.set_title("Corrector necessity: predictor vs. corrected result")
ax.legend(fontsize=8)
ax.set_ylim(bottom=1e-12)

plt.tight_layout()
out_path = "/home/runner/work/landau/landau/scripts/cc_demo.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
