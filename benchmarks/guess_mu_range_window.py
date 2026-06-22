"""Diagnosis benchmark for the chemical-potential window of `guess_mu_range`.

`guess_mu_range` brackets the mu sampling window by inverting the Boltzmann-
averaged concentration c(mu) at `tolerance` away from 0 and 1.  It locates the
ends of the scan with `scipy.optimize.brute`, whose default `finish=fmin` polish
is *unconstrained*: because c(mu) approaches 0 and 1 only asymptotically along
the single-phase tails, the polish walks far outside the (-10, 10) bracket.  The
oversized bracket makes the `linspace` scan grid coarse, and the inverted window
ends up reaching deep into the single-phase regions.

This script builds a small alloy-like system (fcc + liquid solutions plus three
intermetallic line phases, terminal phases being *solutions* so c(mu) has the
asymptotic tails), then compares the window from the old `finish=fmin` call with
the bounded `finish=None` call.  It reports, per temperature:

* the mu window [mu_min, mu_max] and its width,
* the round-trip error |c(mu_edge) - c_edge|: the average concentration actually
  realised at each returned edge versus the (c0, c1) the window claims to span.

It also writes ``guess_mu_range_window.png``: c(mu) with both windows shaded, so
the single-phase over-reach of the old window is visible against the transition.

Run: ``python benchmarks/guess_mu_range_window.py``
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
import scipy.optimize as so

from landau.calculate import _semigrand_average_concentration, guess_mu_range
from landau.phases import LinePhase, RegularSolution


def alloy_phases():
    """fcc + liquid solutions (asymptotic c(mu) tails) plus intermetallics."""
    fcc = RegularSolution(
        "fcc", phases=[LinePhase("Au_fcc", 0, 0.00), LinePhase("m_fcc", 0.5, 0.05), LinePhase("Cu_fcc", 1, 0.00)]
    )
    liquid = RegularSolution(
        "liquid", phases=[LinePhase("Au_l", 0, 0.11), LinePhase("m_l", 0.5, 0.14), LinePhase("Cu_l", 1, 0.12)]
    )
    return [fcc, liquid, LinePhase("Au4Cu2", 0.33, -0.02), LinePhase("AuCu", 0.50, -0.03), LinePhase("AuCu3", 0.75, -0.02)]


def guess_mu_range_unbounded(phases, T, samples, tolerance=1e-2):
    """`guess_mu_range` as it was before the fix: `so.brute` keeps its default
    `finish=fmin` polish, which escapes the (-10, 10) scan bracket."""
    def c(mu):
        return _semigrand_average_concentration(phases, T, mu)

    mu0 = so.brute(lambda x: +c(x[0]), ranges=[(-10, 10)], Ns=200)[0]
    mu1 = so.brute(lambda x: -c(x[0]), ranges=[(-10, 10)], Ns=200)[0]
    mm = np.linspace(mu0, mu1, samples)
    cc = c(mm)
    c0, c1 = min(cc) + tolerance, max(cc) - tolerance
    order = np.argsort(cc)
    cc_s, mm_s = cc[order], mm[order]
    _, uniq = np.unique(cc_s, return_index=True)
    return si.interp1d(cc_s[uniq], mm_s[uniq])(np.linspace(c0, c1, samples)), c0, c1


def report(phases, T, samples=100):
    rows = {}
    for label, fn in (("old finish=fmin", guess_mu_range_unbounded), ("new finish=None", guess_mu_range)):
        mus, c0, c1 = fn(phases, T, samples)
        err_lo = abs(float(_semigrand_average_concentration(phases, T, mus.min())) - c0)
        err_hi = abs(float(_semigrand_average_concentration(phases, T, mus.max())) - c1)
        rows[label] = (mus.min(), mus.max(), mus.max() - mus.min(), max(err_lo, err_hi))
        print(
            f"T={T:6.0f}  {label:16s}  mu in [{mus.min():7.3f}, {mus.max():7.3f}]  "
            f"width={mus.max() - mus.min():6.3f}  round-trip err={max(err_lo, err_hi):.2e}"
        )
    return rows


def plot(phases, T, path):
    mm = np.linspace(-10, 10, 4001)
    cc = _semigrand_average_concentration(phases, T, mm)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mm, cc, color="k", lw=1.5, label="c(mu)")
    old, _, _ = guess_mu_range_unbounded(phases, T, 100)
    new, _, _ = guess_mu_range(phases, T, 100)
    ax.axvspan(old.min(), old.max(), color="tab:red", alpha=0.15, label=f"old window (width {old.max() - old.min():.2f})")
    ax.axvspan(new.min(), new.max(), color="tab:green", alpha=0.25, label=f"new window (width {new.max() - new.min():.2f})")
    ax.set(xlabel="mu", ylabel="Boltzmann-averaged c", title=f"guess_mu_range window, T={T:.0f} K", xlim=(-6, 6))
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def main():
    phases = alloy_phases()
    for T in (1400.0, 800.0, 500.0):
        report(phases, T)
    plot(phases, 1400.0, "guess_mu_range_window.png")


if __name__ == "__main__":
    main()
