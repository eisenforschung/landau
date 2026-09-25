"""Accuracy and speed of ``RedlichKister.fit`` against the exact least-squares fit.

``RedlichKister.fit`` subtracts the chord through the terminals and fits the
Redlich-Kister mixing term, which is linear in its parameters ``L_v``, to the
rest.  This compares two ways of solving that linear problem on random
over-determined fits shaped like line-phase data (6-11 line phases, free
energies around -3 eV/atom at 1000 K with the ideal mixing entropy removed):

* ``curve_fit``: ``scipy.optimize.curve_fit`` on ``_eval_mix`` from ``L = 0``;
* ``lstsq``: ``numpy.linalg.lstsq`` on the same basis, what the fit does now.

Both are measured against the least-squares fit solved in exact rational
arithmetic (the float inputs are exact binary fractions, so the normal
equations can be solved without rounding), as the largest deviation of the
fitted curve on a dense concentration grid.  Run with
``python benchmarks/bench_redlich_kister_fit.py`` from the repo root.
"""

from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
import scipy.optimize as so
from scipy.constants import Avogadro, eV

from landau.interpolate import RedlichKister, RedlichKisterInterpolation
from landau.phases import S, kB

J_PER_MOL = eV * Avogadro
T = 1000.0
GRID = np.linspace(0.0, 1.0, 201)


def sample(rng):
    """Concentrations and entropy-removed free energies of one random set of line phases."""
    n = int(rng.integers(6, 12))
    c = np.concatenate([[0.0], np.sort(rng.uniform(0.02, 0.98, n - 2)), [1.0]])
    energy = rng.uniform(-3.6, -2.8, n)
    entropy = rng.uniform(0.5, 2.6, n) * kB
    return c, energy - T * entropy + T * S(c)


def exact(c, f, nparam):
    """The reference: the least-squares fit from the normal equations in exact rationals."""
    c = [Fraction(x) for x in c]
    f = [Fraction(y) for y in f]
    f0, df = f[0], f[-1] - f[0]
    r = [y - f0 - df * x for x, y in zip(c, f)]
    B = [[x * (1 - x) * (2 * x - 1) ** v for v in range(nparam)] for x in c]
    A = [[sum(row[i] * row[j] for row in B) for j in range(nparam)] for i in range(nparam)]
    b = [sum(row[i] * ri for row, ri in zip(B, r)) for i in range(nparam)]
    for i in range(nparam):  # Gaussian elimination; A is symmetric positive definite
        for k in range(i + 1, nparam):
            factor = A[k][i] / A[i][i]
            A[k] = [a - factor * p for a, p in zip(A[k], A[i])]
            b[k] -= factor * b[i]
    L = [Fraction(0)] * nparam
    for i in reversed(range(nparam)):
        L[i] = (b[i] - sum(A[i][j] * L[j] for j in range(i + 1, nparam))) / A[i][i]

    def curve(x):
        x = Fraction(x)
        return float(f0 + df * x + x * (1 - x) * sum(Lv * (2 * x - 1) ** v for v, Lv in enumerate(L)))

    return np.array([curve(x) for x in GRID])


def curve_fit(c, f, nparam):
    r = f - f[0] - (f[-1] - f[0]) * c
    L, _ = so.curve_fit(RedlichKisterInterpolation._eval_mix, c, r, p0=np.zeros(nparam))
    return RedlichKisterInterpolation(f[-1] - f[0], f[0], L)


def lstsq(c, f, nparam):
    return RedlichKister(nparam).fit(c, f)


def main(cases=300, seed=0):
    rng = np.random.default_rng(seed)
    solvers = {"curve_fit": curve_fit, "lstsq": lstsq}
    errors = {name: [] for name in solvers}
    seconds = dict.fromkeys(solvers, 0.0)
    for _ in range(cases):
        c, f = sample(rng)
        nparam = min(int(rng.integers(3, 6)), len(c) - 2)
        reference = exact(c, f, nparam)
        for name, solve in solvers.items():
            start = time.perf_counter()
            fit = solve(c, f.copy(), nparam)
            seconds[name] += time.perf_counter() - start
            errors[name].append(np.max(np.abs(fit(GRID) - reference)) * J_PER_MOL)
    print(f"{cases} random fits, 6-11 line phases, 3-5 Redlich-Kister parameters")
    print("largest deviation from the least-squares fit on a 201-point grid, J/mol:")
    for name in solvers:
        e = np.array(errors[name])
        print(
            f"  {name:<9} median {np.median(e):8.1e}  95th percentile {np.percentile(e, 95):8.1e}  "
            f"max {e.max():8.1e}  cases above 1 J/mol {np.mean(e > 1) * 100:4.1f}%  "
            f"time per fit {seconds[name] / cases * 1e6:7.1f} us"
        )


if __name__ == "__main__":
    main()
