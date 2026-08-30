"""What ``WhitneyTemperatureInterpolator``'s extension does to an extrapolated ``F(T)``.

Fits an Einstein-oscillator free energy on a truncated temperature window and
carries both ``WhitneyTemperatureInterpolator`` and ``PolyFit(8)`` well past it,
reporting for each extrapolated temperature the signed error against the exact
curve.  Backs the ``Notes`` section of
:class:`~landau.interpolate.whitney.WhitneyTemperatureInterpolator`:

* the Whitney extension is linear, i.e. it freezes the entropy at the fitted
  boundary slope;
* because ``F(T)`` is concave, that tangent bounds the true free energy from
  above, while a polynomial is not bounded either way;
* with ``smoothing=0`` the residual against the training data is at machine
  precision, so it cannot gate fit quality.

Run: ``python benchmarks/bench_whitney_extension.py``
"""

import numpy as np
from scipy.constants import Boltzmann, eV

from landau.interpolate import PolyFit, WhitneyTemperatureInterpolator

kB = Boltzmann / eV

THETA = 350.0  # Einstein temperature, K
N_MODES = 3
E0 = -3.0  # eV/atom
T_LO, T_HI, N_DATA = 100.0, 600.0, 60
T_EXT_HI, N_EXT = 2000.0, 28


def free_energy(T):
    """Einstein-oscillator free energy per atom: concave in T, S -> 0 as T -> 0."""
    T = np.asarray(T, float)
    x = np.divide(THETA, T, out=np.full_like(T, np.inf), where=T > 0)
    zero_point = 0.5 * N_MODES * kB * THETA
    with np.errstate(over="ignore"):
        thermal = N_MODES * kB * T * np.log1p(-np.exp(-x))
    return E0 + zero_point + np.where(T > 0, thermal, 0.0)


def entropy(T, h=1e-3):
    return -(free_energy(T + h) - free_energy(T - h)) / (2 * h)


def main():
    T_data = np.linspace(T_LO, T_HI, N_DATA)
    F_data = free_energy(T_data)
    T_b = T_data.max()

    whitney = WhitneyTemperatureInterpolator(smoothing=0.0).fit(T_data, F_data)
    poly = PolyFit(8).fit(T_data, F_data)

    T_ext = np.linspace(T_b + 50.0, T_EXT_HI, N_EXT)
    F_true, F_whitney, F_poly = free_energy(T_ext), whitney(T_ext), poly(T_ext)

    slopes = np.diff(F_whitney) / np.diff(T_ext)
    print(f"data {T_LO:.0f}-{T_HI:.0f} K ({N_DATA} points), Einstein theta = {THETA:.0f} K\n")
    print("extension slope over the extrapolated range")
    print(f"  spread                 {np.ptp(slopes):.3e} eV/K   (0 = exactly linear)")
    print(f"  frozen S = -f'(T_b)    {-slopes.mean() / kB:7.4f} kB")
    print(f"  true S(T_b)            {entropy(T_b) / kB:7.4f} kB")

    print("\n     T    F_true   F_whitney     F_poly8   whitney-true     poly-true")
    for row in zip(T_ext, F_true, F_whitney, F_poly):
        T, ft, fw, fp = row
        print(f"{T:6.0f} {ft:9.5f} {fw:11.5f} {fp:11.5f} {fw - ft:+14.5f} {fp - ft:+13.5f}")

    print(f"\nwhitney >= true on the whole extrapolated range: {bool(np.all(F_whitney >= F_true))}")
    print(f"  smallest margin  {np.min(F_whitney - F_true):+.3e} eV at {T_ext[np.argmin(F_whitney - F_true)]:.0f} K")
    print(f"  largest  margin  {np.max(F_whitney - F_true):+.3e} eV at {T_ext[np.argmax(F_whitney - F_true)]:.0f} K")
    print(f"poly8 stays above true: {bool(np.all(F_poly >= F_true))}")

    print("\nresidual against own training data (RMS)")
    print(f"  whitney (smoothing=0)  {np.sqrt(np.mean((whitney(T_data) - F_data) ** 2)):.3e} eV")
    print(f"  PolyFit(8)             {np.sqrt(np.mean((poly(T_data) - F_data) ** 2)):.3e} eV")


if __name__ == "__main__":
    main()
