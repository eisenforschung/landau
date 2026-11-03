"""Refined phase diagrams of a fixed set of systems, for comparing two checkouts.

Used to check that parametrising the refiners over an ``Axes`` left their
output unchanged. Run once per checkout, each with its own ``landau`` first on
the path, then compare the two pickles::

    PYTHONPATH=<base checkout> python benchmarks/refine_axes_equivalence.py dump base.pkl
    PYTHONPATH=<this checkout> python benchmarks/refine_axes_equivalence.py dump branch.pkl
    python benchmarks/refine_axes_equivalence.py compare base.pkl branch.pkl

``compare`` asserts the frames are identical (``check_exact=True``) and prints
how many rows each refiner contributed, per locus.

The systems cover every refiner on its default plane: a three-phase eutectic
(triple point, Clausius-Clapeyron lines, terminal melting points), a regular
solution's miscibility gap, a monotectic, both 1-D scans, a line phase melting
congruently from an ideal liquid (the exact congruent solve), and a symmetric
and an asymmetric gap dome above a terminal line phase (the gap closure).
"""

import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from landau.calculate import calc_phase_diagram
from landau.interpolate import PolyFit
from landau.phases import FastInterpolatingPhase, IdealSolution, LinePhase, RegularSolution, kB


def _ideal(name, eA, sA, eB, sB):
    return IdealSolution(name, LinePhase(name + "A", 0, eA, sA * kB), LinePhase(name + "B", 1, eB, sB * kB))


def _regular(name, L0, s=0.0, eA=0.0, eB=0.0):
    pts = [LinePhase(name + "0", 0.0, eA, s), LinePhase(name + "h", 0.5, (eA + eB) / 2 + L0 / 4, s),
           LinePhase(name + "1", 1.0, eB, s)]
    return RegularSolution(name=name, phases=pts, num_coeffs=1, add_entropy=True)


def _gap_liquid(L0, L1=0.0, E_A=-2.8, E_B=-2.35, S=3.0):
    def E_mix(c):
        return c * (1 - c) * (L0 + L1 * (2 * c - 1))

    controls = [LinePhase(f"l{i}", c, (1 - c) * E_A + c * E_B + E_mix(c), S * kB)
                for i, c in enumerate([0, 0.25, 0.5, 0.75, 1])]
    return FastInterpolatingPhase("liquid", controls, add_entropy=True, interpolator=PolyFit(4))


def diagrams():
    eutectic = [_ideal("hcp", -2.975, 1.8, -1.95, 1.1), _ideal("fcc", -3.0, 1.0, -2.0, 1.1),
                _ideal("liquid", -2.75, 5.0, -1.75, 4.4)]
    monotectic = [_regular("liquid", 0.12, s=3 * kB),
                  IdealSolution("solid", LinePhase("sA", 0, -0.08, 1 * kB), LinePhase("sB", 1, 0.05, 1 * kB))]
    congruent = [_ideal("liquid", -2.8, 3.0, -2.35, 3.0), LinePhase("γ", 0.5, -2.85, 1.0 * kB),
                 LinePhase("sA", 0, -2.9, 1 * kB), LinePhase("sB", 1, -2.45, 1 * kB)]
    alpha = LinePhase("α", 0, -3.0, 1.0 * kB)
    return {
        "eutectic": calc_phase_diagram(eutectic, np.linspace(200, 1000, 25), mu=50, keep_unstable=True),
        "gap": calc_phase_diagram([_regular("sol", 0.1)], np.linspace(150, 900, 20), mu=np.linspace(-0.05, 0.05, 21)),
        "monotectic": calc_phase_diagram(monotectic, np.linspace(100, 1500, 25), mu=np.linspace(-0.4, 0.4, 41)),
        "scan_mu": calc_phase_diagram(eutectic, 600.0, mu=60),
        "scan_T": calc_phase_diagram(eutectic, np.linspace(200, 1000, 41), mu=0.6),
        "congruent": calc_phase_diagram(congruent, np.linspace(300.0, 2000.0, 40), mu=60),
        "gap_sym": calc_phase_diagram([alpha, _gap_liquid(0.25)], np.linspace(300.0, 1600.0, 80), mu=100),
        "gap_asym": calc_phase_diagram([alpha, _gap_liquid(0.25, 0.08)], np.linspace(300.0, 1600.0, 80), mu=100),
    }


def main(argv):
    warnings.simplefilter("ignore")
    if argv[:1] == ["dump"]:
        with open(argv[1], "wb") as f:
            pickle.dump(diagrams(), f)
    elif argv[:1] == ["compare"]:
        with open(argv[1], "rb") as f:
            a = pickle.load(f)
        with open(argv[2], "rb") as f:
            b = pickle.load(f)
        assert a.keys() == b.keys()
        for k in a:
            pd.testing.assert_frame_equal(a[k], b[k], check_exact=True)
            counts = a[k][a[k]["refined"] != "no"].groupby(["refined", "locus"]).size()
            print(f"{k:11s} identical, {len(a[k]):5d} rows;",
                  ", ".join(f"{r}/{loc}: {n}" for (r, loc), n in counts.items()))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
