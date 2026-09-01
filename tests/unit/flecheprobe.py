"""Digests a representative landau object of each kind, twice, and prints JSON.

Run as a script by ``tests/unit/test_fleche.py`` in a subprocess with an
explicit ``PYTHONHASHSEED``; see the note there for why a process pool cannot
stand in for that.  The two passes bracket ``load_entry_points()`` so the caller
can tell whether any digest depends on fleche having loaded its hooks yet.

``salt`` reports ``hash("x")`` so the caller can assert that two runs really did
get different hash seeds -- without it, a harness that failed to vary the seed
would make every assertion here vacuously true.
"""

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from fleche.digest import Indigestible, digest, load_entry_points

import landau.poly as poly
import landau.refine as refine
from landau import (
    SGTE,
    FastInterpolatingPhase,
    IdealSolution,
    LinePhase,
    PolyFit,
    RedlichKister,
    RegularSolution,
    SplineFit,
    TemperatureDependentLinePhase,
)
from landau.interpolate import NumericalDerivative
from landau.phases import pointdefects

T = np.linspace(100, 1000, 20)
A = TemperatureDependentLinePhase("A", 0.0, T, -T * 1e-3, SGTE(3))
B = TemperatureDependentLinePhase("B", 1.0, T, -T * 1.1e-3, SGTE(3))
M = TemperatureDependentLinePhase("M", 0.5, T, -T * 1.2e-3 - 0.05, SGTE(3))
L = LinePhase("L", 0.3, -0.1, 1e-4)
defect = pointdefects.ConstantPointDefect("d", 0.1, 1e-4, 0.5)
sublattice = pointdefects.PointDefectSublattice("s", 0, 1.0, [defect])


def build():
    """A fresh instance of every type the hooks are meant to cover."""
    return {
        "LinePhase": L,
        "TemperatureDependentLinePhase": A,
        "IdealSolution": IdealSolution("I", A, B),
        "RegularSolution": RegularSolution("R", [A, M, B]),
        "FastInterpolatingPhase": FastInterpolatingPhase("F", [A, M, B]),
        "PolyFit": PolyFit(3),
        "SGTE": SGTE(3),
        "RedlichKister": RedlichKister(3),
        "SplineFit": SplineFit(),
        "PolynomialInterpolation": PolyFit(3).fit(T, T * 2.0),
        "SGTEInterpolation": SGTE(3).fit(T, -T * 1e-3),
        "NumericalDerivative": NumericalDerivative(PolyFit(3).fit(T, T * 2.0)),
        "ConstantPointDefect": defect,
        "PointDefectSublattice": sublattice,
        "PointDefectedPhase": pointdefects.PointDefectedPhase("PD", L, [sublattice]),
        "ScanRefiner": refine.ScanRefiner("mu"),
        "DelaunayTripleRefiner": refine.DelaunayTripleRefiner(),
        "ClausiusClapeyronRefiner": refine.ClausiusClapeyronRefiner(),
        "MiscibilityGapRefiner": refine.MiscibilityGapRefiner(),
        "Concave": poly.Concave(),
        "Segments": poly.Segments(),
        **ase_objects(),
    }


def ase_objects():
    """``AsePhase`` if ASE is installed; it has a hook of its own."""
    try:
        from ase.thermochemistry import HarmonicThermo

        from landau import AsePhase
    except ImportError:
        return {}
    return {"AsePhase": AsePhase("ase", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.03])))}


def one(value):
    try:
        return str(digest(value))
    except Indigestible:
        return "REFUSED"


def main():
    cold = {k: one(v) for k, v in build().items()}
    load_entry_points()
    warm = {k: one(v) for k, v in build().items()}
    print(json.dumps({"salt": hash("x"), "cold": cold, "warm": warm}))


if __name__ == "__main__":
    main()
