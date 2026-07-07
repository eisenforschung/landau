"""Cost of ``_delaunay_simplices`` / refiner ``propose``.

Every Delaunay refiner (`DelaunayLineRefiner`, `DelaunayTripleRefiner`,
`ClausiusClapeyronRefiner`, `MiscibilityGapRefiner`) walks the (mu, T)
tessellation via ``_delaunay_simplices``. That helper used to materialize one
``df.iloc[simplex]`` DataFrame per simplex (~960 three-row frames on a 25x21
grid), which dominated ``propose`` and hence ``run``. Yielding a numpy-backed
``_Simplex`` view of the four columns the refiners read (T, mu, phase, c)
instead removes that cost with no change to the located transitions.

Run ``python benchmarks/bench_delaunay_propose.py`` from the repo root.

Reference numbers on a 25x21 grid (960 simplices), 5-run best:

    measurement                         before      after   speedup
    _delaunay_simplices (materialize)   157 ms     9.7 ms      16x
    ClausiusClapeyronRefiner.propose    191 ms    12.5 ms      15x
    ClausiusClapeyronRefiner.run        258 ms      81 ms     3.2x
    default_refiners full refine        969 ms     120 ms       8x
"""
from __future__ import annotations

import time

import numpy as np
from scipy.constants import Boltzmann, eV

from landau.phases import LinePhase, IdealSolution
from landau.calculate import calc_phase_diagram, refine_phase_diagram
from landau.refine import ClausiusClapeyronRefiner, _delaunay_simplices

kB = Boltzmann / eV


def _best(fn, repeats=5):
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def main() -> None:
    solid = IdealSolution(
        "solid", LinePhase("A", 0, -2.0, 1.0 * kB), LinePhase("B", 1, -3.0, 1.5 * kB))
    liquid = IdealSolution(
        "liquid", LinePhase("A(l)", 0, -1.9, 2.5 * kB), LinePhase("B(l)", 1, -2.9, 2.2 * kB))
    mapping = {"solid": solid, "liquid": liquid}
    Ts = np.linspace(200, 1800, 25)
    mus = np.linspace(-0.3, 0.3, 21)
    df = calc_phase_diagram([solid, liquid], Ts=Ts, mu=mus, refine=False, keep_unstable=False)
    n_simplices = len(list(_delaunay_simplices(df)))
    r = ClausiusClapeyronRefiner()

    print(f"grid={len(df)} points, {n_simplices} simplices")
    print(f"  _delaunay_simplices (materialize): {_best(lambda: list(_delaunay_simplices(df))):7.1f} ms")
    print(f"  CC propose (list):                 {_best(lambda: list(r.propose(df))):7.1f} ms")
    print(f"  CC full run():                     {_best(lambda: r.run(df, mapping)):7.1f} ms")
    print(f"  default_refiners full refine:      {_best(lambda: refine_phase_diagram(df, mapping)):7.1f} ms")


if __name__ == "__main__":
    main()
