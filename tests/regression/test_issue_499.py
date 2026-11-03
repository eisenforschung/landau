"""Regression test for #499: f_excess of refined rows.

The 2-D refiners emit rows at temperatures off the sampling grid.  When the
endpoint references were taken per ``T`` group of the final frame, such a
group held only the phases coexisting at that point, which lie on a common
tangent, so every refined row came out with ``f_excess ~ 0``.  The reference
is now the pure components' free energy at each row's own temperature,
evaluated from the phases.
"""

import numpy as np
import pytest

from landau.calculate import calc_phase_diagram
from landau.features import Locus
from landau.phases import IdealSolution, LinePhase, kB

ATOL = 1e-12


@pytest.fixture(scope="module")
def system():
    fcc = IdealSolution("fcc", LinePhase("fccA", 0, -3.00, 1.0 * kB), LinePhase("fccB", 1, -2.00, 1.1 * kB))
    hcp = IdealSolution("hcp", LinePhase("hcpA", 0, -2.975, 1.8 * kB), LinePhase("hcpB", 1, -1.95, 1.1 * kB))
    liq = IdealSolution("liquid", LinePhase("liquidA", 0, -2.75, 5.0 * kB),
                        LinePhase("liquidB", 1, -1.75, 4.4 * kB))
    phases = [hcp, fcc, liq]
    mus = np.linspace(0.5, 1.5, 60)
    df = calc_phase_diagram(phases, np.linspace(200, 1000, 25), mus, keep_unstable=True)
    return phases, mus, df


def test_refined_rows_match_a_grid_at_their_temperature(system):
    """Every refined row has the f_excess a grid sampled at its own T assigns it."""
    phases, mus, df = system
    refined = df[df.locus != Locus.INTERIOR]
    assert len(refined) > 100
    assert refined.f_excess.abs().max() > 1e-2, "refined rows must carry a real excess"
    for _, row in refined.iterrows():
        grid = calc_phase_diagram(phases, [row["T"]], np.r_[mus, row.mu], refine=False, keep_unstable=True)
        same = grid[(grid.phase == row.phase) & (grid.mu == row.mu)]
        assert len(same) == 1
        np.testing.assert_allclose(row.f_excess, same.f_excess.iloc[0], atol=ATOL)
