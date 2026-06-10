"""Border-coverage integration check for polygon construction.

Every refined border point of a phase must lie within ``EPS`` (as a fraction
of the axis data ranges) of that phase's polygon outline.  If polygon
construction goes wrong — e.g. a self-intersecting TSP tour whose repair cuts
off part of the phase region, or a dropped phase unit — the border points of
the missing part are far from the outline and flag immediately.
"""

import numpy as np
import pytest
import shapely

import landau.calculate as ldc
import landau.phases as ldp
from landau.plot import get_polygons

try:
    import python_tsp  # noqa: F401

    HAS_PYTHON_TSP = True
except ImportError:
    HAS_PYTHON_TSP = False

try:
    import fast_tsp  # noqa: F401

    HAS_FAST_TSP = True
except ImportError:
    HAS_FAST_TSP = False

# Only methods whose polygons trace the border points exactly; the concave
# hull may legitimately cut corners between them.
METHODS = [
    "segments",
    pytest.param("tsp", marks=pytest.mark.skipif(not HAS_PYTHON_TSP, reason="python-tsp not installed")),
    pytest.param("segment-tsp", marks=pytest.mark.skipif(not HAS_PYTHON_TSP, reason="python-tsp not installed")),
    pytest.param("fasttsp", marks=pytest.mark.skipif(not HAS_FAST_TSP, reason="fast-tsp not installed")),
    pytest.param("segment-fasttsp", marks=pytest.mark.skipif(not HAS_FAST_TSP, reason="fast-tsp not installed")),
]

# The outline runs through every border point, but is inflated by
# min_c_width/2 and trimmed back by up to the same amount where phases touch,
# so allow a small fraction of the axis range.
EPS = 0.02


@pytest.fixture(scope="module")
def hcp_fcc_liquid_df():
    """hcp/fcc/liquid ideal solutions on a T-mu grid (the Basics.ipynb system)."""
    fcca = ldp.LinePhase("fccA", fixed_concentration=0, line_energy=-3.00, line_entropy=1.0 * ldp.kB)
    fccb = ldp.LinePhase("fccB", fixed_concentration=1, line_energy=-2.00, line_entropy=1.1 * ldp.kB)
    hcpa = ldp.LinePhase("hcpA", fixed_concentration=0, line_energy=-2.975, line_entropy=1.8 * ldp.kB)
    hcpb = ldp.LinePhase("hcpB", fixed_concentration=1, line_energy=-1.95, line_entropy=1.1 * ldp.kB)
    lqda = ldp.LinePhase("liquidA", fixed_concentration=0, line_energy=-2.75, line_entropy=5.0 * ldp.kB)
    lqdb = ldp.LinePhase("liquidB", fixed_concentration=1, line_energy=-1.75, line_entropy=4.4 * ldp.kB)
    fcc = ldp.IdealSolution("fcc", fcca, fccb)
    hcp = ldp.IdealSolution("hcp", hcpa, hcpb)
    lqd = ldp.IdealSolution("liquid", lqda, lqdb)

    Ts = np.linspace(200, 1000, 50)
    mus = np.linspace(0.5, 1.5, 50)
    return ldc.calc_phase_diagram([hcp, fcc, lqd], Ts, mu=mus)


def max_border_distance(df, polys, variables):
    """Largest range-normalized distance from any stable border point to its
    phase's polygon outline.  Non-finite points (line-phase endpoints diverge
    in mu) are skipped, as polygon construction does."""
    coords = df[variables].to_numpy()
    finite = np.isfinite(coords).all(axis=-1)
    norm = np.ptp(coords[finite], axis=0)
    outlines = {}
    for (phase, _unit), poly in polys.items():
        outlines.setdefault(phase, []).append(shapely.LinearRing(poly.get_xy() / norm))
    worst = 0.0
    for phase, dd in df[finite].query("stable and border").groupby("phase"):
        assert phase in outlines, f"no polygon for phase {phase!r}"
        pts = shapely.points(dd[variables].to_numpy() / norm)
        dist = np.min([shapely.distance(pts, ring) for ring in outlines[phase]], axis=0)
        worst = max(worst, dist.max())
    return worst


@pytest.mark.parametrize("variables", [["c", "T"], ["mu", "T"]], ids=["c-T", "mu-T"])
@pytest.mark.parametrize("poly_method", METHODS)
def test_border_points_on_outline(hcp_fcc_liquid_df, poly_method, variables):
    if poly_method == "segments" and variables[0] == "mu":
        pytest.xfail(
            "the greedy stitcher's min-x head heuristic fails for phases stable "
            "at the diagram edge (documented in Segments._sort_segments)"
        )
    polys = get_polygons(hcp_fcc_liquid_df, poly_method=poly_method, variables=variables)
    assert max_border_distance(hcp_fcc_liquid_df, polys, variables) < EPS
