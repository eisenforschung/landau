"""The two-liquid dome above a syntectic stays open in every segment polygon method.

The syntectic isotherm carries the liquid twice (its two branch compositions)
plus the compound; before ``_absorb_repeated_invariant_rows`` those two rows
formed a chord segment that the stitch took in place of the dome, filling the
two-phase field between the liquids (reported on the TransitionTemperatures
notebook).
"""
import numpy as np
import pytest
import shapely

from landau.calculate import calc_phase_diagram
from landau.interpolate import PolyFit
from landau.phases import FastInterpolatingPhase, LinePhase, kB
from landau.plot import get_polygons
from landau.poly import __all__ as _poly_methods


def _line(name, c, E, S):
    return LinePhase(name, fixed_concentration=c, line_energy=E, line_entropy=S * kB)


@pytest.fixture(scope="module")
def syntectic_diagram():
    """alpha / beta line phases, a compound at c=0.5 under the gap of a repulsive
    liquid (TransitionTemperatures.ipynb's syntectic system)."""
    L0 = 0.25
    controls = [
        _line(f"l{i}", c, (1 - c) * -2.8 + c * -2.35 + L0 * c * (1 - c), (1 - c) * 3.0 + c * 3.0)
        for i, c in enumerate([0, 0.25, 0.5, 0.75, 1])
    ]
    liquid = FastInterpolatingPhase("liquid", controls, add_entropy=True, interpolator=PolyFit(4))
    phases = [_line("α", 0, -3.0, 1.0), _line("β", 1, -2.5, 1.0), _line("γ", 0.5, -2.78, 1.2), liquid]
    return calc_phase_diagram(phases, np.linspace(300.0, 1600.0, 80), mu=100)


@pytest.mark.parametrize("poly_method", [m for m in ("segments", "segment-fasttsp", "segment-tsp")
                                         if m.replace("-", "").replace("segment", "Segment").replace("fasttsp", "FastTsp").replace("tsp", "PythonTsp") in _poly_methods])
def test_liquid_polygon_leaves_the_dome_open(syntectic_diagram, poly_method):
    polys = get_polygons(syntectic_diagram, poly_method=poly_method)
    liquid = shapely.Polygon(polys.loc["liquid"].iloc[0].get_xy())
    triple = syntectic_diagram[syntectic_diagram["locus"] == "triple"]
    T_syn = triple.loc[triple["phase"] == "γ", "T"].max()
    # the two-liquid field: above the syntectic isotherm, below the dome's top
    assert not liquid.contains(shapely.Point(0.5, T_syn + 60.0))
    # the one-phase liquid: above the critical point
    assert liquid.contains(shapely.Point(0.5, 1550.0))
