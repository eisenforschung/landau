
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from landau.phases import LinePhase, IdealSolution
from landau.calculate import calc_phase_diagram
from landau.plot import plot_phase_diagram, plot_mu_phase_diagram, get_polygons, plot_polygons, get_phase_colors
import landau.poly as poly
from hypothesis import given, strategies as st, settings, HealthCheck
import pytest
import matplotlib
matplotlib.use('Agg')

@st.composite
def eutectic_system(draw):
    # Energy of liquid at c=0 and c=1
    u_l0 = draw(st.floats(min_value=0.5, max_value=1.0))
    u_l1 = draw(st.floats(min_value=0.5, max_value=1.0))

    # Entropy of liquid (actually just making it stable at high T)
    s_l = draw(st.floats(min_value=1e-4, max_value=1e-3))

    p_a = LinePhase("A", 0, 0, 0)
    p_b = LinePhase("B", 1, 0, 0)
    p_l = IdealSolution(name="L", phase1=LinePhase("AL", 0, u_l0, s_l), phase2=LinePhase("BL", 1, u_l1, s_l))

    return [p_a, p_b, p_l]

def get_polygons_from_ax(ax):
    polys = []
    for patch in ax.patches:
        if isinstance(patch, matplotlib.patches.Polygon):
            polys.append(patch.get_xy())
    return polys

def get_available_poly_methods():
    methods = ['concave', 'segments']
    if 'PythonTsp' in poly.__all__:
        methods.append('tsp')
    if 'FastTsp' in poly.__all__:
        methods.append('fasttsp')
    return methods

@pytest.mark.parametrize("poly_method", get_available_poly_methods())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=5)
@given(phases=eutectic_system())
def test_plot_phase_diagram_consistency(phases, poly_method):
    Ts = np.linspace(300, 1500, 10)
    df = calc_phase_diagram(phases, Ts, mu=50, refine=True)

    # Get expected polygons directly
    expected_polys = get_polygons(df, poly_method=poly_method, variables=["c", "T"])

    plt.figure()
    plot_phase_diagram(df, poly_method=poly_method)
    ax = plt.gca()
    plotted_polys = get_polygons_from_ax(ax)
    plt.close()

    # Filter out empty polygons if any (though get_polygons should have dropped them)
    expected_coords = [p.get_xy() for p in expected_polys]

    assert len(plotted_polys) == len(expected_coords)
    # Sort to match order as plot_polygons uses polys.items()
    for p, e in zip(plotted_polys, expected_coords):
        np.testing.assert_allclose(p, e)

@pytest.mark.parametrize("poly_method", get_available_poly_methods())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=5)
@given(phases=eutectic_system())
def test_plot_mu_phase_diagram_consistency(phases, poly_method):
    Ts = np.linspace(300, 1500, 10)
    df = calc_phase_diagram(phases, Ts, mu=50, refine=True)

    # Get expected polygons directly
    expected_polys = get_polygons(df, poly_method=poly_method, variables=["mu", "T"])

    plt.figure()
    plot_mu_phase_diagram(df, poly_method=poly_method)
    ax = plt.gca()
    plotted_polys = get_polygons_from_ax(ax)
    plt.close()

    expected_coords = [p.get_xy() for p in expected_polys]

    assert len(plotted_polys) == len(expected_coords)
    for p, e in zip(plotted_polys, expected_coords):
        np.testing.assert_allclose(p, e)

@pytest.mark.parametrize("poly_method", get_available_poly_methods())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=5)
@given(phases=eutectic_system())
def test_direct_plot_polygons_consistency(phases, poly_method):
    Ts = np.linspace(300, 1500, 10)
    df = calc_phase_diagram(phases, Ts, mu=50, refine=True)

    polys = get_polygons(df, poly_method=poly_method, variables=["c", "T"])
    color_map = get_phase_colors(df.query("stable").phase.unique())

    plt.figure()
    plot_polygons(polys, color_map)
    ax = plt.gca()
    plotted_polys = get_polygons_from_ax(ax)
    plt.close()

    expected_coords = [p.get_xy() for p in polys]

    assert len(plotted_polys) == len(expected_coords)
    for p, e in zip(plotted_polys, expected_coords):
        np.testing.assert_allclose(p, e)
