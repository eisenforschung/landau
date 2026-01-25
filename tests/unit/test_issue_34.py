
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from landau.phases import LinePhase, IdealSolution
from landau.calculate import calc_phase_diagram
from landau.plot import plot_phase_diagram, plot_mu_phase_diagram
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

@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(phases=eutectic_system())
def test_consistent_polygons(phases):
    from landau.plot import get_polygons, plot_polygons, get_phase_colors
    Ts = np.linspace(300, 1500, 10)
    df = calc_phase_diagram(phases, Ts, mu=50, refine=True)

    # Test plot_phase_diagram
    plt.figure()
    plot_phase_diagram(df)
    ax_c = plt.gca()
    polys_c = get_polygons_from_ax(ax_c)
    plt.close()

    assert len(polys_c) > 0

    # Test plot_mu_phase_diagram
    plt.figure()
    plot_mu_phase_diagram(df)
    ax_mu = plt.gca()
    polys_mu = get_polygons_from_ax(ax_mu)
    plt.close()

    assert len(polys_mu) > 0

    # Test get_polygons and plot_polygons directly
    polys_direct = get_polygons(df, variables=["c", "T"])
    assert len(polys_direct) > 0

    color_map = get_phase_colors(df.query("stable").phase.unique())
    plt.figure()
    plot_polygons(polys_direct, color_map)
    polys_plotted = get_polygons_from_ax(plt.gca())
    plt.close()

    assert len(polys_plotted) == len(polys_direct)
