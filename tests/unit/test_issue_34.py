
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from warnings import warn
from landau.phases import LinePhase, IdealSolution
from landau.calculate import calc_phase_diagram, get_transitions, cluster
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

# Original code before refactor
def original_cluster_phase(df):
    df["phase_unit"] = df.groupby("phase", group_keys=False).apply(
            cluster, use_mu=False,
            include_groups=False
    )
    df["phase_id"] = df[["phase", "phase_unit"]].apply(
        lambda r: "_".join(map(str, r.tolist())), axis="columns"
    )
    return df

def original_plot_phase_diagram(
    df, alpha=0.1, element=None, min_c_width=1e-2, color_override: dict[str, str] = {}, tielines=False,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp"] | poly.AbstractPolyMethod | None = None
):
    df = df.query("stable").copy()

    # the default map
    color_map = dict(zip(df.phase.unique(), sns.palettes.SEABORN_PALETTES["pastel"]))
    # disregard overriden phases that are not present
    color_override = {p: c for p, c in color_override.items() if p in color_map}
    # if the override uses the same colors as the default map, multiple phases
    # would be mapped to the same color; so instead let's update the color map of phases that would
    # use the same color as a phase in the override to use the default colors of the overriden phases
    # instead
    duplicates_map = {c: color_map[o] for o, c in color_override.items()}
    diff = {k: duplicates_map[c] for k, c in color_map.items() if c in duplicates_map}
    color_map.update(diff | color_override)

    df = original_cluster_phase(df)
    if (df.phase_unit==-1).any():
        warn("Clustering of phase points failed for some points, dropping them.")
        df = df.query('phase_unit>=0')
    poly_method = poly.handle_poly_method(poly_method, min_c_width=min_c_width, alpha=alpha)
    polys = poly_method.apply(df, variables=["c", "T"])

    ax = plt.gca()
    for i, (phase, p) in enumerate(polys.items()):
        p.zorder = 1/p.get_extents().size.prod()
        if isinstance(phase, tuple):
            phase, rep = phase
        else:
            rep = 0
        p.set_color(color_map[phase])
        p.set_edgecolor("k")
        p.set_label(phase + '\'' * rep)
        ax.add_patch(p)

def original_plot_mu_phase_diagram(
    df, alpha=0.1, element=None, color_override: dict[str, str] = {},
    poly_method: Literal["concave", "segments", "fasttsp", "tsp"] | poly.AbstractPolyMethod | None = None,
):
    df = df.query("stable").copy()

    color_map = get_phase_colors(df.phase.unique(), color_override)

    df = original_cluster_phase(df)
    if (df.phase_unit==-1).any():
        warn("Clustering of phase points failed for some points, dropping them.")
        df = df.query('phase_unit>=0')
    poly_method = poly.handle_poly_method(poly_method, alpha=alpha)
    polys = poly_method.apply(df, variables=["mu", "T"])

    ax = plt.gca()
    for i, (phase, p) in enumerate(polys.items()):
        p.zorder = 1/p.get_extents().size.prod()
        if isinstance(phase, tuple):
            phase, rep = phase
        else:
            rep = 0
        p.set_color(color_map[phase])
        p.set_edgecolor("k")
        p.set_label(phase + '\'' * rep)
        ax.add_patch(p)

@pytest.mark.parametrize("poly_method", get_available_poly_methods())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=5)
@given(phases=eutectic_system())
def test_plot_phase_diagram_consistency(phases, poly_method):
    Ts = np.linspace(300, 1500, 10)
    df = calc_phase_diagram(phases, Ts, mu=50, refine=True)

    # Get polygons from original code
    plt.figure()
    original_plot_phase_diagram(df, poly_method=poly_method)
    orig_polys = get_polygons_from_ax(plt.gca())
    plt.close()

    # Get polygons from current code
    plt.figure()
    plot_phase_diagram(df, poly_method=poly_method)
    curr_polys = get_polygons_from_ax(plt.gca())
    plt.close()

    assert len(curr_polys) == len(orig_polys)
    # Both use polys.items() so order should be identical
    for p, e in zip(curr_polys, orig_polys):
        np.testing.assert_allclose(p, e)

@pytest.mark.parametrize("poly_method", get_available_poly_methods())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=5)
@given(phases=eutectic_system())
def test_plot_mu_phase_diagram_consistency(phases, poly_method):
    Ts = np.linspace(300, 1500, 10)
    df = calc_phase_diagram(phases, Ts, mu=50, refine=True)

    # Get polygons from original code
    plt.figure()
    original_plot_mu_phase_diagram(df, poly_method=poly_method)
    orig_polys = get_polygons_from_ax(plt.gca())
    plt.close()

    # Get polygons from current code
    plt.figure()
    plot_mu_phase_diagram(df, poly_method=poly_method)
    curr_polys = get_polygons_from_ax(plt.gca())
    plt.close()

    assert len(curr_polys) == len(orig_polys)
    for p, e in zip(curr_polys, orig_polys):
        np.testing.assert_allclose(p, e)

@pytest.mark.parametrize("poly_method", get_available_poly_methods())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow], max_examples=5)
@given(phases=eutectic_system())
def test_direct_get_polygons_consistency(phases, poly_method):
    Ts = np.linspace(300, 1500, 10)
    df = calc_phase_diagram(phases, Ts, mu=50, refine=True)

    # Use original_plot_phase_diagram to capture expected polygons
    plt.figure()
    original_plot_phase_diagram(df, poly_method=poly_method)
    orig_polys = get_polygons_from_ax(plt.gca())
    plt.close()

    # Use get_polygons + plot_polygons directly
    polys_direct = get_polygons(df, poly_method=poly_method, variables=["c", "T"])
    color_map = get_phase_colors(df.query("stable").phase.unique())

    plt.figure()
    plot_polygons(polys_direct, color_map)
    curr_polys = get_polygons_from_ax(plt.gca())
    plt.close()

    assert len(curr_polys) == len(orig_polys)
    for p, e in zip(curr_polys, orig_polys):
        np.testing.assert_allclose(p, e)
