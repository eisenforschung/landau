"""The plots in ``landau.plot`` draw each kind of element alike, as ``landau.plot.style.theme`` says.

Each test draws the same element in every plot that has one and asserts the
artists agree, that changing the theme changes all of them, and that sizes come
from the theme or rcParams rather than per-plot constants.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from landau import LinePhase
from landau.calculate import calc_phase_diagram
from landau.phases import kB
from landau.plot import (
    plot_1d_mu_phase_diagram,
    plot_excess_free_energy,
    plot_mu_phase_diagram,
    plot_phase_diagram,
)
from landau.plot.style import Theme, styled, theme

BLACK = to_rgba("k")


@pytest.fixture(autouse=True)
def _restore_theme():
    yield
    theme.reset()


@pytest.fixture(scope="module")
def cut(eutectic_phases):
    """Isothermal cut through the hcp-fcc two-phase field of the eutectic system."""
    return calc_phase_diagram(eutectic_phases, Ts=700.0, mu=50, keep_unstable=True)


@pytest.fixture
def figures(cut, eutectic_diagram):
    """Draw every plot kind once under the current theme and rcParams; close them afterwards."""
    def draw():
        _, ax1d = plt.subplots()
        plot_1d_mu_phase_diagram(cut, ax=ax1d)
        plt.figure()
        axex = plot_excess_free_energy(cut)
        _, axct = plt.subplots()
        plot_phase_diagram(eutectic_diagram, ax=axct, triplepoints=True, poly_method="concave")
        _, axmt = plt.subplots()
        plot_mu_phase_diagram(eutectic_diagram, ax=axmt, triplepoints=True, poly_method="concave")
        return {"1d": ax1d, "excess": axex, "cT": axct, "muT": axmt}
    yield draw
    plt.close("all")


@pytest.fixture
def matplotlib_defaults():
    """matplotlib's built-in rcParams, as for a caller who set nothing."""
    with plt.rc_context():
        matplotlib.rcdefaults()
        yield
    plt.close("all")


def _is_black(color):
    return to_rgba(color) == BLACK


def _phase_curves(ax):
    """Coloured data lines: the phase curves (not the black tie lines, guides or zero line)."""
    return [
        line for line in ax.lines
        if not _is_black(line.get_color()) and len(line.get_xdata()) > 2
    ]


def _dash(line):
    # Line2D exposes no public getter for the scaled on/off sequence it renders with.
    offset, seq = line._dash_pattern
    return offset, None if seq is None else tuple(np.round(seq, 9))


def _isotherm(ax):
    isotherms = [c for c in ax.collections if type(c).__name__ == "LineCollection"]
    assert len(isotherms) == 1
    return isotherms[0]


def _tangents(ax):
    """Common tangents: two-point lines in the tie-line alpha (the zero line is fully opaque)."""
    return [line for line in ax.lines if line.get_alpha() == theme.tie_line["alpha"] and len(line.get_xdata()) == 2]


def _phase_labels(ax, names):
    return [t for t in ax.texts if t.get_text() in names]


def _points(size):
    return matplotlib.font_manager.FontProperties(size=size).get_size_in_points()


# --- the theme object -------------------------------------------------------


def test_theme_rc_keys_differ_from_matplotlib_defaults():
    """A themed key equal to matplotlib's default would only override the caller's setting."""
    assert all(theme.rc[k] != matplotlib.rcParamsDefault[k] for k in theme.rc)


def test_theme_rc_validates_like_rcparams():
    with pytest.raises(KeyError):
        theme.rc.update({"lines.linewdith": 3.0})
    with pytest.raises(ValueError):
        theme.rc["lines.linewidth"] = "fat"
    theme.rc["lines.linewidth"] = "3"
    assert theme.rc["lines.linewidth"] == 3.0


def test_reset_restores_every_field():
    default = Theme()
    theme.rc["lines.linewidth"] = 0.5
    del theme.rc["figure.figsize"]
    theme.tie_line["color"] = "r"
    theme.line_phase_marker_scale = 3.0
    theme.reset()
    assert theme == default
    assert dict(theme.rc) == dict(default.rc)


# --- styled -------------------------------------------------------------------


def test_styled_context_manager_applies_theme_to_own_plots(matplotlib_defaults):
    with styled():
        line, = plt.plot([0, 1], [0, 1])
        assert plt.rcParams["savefig.dpi"] == theme.rc["savefig.dpi"]
    assert line.get_linewidth() == theme.rc["lines.linewidth"]
    assert plt.rcParams["lines.linewidth"] == matplotlib.rcParamsDefault["lines.linewidth"]


def test_styled_decorator_reads_theme_at_each_call(matplotlib_defaults):
    @styled()
    def width():
        return plt.rcParams["lines.linewidth"]

    assert width() == theme.rc["lines.linewidth"]
    theme.rc["lines.linewidth"] = 4.0
    assert width() == 4.0


def test_plot_calls_leave_rcparams_untouched(cut, eutectic_diagram, matplotlib_defaults):
    before = dict(plt.rcParams)
    plot_1d_mu_phase_diagram(cut)
    plot_excess_free_energy(cut)
    plot_phase_diagram(eutectic_diagram, poly_method="concave")
    assert dict(plt.rcParams) == before


@pytest.mark.parametrize("caller_rc", [{}, {"lines.linewidth": 0.9, "figure.figsize": [3.0, 2.0]}])
def test_theme_rc_applies_whatever_rcparams_hold(cut, matplotlib_defaults, caller_rc):
    plt.rcParams.update(caller_rc)
    ax1d = plot_1d_mu_phase_diagram(cut)
    axex = plot_excess_free_energy(cut)
    for ax in (ax1d, axex):
        assert {line.get_linewidth() for line in _phase_curves(ax)} == {theme.rc["lines.linewidth"]}
    assert list(ax1d.figure.get_size_inches()) == theme.rc["figure.figsize"]


def test_deleted_theme_key_hands_control_back_to_rcparams(cut, matplotlib_defaults):
    plt.rcParams["lines.linewidth"] = 0.9
    del theme.rc["lines.linewidth"]
    ax = plot_1d_mu_phase_diagram(cut)
    assert {line.get_linewidth() for line in _phase_curves(ax)} == {0.9}
    assert list(ax.figure.get_size_inches()) == theme.rc["figure.figsize"]


def test_cleared_theme_rc_leaves_the_plot_to_rcparams(cut, matplotlib_defaults):
    plt.rcParams["lines.linewidth"] = 0.9
    theme.rc.clear()
    ax = plot_1d_mu_phase_diagram(cut)
    assert {line.get_linewidth() for line in _phase_curves(ax)} == {0.9}
    assert list(ax.figure.get_size_inches()) == matplotlib.rcParamsDefault["figure.figsize"]


# --- one element, one look ----------------------------------------------------


@pytest.mark.parametrize("linestyle", ["--", ":"])
def test_metastable_branches_follow_theme_in_1d_and_excess(figures, linestyle):
    theme.metastable["linestyle"] = linestyle
    axes = figures()
    with styled():
        _, ref_ax = plt.subplots()
        reference, = ref_ax.plot([0, 1], [0, 1], **theme.metastable)
    patterns = {}
    for kind in ("1d", "excess"):
        dashed = [_dash(line) for line in _phase_curves(axes[kind]) if line.get_linestyle() != "-"]
        assert dashed, f"no metastable branch drawn in the {kind} plot"
        patterns[kind] = set(dashed)
    assert patterns["1d"] == patterns["excess"] == {_dash(reference)}


@pytest.mark.parametrize("color", ["k", "tab:red"])
def test_tie_lines_follow_theme_in_cT_and_excess(figures, color):
    theme.tie_line["color"] = color
    axes = figures()
    iso, tangents = _isotherm(axes["cT"]), _tangents(axes["excess"])
    assert tangents
    assert all(to_rgba(c)[:3] == to_rgba(color)[:3] for c in iso.get_colors())
    assert {to_rgba(line.get_color())[:3] for line in tangents} == {to_rgba(color)[:3]}
    assert iso.get_alpha() == theme.tie_line["alpha"]
    assert iso.get_linestyle() == [(0, None)]
    assert {line.get_linestyle() for line in tangents} == {"-"}
    widths = {line.get_linewidth() for line in tangents} | set(iso.get_linewidths())
    assert widths == {theme.rc["lines.linewidth"]}


def test_transition_guides_dotted_and_faded(figures):
    axes = figures()
    guides = [line for line in axes["1d"].lines if _is_black(line.get_color()) and line.get_linestyle() == ":"]
    assert guides
    assert {line.get_alpha() for line in guides} == {theme.transition_guide["alpha"]}
    # A guide is not a tie line: the two must stay distinguishable.
    assert theme.transition_guide["linestyle"] != theme.tie_line["linestyle"]


def test_invariant_markers_one_size_from_rc(figures):
    """lines.markersize is not themed, so the caller's value sizes every invariant marker."""
    with plt.rc_context({"lines.markersize": 4.0}):
        axes = figures()
        sizes = []
        for kind in ("1d", "excess"):
            dots = [c for c in axes[kind].collections if np.allclose(c.get_facecolors(), BLACK)]
            assert dots, f"no invariant marker in the {kind} plot"
            sizes.extend(np.unique(np.concatenate([c.get_sizes() for c in dots])))
        triple, = [line for line in axes["muT"].lines if line.get_marker() == "o"]
    assert sizes == [16.0, 16.0]
    assert triple.get_markersize() == 4.0


def test_line_phase_dot_scales_with_theme():
    phases = [
        LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB),
        LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB),
        LinePhase("AB", fixed_concentration=0.5, line_energy=-2.8, line_entropy=1.3 * kB),
    ]
    df = calc_phase_diagram(phases, Ts=1000.0, mu=50, keep_unstable=True)
    theme.line_phase_marker_scale = 2.0
    plt.figure()
    try:
        ax = plot_excess_free_energy(df, convex_hull=False)
        sizes = np.concatenate([c.get_sizes() for c in ax.collections])
        assert set(sizes) == {(2.0 * plt.rcParams["lines.markersize"]) ** 2}
    finally:
        plt.close("all")


def test_phase_curves_follow_the_theme_over_rcparams(figures):
    theme.rc["lines.linewidth"] = 3.3
    with plt.rc_context({"lines.linewidth": 0.9}):
        axes = figures()
    for kind in ("1d", "excess"):
        curves = _phase_curves(axes[kind])
        assert curves
        assert {line.get_linewidth() for line in curves} == {3.3}, kind


@pytest.mark.parametrize("kind,region", [
    ("cT", True), ("muT", True),     # field labels
    ("1d", None),                    # top labels are regions, side labels curves
    ("excess", False),               # curve labels
])
def test_region_labels_bold_curve_labels_regular(figures, eutectic_phases, kind, region):
    axes = figures()
    ax = axes[kind]
    labels = _phase_labels(ax, {p.name for p in eutectic_phases})
    assert labels
    for t in labels:
        # In the 1d cut the top labels sit in axes-fraction y, the side labels in data y.
        is_region = region if region is not None else t.get_transform() is not ax.transData
        style = theme.region_label if is_region else theme.curve_label
        assert t.get_fontweight() == style.get("fontweight", "normal"), (kind, t.get_text())
        assert t.get_fontsize() == pytest.approx(_points(style["fontsize"]))
        assert t.get_path_effects(), (kind, t.get_text())
    if region is None:
        assert {t.get_fontweight() for t in labels} == {"bold", "normal"}


def test_theme_colour_overrides_the_phase_colour_of_labels(figures, eutectic_phases):
    """A colour in a label style wins over the phase colour instead of colliding with it."""
    theme.curve_label["color"] = "tab:red"
    theme.region_label["color"] = "tab:blue"
    axes = figures()
    names = {p.name for p in eutectic_phases}
    curve, region = _phase_labels(axes["excess"], names), _phase_labels(axes["cT"], names)
    assert curve and region
    assert {to_rgba(t.get_color()) for t in curve} == {to_rgba("tab:red")}
    assert {to_rgba(t.get_color()) for t in region} == {to_rgba("tab:blue")}


def test_excess_facets_keep_the_full_frame(eutectic_phases):
    df = pd.concat(
        [calc_phase_diagram(eutectic_phases, Ts=T, mu=30, keep_unstable=True) for T in (600.0, 900.0)],
        ignore_index=True,
    )
    g = plot_excess_free_energy(df)
    try:
        for ax in g.axes.flat:
            assert all(ax.spines[side].get_visible() for side in ("top", "right", "bottom", "left"))
    finally:
        plt.close(g.figure)


@pytest.mark.parametrize("temperatures", [(600.0,), (600.0, 900.0)])
def test_excess_zero_line_reads_as_axis_not_curve(eutectic_phases, temperatures):
    """The zero line is solid at frame weight, so it can't pass for a dashed metastable branch."""
    df = pd.concat(
        [calc_phase_diagram(eutectic_phases, Ts=T, mu=30, keep_unstable=True) for T in temperatures],
        ignore_index=True,
    )
    with plt.rc_context({"axes.linewidth": 0.7}):
        plt.figure()
        out = plot_excess_free_energy(df)
    try:
        axes = [out] if isinstance(out, plt.Axes) else list(out.axes.flat)
        for ax in axes:
            zero, = [line for line in ax.lines if list(line.get_ydata()) == [0, 0]]
            assert zero.get_linestyle() == "-"
            assert zero.get_linewidth() == 0.7
    finally:
        plt.close("all")
