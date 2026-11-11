"""The plots in ``landau.plot`` draw each kind of element alike (``landau.plot.style``).

Each test draws the same element in every plot that has one and asserts the
artists agree, and that sizes come from the theme or rcParams rather than
per-plot constants.
"""
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from landau.calculate import calc_phase_diagram
from landau.plot import (
    plot_1d_mu_phase_diagram,
    plot_excess_free_energy,
    plot_mu_phase_diagram,
    plot_phase_diagram,
)
from landau.plot.style import (
    _SHEET,
    CURVE_LABEL,
    METASTABLE,
    REGION_LABEL,
    STYLE,
    TIE_LINE,
    TRANSITION_GUIDE,
    styled,
    theme,
)

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
    """Draw every plot kind once under the current rcParams; close them afterwards."""
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


def _is_black(color):
    return to_rgba(color) == BLACK


def _phase_curves(ax):
    """Coloured data lines: the phase curves (not the black tie lines, guides or axhline)."""
    return [
        line for line in ax.lines
        if not _is_black(line.get_color()) and len(line.get_xdata()) > 2
    ]


def _dash(line):
    # Line2D exposes no public getter for the scaled on/off sequence it renders with.
    offset, seq = line._dash_pattern
    return offset, None if seq is None else tuple(np.round(seq, 9))


def test_style_sheet_loads_by_package_name_and_matches_the_theme(caplog):
    # matplotlib logs (does not raise or warn) on a malformed sheet line and skips it.
    sheet = files("landau.plot") / "landau.mplstyle"
    with caplog.at_level("WARNING", logger="matplotlib"):
        matplotlib.rc_params_from_file(str(sheet), use_default_template=False)
    assert caplog.records == []
    with plt.style.context(STYLE):
        assert {k: plt.rcParams[k] for k in theme} == dict(theme)
    assert dict(theme) == _SHEET and theme["lines.linewidth"] == 2.0


def test_theme_keys_differ_from_matplotlib_defaults():
    """A theme key equal to matplotlib's default would only override the caller's setting."""
    assert all(theme[k] != matplotlib.rcParamsDefault[k] for k in theme)


def test_theme_validates_like_rcparams():
    with pytest.raises(KeyError):
        theme.update({"lines.linewdith": 3.0})
    with pytest.raises(ValueError):
        theme["lines.linewidth"] = "fat"
    theme["lines.linewidth"] = "3"
    assert theme["lines.linewidth"] == 3.0


def test_metastable_branches_dashed_alike_in_1d_and_excess(figures):
    axes = figures()
    _, ref_ax = plt.subplots()
    reference, = styled(ref_ax.plot)([0, 1], [0, 1], **METASTABLE)
    patterns = {}
    for kind in ("1d", "excess"):
        dashed = [_dash(line) for line in _phase_curves(axes[kind]) if line.get_linestyle() == "--"]
        assert dashed, f"no metastable branch drawn in the {kind} plot"
        patterns[kind] = set(dashed)
    assert patterns["1d"] == patterns["excess"] == {_dash(reference)}


def test_tie_lines_drawn_alike_in_cT_and_excess(figures):
    axes = figures()
    isotherms = [c for c in axes["cT"].collections if type(c).__name__ == "LineCollection"]
    tangents = [line for line in axes["excess"].lines if _is_black(line.get_color()) and line.get_alpha() == 0.5]
    assert len(isotherms) == 1 and tangents
    iso = isotherms[0]
    assert all(to_rgba(c)[:3] == BLACK[:3] for c in iso.get_colors())
    assert iso.get_alpha() == TIE_LINE["alpha"]
    assert iso.get_linestyle() == [(0, None)]
    assert {line.get_linestyle() for line in tangents} == {"-"}
    widths = {line.get_linewidth() for line in tangents} | set(iso.get_linewidths())
    assert widths == {theme["lines.linewidth"]}


def test_transition_guides_dotted_and_faded(figures):
    axes = figures()
    guides = [line for line in axes["1d"].lines if _is_black(line.get_color()) and line.get_linestyle() == ":"]
    assert guides
    assert {line.get_alpha() for line in guides} == {TRANSITION_GUIDE["alpha"]}
    # A guide is not a tie line: the two must stay distinguishable.
    assert TRANSITION_GUIDE["linestyle"] != TIE_LINE["linestyle"]


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


def test_phase_curves_follow_the_theme_over_rcparams(figures):
    theme["lines.linewidth"] = 3.3
    with plt.rc_context({"lines.linewidth": 0.9}):
        axes = figures()
    for kind in ("1d", "excess"):
        curves = _phase_curves(axes[kind])
        assert curves
        assert {line.get_linewidth() for line in curves} == {3.3}, kind


def _phase_labels(ax, names):
    return [t for t in ax.texts if t.get_text() in names]


def _points(size):
    return matplotlib.font_manager.FontProperties(size=size).get_size_in_points()


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
        role = REGION_LABEL if is_region else CURVE_LABEL
        assert t.get_fontweight() == role.get("fontweight", "normal"), (kind, t.get_text())
        assert t.get_fontsize() == pytest.approx(_points(role["fontsize"]))
        assert t.get_path_effects(), (kind, t.get_text())
    if region is None:
        weights = {t.get_fontweight() for t in labels}
        assert weights == {"bold", "normal"}


def test_excess_facets_keep_the_full_frame(eutectic_phases):
    import pandas as pd

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
    import pandas as pd

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


@pytest.fixture
def matplotlib_defaults():
    """matplotlib's built-in rcParams, as for a caller who set nothing."""
    with plt.rc_context():
        matplotlib.rcdefaults()
        yield
    plt.close("all")


@pytest.mark.parametrize("caller_rc", [{}, {"lines.linewidth": 0.9, "figure.figsize": [3.0, 2.0]}])
def test_theme_applies_whatever_rcparams_hold(cut, matplotlib_defaults, caller_rc):
    """Themed keys come from the theme inside the call, set by the caller or not."""
    plt.rcParams.update(caller_rc)
    ax1d = plot_1d_mu_phase_diagram(cut)
    axex = plot_excess_free_energy(cut)
    for ax in (ax1d, axex):
        assert {line.get_linewidth() for line in _phase_curves(ax)} == {theme["lines.linewidth"]}
    assert list(ax1d.figure.get_size_inches()) == theme["figure.figsize"]


def test_deleted_theme_key_hands_control_back_to_rcparams(cut, matplotlib_defaults):
    plt.rcParams["lines.linewidth"] = 0.9
    del theme["lines.linewidth"]
    ax = plot_1d_mu_phase_diagram(cut)
    assert {line.get_linewidth() for line in _phase_curves(ax)} == {0.9}
    assert list(ax.figure.get_size_inches()) == theme["figure.figsize"]


def test_cleared_theme_leaves_the_plot_to_rcparams(cut, matplotlib_defaults):
    plt.rcParams["lines.linewidth"] = 0.9
    theme.clear()
    ax = plot_1d_mu_phase_diagram(cut)
    assert {line.get_linewidth() for line in _phase_curves(ax)} == {0.9}
    assert list(ax.figure.get_size_inches()) == matplotlib.rcParamsDefault["figure.figsize"]
    theme.reset()
    assert dict(theme) == _SHEET


def test_plot_calls_leave_rcparams_untouched(cut, eutectic_diagram, matplotlib_defaults):
    before = dict(plt.rcParams)
    plot_1d_mu_phase_diagram(cut)
    plot_excess_free_energy(cut)
    plot_phase_diagram(eutectic_diagram, poly_method="concave")
    assert dict(plt.rcParams) == before
