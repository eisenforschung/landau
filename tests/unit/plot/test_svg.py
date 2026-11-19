"""Tests for the layered SVG export, landau.plot.save_layered_svg.

The file is parsed back and checked for the documented layout: one layer group
per kind, one element per phase field / label / invariant / region / boundary
carrying the phase names in data attributes, ticks and tick labels in separate
sublayers. The two-phase regions are checked against the boundaries in the
phase-diagram frame, and the figure must come out of the save as it went in.
"""
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely
from matplotlib.colors import to_hex

import landau.calculate as ldc
from landau.features import Locus
from landau.phases import AbstractLinePhase, IdealSolution, LinePhase, kB
from landau.plot import (
    get_phase_colors,
    plot_1d_mu_phase_diagram,
    plot_1d_T_phase_diagram,
    plot_mu_phase_diagram,
    plot_phase_diagram,
    save_layered_svg,
)
from landau.plot.style import theme
from landau.plot.svg import _add_coexistence, _name, tag

SVG = "{http://www.w3.org/2000/svg}"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
INKSCAPE = "{http://www.inkscape.org/namespaces/inkscape}"

# Every id the file defines must be usable as a CSS id selector without escaping.
ID_RE = re.compile(r"[^\W\d][\w-]*")
# Field polygons are buffered by min_c_width/2 = 0.005 each side, so a region
# edge sits that far inside the coexisting compositions; allow twice that.
C_TOL = 0.01


def _save(fig, **kwargs):
    buf = io.BytesIO()
    elements = save_layered_svg(fig, buf, **kwargs)
    return ET.fromstring(buf.getvalue()), elements, buf.getvalue()


def _groups(root, cls):
    return [g for g in root.iter(f"{SVG}g") if g.get("class") == cls]


def _by_id(root):
    return {el.get("id"): el for el in root.iter() if el.get("id") is not None}


def _painted(root, skip=()):
    """Every drawing leaf (path, use, image) outside <defs>, in the order it is painted.

    ``skip`` holds group-id prefixes whose content is left out; the second
    value returned is what was left out.
    """
    painted, skipped = [], []

    def walk(el, out):
        tag = el.tag.removeprefix(SVG)
        if tag == "defs":
            return
        if tag == "g" and (el.get("id") or "").startswith(tuple(skip)) and skip:
            out = skipped
        if tag in ("path", "use", "image"):
            out.append((tag, tuple(sorted(el.attrib.items()))))
        for child in el:
            walk(child, out)

    walk(root, painted)
    return painted, skipped


def _assert_painted_as_savefig(fig):
    """The layered file paints what a plain savefig paints, in the same order.

    Ticks and tick labels are regrouped inside each axis, so axis content is
    only checked for being all there.
    """
    with mpl.rc_context({"svg.hashsalt": "landau"}):
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", metadata={"Date": None})
    plain, axes_content = _painted(ET.fromstring(buf.getvalue()), skip=("matplotlib.axis",))
    root, _, _ = _save(fig, coexistence=False, text_as_text=False)
    layered, _ = _painted(root)
    assert len(plain) > 10
    assert Counter(layered) == Counter(plain) + Counter(axes_content)
    axes_content = set(axes_content)
    assert [leaf for leaf in layered if leaf not in axes_content] == plain
    return root


def _render(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.fixture(autouse=True)
def _restore_theme():
    yield
    theme.reset()


@pytest.fixture
def ct_figure(eutectic_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram, ax=ax, transition_temperatures=True)
    yield fig
    plt.close(fig)


@pytest.fixture(scope="module")
def unary_diagram():
    phases = [
        LinePhase("bcc", 0, -2.0, 1.0 * kB),
        LinePhase("fcc", 0, -2.02, 0.5 * kB),
        LinePhase("liquid", 0, -1.8, 3.0 * kB),
    ]
    return ldc.calc_phase_diagram(phases, np.linspace(100, 1500, 80), mu=0.0, keep_unstable=True)


@pytest.fixture
def unary_figure(unary_diagram):
    fig, ax = plt.subplots()
    plot_1d_T_phase_diagram(unary_diagram, ax=ax)
    yield fig
    plt.close(fig)


def _layers(root, scope="ax0"):
    """``data-layer`` of every layer group of one axes, in document order."""
    (axes,) = [g for g in root.iter(f"{SVG}g") if g.get("id") == scope]
    return [g.get("data-layer") for g in axes.iter(f"{SVG}g") if g.get("class") == "landau-layer"]


@pytest.mark.parametrize("plot, kwargs, own", [
    (plot_phase_diagram, {"transition_temperatures": True}, {"labels", "transitions", "transition-labels"}),
    (plot_phase_diagram, {"inline_legend": False}, {"legend"}),
    (plot_mu_phase_diagram, {"transition_temperatures": True}, {"labels", "transitions", "transition-labels"}),
])
def test_2d_diagram_has_one_group_per_layer(eutectic_diagram, plot, kwargs, own):
    fig, ax = plt.subplots()
    plot(eutectic_diagram, ax=ax, **kwargs)
    root = _assert_painted_as_savefig(fig)
    plt.close(fig)
    layers = _layers(root)
    assert len(layers) == len(set(layers))
    assert {"background", "phases", "spines", "xaxis", "yaxis"} | own <= set(layers)


def test_1d_diagram_is_painted_as_savefig(unary_figure):
    root = _assert_painted_as_savefig(unary_figure)
    layers = _layers(root)
    assert len(layers) == len(set(layers))
    assert {"phases", "transitions", "transition-ticks", "transition-points", "labels"} <= set(layers)
    order = [g.get("class") for g in root.iter(f"{SVG}g")
             if g.get("class") in ("landau-curve", "landau-transition-point")]
    # the dots are drawn on top of the curves they mark
    assert order.index("landau-transition-point") > len(order) - 1 - order[::-1].index("landau-curve")


def test_overlays_keep_their_place_in_the_stack(ct_figure):
    """Untagged artists share the layer ``other`` but are painted on either side of the phase fields."""
    ax = ct_figure.axes[0]
    ax.axhline(700, color="0.5", zorder=0)
    ax.plot([0.1, 0.5, 0.9], [300, 600, 400], "ko")
    root = _assert_painted_as_savefig(ct_figure)
    ids = [g.get("id") for g in root.iter(f"{SVG}g")
           if g.get("class") == "landau-layer" and g.get("data-layer") in ("other", "phases")]
    assert ids == ["ax0-layer-other", "ax0-layer-phases", "ax0-layer-other_2"]
    labels = [g.get(f"{INKSCAPE}label") for g in root.iter(f"{SVG}g") if g.get("data-layer") == "other"
              and g.get("class") == "landau-layer"]
    assert labels == ["other", "other (2)"]


@pytest.mark.parametrize("figure", ["ct_figure", "unary_figure"])
def test_returned_elements_are_the_elements_in_the_file(figure, request):
    root, elements, _ = _save(request.getfixturevalue(figure))
    structure = ("landau-layer", "landau-axes", "landau-figure")
    in_file = {g.get("id"): {k: v for k, v in g.attrib.items() if k == "class" or k.startswith("data-")}
               for g in root.iter(f"{SVG}g")
               if (g.get("class") or "").startswith("landau-") and g.get("class") not in structure}
    assert elements == in_file
    assert len(elements) > 30


def test_color_by_fill_is_one_field_element(eutectic_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram, ax=ax, color_by="mu")
    root, elements, raw = _save(fig)
    plt.close(fig)
    (field,) = _groups(root, "landau-field")
    assert field.get("data-column") == "mu"
    assert field.get("data-layer") == "scalar-field"
    assert field.get("pointer-events") == "none"
    assert [child.tag for child in field] == [f"{SVG}image"]
    assert elements[field.get("id")] == {"class": "landau-field", "data-layer": "scalar-field", "data-column": "mu"}
    (layer,) = [g for g in root.iter(f"{SVG}g") if g.get("data-layer") == "scalar-field"
                and g.get("class") == "landau-layer"]
    assert list(layer) == [field]
    phases = _groups(root, "landau-phase")
    assert len(phases) == 3
    assert {g.get("pointer-events") for g in phases} == {"visible"}
    assert b"landau-placeholder" not in raw


def test_image_is_an_element(ct_figure):
    ax = ct_figure.axes[0]
    image = ax.imshow(np.arange(6.0).reshape(2, 3), extent=(0.6, 0.9, 300, 500), aspect="auto")
    tag(image, "micrograph", sample=7)
    root, elements, raw = _save(ct_figure)
    (group,) = _groups(root, "landau-micrograph")
    assert group.get("id") == "ax0-micrograph-0"
    assert group.get("data-sample") == "7"
    assert group.find(f".//{SVG}image") is not None
    assert "ax0-micrograph-0" in elements
    assert b"landau-placeholder" not in raw


def test_hand_set_gid_of_an_image_kept(ct_figure):
    ax = ct_figure.axes[0]
    ax.imshow(np.arange(6.0).reshape(2, 3), extent=(0.6, 0.9, 300, 500), aspect="auto").set_gid("by-hand")
    root, elements, raw = _save(ct_figure)
    assert _by_id(root)["by-hand"].find(f".//{SVG}image") is not None
    assert elements["by-hand"]["data-layer"] == "other"
    assert b"landau-placeholder" not in raw


def test_structural_ids_cannot_be_taken(ct_figure):
    ax = ct_figure.axes[0]
    for gid in ("ax0", "figure", "ax0-layer-phases"):
        ax.plot([0.5, 0.5], [400, 900])[0].set_gid(gid)
    root, elements, _ = _save(ct_figure)
    all_ids = [el.get("id") for el in root.iter() if el.get("id") is not None]
    assert len(all_ids) == len(set(all_ids))
    assert {"ax0_2", "figure_2", "ax0-layer-phases_2"} <= set(elements)
    assert _by_id(root)["ax0"].get("class") == "landau-axes"


def test_layer_groups(ct_figure):
    root, _, _ = _save(ct_figure)
    ids = _by_id(root)
    for layer in ("background", "two-phase", "phases", "boundaries", "transitions", "labels",
                  "transition-labels", "spines", "xaxis", "yaxis",
                  "xaxis-ticks", "xaxis-ticklabels", "xaxis-label"):
        group = ids[f"ax0-layer-{layer}"]
        assert group.get("class") == "landau-layer"
        assert group.get("data-layer") == layer
        assert group.get(f"{INKSCAPE}groupmode") == "layer"
        assert group.get(f"{INKSCAPE}label") == layer
        # every element sits in the layer it names
        for child in group:
            if child.get("class") != "landau-layer":
                assert child.get("data-layer") == layer
    assert ids["ax0"].get(f"{INKSCAPE}groupmode") == "layer"
    assert ids["figure"].get(f"{INKSCAPE}groupmode") == "layer"


def test_ids_unique_and_selectable(ct_figure):
    root, elements, _ = _save(ct_figure)
    all_ids = [el.get("id") for el in root.iter() if el.get("id") is not None]
    assert len(all_ids) == len(set(all_ids))
    assert all(ID_RE.fullmatch(i) for i in elements)
    assert set(elements) <= set(all_ids)


def test_phase_fields_and_labels(ct_figure, eutectic_diagram):
    root, elements, _ = _save(ct_figure)
    phases = set(eutectic_diagram.query("stable").phase.unique())
    fields = _groups(root, "landau-phase")
    assert {g.get("data-phase") for g in fields} == phases
    assert len(fields) == len(phases)
    for g in fields:
        assert g.get("id") == f"ax0-phase-{g.get('data-phase')}-0"
        assert g.get("data-unit") == "0"
        assert g.find(f".//{SVG}path") is not None
    labels = _groups(root, "landau-label")
    assert {g.get("data-phase") for g in labels} == phases
    for g in labels:
        # text_as_text: the label is live text naming its phase, outlined by a stroke
        text = g.find(f".//{SVG}text")
        assert text.text == g.get("data-phase")
        assert "paint-order: stroke" in text.get("style")
        assert elements[g.get("id")]["data-phase"] == g.get("data-phase")


def _style(el):
    """The CSS declarations of an element's style attribute."""
    return dict(d.strip().split(": ", 1) for d in el.get("style", "").split(";") if d.strip())


def _label_styles(root):
    """The CSS declarations of every phase label's <text>, one dict per label."""
    return [_style(g.find(f".//{SVG}text")) for g in _groups(root, "landau-label")]


def test_mathtext_label_is_outlined_by_a_copy_underneath(eutectic_diagram):
    """Stroking the glyph runs of mathtext one by one would paint each run's outline over the run before."""
    names = {"fcc": "Mg$_2$Si", "hcp": r"$\alpha$", "liquid": "liquid"}
    df = eutectic_diagram.copy()
    df["phase"] = df["phase"].map(names)
    fig, ax = plt.subplots()
    plot_phase_diagram(df, ax=ax)
    root, _, _ = _save(fig)
    plt.close(fig)
    texts = {g.get("data-phase"): g.findall(f".//{SVG}text") for g in _groups(root, "landau-label")}
    under, over = texts["Mg$_2$Si"]
    assert (under.get("class"), under.get("aria-hidden")) == ("landau-outline", "true")
    assert {"fill": "none", "stroke": "#ffffff", "stroke-width": "3"}.items() <= _style(under).items()
    assert "stroke" not in _style(over) and over.get("class") is None
    assert "".join(under.itertext()).split() == "".join(over.itertext()).split() == list("MgSi2")
    # one glyph run needs no copy
    for name in (r"$\alpha$", "liquid"):
        (text,) = texts[name]
        assert {"paint-order": "stroke", "stroke": "#ffffff", "stroke-width": "3"}.items() <= _style(text).items()


def test_label_outline_is_the_themes_stroke(eutectic_diagram):
    theme.label_outline = {"linewidth": 5, "foreground": "red", "alpha": 0.25}
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram, ax=ax)
    root, _, _ = _save(fig)
    plt.close(fig)
    styles = _label_styles(root)
    assert len(styles) == 3
    for style in styles:
        assert style["paint-order"] == "stroke"
        assert style["stroke"] == "#ff0000"
        assert float(style["stroke-width"]) == 5
        assert float(style["stroke-opacity"]) == 0.25


@pytest.mark.parametrize("outline", [{}, {"linewidth": 2}, {"foreground": "yellow"},
                                     {"linewidth": 2, "foreground": "k", "capstyle": "round"}])
def test_label_outline_svg_cannot_express_stays_a_path_effect(eutectic_diagram, outline):
    """Such a label keeps matplotlib's stroke: it is drawn as glyph outlines, not as text."""
    theme.label_outline = outline
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram, ax=ax)
    root, _, _ = _save(fig)
    plt.close(fig)
    labels = _groups(root, "landau-label")
    assert len(labels) == 3
    for g in labels:
        assert g.find(f".//{SVG}text") is None
        assert g.find(f".//{SVG}path") is not None


def test_invariants(ct_figure, eutectic_diagram):
    root, _, _ = _save(ct_figure)
    triple = eutectic_diagram[eutectic_diagram["locus"] == Locus.TRIPLE]
    (T,) = triple["T"].unique()
    (line,) = _groups(root, "landau-triple")
    assert json.loads(line.get("data-phases")) == sorted(triple.phase.unique())
    assert float(line.get("data-t")) == pytest.approx(T, rel=1e-5)

    congruent = eutectic_diagram[eutectic_diagram["locus"] == Locus.CONGRUENT]
    expected = {("triple", f"{T:.0f} K")} | {("congruent", f"{t:.0f} K")
                                            for t in congruent.groupby(["mu", "T"]).size().index.get_level_values("T")}
    labels = _groups(root, "landau-transition-label")
    got = {(g.get("data-locus"), g.find(f".//{SVG}text").text) for g in labels}
    assert got == expected
    assert len(labels) == len(expected)


def test_two_phase_regions(ct_figure, eutectic_diagram):
    root, _, _ = _save(ct_figure)
    regions = _groups(root, "landau-two-phase")
    pairs = sorted(tuple(json.loads(g.get("data-phases"))) for g in regions)
    assert pairs == [("fcc", "hcp"), ("fcc", "liquid"), ("hcp", "liquid")]
    for g in regions:
        path = g.find(f".//{SVG}path")
        assert path.get("fill-opacity") == "0" and path.get("style") is None

    boundaries = _groups(root, "landau-boundary")
    sides = sorted((g.get("data-phase"), g.get("data-against")) for g in boundaries)
    assert sides == sorted([(a, b) for p in pairs for a, b in (p, p[::-1])])
    for g in boundaries:
        path = g.find(f".//{SVG}path")
        assert path.get("stroke-opacity") == "0" and path.get("fill") == "none"


def test_two_phase_region_spans_the_coexisting_compositions(ct_figure, eutectic_diagram):
    """The fcc+hcp region at T spans from hcp's to fcc's coexisting composition."""
    ax = ct_figure.axes[0]
    added = _add_coexistence(ax, [])
    try:
        (region,) = [a for a in added if a._landau_svg["data"].get("phases") == ["fcc", "hcp"]]
        shape = shapely.Polygon(region.get_path().vertices)
        rows = eutectic_diagram.query("refined == 'clausius-clapeyron'")
        pair = rows.groupby(["mu", "T"]).filter(lambda g: set(g.phase) == {"fcc", "hcp"})
        checked = 0
        for T, g in pair.groupby("T"):
            if not 400 < T < 750:
                continue
            c_hcp, c_fcc = g.set_index("phase").c[["hcp", "fcc"]]
            cut = shape.intersection(shapely.LineString([(0, T), (1, T)]))
            assert cut.bounds[0] == pytest.approx(c_hcp, abs=C_TOL)
            assert cut.bounds[2] == pytest.approx(c_fcc, abs=C_TOL)
            checked += 1
        assert checked > 5
    finally:
        for a in added:
            a.remove()


def test_ticks_and_ticklabels_separate(ct_figure):
    root, _, _ = _save(ct_figure)
    ids = _by_id(root)
    ticks = ids["ax0-layer-xaxis-ticks"]
    labels = ids["ax0-layer-xaxis-ticklabels"]
    assert {c.get("class") for c in ticks} == {"landau-xtick"}
    assert {c.get("class") for c in labels} == {"landau-xticklabel"}
    assert ticks.find(f".//{SVG}text") is None
    for g in labels:
        assert g.find(f".//{SVG}text").text == g.get("data-text")
    (label,) = list(ids["ax0-layer-yaxis-label"])
    assert label.get("data-text") == "$T$ [K]"


def test_text_as_paths(ct_figure):
    root, _, _ = _save(ct_figure, text_as_text=False)
    assert next(root.iter(f"{SVG}text"), None) is None
    assert len(_groups(root, "landau-label")) == 3


def test_figure_left_unchanged(ct_figure):
    ax = ct_figure.axes[0]
    before = _render(ct_figure)
    children = list(ax.get_children())
    effects = [t.get_path_effects() for t in ax.texts]
    _save(ct_figure)
    assert list(ax.get_children()) == children
    assert [t.get_path_effects() for t in ax.texts] == effects
    assert all(a.get_gid() is None for a in ct_figure.findobj())
    np.testing.assert_array_equal(_render(ct_figure), before)


def test_plotting_sets_no_gids(ct_figure):
    """Tags are only read on save; an ordinary savefig carries none of them."""
    assert all(a.get_gid() is None for a in ct_figure.findobj())
    buf = io.BytesIO()
    ct_figure.savefig(buf, format="svg")
    assert b"landau" not in buf.getvalue()


def test_deterministic(ct_figure):
    assert _save(ct_figure)[2] == _save(ct_figure)[2]


def test_id_prefix(ct_figure):
    root, elements, _ = _save(ct_figure, id_prefix="m1_")
    ids = _by_id(root)
    assert all(i.startswith("m1_") for i in ids)
    assert all(i.startswith("m1_") for i in elements)
    assert "m1_ax0-phase-fcc-0" in ids
    refs = []
    for el in root.iter():
        for key, value in el.attrib.items():
            if key in (XLINK_HREF, "href") and value.startswith("#"):
                refs.append(value[1:])
            refs.extend(re.findall(r"url\(#([^)]+)\)", value))
    assert refs
    assert all(r in ids for r in refs)
    with pytest.raises(ValueError):
        _save(ct_figure, id_prefix="1x")


def test_mu_t_boundaries(eutectic_diagram):
    fig, ax = plt.subplots()
    plot_mu_phase_diagram(eutectic_diagram, ax=ax)
    root, _, _ = _save(fig)
    plt.close(fig)
    assert _groups(root, "landau-two-phase") == []
    pairs = sorted(tuple(json.loads(g.get("data-phases"))) for g in _groups(root, "landau-boundary"))
    assert pairs == [("fcc", "hcp"), ("fcc", "liquid"), ("hcp", "liquid")]


def test_phase_names_sanitised(eutectic_diagram):
    names = {"fcc": "(\u03b1Ca)", "hcp": r"$\mathrm{Mg_{2}Si}$'", "liquid": r"$\gamma$"}
    df = eutectic_diagram.copy()
    df["phase"] = df["phase"].map(names)
    fig, ax = plt.subplots()
    plot_phase_diagram(df, ax=ax)
    root, elements, _ = _save(fig)
    plt.close(fig)
    fields = {g.get("data-phase"): g.get("id") for g in _groups(root, "landau-phase")}
    assert fields == {
        "(\u03b1Ca)": "ax0-phase-\u03b1Ca-0",
        r"$\mathrm{Mg_{2}Si}$'": "ax0-phase-Mg2Si-0",
        r"$\gamma$": "ax0-phase-gamma-0",
    }
    assert all(ID_RE.fullmatch(i) for i in elements)
    pairs = {frozenset(json.loads(g.get("data-phases"))) for g in _groups(root, "landau-two-phase")}
    assert {"(\u03b1Ca)", r"$\mathrm{Mg_{2}Si}$'"} in pairs


@pytest.mark.parametrize("name, expected", [
    (r"$\mathrm{Cs}$", "Cs"),
    (r"$\mathrm{GaAs}$", "GaAs"),
    (r"$\mathrm{Cs_2O}$", "Cs2O"),
    (r"$\mathrm{bcc_{disordered}}$", "bccdisordered"),
    (r"$\alpha$", "alpha"),
    (r"$\gamma'$", "gamma"),
    (r"$\beta$-Sn", "beta-Sn"),
    (r"$\hat{\sigma}$", "sigma"),
    (r"A$\,$B$\ $C", "ABC"),
    ("Ni$_3$Al", "Ni3Al"),
    ("5 $ and 3", "5_and_3"),
    ("\u03b1-Fe", "\u03b1-Fe"),
    ("", "_"),
])
def test_names_keep_the_characters_of_mathtext(name, expected):
    assert _name(name) == expected


def test_legend_entries(eutectic_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram, ax=ax, inline_legend=False)
    root, _, _ = _save(fig)
    plt.close(fig)
    (legend,) = _groups(root, "landau-legend")
    for kind in ("legend-label", "legend-handle"):
        entries = [g for g in legend.iter(f"{SVG}g") if g.get("class") == f"landau-{kind}"]
        assert sorted(g.get("data-phase") for g in entries) == ["fcc", "hcp", "liquid"]
        for g in entries:
            assert g.get("id") == f"ax0-{kind}-{g.get('data-phase')}-0"
            assert g.get("data-unit") == "0"
            assert g.get("data-layer") == "legend"


@pytest.fixture(scope="module")
def split_solid_diagram():
    """A solid solution cut in two by a line compound stable at every T: ``g`` has two fields.

    The second field of ``g`` is labelled ``g'``, like the compound.
    """
    solid = IdealSolution("g", LinePhase("gA", 0, -3.0, 1.0 * kB), LinePhase("gB", 1, -3.0, 1.0 * kB))
    compound = LinePhase("g'", fixed_concentration=0.5, line_energy=-3.25, line_entropy=1.0 * kB)
    return ldc.calc_phase_diagram([solid, compound], np.linspace(300.0, 900.0, 25), mu=60, refine=True)


def test_legend_entry_stands_for_its_artist_not_its_label(split_solid_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(split_solid_diagram, ax=ax, inline_legend=False)
    fields = {(p.get_label(), to_hex(p.get_facecolor())): p._landau_svg["data"] for p in ax.patches}
    assert sorted(label for label, _ in fields) == ["g", "g'", "g'"]
    handles = ax.get_legend().legend_handles
    root, _, _ = _save(fig)
    plt.close(fig)
    entries = _groups(root, "landau-legend-handle")
    assert len(entries) == 3
    for g, handle in zip(entries, handles):
        data = fields[handle.get_label(), to_hex(handle.get_facecolor())]
        assert (g.get("data-phase"), g.get("data-unit")) == (data["phase"], str(data["unit"]))


def test_figure_legend_entries(eutectic_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram, ax=ax)
    fig.legend()
    root, _, _ = _save(fig)
    plt.close(fig)
    entries = _groups(root, "landau-legend-label")
    assert sorted(g.get("data-phase") for g in entries) == ["fcc", "hcp", "liquid"]
    assert {g.get("id") for g in entries} == {f"figure-legend-label-{p}-0" for p in ("fcc", "hcp", "liquid")}


def test_legend_of_other_artists_carries_no_phase(ct_figure):
    ax = ct_figure.axes[0]
    tag(ax.add_patch(plt.Rectangle((0.4, 300), 0.1, 50, label="window")), "overlay", kind="phase")
    ax.legend(["only this"])
    root, _, _ = _save(ct_figure)
    (entry,) = _groups(root, "landau-legend-label")
    assert entry.get("id") == "ax0-legend-label-only_this"
    assert entry.get("data-phase") is None


def test_user_overlay_layer(ct_figure):
    ax = ct_figure.axes[0]
    tag(ax.plot([0.3, 0.3], [400, 900], color="red"), "calphy-liquid", c=0.3)
    root, _, _ = _save(ct_figure)
    (run,) = _groups(root, "landau-calphy-liquid")
    assert run.get("id") == "ax0-calphy-liquid-0"
    assert run.get("data-c") == "0.3"
    assert _by_id(root)["ax0-layer-calphy-liquid"] is not None
    with pytest.raises(ValueError):
        tag(ax.plot([0, 1], [500, 500]), "not a layer")


def test_hand_set_gid_kept(ct_figure):
    ax = ct_figure.axes[0]
    (line,) = ax.plot([0.5, 0.5], [400, 900])
    line.set_gid("calphy-solid-7")
    root, _, _ = _save(ct_figure)
    (g,) = [g for g in root.iter(f"{SVG}g") if g.get("id") == "calphy-solid-7"]
    assert g.get("data-layer") == "other"
    assert line.get_gid() == "calphy-solid-7"


@dataclass(frozen=True)
class _WindowedLinePhase(AbstractLinePhase):
    """A line phase whose free energy is undefined above ``T_max``."""
    name: str
    energy: float
    entropy: float
    T_max: float

    @property
    def line_concentration(self):
        return 0.0

    def line_free_energy(self, T):
        out = np.where(np.asarray(T, dtype=float) <= self.T_max, self.energy - T * self.entropy, np.nan)
        return out.item() if out.ndim == 0 else out


def _polymorphs(E0):
    """Three polymorphs 30 and 60 meV apart, on an energy scale with its zero at ``-E0``."""
    phases = [LinePhase("alpha", 0, E0, 1.0 * kB), LinePhase("beta", 0, E0 + 0.030, 1.2 * kB),
              LinePhase("gamma", 0, E0 + 0.060, 1.3 * kB)]
    return ldc.calc_phase_diagram(phases, np.linspace(300, 1500, 25), mu=0.0, keep_unstable=True), {}


def _mirror_images():
    """Phases whose curves are mirror images of each other once referred to AB."""
    phases = [LinePhase("A", 0, 0.0), LinePhase("B", 1, 0.0), LinePhase("AB", 0.5, -0.5),
              LinePhase("A3B", 0.25, -0.2), LinePhase("AB3", 0.75, -0.2)]
    df = ldc.calc_phase_diagram(phases, 300.0, mu=np.linspace(-2, 2, 41), keep_unstable=True)
    return df, {"reference_phase": "AB"}


def _undefined_above_1500K():
    phases = [LinePhase("bcc", 0, 0.0, 1.0 * kB), LinePhase("liquid", 0, 0.1, 2.0 * kB),
              _WindowedLinePhase("qha", 0.02, 1.1 * kB, T_max=1500.0)]
    return ldc.calc_phase_diagram(phases, np.linspace(300, 2000, 35), mu=0.0, keep_unstable=True), {}


@pytest.mark.parametrize("frame", [
    lambda: _polymorphs(0.0),
    lambda: _polymorphs(-34800.0),  # all-electron total energies
    _mirror_images,
    _undefined_above_1500K,
], ids=["polymorphs", "large-energy-zero", "mirror-images", "nan-free-energy"])
def test_1d_curves_carry_the_phase_and_stability_they_are_drawn_with(frame):
    df, kwargs = frame()
    one_T = df["T"].nunique() == 1
    fig, ax = plt.subplots()
    (plot_1d_mu_phase_diagram if one_T else plot_1d_T_phase_diagram)(df, ax=ax, **kwargs)
    drawn = sum(len(line.get_xdata()) > 0 for line in ax.lines)
    root, _, _ = _save(fig)
    plt.close(fig)
    colors = {phase: to_hex(c) for phase, c in get_phase_colors(df.phase.unique(), palette="muted").items()}
    curves = _groups(root, "landau-curve")
    seen = set()
    for g in curves:
        style = dict(d.strip().split(": ", 1) for d in g.find(f"{SVG}path").get("style").split(";") if d.strip())
        assert style["stroke"] == colors[g.get("data-phase")]
        assert ("stroke-dasharray" in style) == (g.get("data-stable") == "false")
        seen.add(g.get("data-phase"))
    assert seen == set(colors)
    # every line seaborn drew is a curve, the transition guides are the only other lines
    assert len(curves) == drawn - len(_groups(root, "landau-transition"))
    assert _groups(root, "landau-other") == []


def test_unary_curves(unary_figure, unary_diagram):
    df = unary_diagram
    root, _, _ = _save(unary_figure)
    curves = _groups(root, "landau-curve")
    stable = {g.get("data-phase") for g in curves if g.get("data-stable") == "true"}
    assert stable == {"bcc", "fcc", "liquid"}
    # every drawn curve is attributed: nothing is left in "other"
    assert _groups(root, "landau-other") == []
    transitions = sorted(float(g.get("data-value")) for g in _groups(root, "landau-transition"))
    borders = sorted(df.query("border and T > T.min() and T < T.max()")["T"].unique())
    assert transitions == pytest.approx(borders)
