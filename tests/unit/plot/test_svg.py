"""Tests for the layered SVG export, landau.plot.save_layered_svg.

The file is parsed back and checked for the documented layout: one layer group
per kind, one element per phase field / label / invariant / region / boundary
carrying the phase names in data attributes, ticks and tick labels in separate
sublayers. The two-phase regions are checked against the boundaries in the
phase-diagram frame, and the figure must come out of the save as it went in.
"""
import io
import itertools
import json
import pickle
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
from landau.interpolate import RedlichKister
from landau.phases import AbstractLinePhase, FastInterpolatingPhase, IdealSolution, LinePhase, kB
from landau.plot import (
    get_phase_colors,
    plot_1d_mu_phase_diagram,
    plot_1d_T_phase_diagram,
    plot_mu_phase_diagram,
    plot_phase_diagram,
    save_layered_svg,
)
from landau.plot.style import theme
from landau.plot.svg import _add_coexistence, _data_value, _name, tag

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
    assert len(elements) > 30
    assert all(ID_RE.fullmatch(i) for i in elements)


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
    assert float(line.get("data-t")) == pytest.approx(T, rel=1e-9)
    assert float(line.get("data-mu")) == pytest.approx(triple["mu"].unique().item(), rel=1e-9)

    congruent = eutectic_diagram[eutectic_diagram["locus"] == Locus.CONGRUENT]
    expected = {("triple", f"{T:.0f} K")} | {("congruent", f"{t:.0f} K")
                                            for t in congruent.groupby(["mu", "T"]).size().index.get_level_values("T")}
    labels = _groups(root, "landau-transition-label")
    got = {(g.get("data-locus"), g.find(f".//{SVG}text").text) for g in labels}
    assert got == expected
    assert len(labels) == len(expected)
    for g in labels:
        # the label reads the temperature it carries
        assert g.find(f".//{SVG}text").text == f"{float(g.get('data-t')):.0f} K"
    (at_triple,) = [g for g in labels if g.get("data-locus") == "triple"]
    assert at_triple.get("data-mu") == line.get("data-mu")


def test_two_phase_regions(ct_figure):
    root, _, _ = _save(ct_figure)
    regions = _groups(root, "landau-two-phase")
    pairs = sorted(tuple(json.loads(g.get("data-phases"))) for g in regions)
    assert pairs == [("fcc", "hcp"), ("fcc", "liquid"), ("hcp", "liquid")]
    for g in regions:
        assert g.get("id") == "ax0-two-phase-{}-{}-0".format(*json.loads(g.get("data-phases")))
        assert g.get("data-units") == "[0, 0]"
        (path,) = g.findall(f".//{SVG}path")
        assert {k: path.get(k) for k in ("fill", "fill-opacity", "stroke", "style")} == {
            "fill": "#808080", "fill-opacity": "0", "stroke": "none", "style": None}

    boundaries = _groups(root, "landau-boundary")
    sides = sorted((g.get("data-phase"), g.get("data-against")) for g in boundaries)
    assert sides == sorted([(a, b) for p in pairs for a, b in (p, p[::-1])])
    for g in boundaries:
        assert g.get("id") == f"ax0-boundary-{g.get('data-phase')}-{g.get('data-against')}-0"
        (path,) = g.findall(f".//{SVG}path")
        assert {k: path.get(k) for k in ("fill", "stroke", "stroke-opacity", "stroke-width", "style")} == {
            "fill": "none", "stroke": "#000000", "stroke-opacity": "0", "stroke-width": "1", "style": None}


@pytest.fixture(scope="module")
def monotectic_diagram():
    """Line phases alpha and beta under a liquid that demixes: a liquid+liquid dome on a monotectic."""
    E_A, S_A, E_B, S_B, L0 = -2.8, 3.0 * kB, -2.35, 3.0 * kB, 0.25
    alpha = LinePhase("alpha", fixed_concentration=0, line_energy=-3.0, line_entropy=1.0 * kB)
    beta = LinePhase("beta", fixed_concentration=1, line_energy=-2.5, line_entropy=1.0 * kB)
    controls = [LinePhase(f"l{i}", fixed_concentration=c, line_energy=(1 - c) * E_A + c * E_B + L0 * c * (1 - c),
                          line_entropy=(1 - c) * S_A + c * S_B) for i, c in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])]
    liquid = FastInterpolatingPhase("liquid", phases=controls, add_entropy=True, interpolator=RedlichKister(2))
    return ldc.calc_phase_diagram([alpha, beta, liquid], np.linspace(300.0, 1700.0, 30), mu=60, refine=True)


def _frame_tie_lines(df):
    """The two-phase tie-lines of a frame, wherever two states share a finite (mu, T).

    One ``(T, c_left, c_right, phase_left, phase_right)`` each.
    """
    border = df[df["border"] & np.isfinite(df["mu"])]
    ties = []
    for (_mu, T), g in border.groupby(["mu", "T"]):
        if len(g) == 2:  # three states are an invariant, one is the edge of the sampled window
            g = g.sort_values("c")
            ties.append((T, *g["c"], *g["phase"]))
    return ties


def _coexistence(ax):
    """The regions and boundaries save_layered_svg would add to ``ax``, as ``(tag data, shapely geometry)``."""
    added = _add_coexistence(ax, [])
    regions, boundaries = [], []
    for artist in added:
        data, path = artist._landau_svg["data"], artist.get_path()
        starts = [*np.flatnonzero(path.codes == path.MOVETO), len(path.codes)]
        rings = [path.vertices[i:j] for i, j in itertools.pairwise(starts)]
        if artist._landau_svg["kind"] == "two-phase":
            regions.append((data, shapely.Polygon(rings[0], rings[1:])))
        else:
            boundaries.append((data, shapely.MultiLineString(rings)))
        artist.remove()
    return regions, boundaries


@pytest.mark.parametrize("diagram, pairs", [
    ("eutectic_diagram", [("fcc", "hcp"), ("fcc", "liquid"), ("hcp", "liquid")]),
    ("split_solid_diagram", [("g", "g'"), ("g", "g'")]),
    ("monotectic_diagram", [("alpha", "beta"), ("alpha", "liquid"), ("alpha", "liquid"), ("beta", "liquid"),
                            ("liquid", "liquid")]),
])
def test_two_phase_regions_are_where_the_frame_has_tie_lines(diagram, pairs, request):
    """Every tie-line lies in the region of its pair and spans it; no region is without tie-lines."""
    df = request.getfixturevalue(diagram)
    fig, ax = plt.subplots()
    plot_phase_diagram(df, ax=ax)
    regions, boundaries = _coexistence(ax)
    plt.close(fig)
    assert sorted(tuple(data["phases"]) for data, _ in regions) == pairs
    # a tie-line on the edge of the sampled window is on the outline of its region, not inside
    T_min, T_max = df["T"].min(), df["T"].max()
    ties = [t for t in _frame_tie_lines(df) if t[2] - t[1] > 3 * C_TOL and T_min < t[0] < T_max]
    assert len(ties) > 100
    holds = [[] for _ in regions]
    for tie in ties:
        T, c_left, c_right, *phases = tie
        (n,) = [n for n, (_, shape) in enumerate(regions) if shape.contains(shapely.Point((c_left + c_right) / 2, T))]
        data, shape = regions[n]
        assert data["phases"] == sorted(phases)
        holds[n].append(tie)
        cut = shape.intersection(shapely.LineString([(c_left - 1, T), (c_right + 1, T)]))
        assert cut.bounds[0] == pytest.approx(c_left, abs=C_TOL)
        assert cut.bounds[2] == pytest.approx(c_right, abs=C_TOL)
    for (data, shape), held in zip(regions, holds):
        # a region too narrow for the tie-lines checked above still has some of its own pair
        assert held or any(sorted(phases) == data["phases"] and shape.contains(shapely.Point((c_left + c_right) / 2, T))
                           for T, c_left, c_right, *phases in _frame_tie_lines(df))
    # a boundary is its own phase's side of a region: it passes through that phase's end of every tie-line there
    assert len(boundaries) == sum(1 if len(set(data["phases"])) == 1 and len(set(data["units"])) == 1 else 2
                                  for data, _ in regions)
    for data, line in boundaries:
        halfway = line.interpolate(0.5, normalized=True)
        on_outline = [shape.exterior.distance(halfway) for _, shape in regions]
        n = int(np.argmin(on_outline))
        assert on_outline[n] < C_TOL
        assert regions[n][0]["phases"] == sorted((data["phase"], data["against"]))
        for T, c_left, c_right, p_left, p_right in holds[n]:
            crossing = line.intersection(shapely.LineString([(c_left - 1, T), (c_right + 1, T)]))
            ends = [c for c, phase in ((c_left, p_left), (c_right, p_right)) if phase == data["phase"]]
            assert len(ends) == (2 if p_left == p_right and data["unit"] == data["against_unit"] else 1)
            for c in ends:
                assert crossing.distance(shapely.Point(c, T)) < C_TOL


def test_miscibility_gap_is_a_region_of_one_phase(monotectic_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(monotectic_diagram, ax=ax, transition_temperatures=True)
    root, _, _ = _save(fig)
    plt.close(fig)
    (dome,) = [g for g in _groups(root, "landau-two-phase") if json.loads(g.get("data-phases")) == ["liquid", "liquid"]]
    assert (dome.get("id"), dome.get("data-units")) == ("ax0-two-phase-liquid-liquid-0", "[0, 0]")
    # the dome lies inside one field, whose outline along it is the one boundary
    (binodal,) = [g for g in _groups(root, "landau-boundary") if g.get("data-against") == g.get("data-phase")]
    assert (binodal.get("id"), binodal.get("data-phase")) == ("ax0-boundary-liquid-liquid-0", "liquid")
    # the gap phase is two of the three states of the monotectic
    invariants = sorted(json.loads(g.get("data-phases")) for g in _groups(root, "landau-triple"))
    assert invariants == [["alpha", "beta", "liquid"], ["alpha", "liquid", "liquid"]]


def test_regions_of_one_pair_are_numbered(split_solid_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(split_solid_diagram, ax=ax)
    _, elements, _ = _save(fig)
    plt.close(fig)
    regions = {i: a for i, a in elements.items() if a["class"] == "landau-two-phase"}
    assert sorted(regions) == ["ax0-two-phase-g-g-0", "ax0-two-phase-g-g-1"]
    # each lies between the compound and one of the two fields of g
    assert sorted(a["data-units"] for a in regions.values()) == ["[0, 0]", "[1, 0]"]
    assert sorted(i for i, a in elements.items() if a["class"] == "landau-boundary") == [
        "ax0-boundary-g-g-0", "ax0-boundary-g-g-0_2", "ax0-boundary-g-g-1", "ax0-boundary-g-g-1_2"]


@pytest.mark.parametrize("limits", [
    {"xlim": (-0.02, 1.02), "ylim": (150, 1050)},  # margins around the diagram
    {"xlim": (0.4, 1.0), "ylim": (700, 900)},  # the hcp field is out of view
    {"xlim": (1.0, 0.0)},  # inverted
])
def test_regions_do_not_depend_on_the_view(ct_figure, limits):
    def coexistence():
        _, elements, _ = _save(ct_figure)
        return {i: a for i, a in elements.items() if a["class"] in ("landau-two-phase", "landau-boundary")}

    before = coexistence()
    assert len(before) == 9
    ct_figure.axes[0].set(**limits)
    assert coexistence() == before


def test_frame_without_triple_points_gets_no_regions(eutectic_diagram):
    """The three two-phase fields are then one piece of the diagram, which the frame gives no single name."""
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram.drop(columns="locus"), ax=ax)
    root, _, _ = _save(fig)
    plt.close(fig)
    assert len(_groups(root, "landau-phase")) == 3
    assert _groups(root, "landau-two-phase") == [] and _groups(root, "landau-boundary") == []


def test_cleared_axes_has_no_regions(ct_figure, unary_diagram):
    ax = ct_figure.axes[0]
    ax.clear()
    plot_1d_T_phase_diagram(unary_diagram, ax=ax)
    root, _, _ = _save(ct_figure)
    assert len(_groups(root, "landau-curve")) > 0
    assert _groups(root, "landau-two-phase") == [] and _groups(root, "landau-boundary") == []


def test_pickled_figure_saves_the_same(ct_figure):
    _, elements, _ = _save(ct_figure)
    copy = pickle.loads(pickle.dumps(ct_figure))
    assert _save(copy)[1] == elements
    plt.close(copy)


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
    ax = ct_figure.axes[0]
    assert [float(g.get("data-value")) for g in ticks] == pytest.approx(list(ax.get_xticks()), rel=1e-9)
    assert {g.get("data-which") for g in ticks} == {"major"}
    assert [g.get("id") for g in labels] == [f"ax0-xticklabel-{k}" for k in range(len(ax.get_xticks()))]


def test_minor_ticks_grid_and_titles(ct_figure):
    ax = ct_figure.axes[0]
    ax.minorticks_on()
    ax.grid(True)
    ax.set_title("Ca-Mg")
    ax.set_title("draft", loc="right")
    ct_figure.suptitle("fig. 1")
    root, elements, _ = _save(ct_figure)
    ids = _by_id(root)
    minor = [g for g in ids["ax0-layer-xaxis-ticks"] if g.get("data-which") == "minor"]
    assert len(minor) > 0 and all(g.get("id").startswith("ax0-xtick-minor-") for g in minor)
    assert {g.get("class") for g in ids["ax0-layer-xaxis-grid"]} == {"landau-xgrid"}
    assert [g.get("id") for g in ids["ax0-layer-title"]] == ["ax0-title-center", "ax0-title-right"]
    assert [g.get("id") for g in ids["figure-layer-title"]] == ["figure-title-suptitle"]
    assert elements["figure-title-suptitle"] == {"class": "landau-title", "data-layer": "title"}


def test_text_as_paths(ct_figure):
    """Labels are then glyph outlines and keep matplotlib's own outline: two copies, the first stroked."""
    root, _, _ = _save(ct_figure, text_as_text=False)
    assert next(root.iter(f"{SVG}text"), None) is None
    labels = _groups(root, "landau-label")
    assert len(labels) == 3
    for g in labels:
        first, second = list(g)
        assert _style(first)["stroke"] == "#ffffff" and float(_style(first)["stroke-width"]) == 3
        assert "stroke" not in _style(second)


@pytest.mark.parametrize("figure", ["ct_figure", "unary_figure"])
@pytest.mark.parametrize("fails", [False, True])
def test_figure_left_unchanged(figure, fails, request, monkeypatch):
    """Also when drawing fails after the save has put its ids, strokes and hit targets on the figure."""
    fig = request.getfixturevalue(figure)
    ax = fig.axes[0]
    (by_hand,) = ax.plot([], [])
    by_hand.set_gid("by-hand")
    before = _render(fig)
    children = list(ax.get_children())
    effects = [t.get_path_effects() for t in ax.texts]
    assert any(effects)
    if fails:
        def broken(self, renderer):
            raise RuntimeError("no ink")
        monkeypatch.setattr(type(ax.spines["top"]), "draw", broken)
        with pytest.raises(RuntimeError, match="no ink"):
            _save(fig)
        monkeypatch.undo()
    else:
        _save(fig)
    assert list(ax.get_children()) == children
    assert [t.get_path_effects() for t in ax.texts] == effects
    assert [a.get_gid() for a in fig.findobj() if a.get_gid() is not None] == ["by-hand"]
    np.testing.assert_array_equal(_render(fig), before)


def test_plotting_sets_no_gids(ct_figure):
    """Tags are only read on save; an ordinary savefig carries none of them."""
    assert all(a.get_gid() is None for a in ct_figure.findobj())
    buf = io.BytesIO()
    ct_figure.savefig(buf, format="svg")
    assert b"landau" not in buf.getvalue()


def test_deterministic(ct_figure):
    assert _save(ct_figure)[2] == _save(ct_figure)[2]


def test_namespace_prefixes_do_not_depend_on_other_users_of_elementtree(ct_figure):
    ET.register_namespace("", "http://example.com/another-default-namespace")
    ET.register_namespace("link", "http://www.w3.org/1999/xlink")
    _, _, raw = _save(ct_figure)
    assert raw.startswith(b"<?xml version='1.0' encoding='utf-8'?>\n<svg ")
    assert b"<ns0:" not in raw
    assert set(re.findall(rb"\w+:href", raw)) == {b"xlink:href"}


def test_id_prefix(ct_figure):
    _, unprefixed, _ = _save(ct_figure)
    root, elements, _ = _save(ct_figure, id_prefix="m1_")
    ids = _by_id(root)
    assert all(i.startswith("m1_") and not i.startswith("m1_m1_") for i in ids)
    assert elements == {f"m1_{i}": attrs for i, attrs in unprefixed.items()}
    assert {"m1_figure", "m1_ax0", "m1_ax0-layer-phases", "m1_ax0-phase-fcc-0"} <= set(ids)
    refs = []
    for el in root.iter():
        for key, value in el.attrib.items():
            if key in (XLINK_HREF, "href") and value.startswith("#"):
                refs.append(value[1:])
            refs.extend(re.findall(r"url\(#([^)]+)\)", value))
    assert len(refs) > 20  # clip paths and glyphs
    assert all(r in ids for r in refs)
    for bad in ("1x", "a b"):
        with pytest.raises(ValueError):
            _save(ct_figure, id_prefix=bad)


@pytest.mark.parametrize("diagram, pairs", [
    ("eutectic_diagram", [("fcc", "hcp"), ("fcc", "liquid"), ("hcp", "liquid")]),
    ("monotectic_diagram", [("alpha", "beta"), ("alpha", "liquid"), ("beta", "liquid")]),
])
def test_mu_t_boundaries_follow_the_transitions_of_their_pair(diagram, pairs, request):
    df = request.getfixturevalue(diagram)
    fig, ax = plt.subplots()
    plot_mu_phase_diagram(df, ax=ax)
    regions, boundaries = _coexistence(ax)
    root, _, _ = _save(fig)
    plt.close(fig)
    assert regions == [] and _groups(root, "landau-two-phase") == []
    assert sorted(tuple(json.loads(g.get("data-phases"))) for g in _groups(root, "landau-boundary")) == pairs
    border = df[df["border"] & np.isfinite(df["mu"])]
    T_range = np.ptp(df["T"])
    for data, line in boundaries:
        pair = tuple(data["phases"])
        at = [(mu, T) for (mu, T), g in border.groupby(["mu", "T"]) if len(g) == 2 and tuple(sorted(g["phase"])) == pair]
        assert len(at) > 5
        Ts = [T for _, T in at]
        # the boundary runs as far as the pair coexists: to the triple point a tracer step past its last transition
        assert line.bounds[1] == pytest.approx(min(Ts), abs=0.01 * T_range)
        assert line.bounds[3] == pytest.approx(max(Ts), abs=0.01 * T_range)
        mu_range = np.ptp(border["mu"])
        assert max(line.distance(shapely.Point(mu, T)) for mu, T in at) < 0.01 * mu_range


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
    assert list(_by_id(root)["ax0-layer-calphy-liquid"]) == [run]
    for layer, kind in (("not a layer", None), ("layer", "not a kind")):
        with pytest.raises(ValueError):
            tag(ax.plot([0, 1], [500, 500]), layer, kind=kind)


def test_element_names_and_repeated_ids(ct_figure):
    ax = ct_figure.axes[0]
    for _ in range(3):
        tag(ax.plot([0.3, 0.3], [400, 900]), "runs", kind="run", name="Ca-Mg #1")
    tag(ax.plot([0.3, 0.3], [400, 900]), "runs", kind="run", name=("Ca-Mg", 2.5))
    tag(ax.plot([0.3, 0.3], [400, 900]), "runs", kind="run")
    tag(ax.plot([0.3, 0.3], [400, 900]), "runs", kind="run")
    _, elements, _ = _save(ct_figure)
    assert [i for i, a in elements.items() if a["class"] == "landau-run"] == [
        "ax0-run-Ca-Mg_1", "ax0-run-Ca-Mg_1_2", "ax0-run-Ca-Mg_1_3", "ax0-run-Ca_Mg-2_5", "ax0-run-0", "ax0-run-1"]


def test_arguments(ct_figure, tmp_path):
    """``fig`` may be an axes, ``fname`` a path; the two switches switch off what they name."""
    _, default, raw = _save(ct_figure)
    path = tmp_path / "diagram.svg"
    assert save_layered_svg(ct_figure.axes[0], path) == default
    assert path.read_bytes() == raw
    assert save_layered_svg(ct_figure, str(path), inkscape=False, coexistence=False) == {
        i: a for i, a in default.items() if a["class"] not in ("landau-two-phase", "landau-boundary")}
    root = ET.fromstring(path.read_bytes())
    assert all(INKSCAPE not in key for el in root.iter() for key in el.attrib)
    assert b"inkscape:groupmode" in raw
    with pytest.raises(ValueError, match="SVG"):
        _save(ct_figure, format="png")
    _save(ct_figure, format="svg")


def test_tag_takes_nested_containers(ct_figure):
    ax = ct_figure.axes[0]
    container = ax.errorbar([0.2, 0.4], [500, 600], yerr=20, capsize=2)
    assert tag(container, "measured", source="dsc") is container
    drawn = [a for a in mpl.cbook.flatten(container) if a is not None]
    assert len(drawn) == 4  # the data line, two cap lines, the bars
    root, _, _ = _save(ct_figure)
    assert len(_groups(root, "landau-measured")) == len(drawn)
    assert {g.get("data-source") for g in _groups(root, "landau-measured")} == {"dsc"}


def test_tag_rejects_what_it_cannot_write(ct_figure):
    ax = ct_figure.axes[0]
    (line,) = ax.plot([0, 1], [500, 500])
    with pytest.raises(TypeError):
        tag([line, "not an artist"], "overlay")
    for data in ({"my key": 1}, {"T": 900, "t": 300}, {"Layer": "mine"}, {"9": 1}):
        with pytest.raises(ValueError):
            tag(line, "overlay", **data)
    # a rejected call tags nothing
    assert not hasattr(line, "_landau_svg")


@pytest.mark.parametrize("value, written", [
    (0.1 + 0.2, "0.3"),
    (np.float64(748.5423089585613), "748.542309"),
    (float("inf"), "Infinity"),
    (-np.inf, "-Infinity"),
    (float("nan"), "NaN"),
    (True, "true"),
    (np.int64(3), "3"),
    ("liquid", "liquid"),
    (["fcc", "\u03b1"], '["fcc", "\u03b1"]'),
    ((0, np.int64(1)), "[0, 1]"),
    ([748.5423089585613, np.inf, -np.inf, np.nan], '[748.542309, "Infinity", "-Infinity", "NaN"]'),
])
def test_data_values(value, written):
    assert _data_value(value) == written


def test_list_values_are_json(ct_figure, unary_figure):
    """Every list is JSON and a number in it is the number a scalar attribute gives."""
    for fig in (ct_figure, unary_figure):
        _, elements, _ = _save(fig)
        lists = [v for attrs in elements.values() for v in attrs.values() if v.startswith("[")]
        assert len(lists) >= 4
        for v in lists:
            json.loads(v, parse_constant=pytest.fail)
    (ticks,) = [a for a in elements.values() if a["class"] == "landau-transition-ticks"]
    values = [a["data-value"] for a in elements.values() if a["class"] == "landau-transition"]
    assert sorted(json.loads(ticks["data-values"])) == sorted(float(v) for v in values)
    assert len(values) == 2


def test_pure_component_transition_labels_name_their_side(ct_figure, eutectic_diagram):
    """The melting points of the pure components sit at mu = -+inf."""
    root, _, _ = _save(ct_figure)
    congruent = eutectic_diagram[eutectic_diagram["locus"] == Locus.CONGRUENT]
    expected = sorted(float(mu) for mu in congruent["mu"].unique())
    assert set(expected) == {-np.inf, np.inf}
    written = [g.get("data-mu") for g in _groups(root, "landau-transition-label") if g.get("data-locus") == "congruent"]
    assert set(written) == {"-Infinity", "Infinity"}


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
