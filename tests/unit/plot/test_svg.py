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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely

import landau.calculate as ldc
from landau.features import Locus
from landau.phases import LinePhase, kB
from landau.plot import plot_1d_T_phase_diagram, plot_mu_phase_diagram, plot_phase_diagram, save_layered_svg
from landau.plot.svg import _add_coexistence, tag

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


def _render(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.fixture
def ct_figure(eutectic_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram, ax=ax, transition_temperatures=True)
    yield fig
    plt.close(fig)


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
    names = {"fcc": "(\u03b1Ca)", "hcp": r"$\mathrm{Mg_{2}Si}$'", "liquid": "liquid"}
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
        "liquid": "ax0-phase-liquid-0",
    }
    assert all(ID_RE.fullmatch(i) for i in elements)
    pairs = {frozenset(json.loads(g.get("data-phases"))) for g in _groups(root, "landau-two-phase")}
    assert {"(\u03b1Ca)", r"$\mathrm{Mg_{2}Si}$'"} in pairs


def test_legend_entries(eutectic_diagram):
    fig, ax = plt.subplots()
    plot_phase_diagram(eutectic_diagram, ax=ax, inline_legend=False)
    root, _, _ = _save(fig)
    plt.close(fig)
    (legend,) = _groups(root, "landau-legend")
    entries = {g.get("data-phase") for g in legend.iter(f"{SVG}g") if g.get("class") == "landau-legend-label"}
    assert entries == {"fcc", "hcp", "liquid"}


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


def test_unary_curves():
    phases = [
        LinePhase("bcc", 0, -2.0, 1.0 * kB),
        LinePhase("fcc", 0, -2.02, 0.5 * kB),
        LinePhase("liquid", 0, -1.8, 3.0 * kB),
    ]
    df = ldc.calc_phase_diagram(phases, np.linspace(100, 1500, 80), mu=0.0, keep_unstable=True)
    fig, ax = plt.subplots()
    plot_1d_T_phase_diagram(df, ax=ax)
    root, _, _ = _save(fig)
    plt.close(fig)
    curves = _groups(root, "landau-curve")
    stable = {g.get("data-phase") for g in curves if g.get("data-stable") == "true"}
    assert stable == {"bcc", "fcc", "liquid"}
    # every drawn curve is attributed: nothing is left in "other"
    assert _groups(root, "landau-other") == []
    transitions = sorted(float(g.get("data-value")) for g in _groups(root, "landau-transition"))
    borders = sorted(df.query("border and T > T.min() and T < T.max()")["T"].unique())
    assert transitions == pytest.approx(borders)
