"""Layered SVG export of landau figures.

:func:`~landau.plot.save_layered_svg` writes a figure as SVG in which every semantic part
of the plot -- each phase field, phase label, two-phase region, phase
boundary, invariant, the legend, ticks, tick labels and axis labels -- is its
own ``<g>`` element with a stable ``id``, a ``class`` and ``data-*``
attributes, grouped into one layer per kind.  Downstream code (a web page
that highlights a phase on hover, a script that recolours a layer, an editor)
can select on those instead of guessing which anonymous path is which.

The functions plotting phase diagrams record what each artist they draw
*is* (via :func:`tag`); nothing about the drawing changes.
``plot_excess_free_energy`` does not yet, so its artists are written to the
layer ``other``.  The save then gives every artist a group id through :meth:`~matplotlib.artist.Artist.set_gid`, lets
matplotlib's SVG backend write the file, and rearranges the groups in one
pass over the XML.

**Layout of the output.**

``{p}`` is the ``id_prefix`` argument and ``{i}`` the index of an axes in
``fig.axes``.

* ``<g id="{p}figure">`` -- the figure; ``<g id="{p}ax{i}">`` -- one per axes.
  What belongs to the figure rather than to an axes -- its background, a
  ``suptitle``, a figure legend -- is laid out the same way with ``figure``
  in place of ``ax{i}``.  The axes of a ``SubFigure`` are written inside the
  figure's layer ``other``, and a diagram on a child axes (an inset) is one
  element of its parent axes, not broken up further.
* Inside each axes, one layer group per kind, ``<g id="{p}ax{i}-layer-{layer}"
  class="landau-layer" data-layer="{layer}">``, in the order the figure is
  painted.  A layer is one group unless an artist of another layer is
  painted between its members; it is then written as several groups that
  share ``data-layer``, the later ones with ``_2``, ``_3``, ... appended to
  the id, so select a layer by ``data-layer`` rather than by id.  The layers
  landau writes:

  ================== ====================================================
  layer              contents
  ================== ====================================================
  background         the axes patch
  scalar-field       the ``color_by`` fill, one raster image for all fields
  two-phase          two-phase regions of a c-T diagram (added on save)
  phases             phase fields (2d), phase curves (1d)
  boundaries         phase boundaries, one per phase pair (added on save)
  transitions        triple-point isotherms (2d), transition guides (1d)
  transition-ticks   ticks on the top spine at the transitions (1d)
  transition-points  dots at the transitions (1d)
  labels             phase labels
  transition-labels  temperatures of invariants (2d), transition labels (1d)
  legend             the legend box
  xaxis, yaxis       sublayers ``-grid``, ``-ticks``, ``-ticklabels``,
                     ``-label`` and ``-offset``
  spines             the axes frame
  title              axes titles
  other              everything not tagged; an artist given a gid by hand
                     keeps it as its id
  ================== ====================================================

  Overlays tagged with :func:`tag` get the layer named there.  An image
  that stands for several artists (rasterized ones, or neighbouring
  ``imshow`` images matplotlib composites) is one element; it takes their
  tag if they share one.

* Every element is ``<g id="{p}ax{i}-{kind}-{name}" class="landau-{kind}"
  data-layer="{layer}" ...>``.  ``name`` is built from what the element
  shows -- ``{phase}-{field}`` for a phase field or its label, the pair of
  phases plus a running number for two-phase regions and boundaries, a
  running number otherwise -- with mathtext reduced to its characters
  (``$\\mathrm{Mg_{2}Si}$`` becomes ``Mg2Si``, ``$\\alpha$`` becomes
  ``alpha``) and every other character that is not a letter, digit or
  underscore replaced by ``_``.  The ids are therefore stable across
  re-renders of the same diagram but lossy; the original phase name is kept
  in ``data-phase`` (``data-phases`` holds a JSON list where several phases
  meet).  An id that would repeat gets a ``_2``, ``_3``, ... suffix.

  ====================== ==================================================
  kind                   data attributes
  ====================== ==================================================
  phase                  ``phase``, ``unit`` (index of the field among the
                         phase's disconnected fields), ``label``
  field                  ``column``
  two-phase              ``phases``, ``units``
  boundary               c-T: ``phase``, ``unit``, ``against``,
                         ``against-unit``; mu-T: ``phases``, ``units``
  triple                 ``t``, ``mu``, ``phases``
  label                  ``phase``, ``unit`` (2d) or ``position`` (1d)
  transition-label       2d: ``locus`` (``triple``/``congruent``), ``t``,
                         ``mu``; 1d: ``value``
  curve                  ``phase``, ``stable`` (1d)
  transition             ``value``, ``phases`` (1d)
  transition-point       ``value``, ``phases`` (1d)
  transition-ticks       ``values`` (1d)
  legend-label           those of the artist the entry stands for
                         (``phase``, ``unit``, ``label`` of a phase field);
                         none for the legend seaborn draws on a 1d diagram
  legend-handle          as ``legend-label``
  xtick, ytick           ``value``, ``which`` (``major``/``minor``)
  xgrid, ygrid           ``value``, ``which``
  xticklabel, yticklabel ``value``, ``which``, ``text``
  xlabel, ylabel         ``text``
  ====================== ==================================================

  The kinds ``background``, ``spine``, ``title``, ``legend``, ``xoffset``,
  ``yoffset`` and ``other`` carry no data attributes.

Two-phase regions and boundaries are not drawn by the plotting functions.
They are built from the phase fields as drawn when the figure is saved and
written with zero opacity, as presentation attributes rather than inline
style, so a stylesheet can show them (``.landau-boundary path {stroke-opacity:
1}``) and they catch pointer events without changing the picture.  In a c-T
diagram tie-lines are horizontal: the space between two fields is the
two-phase region of the field to its left and the field to its right, and
ends at the three-phase tie-line of an invariant even where the same two
fields continue beyond it.  A miscibility gap has the same field on both sides and is a region of one
phase twice, with one boundary; space with no field drawn on a side -- the
room around the diagram, or beside a field that has thinned out of the
picture -- is no region.  In a mu-T diagram an edge of a field's outline is
its boundary with the field nearest to it.  Neither depends on the axes
limits.

A phase field catches pointer events whether it is filled or not
(``pointer-events="visible"``); the ``color_by`` fill lets them through
(``pointer-events="none"``).

With ``inkscape=True`` every layer group is also an Inkscape layer
(``inkscape:groupmode="layer"`` and ``inkscape:label``).
"""

import copy
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import numpy as np
import shapely
from matplotlib import cbook
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from shapely.ops import split

__all__ = ["save_layered_svg", "tag"]


_SVG = "http://www.w3.org/2000/svg"
_XLINK = "http://www.w3.org/1999/xlink"
_INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
_NAMESPACES = {
    "": _SVG,
    "xlink": _XLINK,
    "inkscape": _INKSCAPE,
    "dc": "http://purl.org/dc/elements/1.1/",
    "cc": "http://creativecommons.org/ns#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
}

_G = f"{{{_SVG}}}g"
_TEXT = f"{{{_SVG}}}text"
_TSPAN = f"{{{_SVG}}}tspan"
_PATH = f"{{{_SVG}}}path"
_DEFS = f"{{{_SVG}}}defs"
_IMAGE = f"{{{_SVG}}}image"

_TAG_ATTR = "_landau_svg"
_PLACEHOLDER = "landau-placeholder-"
# A phase field catches the pointer whether it is filled or, under a
# color_by fill, only outlined; the fill itself lets the pointer through to
# the field and the two-phase regions.
_POINTER_EVENTS = {"phase": "visible", "field": "none"}
_OUTLINE_ATTR = "_landau_outline"
_LAYER_RE = re.compile(r"[A-Za-z0-9_-]+")
_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_PREFIX_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_.-]*)?")

# The axis children, in the order their sublayers are stacked.
_AXIS_PARTS = ("grid", "ticks", "ticklabels", "label", "offset")

# Geometry of the two-phase regions and boundaries, in units of the window the
# phase fields span (both axes scaled to [0, 1], so one tolerance fits a c
# axis and a T axis).
_SNAP = 1e-4  # an outline within this distance of a region borders it
_SLIVER = 1e-6  # a gap between fields narrower than twice this is a rounding error
_MU_SNAP = 5e-3  # mu-T fields are trimmed apart by up to their buffer width
_MIN_LENGTH = 5e-3  # mu-T fields sharing a shorter stretch only touch at a corner


def tag(artists, layer: str, *, kind: str | None = None, name=None, **data):
    """Assign artists to a layer of :func:`~landau.plot.save_layered_svg`'s output.

    landau tags the artists it draws itself; use this to put an overlay drawn
    on top of a diagram into a layer of its own, e.g. ::

        tag(ax.plot([c, c], [T0, T1], color="red"), "calphy-liquid", c=c)

    The tag is stored on the artist and only read when the figure is saved;
    it does not change how the artist is drawn.

    Args:
        artists: One :class:`~matplotlib.artist.Artist` or an iterable of
            them, nested or not (the list :meth:`~matplotlib.axes.Axes.plot`
            returns, the container of
            :meth:`~matplotlib.axes.Axes.errorbar`).
        layer: Layer name, ``[A-Za-z0-9_-]+``.  Artists sharing a layer are
            grouped together.
        kind: Element kind, used in the element id and its class
            ``landau-{kind}``; defaults to ``layer``.
        name: Element name, used in the element id ``ax{i}-{kind}-{name}``.
            Mathtext is reduced to its characters, every run of characters
            other than letters, digits, ``_`` and ``-`` becomes one ``_``,
            and those at either end are dropped; a tuple is joined with
            ``-`` after the same replacement, ``-`` included, in each part.
            Defaults to a running number per kind.
        **data: Written as ``data-*`` attributes, the keys lower-cased and
            their underscores turned into hyphens.  Numbers are written
            with 10 significant digits, non-finite ones as ``Infinity``,
            ``-Infinity`` and ``NaN``; lists and tuples as JSON, with
            non-finite numbers as those strings.

    Returns:
        ``artists``, unchanged, so the call can wrap the drawing call.
    """
    if not _LAYER_RE.fullmatch(layer):
        raise ValueError(f"layer must match [A-Za-z0-9_-]+, got {layer!r}")
    if kind is not None and not _LAYER_RE.fullmatch(kind):
        raise ValueError(f"kind must match [A-Za-z0-9_-]+, got {kind!r}")
    attributes = [_data_attribute(key) for key in data]
    for key, attribute in zip(data, attributes):
        if not _KEY_RE.fullmatch(key) or attribute == "data-layer" or attributes.count(attribute) > 1:
            raise ValueError(f"data key {key!r} does not give an attribute of its own: keys are letters, digits "
                             "and '_', start with a letter, differ in more than case, and 'layer' is taken")
    found = [a for a in ([artists] if isinstance(artists, Artist) else cbook.flatten(artists)) if a is not None]
    for artist in found:
        if not isinstance(artist, Artist):
            raise TypeError(f"tag() takes artists, got {type(artist).__name__}")
    for artist in found:
        setattr(artist, _TAG_ATTR, {"layer": layer, "kind": kind or layer, "name": name, "data": data})
    return artists


def save_layered_svg(
    fig,
    fname,
    *,
    id_prefix: str = "",
    text_as_text: bool = True,
    inkscape: bool = True,
    coexistence: bool = True,
    **savefig_kwargs,
) -> dict[str, dict[str, str]]:
    """Save a figure as SVG with every part of the plot in its own group.

    The layout and the id scheme are described in :mod:`landau.plot.svg`.
    The figure is left as it was found: the ids, path effects and extra
    artists it needs while saving are removed again afterwards.

    Args:
        fig: The :class:`~matplotlib.figure.Figure`, or an axes of it.
        fname: Path or binary file object to write to.
        id_prefix: Prepended to every id in the file -- the semantic ones and
            matplotlib's clip paths, glyphs and markers alike, with all
            references to them rewritten -- so several files can be inlined
            into one HTML page without collisions.  Empty, or a letter or
            underscore followed by letters, digits, ``_``, ``.`` and ``-``.
        text_as_text: Write text as ``<text>`` elements rather than glyph
            outlines (``svg.fonttype = 'none'``), so labels can be selected
            and edited.  The outline landau draws around its labels
            becomes the text's own stroke (``paint-order: stroke``) instead
            of a second, path-drawn copy; a mathtext label of several glyph
            runs gets a stroke-only ``<text class="landau-outline">`` under
            it instead, so its group holds the text twice and ``data-phase``
            is the name to read.  The file then depends on the viewer having
            the fonts.  ``False`` leaves ``svg.fonttype`` as it is configured.
        inkscape: Mark every layer group as an Inkscape layer.
        coexistence: Add the two-phase regions and phase boundaries of every
            2d phase diagram in the figure (see :mod:`landau.plot.svg`).
        **savefig_kwargs: Passed to :meth:`~matplotlib.figure.Figure.savefig`;
            ``format`` is always ``"svg"``.  ``metadata`` defaults to
            ``{"Date": None}`` and matplotlib's ``svg.hashsalt`` to a fixed
            value, so the same figure gives the same file.

    Returns:
        dict: For every element, its id mapped to its attributes (``class``,
        ``data-layer`` and the ``data-*`` attributes).
    """
    if not _PREFIX_RE.fullmatch(id_prefix):
        raise ValueError("id_prefix must be empty or a letter or underscore followed by letters, digits, "
                         f"'_', '.' and '-', got {id_prefix!r}")
    fmt = savefig_kwargs.pop("format", "svg")
    if fmt != "svg":
        raise ValueError(f"save_layered_svg writes SVG, got format={fmt!r}")
    savefig_kwargs.setdefault("metadata", {"Date": None})
    if isinstance(fig, Axes):
        fig = fig.figure
    while getattr(fig, "figure", fig) is not fig:  # a SubFigure
        fig = fig.figure

    extra = []
    session = _Session(fig)
    try:
        if coexistence:
            for ax in fig.axes:
                _add_coexistence(ax, extra)
        # Ticks are created and placed while drawing; draw once so the ones
        # the SVG will contain exist and know their location.
        fig.draw_without_rendering()
        session.tag_figure()
        if text_as_text:
            session.drop_outlines()
        rc = {"svg.fonttype": "none"} if text_as_text else {}
        if mpl.rcParams["svg.hashsalt"] is None:
            rc["svg.hashsalt"] = "landau"
        buffer = io.BytesIO()
        with mpl.rc_context(rc):
            fig.savefig(buffer, format="svg", **savefig_kwargs)
    finally:
        session.restore()
        for artist in extra:
            artist.remove()

    root = ET.fromstring(buffer.getvalue())
    layout = _Layout(session, id_prefix, inkscape)
    elements = layout.apply(root)
    if id_prefix:
        _prefix_ids(root, id_prefix, keep=layout.used)
    # ElementTree keeps one prefix table per process; set it when serialising,
    # so that neither importing landau changes it nor another library's
    # entries turn the root into <ns0:svg>
    for ns_prefix, uri in _NAMESPACES.items():
        ET.register_namespace(ns_prefix, uri)
    out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    if hasattr(fname, "write"):
        fname.write(out)
    else:
        Path(fname).write_bytes(out)
    return elements


def _plain(text) -> str:
    """``text`` with mathtext reduced to its characters.

    ``$\\mathrm{Mg_{2}Si}$`` -> ``Mg2Si``, ``$\\alpha$`` -> ``alpha``.
    """
    parts = str(text).split("$")
    if len(parts) % 2 == 0:  # unbalanced '$': not mathtext
        return str(text)
    for i in range(1, len(parts), 2):
        math = re.sub(r"\\[A-Za-z]+(?=\s*\{)", "", parts[i])  # a command applied to a group: a font, an accent
        math = re.sub(r"\\([A-Za-z]+)", r"\1", math)  # any other command is a symbol and keeps its name
        parts[i] = re.sub(r"\\.|[{}^_\s]", "", math)  # escaped spacing, grouping, scripts
    return "".join(parts)


def _slug(part) -> str:
    # \w is unicode-aware: Greek letters, common in phase names, are valid in ids and CSS selectors
    return re.sub(r"[\W]+", "_", _plain(part)).strip("_") or "_"


def _name(name) -> str:
    if isinstance(name, tuple):
        return "-".join(_slug(p) for p in name)
    return re.sub(r"[^\w-]+", "_", _plain(name)).strip("_-") or "_"


def _data_attribute(key) -> str:
    return f"data-{key.replace('_', '-').lower()}"


def _number(value) -> str:
    """A float in 10 significant digits, non-finite ones spelled as JavaScript's ``Number()`` reads them."""
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return f"{value:.10g}"


def _data_value(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps([_json_item(v) for v in value], ensure_ascii=False)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    return str(value)


def _json_item(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # JSON has no non-finite numbers: those are written as the strings a scalar gives
        return float(_number(value)) if math.isfinite(value) else _number(value)
    return str(value)


class _Session:
    """The gids, tags and path effects set on a figure for one save.

    Every artist that should become a semantic element is given a unique
    placeholder gid, so its ``<g>`` can be found in the SVG again;
    :meth:`restore` puts back whatever was there before.
    """

    def __init__(self, fig):
        self.fig = fig
        self.entries = {}  # placeholder gid -> entry dict
        self._gids = {}  # artist -> original gid
        self._effects = {}  # text -> original path effects
        self._assigned = set()

    def assign(self, artist, scope, layer, kind, name=None, data=None, sub=None, hit=None, keep_id=None):
        if artist is None or id(artist) in self._assigned:
            return
        self._assigned.add(id(artist))
        token = f"{_PLACEHOLDER}{len(self.entries)}"
        self._gids[artist] = artist.get_gid()
        artist.set_gid(token)
        self.entries[token] = {
            "artist": artist, "scope": scope, "layer": layer, "kind": kind,
            "name": name, "data": dict(data or {}), "sub": sub, "hit": hit, "keep_id": keep_id,
        }

    def assign_tagged(self, artist, scope, default_layer="other"):
        info = getattr(artist, _TAG_ATTR, None)
        if info is None:
            # a gid set by hand stays the element's id
            self.assign(artist, scope, default_layer, default_layer, keep_id=artist.get_gid())
        else:
            self.assign(artist, scope, info["layer"], info["kind"], info["name"], info["data"],
                        hit=info.get("hit"))

    def tag_figure(self):
        fig = self.fig
        self.assign(fig, "figure", None, "figure")
        self.assign(fig.patch, "figure", "background", "background")
        for i, ax in enumerate(fig.axes):
            self._tag_axes(ax, f"ax{i}", i)
        suptitle = getattr(fig, "_suptitle", None)
        if suptitle is not None:
            self.assign(suptitle, "figure", "title", "title", "suptitle")
        for legend in fig.legends:
            self._tag_legend(legend, "figure")
        for artist in fig.get_children():
            self.assign_tagged(artist, "figure")

    def _tag_axes(self, ax, scope, index):
        self.assign(ax, scope, None, "axes", data={"axes": index})
        self.assign(ax.patch, scope, "background", "background")
        for side, spine in ax.spines.items():
            self.assign(spine, scope, "spines", "spine", side)
        for letter, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            self._tag_axis(axis, letter, scope)
        for loc, title in (("center", ax.title), ("left", ax._left_title), ("right", ax._right_title)):
            self.assign(title, scope, "title", "title", loc)
        legend = ax.get_legend()
        if legend is not None:
            self._tag_legend(legend, scope)
        for artist in ax.get_children():
            self.assign_tagged(artist, scope)

    def _tag_axis(self, axis, letter, scope):
        layer = f"{letter}axis"
        self.assign(axis, scope, layer, "axis")
        first, second = ("bottom", "top") if letter == "x" else ("left", "right")
        for which, locs, ticks in (
            ("major", axis.get_majorticklocs(), axis.get_major_ticks),
            ("minor", axis.get_minorticklocs(), axis.get_minor_ticks),
        ):
            pre = "" if which == "major" else "minor-"
            for k, tick in enumerate(ticks(len(locs))):
                value = {"value": tick.get_loc(), "which": which}
                self.assign(tick.gridline, scope, layer, f"{letter}grid", f"{pre}{k}", value, sub="grid")
                self.assign(tick.tick1line, scope, layer, f"{letter}tick", f"{pre}{k}", value, sub="ticks")
                self.assign(tick.tick2line, scope, layer, f"{letter}tick", f"{pre}{k}-{second}", value, sub="ticks")
                for side, label in ((first, tick.label1), (second, tick.label2)):
                    name = f"{pre}{k}" if side == first else f"{pre}{k}-{side}"
                    self.assign(label, scope, layer, f"{letter}ticklabel", name,
                                value | {"text": label.get_text()}, sub="ticklabels")
        self.assign(axis.label, scope, layer, f"{letter}label", data={"text": axis.label.get_text()}, sub="label")
        self.assign(axis.offsetText, scope, layer, f"{letter}offset", sub="offset")

    def _tag_legend(self, legend, scope):
        self.assign(legend, scope, "legend", "legend")
        sources = self._legend_sources(legend)
        for i, (text, handle) in enumerate(zip(legend.texts, legend.legend_handles)):
            info = getattr(sources[i], _TAG_ATTR, None) if sources else None
            data = {} if info is None else info["data"]
            name = text.get_text() if info is None or info["name"] is None else info["name"]
            self.assign(text, scope, "legend", "legend-label", name, data)
            self.assign(handle, scope, "legend", "legend-handle", name, data)

    @staticmethod
    def _legend_sources(legend):
        """The artists a legend's entries stand for, or ``None`` if it was not built from the labelled artists."""
        axes = [legend.axes] if legend.axes is not None else legend.figure.axes
        handles, labels = [], []
        for ax in axes:
            h, lab = ax.get_legend_handles_labels()
            handles += h
            labels += lab
        return handles if labels == [text.get_text() for text in legend.texts] else None

    def drop_outlines(self):
        """Replace landau's path-effect outline of every tagged text by a stroke.

        Text drawn with a path effect is written as glyph outlines even with
        ``svg.fonttype = 'none'``; without the effect it stays text, and the
        layout restores the outline as the text's own SVG stroke.
        """
        for entry in self.entries.values():
            artist = entry["artist"]
            outline = getattr(artist, _OUTLINE_ATTR, None)
            if isinstance(artist, Text) and outline is not None and artist.get_path_effects() == outline["effects"]:
                self._effects[artist] = artist.get_path_effects()
                artist.set_path_effects([])
                entry["outline"] = outline

    def restore(self):
        for artist, gid in self._gids.items():
            artist.set_gid(gid)
        for artist, effects in self._effects.items():
            artist.set_path_effects(effects)


class _Layout:
    """Rearranges matplotlib's SVG into the documented layers and ids."""

    def __init__(self, session, prefix, inkscape):
        self.session = session
        self.prefix = prefix
        self.inkscape = inkscape
        self.used = set()  # every id written here
        self.elements = {}
        self.counters = {}
        self.entry_of = {}  # <g> element -> entry dict

    def apply(self, root):
        entries = self.session.entries
        parents = {child: parent for parent in root.iter() for child in parent}
        # matplotlib writes an image's gid on the <image> itself; give it a group like every other artist
        for el in [el for el in root.iter() if el.tag != _G and el.get("id") in entries]:
            _enclosing_group(el, parents).set("id", el.attrib.pop("id"))
        self.entry_of = {el: entries[el.get("id")] for el in root.iter(_G) if el.get("id") in entries}
        containers = [(el, e) for el, e in self.entry_of.items() if e["kind"] in ("figure", "axes")]
        for el, entry in containers:
            if entry["kind"] == "axes":
                self._group_rasters(el, entry["scope"], parents)
        # the figure and axes ids first: nothing else may take them
        for el, entry in containers:
            if entry["kind"] == "figure":
                self._make_layer(el, f"{self.prefix}figure", "figure", "landau-figure")
            else:
                el.set("data-axes", str(entry["data"]["axes"]))
                self._make_layer(el, f"{self.prefix}{entry['scope']}", f"axes {entry['data']['axes']}", "landau-axes")
        for el, entry in list(self.entry_of.items()):
            if entry["kind"] == "axis":
                self._split_axis(el, entry, parents)
        for el, entry in containers:
            self._group_layers(el, entry["scope"])
        for el in root.iter(_G):  # in the order they are painted
            entry = self.entry_of.get(el)
            if entry is not None and entry["kind"] not in ("figure", "axes", "axis"):
                self._annotate(el, entry)
        for el in root.iter():
            if (el.get("id") or "").startswith(_PLACEHOLDER):  # an artist that is no element
                del el.attrib["id"]
        return self.elements

    def _group_rasters(self, axes_el, scope, parents):
        """Make an element of every raster image written directly into an axes.

        Rasterized artists get no group of their own; neighbours are merged
        into one image.  The image stands for the rasterized artists of the
        axes that are missing from the file and takes their tag if they
        share one.
        """
        images = [el for el in axes_el if el.tag == _IMAGE]
        if not images:
            return
        written = {id(entry) for entry in self.entry_of.values()}
        tags = [
            (e["layer"], e["kind"], e["name"], e["data"])
            for e in self.session.entries.values()
            if e["scope"] == scope and id(e) not in written
            and e["artist"].get_rasterized() and e["artist"].get_visible()
        ]
        shared = tags[0] if tags and all(t == tags[0] for t in tags) else None
        for image in images:
            group = _enclosing_group(image, parents)
            if shared is not None:
                layer, kind, name, data = shared
                self.entry_of[group] = {"scope": scope, "layer": layer, "kind": kind, "name": name,
                                        "data": data, "hit": None}

    def _new_id(self, eid):
        """``eid``, or ``eid_2``, ``eid_3``, ... if it is taken."""
        base, k = eid, 2
        while eid in self.used:
            eid = f"{base}_{k}"
            k += 1
        self.used.add(eid)
        return eid

    def _layer_id(self, scope, layer):
        return f"{self.prefix}{scope}-layer-{layer}"

    def _make_layer(self, el, lid, label, cls="landau-layer", layer=None):
        el.set("id", self._new_id(lid))
        el.set("class", cls)
        if layer is not None:
            el.set("data-layer", layer)
        if self.inkscape:
            el.set(f"{{{_INKSCAPE}}}groupmode", "layer")
            el.set(f"{{{_INKSCAPE}}}label", label)

    def _group_layers(self, parent, scope):
        """Collect the children of an axes (or the figure) into layer groups.

        The children keep the order they are painted in; each run of
        neighbours that share a layer becomes one group.  Axes and axis
        children are layers themselves and stay where they are.
        """
        children = []
        for child in list(parent):
            entry = self.entry_of.get(child)
            untagged = entry is None or (entry["kind"] == "other" and entry["layer"] == "other")
            if untagged and child.tag == _G and len(child) == 0:
                # an artist that drew nothing, such as a legend proxy line
                parent.remove(child)
                self.entry_of.pop(child, None)
            else:
                children.append(child)
                parent.remove(child)
        runs = {}
        current = group = None
        for child in children:
            entry = self.entry_of.get(child)
            if child.tag != _G:
                layer = None
            elif entry is None:
                # an untagged matplotlib group: an element all the same
                layer = "other"
                self.entry_of[child] = {"scope": scope, "layer": "other", "kind": "other", "name": None,
                                        "data": {}, "hit": None}
            elif entry["kind"] in ("axes", "figure"):
                layer = None
            elif entry["kind"] == "axis":
                layer = None
                self._make_layer(child, self._layer_id(scope, entry["layer"]), entry["layer"], layer=entry["layer"])
            else:
                layer = entry["layer"]
            if layer is None:
                parent.append(child)
                if child.tag != _DEFS:  # definitions paint nothing and split no layer
                    current = None
                continue
            if layer != current:
                n = runs[layer] = runs.get(layer, 0) + 1
                group = ET.SubElement(parent, _G)
                self._make_layer(group, self._layer_id(scope, layer), layer if n == 1 else f"{layer} ({n})",
                                 layer=layer)
                current = layer
            group.append(child)

    def _split_axis(self, axis_el, entry, parents):
        """Regroup an axis' ticks, tick labels, label and offset text into sublayers."""
        scope, layer = entry["scope"], entry["layer"]
        parts = {part: [] for part in _AXIS_PARTS}
        for el in list(axis_el.iter(_G)):
            sub_entry = self.entry_of.get(el)
            if el is axis_el or sub_entry is None or sub_entry["sub"] is None:
                continue
            parts[sub_entry["sub"]].append(el)
        for members in parts.values():
            for el in members:
                parents[el].remove(el)
        # tick groups emptied by the moves above
        for el in list(axis_el):
            if el.tag == _G and len(el) == 0:
                axis_el.remove(el)
        for part in _AXIS_PARTS:
            if not parts[part]:
                continue
            group = ET.SubElement(axis_el, _G)
            name = f"{layer}-{part}"
            self._make_layer(group, self._layer_id(scope, name), name, layer=name)
            group.extend(parts[part])

    def _annotate(self, el, entry):
        kind, scope = entry["kind"], entry["scope"]
        name = entry["name"]
        if name is None:
            n = self.counters.get((scope, kind), 0)
            self.counters[(scope, kind)] = n + 1
            name = str(n)
        eid = f"{self.prefix}{scope}-{kind}-{_name(name)}"
        if entry.get("keep_id"):
            eid = f"{self.prefix}{entry['keep_id']}"
        eid = self._new_id(eid)
        el.set("id", eid)
        layer = entry["layer"] if entry.get("sub") is None else f"{entry['layer']}-{entry['sub']}"
        attrs = {"class": f"landau-{kind}", "data-layer": layer}
        for key, value in entry["data"].items():
            attrs[_data_attribute(key)] = _data_value(value)
        for key, value in attrs.items():
            el.set(key, value)
        self.elements[eid] = attrs
        if kind in _POINTER_EVENTS:
            el.set("pointer-events", _POINTER_EVENTS[kind])
        if entry.get("outline") is not None:
            _stroke_text(el, entry["outline"])
        if entry.get("hit") is not None:
            _hit_target(el, entry["hit"])


def _enclosing_group(el, parents):
    """The ``<g>`` holding only ``el``: its parent if that is such a group without id, else a new one."""
    parent = parents[el]
    if parent.tag == _G and parent.get("id") is None and len(parent) == 1:
        return parent
    group = ET.Element(_G)
    group.tail, el.tail = el.tail, None
    parent.insert(list(parent).index(el), group)
    parent.remove(el)
    group.append(el)
    parents[group], parents[el] = parent, group
    return group


def _stroke_text(el, outline):
    """Write landau's label outline as an SVG stroke on the text in ``el``.

    Text in one run is stroked in place, behind its own fill.  Mathtext is
    written as one ``<tspan>`` per glyph run, and a viewer strokes and fills
    them one after the other, so each run's stroke would paint over the run
    before it; such text gets a stroke-only copy underneath instead.
    """
    rgba = to_rgba(outline["foreground"], outline.get("alpha"))
    stroke = f"stroke: {to_hex(rgba)}; stroke-width: {outline['linewidth']:g}; stroke-linejoin: round"
    if rgba[3] != 1:
        stroke += f"; stroke-opacity: {rgba[3]:g}"
    parents = {child: parent for parent in el.iter() for child in parent}
    for text in list(el.iter(_TEXT)):
        style = text.get("style", "").rstrip("; ")
        if len(text.findall(_TSPAN)) > 1:
            under = copy.deepcopy(text)
            under.tail = None
            under.set("style", "; ".join(filter(None, [style, "fill: none", stroke])))
            under.set("class", "landau-outline")
            under.set("aria-hidden", "true")
            parent = parents[text]
            parent.insert(list(parent).index(text), under)
        else:
            text.set("style", "; ".join(filter(None, [style, "paint-order: stroke", stroke])))


def _hit_target(el, hit):
    """Write a two-phase region or boundary invisible but addressable.

    Presentation attributes instead of inline style, so any stylesheet rule
    overrides them; zero opacity rather than ``none``, so they still receive
    pointer events.
    """
    for path in el.iter(_PATH):
        path.attrib.pop("style", None)
        if hit == "area":
            path.attrib.update({"fill": "#808080", "fill-opacity": "0", "stroke": "none"})
        else:
            path.attrib.update({"fill": "none", "stroke": "#000000", "stroke-opacity": "0", "stroke-width": "1"})


def _prefix_ids(root, prefix, keep):
    """Prefix every id not yet prefixed, and every local reference to one."""
    renamed = {}
    for el in root.iter():
        old = el.get("id")
        if old is not None and old not in keep:
            renamed[old] = prefix + old
            el.set("id", prefix + old)
    if not renamed:
        return
    url = re.compile(r"url\(#([^)]+)\)")

    def fix(value):
        return url.sub(lambda m: f"url(#{renamed.get(m.group(1), m.group(1))})", value)

    for el in root.iter():
        for key, value in el.attrib.items():
            if key in (f"{{{_XLINK}}}href", "href") and value.startswith("#"):
                el.set(key, "#" + renamed.get(value[1:], value[1:]))
            elif "url(#" in value:
                el.set(key, fix(value))


# ---------------------------------------------------------------------------
# Two-phase regions and boundaries
# ---------------------------------------------------------------------------


def _add_coexistence(ax, added: list) -> list:
    """Add the two-phase regions and boundaries of a 2d phase diagram on ``ax``.

    Built from the phase fields ``_plot_phase_diagram`` drew and recorded on
    the axes, with the tie-lines of the invariants it would draw:

    * c-T: tie-lines are horizontal, so a point between two fields lies in
      the two-phase region of the field to its left and the field to its
      right.  The space the fields leave free, cut at every temperature
      where a field begins or ends, falls into pieces that each have one
      such pair of neighbours; pieces of one pair lying on top of each other
      are one region, except across the three-phase tie-line of an
      invariant, past which the same two fields can continue.  A piece with
      the same field on both sides is a miscibility
      gap, one without a field on a side is no region.  The boundary of a
      phase against the other is the part of its field's outline along the
      region.
    * mu-T: the fields tile the plane, so there are no regions; an edge of a
      field's outline is its boundary with the field nearest to it.

    The artists are appended to ``added`` as they are created, so the caller
    can remove them again even if a later one fails; ``added`` is returned.
    """
    state = getattr(ax, "_landau_phase_diagram", None)
    if state is None:
        return added
    if any(patch.axes is not ax for _, _, patch in state["fields"]):  # cleared or changed since it was drawn
        return added
    outlines = [(phase, unit, np.asarray(patch.get_xy(), dtype=float)) for phase, unit, patch in state["fields"]]
    outlines = [(phase, unit, xy) for phase, unit, xy in outlines if len(xy) and np.isfinite(xy).all()]
    if not outlines:
        return added
    every = np.concatenate([xy for _, _, xy in outlines])
    offset, scale = every.min(axis=0), np.ptp(every, axis=0)
    if not (scale > 0).all():
        return added

    # the window the fields span, scaled to the unit square so that one tolerance fits both axes

    def to_data(xy):
        return np.asarray(xy, dtype=float) * scale + offset

    fields = []
    for phase, unit, xy in outlines:
        shape = shapely.union_all(_polygons(shapely.make_valid(shapely.Polygon((xy - offset) / scale))))
        if not shape.is_empty:
            fields.append((phase, unit, shape))
    if state["variables"][0] == "c":
        tie_lines = [((T - offset[1]) / scale[1], (c0 - offset[0]) / scale[0], (c1 - offset[0]) / scale[0])
                     for T, c0, c1 in state["triples"]]
        found = _regions(fields, tie_lines)
    else:
        found = _mu_boundaries(fields)

    z_lines = max(2.4, max(patch.get_zorder() for _, _, patch in state["fields"]) * (1 + 1e-9))
    for hit, kind, name, data, geom in found:
        path = _mpl_path(geom, to_data)
        if path is None:
            continue
        if hit == "area":
            artist = PathPatch(path, facecolor="0.5", edgecolor="none", zorder=-10)
        else:
            artist = PathPatch(path, fill=False, edgecolor="k", linewidth=1.0, zorder=z_lines)
        tag(artist, "two-phase" if hit == "area" else "boundaries", kind=kind, name=name, **data)
        getattr(artist, _TAG_ATTR)["hit"] = hit
        ax.add_artist(artist)
        added.append(artist)
    return added


def _turning_heights(shape) -> set:
    """The heights at which the outline of ``shape`` turns from rising to falling or back."""
    heights = set()
    for polygon in _polygons(shape):
        for ring in (polygon.exterior, *polygon.interiors):
            y = np.asarray(ring.coords)[:-1, 1]
            y = y[y != np.roll(y, 1)]  # a level stretch turns once
            turns = (y - np.roll(y, 1)) * (np.roll(y, -1) - y) < 0
            heights.update(y[turns].tolist())
    return heights


def _regions(fields, tie_lines) -> list:
    """The two-phase regions of a c-T diagram and the boundaries along them.

    ``tie_lines`` are the three-phase tie-lines of the diagram's invariants,
    each as its height and its two ends.
    """
    shapes = np.array([shape for _, _, shape in fields], dtype=object)
    free = shapely.box(0, 0, 1, 1).difference(shapely.union_all(shapes))
    # between two neighbouring heights no field begins or ends, so the
    # fields left and right of a piece of free space are the same all over it
    heights = sorted(set().union(*(_turning_heights(shape) for shape in shapes)))
    cuts = shapely.MultiLineString([[(-1, y), (2, y)] for y in heights if 0 < y < 1])
    pieces = [q for part in _polygons(free) for q in _polygons(split(part, cuts) if len(cuts.geoms) else part)]

    def key(i):
        return fields[i][0], fields[i][1]

    beside = {}
    for piece in pieces:
        at = piece.representative_point()
        left = shapely.bounds(shapely.intersection(shapely.LineString([(-1, at.y), (at.x, at.y)]), shapes))[:, 2]
        right = shapely.bounds(shapely.intersection(shapely.LineString([(at.x, at.y), (2, at.y)]), shapes))[:, 0]
        if np.isnan(left).all() or np.isnan(right).all():
            continue  # open to one side: the margin of the diagram
        pair = tuple(sorted((int(np.nanargmax(left)), int(np.nanargmin(right))), key=key))
        beside.setdefault(pair, []).append(piece)

    regions = []
    for (i, j), parts in beside.items():
        for whole in _joined(parts, tie_lines):
            # an opening: drops the slivers left where two fields meet a rounding error apart
            whole = whole.buffer(-_SLIVER, join_style="mitre").buffer(_SLIVER, join_style="mitre").intersection(whole)
            regions.extend(((fields[i][0], fields[j][0]), (i, j), part) for part in _polygons(whole))
    regions.sort(key=lambda r: (r[0], r[2].centroid.y, r[2].centroid.x))

    found, count = [], {}
    for (pa, pb), (i, j), part in regions:
        k = count[(pa, pb)] = count.get((pa, pb), -1) + 1
        found.append(("area", "two-phase", (pa, pb, k),
                      {"phases": [pa, pb], "units": [fields[i][1], fields[j][1]]}, part))
        near = part.buffer(_SNAP)
        for a, b in dict.fromkeys(((i, j), (j, i))):
            line = _lines(fields[a][2].boundary.intersection(near))
            if line is not None:
                found.append(("line", "boundary", (fields[a][0], fields[b][0], k),
                              {"phase": fields[a][0], "unit": fields[a][1],
                               "against": fields[b][0], "against_unit": fields[b][1]}, line))
    return found


def _joined(pieces, tie_lines) -> list:
    """Join the pieces of free space between one pair of fields that lie on top of each other.

    Not across the three-phase tie-line of an invariant: the same two fields
    can lie beside the space above it and below it, which are two regions.
    """
    group = list(range(len(pieces)))

    def find(k):
        while group[k] != k:
            k = group[k]
        return k

    for a, lower in enumerate(pieces):
        y = lower.bounds[3]
        for b, upper in enumerate(pieces):
            if upper.bounds[1] != y:
                continue
            shared = lower.intersection(upper)
            if shared.length == 0:
                continue
            x = shared.centroid.x
            if not any(abs(y - height) <= _SNAP and c0 <= x <= c1 for height, c0, c1 in tie_lines):
                group[find(a)] = find(b)
    joined = {}
    for k, piece in enumerate(pieces):
        joined.setdefault(find(k), []).append(piece)
    return [shapely.union_all(parts) for parts in joined.values()]


def _mu_boundaries(fields) -> list:
    """The boundaries between the fields of a mu-T diagram."""
    order = sorted(range(len(fields)), key=lambda i: (fields[i][0], fields[i][1]))
    shapes = np.array([fields[i][2] for i in order], dtype=object)
    found, count = [], {}
    for n, i in enumerate(order[:-1]):
        pa, ua, sa = fields[i]
        edges = np.array([edge for polygon in _polygons(sa) for ring in (polygon.exterior, *polygon.interiors)
                          for edge in zip(ring.coords[:-1], ring.coords[1:])])
        # the field nearest to each edge of the outline, if any is near
        away = shapely.distance(shapes[:, None], shapely.points(edges.mean(axis=1))[None, :])
        away[n] = np.inf
        nearest = np.where(away.min(axis=0) <= _MU_SNAP, away.argmin(axis=0), -1)
        for m in range(n + 1, len(order)):
            pb, ub, _ = fields[order[m]]
            if pb == pa or not (nearest == m).any():
                continue
            line = _lines(shapely.MultiLineString(list(edges[nearest == m])))
            if line is None or line.length < _MIN_LENGTH:
                continue
            k = count[(pa, pb)] = count.get((pa, pb), -1) + 1
            found.append(("line", "boundary", (pa, pb, k), {"phases": [pa, pb], "units": [ua, ub]}, line))
    return found


def _polygons(geom) -> list:
    """The polygons in ``geom``, whatever collection it is."""
    if geom.is_empty:
        return []
    if isinstance(geom, shapely.Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _polygons(g)]
    return []


def _lines(geom):
    """The line parts of ``geom`` merged into one (Multi)LineString, or ``None``."""
    def parts(g):
        if isinstance(g, shapely.LineString):
            return [g]
        if isinstance(g, shapely.MultiLineString) or hasattr(g, "geoms"):
            return [p for s in g.geoms for p in parts(s)]
        return []

    found = [p for p in parts(geom) if p.length > 0]
    if not found:
        return None
    return shapely.line_merge(shapely.MultiLineString(found))


def _mpl_path(geom, to_data) -> MplPath | None:
    """A matplotlib path in data coordinates for a shapely polygon or line geometry."""
    paths = []
    for g in getattr(geom, "geoms", [geom]):
        if isinstance(g, shapely.Polygon):
            for ring in (g.exterior, *g.interiors):
                paths.append(MplPath(to_data(ring.coords), closed=True))
        elif isinstance(g, shapely.LineString):
            paths.append(MplPath(to_data(g.coords)))
    if not paths:
        return None
    return MplPath.make_compound_path(*paths)
