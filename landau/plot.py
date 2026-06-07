from typing import Literal
from warnings import warn

from pyiron_snippets.deprecate import deprecate

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import seaborn as sns
import numpy as np
import pandas as pd
import shapely
from shapely.ops import polylabel
from matplotlib.colors import to_rgba

from .calculate import calc_phase_diagram, get_transitions, cluster, cluster_T_c, _join_phase_unit
import landau.poly as poly


__all__ = [
    "plot_phase_diagram",
    "plot_mu_phase_diagram",
    "plot_1d_mu_phase_diagram",
    "plot_1d_T_phase_diagram",
]


def _text_with_outline(ax, x, y, s, *, outline_width=3, **kwargs):
    """Draw text with a solid white outline so it stays legible over any fill.

    A small reusable wrapper around :meth:`matplotlib.axes.Axes.text` that
    strokes the glyphs with a white outline (via matplotlib path effects)
    instead of drawing an opaque box behind them, keeping a label readable on
    top of coloured regions or tielines.  Extra keyword arguments are forwarded
    to ``ax.text``.

    Returns the created :class:`matplotlib.text.Text`.
    """
    kwargs.setdefault(
        "path_effects",
        [patheffects.withStroke(linewidth=outline_width, foreground="white")],
    )
    return ax.text(x, y, s, **kwargs)


def _largest_inscribed_circle_center(polygon_xy, ax):
    """Centre of the largest circle inscribable in a polygon, in data units.

    Uses shapely's pole of inaccessibility (:func:`shapely.ops.polylabel`),
    which lies inside even concave or crescent-shaped regions.  Phase-diagram
    axes are strongly anisotropic (``c`` spans ~1, ``T`` spans hundreds of
    kelvin), so the coordinates are normalised by the axis data-ranges before
    the search and mapped back afterwards; this yields the visually – rather
    than numerically – largest circle.

    Returns ``(x, y)`` in data coordinates, or ``None`` for a degenerate
    polygon.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = (x1 - x0) or 1.0
    sy = (y1 - y0) or 1.0
    coords = np.asarray(polygon_xy, dtype=float)
    poly = shapely.Polygon(np.column_stack([coords[:, 0] / sx, coords[:, 1] / sy]))
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
        if isinstance(poly, shapely.MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
    if not isinstance(poly, shapely.Polygon) or poly.is_empty or poly.area == 0:
        return None
    point = polylabel(poly, tolerance=1e-3)
    return point.x * sx, point.y * sy


def _add_inline_polygon_labels(ax, polys):
    """Label each phase polygon in place instead of drawing a legend box.

    Every polygon is annotated with its phase name (with trailing apostrophes
    for repeated stability regions, matching :func:`plot_polygons`) at the
    centre of its largest inscribed circle, with a white outline.  The text is
    black: the polygon fill already carries the phase colour, and black with a
    white stroke stays legible even over the pale pastel fills.

    Args:
        ax: matplotlib Axes the polygons were drawn on.
        polys: Series of matplotlib Polygons indexed as in :func:`get_polygons`.
    """
    for key, p in polys.items():
        if isinstance(key, tuple):
            phase, rep = key
        else:
            phase, rep = key, 0
        center = _largest_inscribed_circle_center(p.get_xy(), ax)
        if center is None:
            continue
        _text_with_outline(
            ax, center[0], center[1], phase + "'" * rep,
            ha="center", va="center", fontsize="small", fontweight="bold",
            color="black",
        )


def cluster_phase(df, distance_threshold=0.5):  # 0.5 hand-tuned
    """Cluster the stable, single phase regions.

    When a (e.g solid solution) phase has multiple disconnected regions of stability, the make_poly and
    make_concave_poly functions give wrong results, because they draw a single polygon.
    Instead this function adds two new columns `phase_unit` and `phase_id` and the latter will always refer to only a
    single connected stability region.  `phase_unit` enumerates disconnected regions of one phase.

    Args:
        df: DataFrame with columns 'phase', 'T', 'c'.
        distance_threshold: Passed to :func:`~landau.calculate.cluster_T_c`. Lower values
            (e.g. 0.1) split more aggressively and are needed when two stable segments of
            the same phase are close in concentration space.
    """
    # In pandas 3, ``groupby.apply`` collapses inconsistently when the callable returns a
    # ``Series``: with multiple groups the per-group Series get concatenated into one long
    # row-aligned Series, but with a *single* group the same Series becomes a one-row
    # DataFrame indexed by the group key (columns = original row indices), and the
    # downstream ``df['phase_unit'] = ...`` assignment fails.  Returning a DataFrame from
    # the callable instead is the documented path that combines consistently across
    # pandas 2 and 3 (see "Flexible apply" in the pandas user guide).
    df["phase_unit"] = df.groupby("phase", group_keys=False).apply(
        lambda g: cluster_T_c(g, distance_threshold=distance_threshold).to_frame("phase_unit"),
        include_groups=False,
    )["phase_unit"]
    df["phase_id"] = _join_phase_unit(df["phase"], df["phase_unit"])
    return df

def get_polygons(
    df,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] | None = None,
    distance_threshold: float = 0.5,  # hand-tuned
    **kwargs,
):
    """Turn the stable phase regions in df into polygons.

    Args:
        df (pandas.DataFrame):
            Input data containing columns for the variables and 'phase', 'stable'.
        poly_method (str or poly.AbstractPolyMethod, optional):
            The method to use for polygon construction.
        variables (list of str, optional):
            The columns in df to use as coordinates for the polygons. Defaults to ["c", "T"].
        distance_threshold (float, optional):
            Passed to :func:`cluster_phase`. Lower values split disconnected stable regions
            more aggressively. Default is 0.5.
        **kwargs:
            Passed to poly.handle_poly_method.

    Returns:
        pandas.Series:
            The constructed polygons, indexed by phase and phase_unit.
    """
    if variables is None:
        variables = ["c", "T"]
    df = df.query("stable").copy()
    df = cluster_phase(df, distance_threshold=distance_threshold)
    if (df.phase_unit == -1).any():
        warn("Clustering of phase points failed for some points, dropping them.")
        df = df.query("phase_unit>=0")
    poly_method = poly.handle_poly_method(poly_method, **kwargs)
    return poly_method.apply(df, variables=variables)


def plot_polygons(polys, color_map, ax=None):
    """Plot the given polygons to a matplotlib axis.

    Args:
        polys (pandas.Series):
            The polygons to plot, as returned by get_polygons.
        color_map (dict):
            Mapping from phase names to colors.
        ax (matplotlib.axes.Axes, optional):
            The axis to plot on. If None, plt.gca() is used.
    """
    if ax is None:
        ax = plt.gca()
    for i, (phase, p) in enumerate(polys.items()):
        with np.errstate(divide="ignore"):
            p.zorder = 1 / p.get_extents().size.prod()
        if isinstance(phase, tuple):
            phase, rep = phase
        else:
            rep = 0
        p.set_color(color_map[phase])
        p.set_edgecolor("k")
        p.set_label(phase + "'" * rep)
        ax.add_patch(p)


def _plot_tielines(df, ax=None):
    """Plot tielines for a concentration-based phase diagram.

    Args:
        df (pandas.DataFrame):
            Input data.
        ax (matplotlib.axes.Axes, optional):
            The axis to plot on.
    """
    if ax is None:
        ax = plt.gca()
    # TODO: quite buggy and not nice; can benefit a lot from
    # get_transitions
    if "refined" in df.columns:
        tdf = get_transitions(df)

        def plot_tie(dd):
            Tmin = dd["T"].min()
            Tmax = dd["T"].max()
            di = dd.query("T==@Tmin")
            da = dd.query("T==@Tmax")
            # "artificial" segment at the border of diagram
            # we just want to plot triple lines? so #phases==3
            if len(dd.phase.unique()) in [1, 2]:
                return
            ax.hlines(Tmin, di.c.min(), di.c.max(), color="k", zorder=-2, alpha=0.5, lw=4)
            # current marvin to past marvin: Why is that even necessary?
            if Tmin != Tmax:
                ax.hlines(Tmax, da.c.min(), da.c.max(), color="k", zorder=-2, alpha=0.5, lw=4)

        # FIXME: WARNING reuses local var define in if branch
        tdf.groupby("border_segment").apply(plot_tie, include_groups=False)
    else:
        # count the numbers of distinct phases per T, it changes there *must* be a triple
        # point, draw tie lines only there
        # TODO: figure out how to only draw them between the involved phases not over the whole conc range
        # the refined data points mess this up, because the phases are no longer on
        # the same grid
        chg = df.groupby("T").size().diff()
        T_tie = chg.loc[chg != 0].index[1:]  # skip first temp

        def plot_tie(dd):
            if dd["T"].iloc[0].round(3) not in T_tie.round(3):
                return
            if len(dd) != 2:
                return
            cl, cr = sorted(dd.c)
            ax.plot([cl, cr], dd["T"], color="k", zorder=-2, alpha=0.5, lw=4)

        df.groupby(["T", "mu"]).apply(plot_tie, include_groups=False)


def _set_axis_for(axis_var: str, df_stable, element: str | None, ax) -> None:
    """Configure the x-axis of a phase diagram based on the axis variable.

    Args:
        axis_var: The x-axis variable name, either ``"c"`` (concentration) or ``"mu"``
            (chemical potential).
        df_stable: Stable-phase rows of the diagram DataFrame (used only for ``"mu"``
            to determine finite x-limits).
        element: Optional element symbol used to build the axis label.
        ax: The :class:`matplotlib.axes.Axes` to configure.

    Raises:
        ValueError: If ``axis_var`` is not ``"c"`` or ``"mu"``.
    """
    if axis_var == "c":
        ax.set_xlim(0, 1)
        if element is not None:
            ax.set_xlabel(rf"$c_\mathrm{{{element}}}$")
        else:
            ax.set_xlabel("$c$")
    elif axis_var == "mu":
        mus = df_stable["mu"].unique()
        mus = mus[np.isfinite(mus)]
        if len(mus) > 0:
            ax.set_xlim(mus.min(), mus.max())
        if element is not None:
            ax.set_xlabel(rf"$\Delta\mu_\mathrm{{{element}}}$ [eV]")
        else:
            ax.set_xlabel(r"$\Delta\mu$ [eV]")
    else:
        raise ValueError(
            f"Unknown coordinate system: variables[0]={axis_var!r}. Expected 'c' or 'mu'."
        )


def _plot_phase_diagram(
    df,
    alpha=0.1,
    element=None,
    min_c_width=1e-2,
    color_override: dict[str, str] = {},
    tielines=False,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] | None = None,
    inline_legend=True,
    ax=None,
):
    if variables is None:
        variables = ["c", "T"]
    if ax is None:
        ax = plt.gca()
    df_stable = df.query("stable")
    color_map = get_phase_colors(df_stable.phase.unique(), color_override)

    polys = get_polygons(df, poly_method=poly_method, variables=variables, min_c_width=min_c_width, alpha=alpha)

    plot_polygons(polys, color_map, ax=ax)

    if tielines and variables[0] == "c":
        _plot_tielines(df, ax=ax)

    _set_axis_for(variables[0], df_stable, element, ax)

    ax.set_ylim(df_stable["T"].min(), df_stable["T"].max())
    # Inline labels need the final axis limits to place each label at the centre
    # of its polygon, so this runs after the limits above are set.
    if inline_legend:
        _add_inline_polygon_labels(ax, polys)
    else:
        ax.legend(ncols=2)
    ax.set_ylabel("$T$ [K]")


@deprecate(
    alpha="Pass a poly method from landau.poly to poly_method",
    min_c_width="Pass a poly method from landau.poly to poly_method",
)
def plot_phase_diagram(
    df,
    alpha=0.1,
    element=None,
    min_c_width=1e-2,
    color_override: dict[str, str] = {},
    tielines=False,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] | None = None,
    inline_legend=True,
    ax=None,
):
    return _plot_phase_diagram(
        df,
        alpha=alpha,
        element=element,
        min_c_width=min_c_width,
        color_override=color_override,
        tielines=tielines,
        poly_method=poly_method,
        variables=variables,
        inline_legend=inline_legend,
        ax=ax,
    )


def get_phase_colors(phase_names, override: dict[str, str] | None = None):
    if override is None:
        override = {}
    # the default map
    color_map = dict(zip(phase_names, sns.palettes.SEABORN_PALETTES["pastel"]))
    # disregard overriden phases that are not present
    override = {p: c for p, c in override.items() if p in color_map}
    # if the override uses the same colors as the default map, multiple phases
    # would be mapped to the same color; so instead let's update the color map of phases that would
    # use the same color as a phase in the override to use the default colors of the overriden phases
    # instead
    duplicates_map = {c: color_map[o] for o, c in override.items()}
    diff = {k: duplicates_map[c] for k, c in color_map.items() if c in duplicates_map}
    color_map.update(diff | override)
    return color_map

@deprecate(alpha="Pass a poly method from landau.poly to poly_method")
def plot_mu_phase_diagram(
    df,
    alpha=0.1,
    element=None,
    color_override: dict[str, str] = {},
    poly_method: Literal["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"] | poly.AbstractPolyMethod | None = None,
    inline_legend=True,
    ax=None,
):
    return _plot_phase_diagram(
        df,
        alpha=alpha,
        element=element,
        color_override=color_override,
        poly_method=poly_method,
        variables=["mu", "T"],
        inline_legend=inline_legend,
        ax=ax,
    )


def _assign_segment_ids(df: pd.DataFrame, scan_col: str) -> pd.Series:
    """Assign contiguous-segment IDs for a 1d scan along ``scan_col``.

    Ordering the points by ``scan_col``, a phase begins a new segment whenever its
    ``stable`` flag flips.  This is threshold-free and independent of grid density
    or of how derived quantities (concentration, ...) vary along the cut, so two
    disjoint metastable branches of one phase – e.g. a middle phase that is
    unstable both below and above its stable window – are never joined into one
    line.  Pass the result as ``units=`` to seaborn so it draws each segment
    separately.

    The cut axis is a parameter, so the helper extends unchanged to generalised
    1d diagrams along arbitrary T / mu / ... cuts: pass whichever column orders
    the points along the cut.

    Assumes every phase is sampled at every scan point – as produced by
    :func:`~landau.calculate.calc_phase_diagram` with ``keep_unstable=True`` – so
    a stability flip is the only way a phase's run along the cut can break.

    Args:
        df: DataFrame with 'phase', 'stable', and ``scan_col`` columns.
        scan_col: Column ordering the points along the cut ('mu', 'T', ...).

    Returns:
        String Series aligned with ``df.index``, one unique value per segment.
    """
    ordered = df.sort_values(scan_col)
    seg = ordered.groupby("phase", group_keys=False).apply(
        lambda g: (g["stable"] != g["stable"].shift()).cumsum().to_frame("_seg"),
        include_groups=False,
    )["_seg"]
    return (
        ordered["phase"].astype(str)
        + "_"
        + ordered["stable"].astype(int).astype(str)
        + "_"
        + seg.astype(str)
    ).reindex(df.index)


def _bridge_unstable_segments(df: pd.DataFrame, scan_col: str) -> pd.DataFrame:
    """Extend each unstable branch up to the exact transition point.

    A refined transition (``border``) row sits exactly at a stability flip but is
    marked ``stable``, so it anchors the solid line of both coexisting phases.
    The phase that turns metastable across the flip only resumes at the next
    sampled point, leaving a gap between the transition and the start of its
    dashed branch – the wider the gap, the coarser the grid.  For every such
    border row this duplicates it as an unstable point tagged with the adjacent
    unstable segment's ``_seg_id`` (the metastable side of the flip), so the
    dashed branch is drawn right up to the transition.

    Operates on the ``_seg_id``-tagged frame and returns it with the bridge rows
    appended; the originals are untouched, so the solid line still reaches the
    same point.  A no-op when no ``border`` column is present.

    Args:
        df: DataFrame with 'phase', 'stable', '_seg_id', ``scan_col`` and
            optionally 'border' columns.
        scan_col: Column ordering the points along the cut ('mu', 'T', ...).

    Returns:
        ``df`` with one duplicated unstable row per (border, adjacent-unstable)
        pair appended, reindexed.
    """
    if "border" not in df.columns:
        return df
    src_idx, seg_ids = [], []
    for _, g in df.groupby("phase", sort=False):
        g = g.sort_values(scan_col)
        idx = g.index.to_numpy()
        stable = g["stable"].to_numpy()
        border = g["border"].to_numpy()
        seg = g["_seg_id"].to_numpy()
        for i in range(len(g)):
            if not (border[i] and stable[i]):
                continue
            # The flip is stable on one side, metastable on the other; bridge to
            # whichever neighbour(s) along the cut are unstable.
            for j in (i - 1, i + 1):
                if 0 <= j < len(g) and not stable[j]:
                    src_idx.append(idx[i])
                    seg_ids.append(seg[j])
    if not src_idx:
        return df
    bridges = df.loc[src_idx].copy()
    bridges["stable"] = False
    bridges["_seg_id"] = seg_ids
    return pd.concat([df, bridges], ignore_index=True)


def _subtract_reference_phase(df, scan_col, reference_phase):
    """Subtract reference phase's phi from all phases along scan_col."""
    if reference_phase not in df["phase"].values:
        raise ValueError(f"reference_phase {reference_phase!r} not found in data")
    ref = df.loc[df["phase"] == reference_phase, [scan_col, "phi"]].sort_values(scan_col)
    # np.interp handles border points where the reference phase has no exact row
    # (refinement adds rows only for the two transitioning phases, not all phases).
    df["phi"] = df["phi"] - np.interp(df[scan_col], ref[scan_col], ref["phi"])
    return df


def _bold_math(label: str) -> str:
    """Wrap mathtext segments of ``label`` in ``\\mathbf`` so subscripts render bold.

    matplotlib's ``fontweight="bold"`` only affects the regular-text portions of a
    string; content inside ``$...$`` keeps the regular math weight. Wrapping each
    math segment in ``\\mathbf{...}`` bolds it to match while leaving arbitrary
    LaTeX inside renderable. A malformed string (odd number of ``$``) is returned
    unchanged.
    """
    parts = label.split("$")
    if len(parts) % 2 == 0:  # unbalanced '$' -> leave untouched
        return label
    for i in range(1, len(parts), 2):
        if parts[i]:
            parts[i] = r"\mathbf{" + parts[i] + "}"
    return "$".join(parts)


def _get_renderer(fig):
    """Return a renderer for *fig*, drawing the canvas first so text can be measured.

    ``Text.get_window_extent`` needs a renderer to report the rendered pixel size of a
    label.  ``Figure.canvas.get_renderer`` exists on the Agg backend; other backends
    expose it via the private ``Figure._get_renderer`` (matplotlib >= 3.6).
    """
    fig.canvas.draw()
    if hasattr(fig.canvas, "get_renderer"):
        return fig.canvas.get_renderer()
    return fig._get_renderer()


def _spread_labels(centers, heights, lo, hi, gap=0.0):
    """Nudge label centers apart so their vertical extents never overlap.

    Each label *i* occupies ``[center - heights[i]/2, center + heights[i]/2]``.  The
    returned centers keep the input order's stacking, sit within ``[lo, hi]`` where the
    stack fits, and move as little as possible from the requested ``centers``.  Works in
    whatever 1-d coordinate the caller passes (display pixels are convenient because a
    rendered text height is constant there regardless of the data scale).

    Args:
        centers: Desired center coordinate of each label.
        heights: Full extent of each label in the same units as ``centers``.
        lo, hi: Lower / upper bounds the stack should stay within.
        gap: Extra clearance to keep between adjacent labels.

    Returns:
        List of adjusted centers, one per input label, in the input order.
    """
    n = len(centers)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: centers[i])
    adj = list(centers)
    # Upward pass: push each label up just enough to clear the one below it.
    prev_top = lo
    for i in order:
        half = heights[i] / 2
        c = max(centers[i], prev_top + half)
        adj[i] = c
        prev_top = c + half + gap
    # Downward pass: pull labels down to respect the top bound while staying separated.
    next_bot = hi
    for i in reversed(order):
        half = heights[i] / 2
        c = min(adj[i], next_bot - half)
        adj[i] = c
        next_bot = c - half - gap
    return adj


def _place_side_labels(ax, df, scan_col, phase_colors):
    """Label every phase at the end of its line, reserving room adaptively.

    A phase is labelled at its right-hand line end by default.  The horizontal space
    reserved for the stack is derived from the widest *rendered* label (measured in
    pixels), not from any string-length heuristic, so the axis is widened by exactly as
    much as the labels need.  Within a stack the labels are spread vertically so they do
    not overlap, clamped to the current y-limits.

    If the current y-limit would clip a label off the top (its right-end value lies above
    the visible window), that phase is instead labelled at its left-hand line end on a
    mirrored left-hand stack, which is reserved and laid out by the same rules.
    """
    fig = ax.figure
    x_min, x_max = df[scan_col].min(), df[scan_col].max()
    span0 = x_max - x_min

    # Settle autoscaled limits (and obtain a renderer) before deciding which side each
    # label belongs to or measuring any text.
    renderer = _get_renderer(fig)
    lo_d, hi_d = sorted(ax.get_ylim())
    axbb = ax.get_window_extent(renderer)

    # Split phases: those visible at the right end label on the right; those whose
    # right end is above the window label at their left end instead.
    right, left = [], []
    for phase, group in df.groupby("phase"):
        g = group.sort_values(scan_col)
        right_y = g["phi"].iloc[-1]
        if right_y > hi_d:
            left.append((phase, g["phi"].iloc[0]))
        else:
            right.append((phase, right_y))

    def make_texts(entries, ha):
        return [
            ax.text(
                x_min, y, phase, transform=ax.transData,
                ha=ha, va="center", fontsize="small",
                color=phase_colors.get(phase, "black"), clip_on=True,
            )
            for phase, y in entries
        ]

    right_texts = make_texts(right, "left")
    left_texts = make_texts(left, "right")

    def measure(texts):
        if not texts:
            return 0.0, []
        ext = [t.get_window_extent(renderer) for t in texts]
        return max(e.width for e in ext) / axbb.width, [e.height for e in ext]

    gap_frac, margin_frac = 0.02, 0.01
    f_right, h_right = measure(right_texts)
    f_left, h_left = measure(left_texts)
    reserve_r = (gap_frac + f_right + margin_frac) if right_texts else 0.0
    reserve_l = (gap_frac + f_left + margin_frac) if left_texts else 0.0

    # Widen the axis so the line data occupies the middle (1 - reserve_l - reserve_r) of
    # it and each stack sits in its reserved strip.
    denom = max(1.0 - reserve_l - reserve_r, 0.2)
    total_span = span0 / denom
    x0 = x_min - reserve_l * total_span
    ax.set_xlim(x0, x0 + total_span)

    def place(texts, entries, heights, anchor_frac):
        if not texts:
            return
        anchor_x = x0 + anchor_frac * total_span
        target_px = [
            ax.transData.transform((anchor_x, min(max(y, lo_d), hi_d)))[1]
            for _, y in entries
        ]
        placed_px = _spread_labels(target_px, heights, axbb.y0, axbb.y1)
        inv = ax.transData.inverted()
        for t, py in zip(texts, placed_px):
            y_d = inv.transform((axbb.x0, py))[1]
            t.set_position((anchor_x, y_d))

    place(right_texts, right, h_right, 1.0 - margin_frac - f_right)
    place(left_texts, left, h_left, f_left + margin_frac)


def _add_1d_phase_legend(ax, df, scan_col, top_labels=True, side_labels=True, ylim=None):
    """Annotate a 1d phase diagram with inline phase labels.

    top_labels
        Ticks on the top spine mark each transition boundary (positions where
        ``border == True``) and the stable phase name is placed near the top of
        the axis, centered between adjacent boundaries.

    side_labels
        The default seaborn legend is removed and, when unstable lines are
        present, every phase is annotated by name at the right end of its final
        line segment inside the axis.

    Args:
        ax: matplotlib Axes with a seaborn lineplot already rendered.
        df: DataFrame with 'phase', 'stable', ``scan_col``, 'phi', and
            (if available) 'border' columns.
        scan_col: Scan-axis column ('mu' or 'T').
        top_labels: If True, add the top-spine ticks and stable-phase labels.
        side_labels: If True, remove the default seaborn legend and add the
            right-end side labels.
        ylim: If given, applied via ``ax.set_ylim`` before the labels are placed,
            so the side labels are clamped to (and spread within) this window.
            A scalar is treated as ``(None, ylim)`` — an upper bound only.
    """
    if ylim is not None:
        if np.isscalar(ylim):
            ylim = (None, ylim)
        ax.set_ylim(ylim)

    # Extract phase → color from the seaborn legend (used by both label sets).
    phase_colors = {}
    legend = ax.get_legend()
    if legend is not None:
        white = to_rgba("w")
        gray2 = to_rgba(".2")
        for handle, text in zip(legend.legend_handles, legend.texts):
            try:
                c = handle.get_color()
            except AttributeError:
                continue
            if to_rgba(c) in (white, gray2):
                continue
            phase_colors[text.get_text()] = c
        # The side labels replace the default seaborn legend.
        if side_labels:
            legend.remove()

    # Store so tests (or user code) can still access the color map.
    ax._landau_phase_colors = phase_colors

    if top_labels and "border" in df.columns:
        # Interior transition positions only.
        x_min = df[scan_col].min()
        x_max = df[scan_col].max()
        transitions = sorted(
            t for t in df.loc[df["border"], scan_col].unique()
            if x_min < t < x_max
        )
        boundaries = [x_min] + transitions + [x_max]

        # Top-spine ticks at transition boundaries (no tick labels; just marks).
        ax2 = ax.secondary_xaxis("top")
        ax2.set_ticks(transitions)
        ax2.set_xticklabels([])
        ax2.tick_params(direction="in", length=6)

        # Phase-name labels near the top of the axis, centered between boundaries.
        # get_xaxis_transform(): x in data coords, y in axes [0,1] fraction.
        xform = ax.get_xaxis_transform()
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            mid = (lo + hi) / 2
            # Strict inequalities exclude the border rows themselves, which can have
            # two phases marked stable simultaneously (the transition point).
            mask = (df[scan_col] > lo) & (df[scan_col] < hi) & df["stable"]
            stable_phases = df.loc[mask, "phase"].unique()
            if len(stable_phases) != 1:
                raise RuntimeError(
                    f"expected exactly one stable phase in [{lo}, {hi}], "
                    f"got {list(stable_phases)}"
                )
            phase = stable_phases[0]
            # White outline keeps the bold label legible on top of tielines.
            _text_with_outline(
                ax, mid, 0.97, _bold_math(phase),
                transform=xform,
                ha="center", va="top", fontsize="small", fontweight="bold",
                color=phase_colors.get(phase, "black"),
            )

    if side_labels and not df["stable"].all():
        _place_side_labels(ax, df, scan_col, phase_colors)


def plot_1d_mu_phase_diagram(
        df,
        ax=None,
        show=True,
        mark_transitions=True,
        reference_phase=None,
        top_labels=True,
        side_labels=True,
        ylim=None):
    """
    Plot a one dimensional isothermal phase diagram of the semi-grandcanonical
    potential as function of the chemical potential difference.

    Args:
        df (pandas.DataFrame):
            Input data containing columns for chemical potential difference ('mu'),
            semi-grandcanonical potential ('phi'), phase name ('phase'), stability
            ('stable'), and optionally a 'border' column indicating phase transition.
        ax (matplotlib.axes.Axes, optional):
            Existing matplotlib Axes to plot on. If None, a new figure and axes are created.
        mark_transitions (bool, optional):
            If True, all transition temperatures are marked on the plot. Defaults to True.
        reference_phase (str, optional):
            If given, subtract this phase's potential from all other phases before
            plotting so that the reference phase lies at zero throughout.
        top_labels (bool, optional):
            If True, label the stable phase of each segment near the top of the
            axis. Defaults to True.
        side_labels (bool, optional):
            If True, remove the default seaborn legend and label every phase at
            the right end of its line instead. Defaults to True.
        ylim (tuple or float, optional):
            If given, applied like :func:`matplotlib.pyplot.ylim`. A scalar is
            treated as ``(None, ylim)`` (upper bound only). It also bounds the
            side-label stack: a label whose line end is above the window is moved
            to a mirrored stack on the left.

    Returns:
        matplotlib.axes.Axes:
            The Axes object with the phase diagram plot.
    """

    if len(df['T'].unique()) > 1:
        raise ValueError("data contains more than one temperature!")
    if ax is None:
        fig, ax = plt.subplots()

    df = df.sort_values("mu").copy()

    if reference_phase is not None:
        df = _subtract_reference_phase(df, "mu", reference_phase)

    df["_seg_id"] = _assign_segment_ids(df, scan_col="mu")
    sns.lineplot(
        data=_bridge_unstable_segments(df, scan_col="mu"),
        x='mu', y='phi',
        hue='phase', hue_order=sorted(df.phase.unique()),
        style='stable', style_order=[True, False],
        units='_seg_id', estimator=None, errorbar=None,
        ax=ax,
    )

    _add_1d_phase_legend(ax, df, scan_col="mu", top_labels=top_labels, side_labels=side_labels, ylim=ylim)

    if 'border' not in df.columns:
        return ax

    dfa = np.ptp(df['phi'].dropna())
    dfm = np.ptp(df['mu'].dropna())

    if mark_transitions:
        for mt, dd in df.query("mu.min()<mu<mu.max() and border").groupby("mu"):
            ft = dd['phi'].iloc[0]
            ax.axvline(mt, color='k', linestyle='dotted', alpha=.5)
            ax.scatter(mt, ft, marker='o', c='k', zorder=10)

            ax.text(mt - .05 * dfm, ft - dfa * .1, rf"$\Delta\mu = {mt:.03f}\,\mathrm{{eV}}$",
                    rotation='vertical', ha='center', va='top')
    ax.set_xlabel("Chemical Potential Difference [eV]")
    ylabel = "Semi-grandcanonical Potential [eV/atom]"
    if reference_phase is not None:
        ylabel = f"Semi-grandcanonical Potential\nrelative to {reference_phase} [eV/atom]"
    ax.set_ylabel(ylabel)

    return ax

def plot_1d_T_phase_diagram(
        df,
        ax=None,
        mark_transitions=True,
        show=True,
        reference_phase=None,
        top_labels=True,
        side_labels=True,
        ylim=None,
        ):
    """
    Plots a one-dimensional equipotential phase diagram as a function of temperature.

    Args:
        df (pandas.DataFrame):
            Input data containing columns for temperature ('T'), semi-grandcanonical
            potential ('phi'), phase name ('phase'), and optionally a 'border' column
            indicating phase transition.
        ax (matplotlib.axes.Axes, optional):
            Existing matplotlib Axes to plot on. If None, a new figure and axes are created.
        mark_transitions (bool, optional):
            If True, all transition temperatures are marked on the plot. Defaults to True.
        reference_phase (str, optional):
            If given, subtract this phase's potential from all other phases before
            plotting so that the reference phase lies at zero throughout.
        top_labels (bool, optional):
            If True, label the stable phase of each segment near the top of the
            axis. Defaults to True.
        side_labels (bool, optional):
            If True, remove the default seaborn legend and label every phase at
            the right end of its line instead. Defaults to True.
        ylim (tuple or float, optional):
            If given, applied like :func:`matplotlib.pyplot.ylim`. A scalar is
            treated as ``(None, ylim)`` (upper bound only). It also bounds the
            side-label stack: a label whose line end is above the window is moved
            to a mirrored stack on the left.

    Returns:
        matplotlib.axes.Axes:
            The Axes object with the phase diagram plot.
    """

    if len(df.mu.unique()) > 1:
        raise ValueError("Data contains more than one chemical potential!")

    if ax is None:
        fig, ax = plt.subplots()

    df = df.copy()

    if reference_phase is not None:
        df = _subtract_reference_phase(df, "T", reference_phase)

    df["_seg_id"] = _assign_segment_ids(df, scan_col="T")

    sns.lineplot(
        data=_bridge_unstable_segments(df, scan_col="T"),
        x='T', y='phi',
        hue='phase', hue_order=sorted(df.phase.unique()),
        style='stable', style_order=[True, False],
        units='_seg_id', estimator=None, errorbar=None,
        ax=ax,
    )

    _add_1d_phase_legend(ax, df, scan_col="T", top_labels=top_labels, side_labels=side_labels, ylim=ylim)

    if 'border' not in df.columns:
        return ax

    dfa = np.ptp(df['phi'].dropna())
    dft = np.ptp(df['T'].dropna())

    if mark_transitions:
        for Tt, dd in df.query("T.min()<T<T.max() and border").groupby("T"):
            ft = dd['phi'].iloc[0]
            ax.axvline(Tt, color='k', linestyle='dotted', alpha=.5)
            ax.scatter(Tt, ft, marker='o', c='k', zorder=10)

            ax.text(Tt + .05 * dft, ft + dfa * .1, rf"$T = {Tt:.0f}\,\mathrm{{K}}$", rotation='vertical', ha='center')

    ax.set_xlabel("Temperature [K]")
    ylabel = "Semi-grandcanonical potential [eV/atom]"
    if reference_phase is not None:
        ylabel = f"Semi-grandcanonical potential\nrelative to {reference_phase} [eV/atom]"
    ax.set_ylabel(ylabel)

    return ax


# ---------------------------------------------------------------------------
# Excess free energy plot
# ---------------------------------------------------------------------------

def plot_excess_free_energy(
    df,
    col_wrap=3,
    height=3.0,
    aspect=1.3,
    color_override=None,
    convex_hull=True,
):
    """Plot excess free energy vs concentration for competing phases.

    Takes a pre-computed DataFrame from ``calc_phase_diagram(..., keep_unstable=True)``
    and delegates to seaborn ``relplot``.

    When ``convex_hull=True``, stable solution phases render as solid curves,
    metastable/unstable regions as faded lines (same colour, alpha=0.4), and
    the common-tangent construction as black dotted segments — one segment per
    coexistence region (grouped by ``mu``) — with black vertex markers.
    Line phases always render as a single coloured scatter dot.
    When ``convex_hull=False``, all solution phases render as plain solid curves
    regardless of stability.

    Args:
        df: DataFrame from ``calc_phase_diagram(..., keep_unstable=True)`` with
            columns ``c``, ``f_excess``, ``phase``, ``T``, ``stable``, and
            optionally ``border`` and ``mu``.
        col_wrap: Maximum subplot columns per row.
        height: Height of each facet in inches.
        aspect: Width-to-height ratio of each facet.
        color_override: Optional ``dict[name -> color]`` overriding phase colours.
        convex_hull: If True, distinguish stable (solid curves) from metastable
            (faded lines) and overlay the common-tangent segments in black.
            If False, all solution phases render as plain solid curves.

    Returns:
        seaborn.FacetGrid: FacetGrid with one column per temperature.
        Access the figure via ``.fig`` and individual axes via ``.axes``.
    """
    import matplotlib.lines as mlines

    df = df.copy()
    if df.empty:
        raise ValueError("df is empty.")
    if "border" not in df.columns:
        df["border"] = False

    temperatures = sorted(df["T"].unique())
    col_wrap = min(col_wrap, len(temperatures))

    # Line phases have a fixed concentration — detect by zero range across all rows.
    phase_names = list(df["phase"].unique())
    c_range = df.groupby("phase")["c"].apply(lambda s: s.max() - s.min())
    line_phase_names = set(c_range[c_range < 1e-9].index)

    muted_colors = sns.color_palette("muted")
    palette = {name: muted_colors[i % len(muted_colors)] for i, name in enumerate(phase_names)}
    if color_override:
        palette.update({k: v for k, v in color_override.items() if k in palette})

    df_sol = df[~df["phase"].isin(line_phase_names)].copy()
    df_lp = df[df["phase"].isin(line_phase_names)].copy()

    sol_palette = {k: v for k, v in palette.items() if k not in line_phase_names}
    sol_hue_order = [n for n in phase_names if n not in line_phase_names]

    if convex_hull:
        base_data = df_sol[df_sol["stable"]].copy()
        if not base_data.empty:
            base_data = cluster_phase(base_data, distance_threshold=0.1)
            units_col = "phase_id"
        else:
            units_col = None
    else:
        base_data = df_sol
        units_col = None

    if base_data.empty:
        # No solution-phase rows (e.g. all phases are line phases, or all solution rows are
        # unstable).  sns.relplot cannot create facets from an empty DataFrame, so build
        # the grid structure directly and skip the line-drawing step.
        g = sns.FacetGrid(
            data=df[["T"]].drop_duplicates(),
            col="T",
            col_order=temperatures,
            col_wrap=col_wrap,
            height=height,
            aspect=aspect,
        )
    else:
        g = sns.relplot(
            data=base_data,
            x="c",
            y="f_excess",
            hue="phase",
            hue_order=sol_hue_order,
            units=units_col,
            col="T",
            col_wrap=col_wrap,
            palette=sol_palette,
            height=height,
            aspect=aspect,
            kind="line",
            estimator=None,
            errorbar=None,
            linewidth=2.5,
        )

    for ax, T_val in zip(g.axes.flat, temperatures):
        sub_all = df[df["T"] == T_val]
        sub_sol = df_sol[df_sol["T"] == T_val]
        sub_lp = df_lp[df_lp["T"] == T_val]

        if convex_hull:
            # Metastable solution phases: faded lines, same colour as stable.
            # T is constant within this facet, so cluster_T_c reduces to c-only
            # clustering, correctly splitting disjoint metastable c-ranges.
            unstable = sub_sol[~sub_sol["stable"]]
            for pname, grp in unstable.groupby("phase"):
                seg_ids = cluster_T_c(grp, distance_threshold=0.1)
                color = sol_palette.get(pname, "gray")
                for seg_id in seg_ids.unique():
                    seg = grp.loc[seg_ids == seg_id].sort_values("c")
                    ax.plot(
                        seg["c"].values, seg["f_excess"].values,
                        color=color, alpha=0.4, lw=2.5, zorder=2,
                    )

        # Line phases: single colored dot at fixed concentration.
        for _, row in sub_lp.drop_duplicates("phase").iterrows():
            ax.scatter(
                [row["c"]], [row["f_excess"]],
                color=palette.get(row["phase"], "k"),
                zorder=5, s=80,
            )

        if convex_hull:
            # Common-tangent lines: one dotted segment per coexistence region.
            # Rows sharing the same mu value belong to one two-phase equilibrium,
            # so grouping by mu gives one tangent line per coexistence pair.
            bd_all = sub_all[sub_all["border"]]
            for _mu, grp in bd_all.groupby("mu"):
                grp_sorted = grp.drop_duplicates(subset=["c", "f_excess"]).sort_values("c")
                if len(grp_sorted) >= 2:
                    ax.plot(
                        grp_sorted["c"].values, grp_sorted["f_excess"].values,
                        ls="dotted", color="k", zorder=3, lw=1.5,
                    )
            # Hull vertex markers: deduplicated across all mu values.
            bd_unique = bd_all.drop_duplicates(subset=["c", "f_excess"]).sort_values("c")
            if not bd_unique.empty:
                ax.scatter(
                    bd_unique["c"].values, bd_unique["f_excess"].values,
                    color="k", s=25, zorder=7,
                )

    # Add line phases to the figure legend.
    if not df_lp.empty:
        lp_handles = [
            mlines.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=palette.get(n, "k"), markersize=8, label=n,
            )
            for n in sorted(line_phase_names)
        ]
        if g._legend is not None:
            existing_handles = list(g._legend.legend_handles)
            existing_labels = [t.get_text() for t in g._legend.texts]
            g._legend.remove()
        else:
            existing_handles = []
            existing_labels = []
        g.figure.legend(
            existing_handles + lp_handles,
            existing_labels + [h.get_label() for h in lp_handles],
            title="phase",
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            borderaxespad=0,
        )

    g.refline(y=0)
    g.set_titles("T = {col_name:.0f} K")
    g.set(xlabel="Concentration", ylabel="Free Energy of Formation")

    return g
