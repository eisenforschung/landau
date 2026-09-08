from collections.abc import Callable
from dataclasses import dataclass
from itertools import cycle
from typing import Literal
from warnings import warn

from pyiron_snippets.deprecate import deprecate

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from ..calculate import calc_phase_diagram, cluster, cluster_T_c, _join_phase_unit, _apply_series
from ..features import Locus
from .labels import (
    _add_1d_phase_legend,
    _add_inline_curve_labels,
    _add_inline_polygon_labels,
    _annotate_transition_temperatures,
    _place_transition_labels,
)
import landau.poly as poly


__all__ = [
    "plot_phase_diagram",
    "plot_mu_phase_diagram",
    "plot_1d_mu_phase_diagram",
    "plot_1d_T_phase_diagram",
]


def cluster_phase(df, distance_threshold=0.2):  # hand-tuned, issue #456
    """Cluster the stable, single phase regions.

    When a (e.g solid solution) phase has multiple disconnected regions of stability, the make_poly and
    make_concave_poly functions give wrong results, because they draw a single polygon.
    Instead this function adds two new columns `phase_unit` and `phase_id` and the latter will always refer to only a
    single connected stability region.  `phase_unit` enumerates disconnected regions of one phase.

    Args:
        df: DataFrame with columns 'phase', 'T', 'c'.
        distance_threshold: Passed to :func:`~landau.calculate.cluster_T_c`. Lower values
            split more aggressively and are needed when two disconnected stable segments of
            the same phase sit close together in (normalised T, c) space; raise it (towards
            0.5) if a single continuous region is instead being split apart on its own
            sampling grid. The 0.2 default (issue #456) sits inside the safe band measured
            empirically on a real diagram (a Y-Zn liquid whose two stable fields are 0.39
            apart in normalised space stays split for any threshold from ~0.01 to ~0.3) and
            on the coarser synthetic grids in the test suite (a single continuous region
            starts fragmenting somewhere between 0.05 and 0.1).
    """
    df["phase_unit"] = _apply_series(
        df.groupby("phase", group_keys=False),
        lambda g: cluster_T_c(g, distance_threshold=distance_threshold),
        "phase_unit",
    )
    df["phase_id"] = _join_phase_unit(df["phase"], df["phase_unit"])
    return df

def get_polygons(
    df,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] | None = None,
    distance_threshold: float = 0.2,  # hand-tuned, issue #456
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
            more aggressively. Default is 0.2 — see :func:`cluster_phase` for the empirical
            margin behind that choice.
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


def _plot_triplepoints(df, ax=None, variables=None):
    """Mark the three-phase invariants of a phase diagram.

    Triple points are tagged :attr:`~landau.features.Locus.TRIPLE` in the
    ``locus`` column of a refined :func:`~landau.calculate.calc_phase_diagram`
    frame; the three coexisting phases share one ``(mu, T)``.

    The mark depends on the axes:

    * In a concentration-temperature diagram (``variables[0] == "c"``) a triple
      point is an isothermal line joining the three coexisting compositions, so
      one horizontal line is drawn across the concentration span of each
      ``(mu, T)`` group.
    * In a chemical-potential-temperature diagram (``variables[0] == "mu"``) the
      three phases collapse onto a single ``(mu, T)`` point, so a black marker is
      drawn there.

    Args:
        df (pandas.DataFrame):
            Phase-diagram frame carrying a ``locus`` column. Unrefined frames
            have no triple points and draw nothing.
        ax (matplotlib.axes.Axes, optional):
            The axis to plot on.
        variables (list[str], optional):
            The ``[x, y]`` axis variables; defaults to ``["c", "T"]``.
    """
    if variables is None:
        variables = ["c", "T"]
    if ax is None:
        ax = plt.gca()
    if "locus" not in df.columns:
        return
    triple = df[df["locus"] == Locus.TRIPLE]
    if variables[0] == "c":
        for (_mu, T), grp in triple.groupby(["mu", "T"], sort=False)[["c"]]:
            ax.hlines(T, grp["c"].min(), grp["c"].max(), color="k", zorder=-2, alpha=0.5, lw=2)
    elif variables[0] == "mu":
        for (mu, T), _grp in triple.groupby(["mu", "T"], sort=False):
            ax.plot(mu, T, marker="o", color="k", zorder=3)


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
    triplepoints=None,
    transition_temperatures=False,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] | None = None,
    inline_legend=True,
    legend=True,
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

    # triplepoints=None means the caller did not say, so the marks follow
    # whatever else is being drawn -- today, the temperature labels, which
    # annotate them. Passing it explicitly settles it either way: False keeps
    # the marks off even with the labels on, rather than having one keyword
    # quietly switch on what another turned off.
    if transition_temperatures if triplepoints is None else triplepoints:
        _plot_triplepoints(df, ax=ax, variables=variables)

    _set_axis_for(variables[0], df_stable, element, ax)

    ax.set_ylim(df_stable["T"].min(), df_stable["T"].max())
    # Inline labels need the final axis limits to place each label at the centre
    # of its polygon, so this runs after the limits above are set.
    if legend:
        if inline_legend:
            _add_inline_polygon_labels(ax, polys)
        else:
            ax.legend(ncols=2)
    # Last, so the phase labels just placed are obstacles it keeps clear of.
    if transition_temperatures:
        _annotate_transition_temperatures(df, polys, ax=ax, variables=variables)
    ax.set_ylabel("$T$ [K]")


@deprecate(
    alpha="Pass a poly method from landau.poly to poly_method",
    min_c_width="Pass a poly method from landau.poly to poly_method",
    tielines="Use triplepoints instead",
)
def plot_phase_diagram(
    df,
    alpha=0.1,
    element=None,
    min_c_width=1e-2,
    color_override: dict[str, str] = {},
    triplepoints=None,
    transition_temperatures=False,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] | None = None,
    inline_legend=True,
    legend=True,
    ax=None,
    tielines=None,
):
    if tielines is not None:
        triplepoints = tielines
    return _plot_phase_diagram(
        df,
        alpha=alpha,
        element=element,
        min_c_width=min_c_width,
        color_override=color_override,
        triplepoints=triplepoints,
        transition_temperatures=transition_temperatures,
        poly_method=poly_method,
        variables=variables,
        inline_legend=inline_legend,
        legend=legend,
        ax=ax,
    )


def get_phase_colors(phase_names, override: dict[str, str] | None = None):
    if override is None:
        override = {}
    # the default map; cycle the palette so every phase gets a color even
    # past the pastel palette's 10 entries
    color_map = dict(zip(phase_names, cycle(sns.palettes.SEABORN_PALETTES["pastel"])))
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
    triplepoints=None,
    transition_temperatures=False,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp", "segment-fasttsp", "segment-tsp"] | poly.AbstractPolyMethod | None = None,
    inline_legend=True,
    legend=True,
    ax=None,
):
    return _plot_phase_diagram(
        df,
        alpha=alpha,
        element=element,
        color_override=color_override,
        triplepoints=triplepoints,
        transition_temperatures=transition_temperatures,
        poly_method=poly_method,
        variables=["mu", "T"],
        inline_legend=inline_legend,
        legend=legend,
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
    """Subtract reference phase's phi from all phases along scan_col.

    The reference phase must be sampled at every grid point of the scan. Otherwise
    the ``np.interp`` below silently fabricates the missing reference values --
    clamping to a constant beyond the reference's range, or bridging an interior
    gap with a chord -- and corrupts every phase's curve. A frame keeping only
    stable rows triggers this: the reference is then present only within its own
    stability window(s), so recompute the diagram with ``keep_unstable=True``.
    """
    if reference_phase not in df["phase"].values:
        raise ValueError(f"reference_phase {reference_phase!r} not found in data")
    ref = df.loc[df["phase"] == reference_phase, [scan_col, "phi"]].sort_values(scan_col)
    # Refined border rows are excluded: the reference legitimately has no row there
    # (refinement adds rows only for the two transitioning phases), and np.interp
    # interpolates them from the dense grid neighbours on either side.
    grid = df.loc[~df["border"], scan_col] if "border" in df.columns else df[scan_col]
    grid = np.unique(grid)
    missing = np.setdiff1d(grid, ref[scan_col].to_numpy())
    if missing.size:
        raise ValueError(
            f"reference_phase {reference_phase!r} is not sampled across the full "
            f"{scan_col} range ({missing.size} of {grid.size} grid points have no "
            f"reference row); recompute the diagram with keep_unstable=True."
        )
    df["phi"] = df["phi"] - np.interp(df[scan_col], ref[scan_col], ref["phi"])
    return df


# Both cuts plot the same quantity on y, so only the x side is per-cut.
_YLABEL_1D = "Semi-grandcanonical Potential"


@dataclass(frozen=True)
class _Axis1D:
    """Everything that differs between the two 1d phase diagrams.

    The body of :func:`_plot_1d_phase_diagram` is the same for both cuts; only the
    column held fixed along the cut, the x label and the text of the transition
    annotation depend on which variable is scanned.
    """

    fixed_col: str
    fixed_error: str
    xlabel: str
    transition_label: Callable[[float], str]


_AXES_1D = {
    "mu": _Axis1D(
        fixed_col="T",
        fixed_error="Data contains more than one temperature!",
        xlabel="Chemical Potential Difference [eV]",
        transition_label=lambda mu: rf"$\Delta\mu = {mu:.03f}\,\mathrm{{eV}}$",
    ),
    "T": _Axis1D(
        fixed_col="mu",
        fixed_error="Data contains more than one chemical potential!",
        xlabel="Temperature [K]",
        transition_label=lambda T: rf"$T = {T:.0f}\,\mathrm{{K}}$",
    ),
}


def _plot_1d_phase_diagram(
        df,
        scan_col,
        ax=None,
        mark_transitions=True,
        reference_phase=None,
        top_labels=True,
        side_labels=True,
        ylim=None):
    """Draw the semi-grandcanonical potential of every phase along a 1d cut.

    Shared body of :func:`plot_1d_mu_phase_diagram` and
    :func:`plot_1d_T_phase_diagram`; ``scan_col`` selects the cut axis and, via
    :data:`_AXES_1D`, its texts and transition annotation.
    """
    axis = _AXES_1D[scan_col]
    if len(df[axis.fixed_col].unique()) > 1:
        raise ValueError(axis.fixed_error)
    if ax is None:
        fig, ax = plt.subplots()

    df = df.sort_values(scan_col).copy()

    if reference_phase is not None:
        df = _subtract_reference_phase(df, scan_col, reference_phase)

    df["_seg_id"] = _assign_segment_ids(df, scan_col=scan_col)
    sns.lineplot(
        data=_bridge_unstable_segments(df, scan_col=scan_col),
        x=scan_col, y='phi',
        hue='phase', hue_order=sorted(df.phase.unique()),
        style='stable', style_order=[True, False],
        units='_seg_id', estimator=None, errorbar=None,
        ax=ax,
    )

    _add_1d_phase_legend(ax, df, scan_col=scan_col, top_labels=top_labels, side_labels=side_labels, ylim=ylim)

    if mark_transitions and 'border' in df.columns:
        # The marker dot stays in data coords and is simply clipped if the crossing
        # lies outside the window; the labels are placed (and spread) by
        # _place_transition_labels.
        positions, labels = [], []
        for xt, dd in df.query(f"{scan_col}.min()<{scan_col}<{scan_col}.max() and border").groupby(scan_col):
            ft = dd['phi'].iloc[0]
            ax.axvline(xt, color='k', linestyle='dotted', alpha=.5)
            ax.scatter(xt, ft, marker='o', c='k', zorder=10)
            positions.append(xt)
            labels.append(axis.transition_label(xt))
        _place_transition_labels(ax, positions, labels, side="left")

    ax.set_xlabel(axis.xlabel)
    ylabel = f"{_YLABEL_1D} [eV/atom]"
    if reference_phase is not None:
        ylabel = f"{_YLABEL_1D}\nRelative to {reference_phase} [eV/atom]"
    ax.set_ylabel(ylabel)

    return ax


@deprecate(show="No longer read; figure display is left to the caller")
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
            to a mirrored stack on the left, and a phase whose line is pushed
            entirely out of view is not labelled.

    Returns:
        matplotlib.axes.Axes:
            The Axes object with the phase diagram plot.
    """

    return _plot_1d_phase_diagram(
        df,
        "mu",
        ax=ax,
        mark_transitions=mark_transitions,
        reference_phase=reference_phase,
        top_labels=top_labels,
        side_labels=side_labels,
        ylim=ylim,
    )


@deprecate(show="No longer read; figure display is left to the caller")
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
            to a mirrored stack on the left, and a phase whose line is pushed
            entirely out of view is not labelled.

    Returns:
        matplotlib.axes.Axes:
            The Axes object with the phase diagram plot.
    """

    return _plot_1d_phase_diagram(
        df,
        "T",
        ax=ax,
        mark_transitions=mark_transitions,
        reference_phase=reference_phase,
        top_labels=top_labels,
        side_labels=side_labels,
        ylim=ylim,
    )


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
    inline_legend=True,
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
        height: Height of each facet in inches (multiple temperatures only).
        aspect: Width-to-height ratio of each facet (multiple temperatures only).
        color_override: Optional ``dict[name -> color]`` overriding phase colours.
        convex_hull: If True, distinguish stable (solid curves) from metastable
            (faded lines) and overlay the common-tangent segments in black.
            If False, all solution phases render as plain solid curves.
        inline_legend: If True (default), drop the figure legend and label each
            phase just off its line -- solution curves above the centre of their
            largest continuous region, line phases below their dot -- white-outlined
            and colour-matched, with overlapping labels spread apart vertically. If
            False, keep the figure legend box on the right.

    Returns:
        For a single temperature, the current :class:`matplotlib.axes.Axes`
        holding the plain lineplot; no figure is allocated, so pre-create one
        to control its size.  For multiple temperatures, a
        :class:`seaborn.FacetGrid` with one column per temperature (figure via
        ``.fig``, axes via ``.axes``).
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

    # A single temperature does not need a facet grid; draw a plain lineplot onto
    # the current axes (the caller controls figure allocation) and return that axes.
    # Multiple temperatures fan out into a FacetGrid.
    single = len(temperatures) == 1
    if single:
        single_ax = plt.gca()
        if not base_data.empty:
            sns.lineplot(
                data=base_data,
                x="c",
                y="f_excess",
                hue="phase",
                hue_order=sol_hue_order,
                units=units_col,
                palette=sol_palette,
                estimator=None,
                errorbar=None,
                linewidth=2.5,
                ax=single_ax,
            )
        g = None
        axes = [single_ax]
    elif base_data.empty:
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
        axes = list(g.axes.flat)
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
        axes = list(g.axes.flat)

    facet_entries = []  # (ax, [(label, x, y, color), ...]) for inline labelling
    for ax, T_val in zip(axes, temperatures):
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

        if inline_legend:
            # Label each solution phase above its largest continuous region: a
            # free-energy curve is convex, so the space above it is open whether the
            # phase is stable (lower-hull) or metastable (upper arc).  Anchoring on
            # the largest continuous c-region keeps the label on the prominent stretch
            # rather than a sliver (a phase stable only in the dilute corners would
            # otherwise pull its label into a corner).  Within that arc the x-anchor
            # is fanned out by the phase's rank instead of always sitting at the arc
            # centre, so phases whose arcs all span the range don't pile their labels
            # in the middle.  Line phases sit on the lower hull, so their labels go
            # below the dot.  T is constant within a facet, so cluster_T_c splits
            # regions by c alone.  Placement is deferred until refline/limits settle.
            arcs = []
            for pname in sol_hue_order:
                pg = sub_sol[sub_sol["phase"] == pname].dropna(subset=["f_excess"])
                if pg.empty:
                    continue
                region = pg
                if len(region) > 1:
                    seg = cluster_T_c(region, distance_threshold=0.1)
                    extent = region.groupby(seg)["c"].agg(lambda c: c.max() - c.min())
                    region = region.loc[seg == extent.idxmax()]
                region = region.sort_values("c")
                arcs.append((
                    pname, region["c"].to_numpy(), region["f_excess"].to_numpy(),
                    sol_palette.get(pname, "k"),
                ))
            # Spread the anchors left-to-right: order the arcs by their centre, then
            # place each phase's label at the rank-derived fraction of its own arc.
            # The fraction is mapped into the arc's inner 60% so a label never lands
            # on the curve's edge where it turns up.
            arcs.sort(key=lambda a: (0.5 * (a[1][0] + a[1][-1]), a[0]))
            n = len(arcs)
            margin = 0.2  # keep anchors clear of each arc's turning-up ends
            entries = []
            for i, (pname, cs, fs, color) in enumerate(arcs):
                frac = margin + (1 - 2 * margin) * (i + 0.5) / n
                x = cs[0] + frac * (cs[-1] - cs[0])
                y = np.interp(x, cs, fs)
                entries.append((pname, x, y, color, "above"))
            for _, row in sub_lp.drop_duplicates("phase").iterrows():
                entries.append((row["phase"], row["c"], row["f_excess"], palette.get(row["phase"], "k"), "below"))
            facet_entries.append((ax, entries))

    # The single-temperature path draws onto one axes whose legend lives on the axes
    # itself; the FacetGrid path keeps its legend on ``g._legend``.  Resolve both here.
    legend = g._legend if g is not None else axes[0].get_legend()
    figure = g.figure if g is not None else axes[0].figure

    if inline_legend:
        # Inline labels replace the legend box; drop seaborn's legend.
        if legend is not None:
            legend.remove()
    # Add line phases to the figure legend.
    elif not df_lp.empty:
        lp_handles = [
            mlines.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=palette.get(n, "k"), markersize=8, label=n,
            )
            for n in sorted(line_phase_names)
        ]
        if legend is not None:
            existing_handles = list(legend.legend_handles)
            existing_labels = [t.get_text() for t in legend.texts]
            legend.remove()
        else:
            existing_handles = []
            existing_labels = []
        figure.legend(
            existing_handles + lp_handles,
            existing_labels + [h.get_label() for h in lp_handles],
            title="phase",
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            borderaxespad=0,
        )

    if g is not None:
        g.refline(y=0)
        g.set_titles("T = {col_name:.0f} K")
        g.set(xlabel="Concentration", ylabel="Free Energy of Formation")
    else:
        ax = axes[0]
        ax.axhline(0, color=".5", linestyle="--")
        ax.set(xlabel="Concentration", ylabel="Free Energy of Formation")

    # Place inline labels last so they see the final axis limits (refline/relplot).
    for ax, entries in facet_entries:
        _add_inline_curve_labels(ax, entries)

    return g if g is not None else axes[0]
