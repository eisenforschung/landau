from typing import Literal
from warnings import warn

from pyiron_snippets.deprecate import deprecate

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .calculate import calc_phase_diagram, get_transitions, cluster, cluster_T_c, _join_phase_unit
import landau.poly as poly


__all__ = [
    "plot_phase_diagram",
    "plot_mu_phase_diagram",
    "plot_1d_mu_phase_diagram",
    "plot_1d_T_phase_diagram",
]


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
    ax=None,
):
    return _plot_phase_diagram(
        df,
        alpha=alpha,
        element=element,
        color_override=color_override,
        poly_method=poly_method,
        variables=["mu", "T"],
        ax=ax,
    )

def plot_1d_mu_phase_diagram(
        df,
        ax=None, 
        show=True, 
        mark_transitions=True):
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

    Returns:
        matplotlib.axes.Axes: 
            The Axes object with the phase diagram plot.
    """

    if len(df['T'].unique()) > 1:
        raise ValueError("data contains more than one temperature!")
    if ax is None:
        fig, ax = plt.subplots()

    if 'border' not in df.columns:
        sns.lineplot(
            data=df,
            x='mu', y='phi',
            hue='phase',
            style='stable', style_order=[True, False],
            ax=ax,
        )
        return ax

    df_sorted = df.sort_values("mu").reset_index(drop=True)
    border_rows = df_sorted.query("border")
    border_mus = np.sort(border_rows['mu'])

    split_points = np.concatenate(([-np.inf], border_mus, [np.inf]))

    for i in range(len(split_points) - 1):
        left = split_points[i]
        right = split_points[i + 1]

        seg = df_sorted.query("@left < mu <= @right")
        if not seg.empty:
            sns.lineplot(
                data=seg,
                x='mu', y='phi',
                hue='phase', hue_order=sorted(df.phase.unique()),
                style='stable', style_order=[True, False],
                legend='auto' if i == 0 else False,
                ax=ax,
            )

    dfa = np.ptp(df['phi'].dropna())
    dfm = np.ptp(df['mu'].dropna())

    if mark_transitions and 'border' in df.columns:
        for mt, dd in df.query("mu.min()<mu<mu.max() and border").groupby("mu"):
            ft = dd['phi'].iloc[0]
            ax.axvline(mt, color='k', linestyle='dotted', alpha=.5)
            ax.scatter(mt, ft, marker='o', c='k', zorder=10)

            ax.text(mt - .05 * dfm, ft - dfa * .1, rf"$\Delta\mu = {mt:.03f}\,\mathrm{{eV}}$",
                    rotation='vertical', ha='center', va='top')
    ax.set_xlabel("Chemical Potential Difference [eV]")
    ax.set_ylabel("Semi-grandcanonical Potential [eV/atom]")

    return ax

def plot_1d_T_phase_diagram(
        df, 
        ax=None, 
        mark_transitions=True,
        show=True
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

    Returns:
        matplotlib.axes.Axes: 
            The Axes object with the phase diagram plot.
    """

    if len(df.mu.unique()) > 1:
        raise ValueError("Data contains more than one chemical potential!")

    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x='T', y='phi',
        hue='phase', hue_order=sorted(df.phase.unique()),
        style='stable', style_order=[True, False],
        ax=ax,
    )

    if 'border' not in df.columns:
        return ax

    dfa = np.ptp(df['phi'].dropna())
    dft = np.ptp(df['T'].dropna())

    if mark_transitions and 'border' in df.columns:
        for Tt, dd in df.query("T.min()<T<T.max() and border").groupby("T"):
            ft = dd['phi'].iloc[0]
            ax.axvline(Tt, color='k', linestyle='dotted', alpha=.5)
            ax.scatter(Tt, ft, marker='o', c='k', zorder=10)

            ax.text(Tt + .05 * dft, ft + dfa * .1, rf"$T = {Tt:.0f}\,\mathrm{{K}}$", rotation='vertical', ha='center')

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Semi-grandcanonical potential [eV/atom]")

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
        base_data = cluster_phase(base_data, distance_threshold=0.1)
        units_col = "phase_id"
    else:
        base_data = df_sol
        units_col = None

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
            unstable = sub_sol[~sub_sol["stable"]]
            for pname, grp in unstable.groupby("phase"):
                grp_sorted = grp.sort_values("c")
                diffs = grp_sorted["c"].diff().fillna(0)
                pos_diffs = diffs[diffs > 0]
                gap_thr = max(0.05, 5 * pos_diffs.quantile(0.9)) if not pos_diffs.empty else 0.05
                seg_ids = (diffs > gap_thr).cumsum()
                color = sol_palette.get(pname, "gray")
                for _, seg in grp_sorted.groupby(seg_ids):
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
