from typing import Literal
from warnings import warn

from pyiron_snippets.deprecate import deprecate

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .calculate import get_transitions, cluster
import landau.poly as poly


__all__ = [
    "plot_phase_diagram",
    "plot_mu_phase_diagram",
    "plot_1d_mu_phase_diagram",
    "plot_1d_T_phase_diagram",
    "get_polygons",
    "plot_polygons",
]


def cluster_phase(df):
    """Cluster the stable, single phase regions.

    When a (e.g solid solution) phase has multiple disconnected regions of stability, the make_poly and
    make_concave_poly functions give wrong results, because they draw a single polygon.
    Instead this function adds two new columns `phase_unit` and `phase_id` and the latter will always refer to only a
    single connected stability region.  `phase_unit` enumerates disconnected regions of one phase.
    """
    df["phase_unit"] = df.groupby("phase", group_keys=False).apply(
            cluster, use_mu=False,
            include_groups=False
    )
    df["phase_id"] = df[["phase", "phase_unit"]].apply(
        lambda r: "_".join(map(str, r.tolist())), axis="columns"
    )
    return df

def get_polygons(
    df,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] = ["c", "T"],
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
        **kwargs:
            Passed to poly.handle_poly_method.

    Returns:
        pandas.Series:
            The constructed polygons, indexed by phase and phase_unit.
    """
    df = df.query("stable").copy()
    df = cluster_phase(df)
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
        tdf.groupby("border_segment").apply(plot_tie)
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

        df.groupby(["T", "mu"]).apply(plot_tie)


def _plot_phase_diagram(
    df,
    alpha=0.1,
    element=None,
    min_c_width=1e-2,
    color_override: dict[str, str] = {},
    tielines=False,
    poly_method: Literal["concave", "segments", "fasttsp", "tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] = ["c", "T"],
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    df_stable = df.query("stable")
    color_map = get_phase_colors(df_stable.phase.unique(), color_override)

    polys = get_polygons(df, poly_method=poly_method, variables=variables, min_c_width=min_c_width, alpha=alpha)

    plot_polygons(polys, color_map, ax=ax)

    if tielines and variables[0] == "c":
        _plot_tielines(df, ax=ax)

    if variables[0] == "c":
        ax.set_xlim(0, 1)
        if element is not None:
            ax.set_xlabel(rf"$c_\mathrm{{{element}}}$")
        else:
            ax.set_xlabel("$c$")
    elif variables[0] == "mu":
        mus = df_stable["mu"].unique()
        mus = mus[np.isfinite(mus)]
        if len(mus) > 0:
            ax.set_xlim(mus.min(), mus.max())
        ax.set_xlabel(r"$\Delta\mu$ [eV]")

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
    poly_method: Literal["concave", "segments", "fasttsp", "tsp"] | poly.AbstractPolyMethod | None = None,
    variables: list[str] = ["c", "T"],
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
    # the default map
    color_map = dict(zip(phase_names, sns.palettes.SEABORN_PALETTES["pastel"]))
    if override is None:
        return color_map
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
    poly_method: Literal["concave", "segments", "fasttsp", "tsp"] | poly.AbstractPolyMethod | None = None,
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
        show (bool, optional): 
            If True, the plot is displayed immediately. Defaults to True.
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
        )
        return

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
                legend='auto' if i == 0 else False
            )

    dfa = np.ptp(df['phi'].dropna())
    dfm = np.ptp(df['mu'].dropna())

    if mark_transitions and 'border' in df.columns:
        for mt, dd in df.query("mu.min()<mu<mu.max() and border").groupby("mu"):
            ft = dd['phi'].iloc[0]
            plt.axvline(mt, color='k', linestyle='dotted', alpha=.5)
            plt.scatter(mt, ft, marker='o', c='k', zorder=10)

            plt.text(mt - .05 * dfm, ft - dfa * .1, rf"$\Delta\mu = {mt:.03f}\,\mathrm{{eV}}$",
                    rotation='vertical', ha='center', va='top')
    plt.xlabel("Chemical Potential Difference [eV]")
    plt.ylabel("Semi-grandcanonical Potential [eV/atom]")

    if show==True:
        plt.show()

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
        show (bool, optional): 
            If True, the plot is displayed immediately. Defaults to True.

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
    )

    if 'border' not in df.columns: return

    dfa = np.ptp(df['phi'].dropna())
    dft = np.ptp(df['T'].dropna())

    if mark_transitions and 'border' in df.columns:
        for Tt, dd in df.query("T.min()<T<T.max() and border").groupby("T"):
            ft = dd['phi'].iloc[0]
            plt.axvline(Tt, color='k', linestyle='dotted', alpha=.5)
            plt.scatter(Tt, ft, marker='o', c='k', zorder=10)

            plt.text(Tt + .05 * dft, ft + dfa * .1, rf"$T = {Tt:.0f}\,\mathrm{{K}}$", rotation='vertical', ha='center')

    plt.xlabel("Temperature [K]")
    plt.ylabel("Semi-grandcanonical potential [eV/atom]")

    if show==True:
        plt.show()

    return ax
