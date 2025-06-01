from typing import Literal

from matplotlib.patches import Polygon
import shapely
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from .calculate import get_transitions, cluster


def make_concave_poly(dd, alpha=0.1, min_c_width=1e-3, variables=["c", "T"]):
    # concave hull algo seems more stable when both variables are of the same order

    if "border" in dd.columns:
        dd = dd.query("border")

    pp = dd.sort_values(variables[0])[variables].to_numpy()
    pp = np.unique(pp[np.isfinite(pp).all(axis=-1)], axis=0)

    refnorm = {}
    for i, var in enumerate(variables):
        refnorm[var] = pp[:, i].min(), (np.ptp(pp[:, i]) or 1)
        pp[:, i] -= refnorm[var][0]
        pp[:, i] /= refnorm[var][1]
    points = shapely.MultiPoint(pp)
    # check for c-degenerate line phase
    shape = shapely.convex_hull(points)
    if variables[0] == "c" and isinstance(shape, shapely.LineString):
        coords = np.asarray(shape.coords)
        if np.allclose(coords[:, 0], coords[0, 0]):
            match coords[0, 0]:
                case 0.0:
                    bias = +min_c_width / 2
                case 1.0:
                    bias = -min_c_width / 2
                case _:
                    bias = 0
            # artificially widen the line phase in c, so that we can make a
            # "normal" polygon for it.
            coords = np.concatenate(
                [
                    # inverting the order for the second half of the array, makes
                    # it so that the points are in the correct order for the
                    # polygon
                    coords[::+1] - [min_c_width / 2, 0],
                    coords[::-1] + [min_c_width / 2, 0],
                ],
                axis=0,
            )
            coords[:, 0] += bias
    else:
        shape = shapely.concave_hull(points, ratio=alpha)
        coords = np.asarray(shape.exterior.coords)
    for i, var in enumerate(variables):
        coords[:, i] *= refnorm[var][1]
        coords[:, i] += refnorm[var][0]
    return Polygon(coords)


def sort_segments(df, x_col="c", y_col="T", segment_label="border_segment"):
    """
    Sorts the points in df such that they can be used as the bounding polygon of a phase in a binary diagram.

    Assumptions:
    1. df contains only data on a single, coherent phase, i.e. the c/T points are "connected"

    Algorithm:
    1. Subset the data according to the column given by `segment_label`.  These should label connected points on a single two-phase boundary. Such a subset is called a segment.
    2. Sort points in each segment by a 1D PCA. (Sorting by c or T alone fails when the segment is either vertical or horizontal.)
    3. Sort the segments so that they "easily" fit together:
        a. Pick the segment with minimum `x` as the "head"
        b. Go over all other segments, s, and:
            b0. Get the distance from endpoint of "head" to either the starting point or the end point of s
            b1. if the distance to the end point is shorter than to the starting point, invert order of s
            b2. return the minimum of both distances
        c. the segment with smallest distance to the current "head" is the next "head" and removed from the pool of segments
        d. break if no segments left
    4. return the segments in the order they were picked as "head"s.

    a) is a heuristic for "normal" phase diagrams, starting from the left (or right) we can often make a full circle.
    Picking a random segments breaks for phases that are stable at the lower or upper edge of the diagram, where we technically do not compute
    a "segment".  A "proper" fix would be to modify b to allow joining also to the start of "head" rather than just the end.
    """

    com = df[[x_col, y_col]].mean()
    norm = np.ptp(df[[x_col, y_col]], axis=0).values


    # Step 1: PCA Projection
    def pca_projection(group):
        # avoid warnings when clustering only found one or two points
        if len(group) < 2:
            return group
        pca = PCA(n_components=1)
        projected = pca.fit_transform(group[[x_col, y_col]])
        group["projected"] = projected
        return group.sort_values("projected").copy().drop("projected", axis="columns").reset_index(drop=True)

    segments = []
    for label, dd in df.groupby(segment_label):
        segments.append(pca_projection(dd))

    # initial sorting by center of mass angle
    segments = sorted(
            segments,
            key=lambda s: np.arctan2( (s[y_col].mean() - com[y_col]) / norm[1],
                                      (s[x_col].mean() - com[x_col]) / norm[0])
    )

    def start(s):
        return s.iloc[0][[x_col, y_col]]

    def end(s):
        return s.iloc[-1][[x_col, y_col]]

    def dist(p1, p2):
        return np.linalg.norm((p2 - p1) / norm)

    def flip(s):
        s.reset_index(drop=True, inplace=True)
        s.loc[:] = s.loc[::-1].reset_index(drop=True)
        return s

    head, *remaining = sorted(segments, key=lambda s: s[x_col].min())

    def find_distance(head, segment):
        head2tail = dist(end(head), start(segment))
        tail2tail = dist(end(head), end(segment))
        if tail2tail < head2tail:
            flip(segment)
            return tail2tail
        else:
            return head2tail

    segments = [head]
    while len(remaining) > 0:
        head, *remaining = sorted(remaining, key=lambda s: find_distance(head, s))
        segments.append(head)

    return pd.concat(segments, ignore_index=True)


def make_poly(td, min_c_width=1e-3, variables=["c", "T"]):
    """
    Requires a grouped dataframe from get_transitions (by phase).
    """
    if "c" in variables and np.ptp(td.c) < min_c_width:
        meanc = td.c.mean()
        Tmin = td["T"].min()
        Tmax = td["T"].max()
        return Polygon(
            [
                [meanc - min_c_width / 2, Tmin],
                [meanc + min_c_width / 2, Tmin],
                [meanc + min_c_width / 2, Tmax],
                [meanc - min_c_width / 2, Tmax],
            ]
        )
    td = td.loc[ np.isfinite(td[variables[0]]) & np.isfinite(td[variables[1]]) ]
    sd = sort_segments(td, x_col=variables[0], y_col=variables[1])
    # sd = sd.loc[ np.isfinite(sd[variables[0]]) & np.isfinite(sd[variables[1]]) ]
    return Polygon(np.transpose([sd[v] for v in variables]))

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

def plot_phase_diagram(
    df, alpha=0.1, element=None, min_c_width=5e-3, color_override: dict[str, str] = {}, tielines=False,
    poly_method: Literal["concave", "segments"] = 'concave',
):
    df = df.query("stable").copy()

    # the default map
    color_map = dict(zip(df.phase.unique(), sns.palettes.SEABORN_PALETTES["pastel"]))
    # disregard overriden phases that are not present
    color_override = {p: c for p, c in color_override.items() if p in color_map}
    # if the override uses the same colors as the default map, multiple phases
    # would be mapped to the same color; so instead let's update the color map of phases that would
    # use the same color as a phase in the override to use the default colors of the overriden phases
    # instead
    duplicates_map = {c: color_map[o] for o, c in color_override.items()}
    diff = {k: duplicates_map[c] for k, c in color_map.items() if c in duplicates_map}
    color_map.update(diff | color_override)

    df = cluster_phase(df)
    if "refined" in df.columns and poly_method == "segments":
        df.loc[:, "phase"] = df.phase_id
        tdf = get_transitions(df)
        tdf["phase_unit"] = tdf.phase.str.rsplit('_', n=1).map(lambda x: int(x[1]))
        tdf["phase"] = tdf.phase.str.rsplit('_', n=1).map(lambda x: x[0])
        polys = tdf.groupby(["phase", "phase_unit"]).apply(
            make_poly,
            min_c_width=min_c_width,
        )
    else:
        polys = df.groupby(["phase", "phase_unit"]).apply(
            make_concave_poly,
            alpha=alpha,
            min_c_width=min_c_width,
        )

    ax = plt.gca()
    for i, (phase, p) in enumerate(polys.items()):
        p.zorder = 1/p.get_extents().size.prod()
        if isinstance(phase, tuple):
            phase, rep = phase
        else:
            rep = 0
        p.set_color(color_map[phase])
        p.set_edgecolor("k")
        p.set_label(phase + '\'' * rep)
        # p.set_label(polys.index[i])
        ax.add_patch(p)

    if tielines:
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
                plt.hlines(Tmin, di.c.min(), di.c.max(), color="k", zorder=-2, alpha=0.5, lw=4)
                # current marvin to past marvin: Why is that even necessary?
                if Tmin != Tmax:
                    plt.hlines(Tmax, da.c.min(), da.c.max(), color="k", zorder=-2, alpha=0.5, lw=4)

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
                plt.plot([cl, cr], dd["T"], color="k", zorder=-2, alpha=0.5, lw=4)

            df.groupby(["T", "mu"]).apply(plot_tie)

    plt.xlim(0, 1)
    plt.ylim(df["T"].min(), df["T"].max())
    plt.legend(ncols=2)
    if element is not None:
        plt.xlabel(rf"$c_\mathrm{{{element}}}$")
    else:
        plt.xlabel("$c$")
    plt.ylabel("$T$ [K]")

def get_phase_colors(phase_names, override: dict[str, str] | None = None):
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

def plot_mu_phase_diagram(
    df, alpha=0.1, element=None, color_override: dict[str, str] = {},
    poly_method: Literal["concave", "segments"] = 'concave',
):
    df = df.query("stable").copy()

    color_map = get_phase_colors(df.phase.unique(), color_override)

    df = cluster_phase(df)
    if "refined" in df.columns and poly_method == "segments":
        df.loc[:, "phase"] = df.phase_id
        tdf = get_transitions(df)
        tdf["phase_unit"] = tdf.phase.str.rsplit('_', n=1).map(lambda x: int(x[1]))
        tdf["phase"] = tdf.phase.str.rsplit('_', n=1).map(lambda x: x[0])
        polys = tdf.groupby(["phase", "phase_unit"]).apply(
            make_poly,
            variables=["mu", "T"],
        )
    else:
        polys = df.groupby("phase").apply(
            make_concave_poly,
            alpha=alpha,
            variables=["mu", "T"],
        )

    ax = plt.gca()
    for i, (phase, p) in enumerate(polys.items()):
        p.zorder = 1/p.get_extents().size.prod()
        if isinstance(phase, tuple):
            phase, rep = phase
        else:
            rep = 0
        p.set_color(color_map[phase])
        p.set_edgecolor("k")
        p.set_label(phase + '\'' * rep)
        ax.add_patch(p)

    mus = df["mu"].unique()
    mus = mus[np.isfinite(mus)]
    plt.xlim(mus.min(), mus.max())
    plt.ylim(df["T"].min(), df["T"].max())
    plt.legend(ncols=2)
    plt.xlabel(r"$\Delta\mu$ [eV]")
    plt.ylabel("$T$ [K]")

def plot_1d_mu_phase_diagram(df):
    """Plot a one dimensional isothermal phase diagram."""
    if len(df['T'].unique()) > 1:
        raise ValueError("data contains more than one temperature!")
    sns.lineplot(
        data=df,
        x='mu', y='phi',
        hue='phase',
        style='stable', style_order=[True, False],
    )
    if 'border' not in df.columns: return

    dfa = np.ptp(df['phi'].dropna())
    dfm = np.ptp(df['mu'].dropna())

    for mt, dd in df.query("mu.min()<mu<mu.max() and border").groupby("mu"):
        ft = dd['phi'].iloc[0]
        plt.axvline(mt, color='k', linestyle='dotted', alpha=.5)
        plt.scatter(mt, ft, marker='o', c='k', zorder=10)

        plt.text(mt - .05 * dfm, ft - dfa * .1, rf"$\Delta\mu = {mt:.03f}\,\mathrm{{eV}}$",
                 rotation='vertical', ha='center', va='top')
    plt.xlabel("Chemical Potential Difference [ev]")
    plt.ylabel("Semi-grandcanonical Potential [eV/atom]")

def plot_1d_T_phase_diagram(df):
    """Plot a one dimensional equipotential phase diagram."""
    if len(df.mu.unique()) > 1:
        raise ValueError("data contains more than one chemical potential!")
    sns.lineplot(
        data=df,
        x='T', y='phi',
        hue='phase',
        style='stable', style_order=[True, False],
    )
    if 'border' not in df.columns: return

    dfa = np.ptp(df['phi'].dropna())
    dft = np.ptp(df['T'].dropna())

    for Tt, dd in df.query("T.min()<T<T.max() and border").groupby("T"):
        ft = dd['phi'].iloc[0]
        plt.axvline(Tt, color='k', linestyle='dotted', alpha=.5)
        plt.scatter(Tt, ft, marker='o', c='k', zorder=10)

        plt.text(Tt + .05 * dft, ft + dfa * .1, rf"$T = {Tt:.0f}\,\mathrm{{K}}$", rotation='vertical', ha='center')
    plt.xlabel("Temperature [K]")
    plt.ylabel("Semi-grandcanonical potential [eV/atom]")
