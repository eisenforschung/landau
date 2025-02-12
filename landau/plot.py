import math

from matplotlib.patches import Polygon
import shapely
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from .calculate import (
        get_transitions
)

def make_concave_poly(dd, alpha=.1, plot=False, min_c_width=1e-3):
    # concave hull algo seems more stable when both variables are of the same order
    # since c in [0, 1]; so let's rescale T relative to the extrema as well
    Tmin = dd['T'].min()
    Tmax = dd['T'].max()

    # pp = dd.sort_values('mu')[['mu', 'T']].to_numpy()
    pp = dd.sort_values('c')[['c', 'T']].to_numpy()
    if plot:
        plt.scatter(*pp.T, marker='.')
    pp[:, 0] = np.clip(pp[:, 0], -1e3, 1e3)
    pp[:, 1] -= Tmin
    pp[:, 1] /= (Tmax-Tmin)
    shape = shapely.concave_hull(shapely.MultiPoint(pp), ratio=alpha)
    # check for c-degenerate line phase
    if isinstance(shape, shapely.LineString):
        coords = np.asarray(shape.coords)
        if np.allclose(coords[:, 0], coords[0, 0]):
            match coords[0, 0]:
                case 0.0:
                    bias = +min_c_width/2
                case 1.0:
                    bias = -min_c_width/2
                case _:
                    bias = 0
            # artificially widen the line phase in c, so that we can make a
            # "normal" polygon for it.
            coords = np.concatenate([
                # inverting the order for the second half of the array, makes
                # it so that the points are in the correct order for the
                # polygon
                    coords[::+1] - [min_c_width/2, 0],
                    coords[::-1] + [min_c_width/2, 0]
                ], axis=0
            )
            coords[:, 0] += bias
    else:
        coords = np.asarray(shape.exterior.coords)
    coords[:,1] *= (Tmax-Tmin)
    coords[:,1] += Tmin
    # doesn't work nicely because of our regular grid; or rather a regular grid
    # with a few "refined" points on top.  This seems to confuse the
    # interpolators a lot
    # coords[:,0] = si.griddata(pp, dd['c'], coords, method='nearest', rescale=True)
    return Polygon(coords)

def sort_segments(df, x_col='c', y_col='T', segment_label='border_segment'):
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

    total_center_of_mass = df[[x_col, y_col]].mean().values

    # Step 1: PCA Projection
    def pca_projection(group):
        pca = PCA(n_components=1)
        projected = pca.fit_transform(group[[x_col, y_col]])
        group['projected'] = projected
        return group.sort_values('projected').copy().drop('projected', axis='columns').reset_index(drop=True)

    segments = []
    for label, dd in df.groupby(segment_label):
        segments.append(pca_projection(dd))

    def start(s):
        return s.iloc[0][[x_col, y_col]]
    def end(s):
        return s.iloc[-1][[x_col, y_col]]
    def dist(p1, p2):
        return np.linalg.norm( (p2 - p1) / np.ptp(df[[x_col, y_col]], axis=0))
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

def make_poly(td, min_c_width=1e-3):
    """
    Requires a grouped dataframe from find_transitions (by phase).
    """
    if np.ptp(td.c) < min_c_width:
        meanc = td.c.mean()
        Tmin = td['T'].min()
        Tmax = td['T'].max()
        return Polygon([
            [meanc - min_c_width/2, Tmin],
            [meanc + min_c_width/2, Tmin],
            [meanc + min_c_width/2, Tmax],
            [meanc - min_c_width/2, Tmax],
        ])
    sd = sort_segments(td)
    return Polygon( np.transpose([sd.c, sd['T']]) )

def plot_phase_diagram(
        df, alpha=.1, element=None, min_c_width=5e-3,
        color_override: dict[str, str] = {},
        tielines=False
):
    df = df.query('stable')

    # the default map
    color_map = dict(zip(df.phase.unique(), sns.palettes.SEABORN_PALETTES['pastel']))
    # disregard overriden phases that are not present
    color_override = {p: c for p, c in color_override.items() if p in color_map}
    # if the override uses the same colors as the default map, multiple phases
    # would be mapped to the same color; so instead let's update the color map of phases that would
    # use the same color as a phase in the override to use the default colors of the overriden phases
    # instead
    duplicates_map = {c: color_map[o] for o, c in color_override.items()}
    diff = {k: duplicates_map[c] for k, c in color_map.items() if c in duplicates_map}
    color_map.update(diff | color_override)

    if 'refined' in df.columns:
        tdf = get_transitions(df)
        polys = tdf.groupby('phase').apply(
                make_poly, min_c_width=min_c_width,
        )
    else:
        polys = df.groupby('phase').apply(
                make_concave_poly, alpha=alpha, min_c_width=min_c_width,
        )

    ax = plt.gca()
    for i, (phase, p) in enumerate(polys.items()):
        p.zorder = -1
        p.set_color(color_map[phase])
        p.set_edgecolor('k')
        # p.set_alpha(.8)
        p.set_label(polys.index[i])
        ax.add_patch(p)

    if tielines:
        # TODO: quite buggy and not nice; can benefit a lot from
        # get_transitions
        if 'refined' in df.columns:
            def plot_tie(dd):
                Tmin = dd['T'].min()
                Tmax = dd['T'].max()
                di = dd.query('T==@Tmin')
                da = dd.query('T==@Tmax')
                # "artificial" segment at the border of diagram
                # we just want to plot triple lines? so #phases==3
                if len(dd.phase.unique()) in [1, 2]:
                    return
                plt.hlines(Tmin, di.c.min(), di.c.max(),
                           color='k', zorder=-2,
                           alpha=0.5, lw=4)
                plt.hlines(Tmax, da.c.min(), da.c.max(),
                           color='k', zorder=-2,
                           alpha=0.5, lw=4)
            # FIXME: WARNING reuses local var define in if branch
            tdf.groupby('border_segment').apply(plot_tie)
        else:
            # count the numbers of distinct phases per T, it changes there *must* be a triple
            # point, draw tie lines only there
            # TODO: figure out how to only draw them between the involved phases not over the whole conc range
            # the refined data points mess this up, because the phases are no longer on
            # the same grid
            chg = df.groupby('T').size().diff()
            T_tie = chg.loc[chg!=0].index[1:] # skip first temp
            def plot_tie(dd):
                if dd['T'].iloc[0].round(3) not in T_tie.round(3):
                    return
                if len(dd)!=2:
                    return
                cl, cr = sorted(dd.c)
                plt.plot([cl, cr], dd['T'], color='k',
                        zorder=-2,
                        alpha=0.5,
                        lw=4)
            df.groupby(['T','mu']).apply(plot_tie)

    plt.xlim(0, 1)
    plt.ylim(df['T'].min(), df['T'].max())
    plt.legend(ncols=2)
    if element is not None:
        plt.xlabel(rf'$c_\mathrm{{{element}}}$')
    else:
        plt.xlabel('$c$')
    plt.ylabel('$T$ [K]')
