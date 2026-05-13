"""Demonstrator: option 1 (pairwise half-overlap subtraction) vs option 2
(Voronoi clipping) for symmetrically trimming buffered phase polygons.

We make two scenes:

    c-T scene -- like a real T-vs-concentration diagram: there is *empty
                 space* between phases along the c axis (line phases at
                 fixed compositions); the buffer is what makes them
                 visible at all.  Originals do not touch -- they have
                 gaps -- so option 1's "subtract neighbour's un-buffered
                 shape" leaves a thin overlap strip.

    T-mu scene -- like a T-vs-chemical-potential diagram: the phase
                 regions tile the whole plane with no gaps.  Originals
                 touch at sharp boundaries, so option 1 and option 2
                 produce identical, crisp seams.
"""

import numpy as np
import shapely
from shapely.ops import voronoi_diagram, unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------


def scene_cT():
    """Three near-line phases at fixed c, separated by empty space."""
    # narrow vertical strips with small width; gaps between them in c
    a = shapely.Polygon([(0.05, 200), (0.07, 200), (0.07, 900), (0.05, 900)])
    b = shapely.Polygon([(0.30, 250), (0.32, 250), (0.32, 850), (0.30, 850)])
    c = shapely.Polygon([(0.55, 200), (0.57, 200), (0.57, 900), (0.55, 900)])
    return {"alpha": a, "beta": b, "gamma": c}, ("c", "T")


def scene_Tmu():
    """Three phase regions that tile a T-mu plane with no gaps."""
    a = shapely.Polygon([(-0.5, 200), (-0.1, 200), (-0.1, 900), (-0.5, 900)])
    b = shapely.Polygon([(-0.1, 200), (0.2, 200), (0.2, 900), (-0.1, 900)])
    c = shapely.Polygon([(0.2, 200), (0.6, 200), (0.6, 900), (0.2, 900)])
    return {"alpha": a, "beta": b, "gamma": c}, ("mu", "T")


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


def buffer_all(polys, r_x, r_y):
    """Buffer in scaled space (so x and y get different absolute buffers).

    Phase-diagram axes have wildly different units (c ~ O(1), T ~ O(100));
    we mimic landau's StandardScaler trick by buffering in a scaled frame.
    """
    sx, sy = 1.0 / max(r_x, 1e-9), 1.0 / max(r_y, 1e-9)
    out = {}
    for k, p in polys.items():
        scaled = shapely.affinity.scale(p, xfact=sx, yfact=sy, origin=(0, 0))
        buf = scaled.buffer(0.5)  # 0.5 in scaled units
        out[k] = shapely.affinity.scale(buf, xfact=1 / sx, yfact=1 / sy, origin=(0, 0))
    return out


def option1(buffered, originals):
    """Subtract each neighbour's *un-buffered* original from every polygon."""
    out = {}
    for k, a in buffered.items():
        trimmed = a
        for k2, b_orig in originals.items():
            if k2 == k:
                continue
            trimmed = trimmed.difference(b_orig)
        out[k] = trimmed
    return out


def option2(buffered, originals, grid_res=400):
    """Generalized-Voronoi clip via rasterized distance fields.

    For every pixel in a fine grid over the bounding box, label it with
    the phase whose original is closest.  Then for each phase build the
    polygon of pixels labelled with that phase and intersect with the
    buffered polygon.  This is the "honest" generalized Voronoi diagram
    of the original polygons (not just of sampled boundary points).
    """
    union_buf = unary_union(list(buffered.values()))
    minx, miny, maxx, maxy = union_buf.buffer(1e-3).bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    minx -= pad_x; maxx += pad_x; miny -= pad_y; maxy += pad_y
    xs = np.linspace(minx, maxx, grid_res)
    ys = np.linspace(miny, maxy, grid_res)
    X, Y = np.meshgrid(xs, ys)
    coords = np.column_stack([X.ravel(), Y.ravel()])

    # scale into a frame where x and y span the same range so the
    # "closest original" makes geometric sense across mixed units
    sx = 1.0 / (maxx - minx)
    sy = 1.0 / (maxy - miny)
    scaled_coords = coords * np.array([sx, sy])

    keys = list(originals.keys())
    # signed distance: <0 inside, >0 outside; we want unsigned-from-boundary
    # but for "closest original" we want shapely.distance which returns 0
    # for points inside any original, and positive distance otherwise.
    # rescale the originals into the same frame.
    scaled_orig = {
        k: shapely.affinity.scale(p, xfact=sx, yfact=sy, origin=(0, 0))
        for k, p in originals.items()
    }
    dists = np.empty((len(keys), len(coords)))
    for i, k in enumerate(keys):
        op = scaled_orig[k]
        dists[i] = [op.distance(shapely.Point(x, y)) for x, y in scaled_coords]
    label = np.argmin(dists, axis=0).reshape(grid_res, grid_res)

    out = {}
    for i, k in enumerate(keys):
        mask = (label == i).astype(np.uint8)
        # convert mask to polygon via marching-squares-ish trick using
        # matplotlib contour
        import matplotlib.pyplot as _plt
        fig = _plt.figure()
        cs = _plt.contourf(xs, ys, mask, levels=[0.5, 1.5])
        _plt.close(fig)
        polys = []
        for path in cs.get_paths():
            for poly_verts in path.to_polygons():
                if len(poly_verts) >= 3:
                    polys.append(shapely.Polygon(poly_verts))
        if not polys:
            out[k] = shapely.Polygon()
            continue
        region = unary_union(polys)
        out[k] = buffered[k].intersection(region)
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

COLORS = {"alpha": "#9bbcdb", "beta": "#f7c59f", "gamma": "#b6d7a8"}


def add_shape(ax, geom, color, alpha=1.0):
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    for g in geoms:
        if isinstance(g, shapely.Polygon) and not g.is_empty:
            coords = np.asarray(g.exterior.coords)
            ax.add_patch(MplPolygon(coords, facecolor=color, edgecolor="k",
                                    linewidth=0.8, alpha=alpha))


def render(scene_name, originals, buffered, opt1, opt2, axis_labels, fname):
    xs = [p.bounds for p in buffered.values()]
    xmin = min(b[0] for b in xs) - 0.02
    xmax = max(b[2] for b in xs) + 0.02
    ymin = min(b[1] for b in xs) - 20
    ymax = max(b[3] for b in xs) + 20

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["buffered (overlap)", "option 1: subtract neighbour original",
              "option 2: Voronoi clip"]
    sets = [buffered, opt1, opt2]
    for ax, title, s in zip(axes, titles, sets):
        for k, geom in s.items():
            add_shape(ax, geom, COLORS[k], alpha=0.85)
        # overlay originals as dashed outlines for reference
        for k, o in originals.items():
            x, y = o.exterior.xy
            ax.plot(x, y, color="k", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(title)
        ax.set_aspect("auto")
    fig.suptitle(f"{scene_name}  (dashed = original un-buffered shapes)")
    fig.tight_layout()
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    print(f"wrote {fname}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run():
    for name, (scene, axes) in [("c-T (sparse: gaps between phases)", scene_cT()),
                                ("T-mu (dense: phases tile the plane)", scene_Tmu())]:
        originals, axis_labels = scene, axes
        # buffer amounts representative of landau defaults (min_c_width=0.01)
        # buffer in scaled space using the data range as the scale
        x_range = max(p.bounds[2] for p in originals.values()) - min(p.bounds[0] for p in originals.values())
        y_range = max(p.bounds[3] for p in originals.values()) - min(p.bounds[1] for p in originals.values())
        # the buffer in scaled units of 0.5 corresponds to half the data
        # range; pick a smaller fraction so the effect is visible without
        # being absurd
        r_x = x_range * 0.04
        r_y = y_range * 0.04
        buffered = buffer_all(originals, r_x, r_y)
        opt1 = option1(buffered, originals)
        opt2 = option2(buffered, originals)
        slug = "cT" if "c-T" in name else "Tmu"
        render(name, originals, buffered, opt1, opt2, axis_labels,
               f"/tmp/voronoi_demo/{slug}.png")


if __name__ == "__main__":
    run()
