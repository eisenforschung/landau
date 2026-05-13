"""Demonstrator: option 1 (pairwise half-overlap subtraction) vs option 2
(Voronoi clipping) for symmetrically trimming buffered phase polygons.

We make two scenes:

    c-T scene -- two solid-solution phases shaped like Ag-Cu's alpha/beta:
                 each is widest along its pure-element axis and tapers
                 to (nearly) zero solubility at a common eutectic point.
                 Buffered, the tips overlap right where the buffer was
                 doing the most work (visibility), which is exactly the
                 ugly case the trimming has to solve.

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
    """Two solid-solution phases approaching a common eutectic terminal.

    Loosely modelled on Ag-Cu: two triangular regions widest along their
    respective pure-element axes and narrowing to (nearly) zero
    solubility at a shared eutectic point in the middle.  After buffering
    the tips overlap right where the buffer matters most for visibility
    -- exactly the messy case the trim has to handle.
    """
    eutectic = (0.5, 800)
    alpha = shapely.Polygon([(0.0, 200), eutectic, (0.0, 900)])
    beta = shapely.Polygon([(1.0, 200), (1.0, 900), eutectic])
    return {"alpha": alpha, "beta": beta}, ("c", "T")


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
    """Subtract each neighbour's un-buffered original from every polygon,
    then resolve any residual pairwise overlap by assigning each piece to
    the polygon whose *original* is closer.

    The first step alone leaves slivers where the buffer rounds off a
    point-tangent corner (the buffered apex extends into a region that's
    not inside any neighbour's un-buffered shape).  The second step
    fixes those slivers: residual a' ∩ b' is split into two pieces by
    closer-original, and the foreign half is removed from each side.
    """
    out = {}
    for k, a in buffered.items():
        trimmed = a
        for k2, b_orig in originals.items():
            if k2 == k:
                continue
            trimmed = trimmed.difference(b_orig)
        out[k] = trimmed

    # post-process: any residual mutual overlap is split with a mini
    # rasterized Voronoi diagram on just the (tiny) intersection bbox.
    # The "foreign" half (pixels closer to the other original) is then
    # subtracted from each side.
    keys = list(out.keys())
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            inter = out[ki].intersection(out[kj])
            if inter.is_empty or inter.area == 0:
                continue
            minx, miny, maxx, maxy = inter.bounds
            width = maxx - minx
            height = maxy - miny
            if width == 0 or height == 0:
                continue
            n = 80
            if width >= height:
                nx, ny = n, max(8, int(n * height / width))
            else:
                ny, nx = n, max(8, int(n * width / height))
            xs = np.linspace(minx, maxx, nx)
            ys = np.linspace(miny, maxy, ny)
            X, Y = np.meshgrid(xs, ys)
            sx = 1.0 / width
            sy = 1.0 / height
            pts = shapely.points(X.ravel() * sx, Y.ravel() * sy)
            oa = shapely.affinity.scale(originals[ki], xfact=sx, yfact=sy, origin=(0, 0))
            ob = shapely.affinity.scale(originals[kj], xfact=sx, yfact=sy, origin=(0, 0))
            da = shapely.distance(oa, pts)
            db = shapely.distance(ob, pts)
            b_closer = (db < da).astype(np.uint8).reshape(ny, nx)

            for level, owner in [(1, ki), (0, kj)]:
                fig = plt.figure()
                cs = plt.contourf(xs, ys, b_closer, levels=[level - 0.5, level + 0.5])
                plt.close(fig)
                polys = []
                for path in cs.get_paths():
                    for poly_verts in path.to_polygons():
                        if len(poly_verts) >= 3:
                            polys.append(shapely.Polygon(poly_verts))
                if not polys:
                    continue
                cut = unary_union(polys).intersection(inter)
                out[owner] = out[owner].difference(cut)
    return out


def option2(buffered, originals, grid_res=400):
    """Generalized-Voronoi clip via global rasterized distance fields.

    Sample a fine grid over the *full* bounding box, label each pixel
    with the phase whose original is closest, build per-phase region
    polygons via contour extraction, intersect with the buffered shape.
    """
    union_buf = unary_union(list(buffered.values()))
    minx, miny, maxx, maxy = union_buf.buffer(1e-3).bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    minx -= pad_x; maxx += pad_x; miny -= pad_y; maxy += pad_y
    xs = np.linspace(minx, maxx, grid_res)
    ys = np.linspace(miny, maxy, grid_res)
    X, Y = np.meshgrid(xs, ys)

    # scaled frame so x and y span comparable ranges
    sx = 1.0 / (maxx - minx)
    sy = 1.0 / (maxy - miny)
    scaled_pts = shapely.points(X.ravel() * sx, Y.ravel() * sy)

    keys = list(originals.keys())
    scaled_orig = [
        shapely.affinity.scale(originals[k], xfact=sx, yfact=sy, origin=(0, 0))
        for k in keys
    ]
    dists = np.stack([shapely.distance(op, scaled_pts) for op in scaled_orig])
    label = np.argmin(dists, axis=0).reshape(grid_res, grid_res)

    out = {}
    for i, k in enumerate(keys):
        mask = (label == i).astype(np.uint8)
        fig = plt.figure()
        cs = plt.contourf(xs, ys, mask, levels=[0.5, 1.5])
        plt.close(fig)
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


def option2_local(buffered, originals, grid_res=200):
    """Generalized-Voronoi clip, rasterized only on pairwise overlap regions.

    For each pair (a, b) whose buffered shapes overlap, rasterize *only*
    the intersection bbox, classify pixels by closer-original-of-the-pair,
    extract the "b-side" sub-polygon of the overlap, and subtract it from
    a.  Pixels outside any overlap require no work.  ``grid_res`` is the
    number of samples along the longer axis of each pairwise overlap.
    """
    out = dict(buffered)
    keys = list(buffered.keys())
    for ki in keys:
        for kj in keys:
            if ki >= kj:
                continue
            a, b = out[ki], out[kj]
            inter = a.intersection(b)
            if inter.is_empty or inter.area == 0:
                continue
            minx, miny, maxx, maxy = inter.bounds
            width = maxx - minx
            height = maxy - miny
            if width == 0 or height == 0:
                continue
            # adapt grid so the longer axis has grid_res samples
            if width >= height:
                nx = grid_res
                ny = max(8, int(grid_res * height / width))
            else:
                ny = grid_res
                nx = max(8, int(grid_res * width / height))
            xs = np.linspace(minx, maxx, nx)
            ys = np.linspace(miny, maxy, ny)
            X, Y = np.meshgrid(xs, ys)

            sx = 1.0 / width
            sy = 1.0 / height
            pts = shapely.points(X.ravel() * sx, Y.ravel() * sy)
            oa = shapely.affinity.scale(originals[ki], xfact=sx, yfact=sy, origin=(0, 0))
            ob = shapely.affinity.scale(originals[kj], xfact=sx, yfact=sy, origin=(0, 0))
            da = shapely.distance(oa, pts)
            db = shapely.distance(ob, pts)
            # pixel belongs to "b-side" of the overlap if it's closer to
            # original b; this is the piece we must subtract from a
            b_side = (db < da).astype(np.uint8).reshape(ny, nx)

            for label_val, owner, other in [(1, ki, kj), (0, kj, ki)]:
                fig = plt.figure()
                cs = plt.contourf(xs, ys, b_side, levels=[label_val - 0.5, label_val + 0.5])
                plt.close(fig)
                polys = []
                for path in cs.get_paths():
                    for poly_verts in path.to_polygons():
                        if len(poly_verts) >= 3:
                            polys.append(shapely.Polygon(poly_verts))
                if not polys:
                    continue
                cut = unary_union(polys).intersection(inter)
                # subtract the foreign-owned half of the overlap from `owner`
                out[owner] = out[owner].difference(cut)
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


def render(scene_name, originals, buffered, opt1, opt2, opt2_local, axis_labels, fname):
    xs = [p.bounds for p in buffered.values()]
    xmin = min(b[0] for b in xs) - 0.02
    xmax = max(b[2] for b in xs) + 0.02
    ymin = min(b[1] for b in xs) - 20
    ymax = max(b[3] for b in xs) + 20

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["buffered (overlap)", "option 1: subtract neighbour original",
              "option 2: global Voronoi", "option 2 (local): pairwise overlap"]
    sets = [buffered, opt1, opt2, opt2_local]
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


def time_it(fn, *args, n_runs=5, **kw):
    import time
    # warm up
    result = fn(*args, **kw)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn(*args, **kw)
    elapsed = (time.perf_counter() - t0) / n_runs
    return result, elapsed


def run():
    timings = {}
    for name, (scene, axes) in [("c-T (eutectic-like: triangles approaching a terminal)", scene_cT()),
                                ("T-mu (dense: phases tile the plane)", scene_Tmu())]:
        originals, axis_labels = scene, axes
        x_range = max(p.bounds[2] for p in originals.values()) - min(p.bounds[0] for p in originals.values())
        y_range = max(p.bounds[3] for p in originals.values()) - min(p.bounds[1] for p in originals.values())
        r_x = x_range * 0.04
        r_y = y_range * 0.04
        buffered = buffer_all(originals, r_x, r_y)
        opt1, t1 = time_it(option1, buffered, originals)
        opt2, t2 = time_it(option2, buffered, originals)
        opt2l, t2l = time_it(option2_local, buffered, originals)
        timings[name] = (t1, t2, t2l)
        slug = "cT" if "c-T" in name else "Tmu"
        render(name, originals, buffered, opt1, opt2, opt2l, axis_labels,
               f"docs/issue_125/{slug}.png")

    print("\nTimings (mean over 5 runs, seconds):")
    print(f"{'scene':<55} {'opt1':>10} {'opt2':>10} {'opt2_local':>12}")
    for name, (t1, t2, t2l) in timings.items():
        print(f"{name:<55} {t1*1000:>8.2f}ms {t2*1000:>8.2f}ms {t2l*1000:>10.2f}ms")


if __name__ == "__main__":
    run()
