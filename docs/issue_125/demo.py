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


def _shared_coords(shared):
    """Extract ordered coordinates from the shapely intersection of two
    unbuffered originals.  Returns a list of (x, y) tuples in geometric
    order along the shared feature (point, linestring, or collection)."""
    if shared.is_empty:
        return []
    if shared.geom_type == "Point":
        return [(shared.x, shared.y)]
    if shared.geom_type == "MultiPoint":
        return [(g.x, g.y) for g in shared.geoms]
    if shared.geom_type == "LineString":
        return list(shared.coords)
    if shared.geom_type == "MultiLineString":
        return [c for g in shared.geoms for c in g.coords]
    if shared.geom_type == "GeometryCollection":
        coords = []
        for g in shared.geoms:
            coords.extend(_shared_coords(g))
        return coords
    return []


def option3(buffered, originals, n_samples=200):
    """Boundary-only Voronoi: O(N) work on the *boundary* of each pairwise
    overlap union, instead of an O(N^2) raster.

    For each overlapping pair (a, b):
      1. Compute the union of the buffered shapes (per connected component).
      2. Sample the union's exterior boundary uniformly into B_U.
      3. Label each B_U point by which un-buffered original (a or b) is
         closer (shapely.distance to the polygon).  This is the "Voronoi
         query"; with so few points (~100s) a KDTree is overkill.
      4. The boundary is now a sequence of label runs; transitions between
         runs are the seam endpoints.
      5. Route the cut through any shared geometry of the two un-buffered
         originals (a touching point, a coincident edge).  This is what
         pins the cut to the true equidistance locus at point-tangent
         apices -- the apex is at distance 0 from both originals, so the
         seam must pass through it.
      6. Split the union with that polyline; assign each piece to its
         closer original and subtract the foreign piece from the *other*
         buffered polygon.

    No rasterization, no contour extraction.
    """
    out = dict(buffered)
    keys = list(buffered.keys())
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            a, b = out[ki], out[kj]
            inter = a.intersection(b)
            if inter.is_empty or inter.area == 0:
                continue

            oa, ob = originals[ki], originals[kj]
            shared = oa.intersection(ob)
            shared_pts = _shared_coords(shared)

            union = unary_union([a, b])
            comps = list(union.geoms) if hasattr(union, "geoms") else [union]
            for comp in comps:
                if not comp.intersects(inter):
                    continue
                ext = comp.exterior
                L = ext.length
                ts = np.linspace(0, L, n_samples, endpoint=False)
                B_U = np.array([(p.x, p.y) for p in (ext.interpolate(t) for t in ts)])

                # scale-normalised distances so x/y units are comparable
                minx, miny, maxx, maxy = comp.bounds
                sx = 1.0 / max(maxx - minx, 1e-12)
                sy = 1.0 / max(maxy - miny, 1e-12)
                oa_s = shapely.affinity.scale(oa, xfact=sx, yfact=sy, origin=(0, 0))
                ob_s = shapely.affinity.scale(ob, xfact=sx, yfact=sy, origin=(0, 0))
                pts_s = shapely.points(B_U[:, 0] * sx, B_U[:, 1] * sy)
                da = shapely.distance(oa_s, pts_s)
                db = shapely.distance(ob_s, pts_s)
                label = (db < da).astype(int)  # 0 = closer to a, 1 = closer to b

                # cyclic transitions: indices where label[t] != label[t+1]
                next_label = np.roll(label, -1)
                trans_idx = np.where(label != next_label)[0]
                if len(trans_idx) < 2:
                    continue

                # seam endpoints: refine each label-transition with a
                # binary search along the boundary segment so the seam
                # point sits where da == db, not just at the discrete
                # sample midpoint.  20 iterations gives ~1e-6 relative
                # precision and is essentially free.
                seam_pts = []
                for t in trans_idx:
                    lo_t = ts[t]
                    hi_t = ts[(t + 1) % len(ts)]
                    if hi_t < lo_t:
                        hi_t += L
                    lbl_lo = label[t]
                    for _ in range(20):
                        mid_t = 0.5 * (lo_t + hi_t)
                        p = ext.interpolate(mid_t % L)
                        ps = shapely.Point(p.x * sx, p.y * sy)
                        lbl_mid = 1 if ob_s.distance(ps) < oa_s.distance(ps) else 0
                        if lbl_mid == lbl_lo:
                            lo_t = mid_t
                        else:
                            hi_t = mid_t
                    p = ext.interpolate(0.5 * (lo_t + hi_t) % L)
                    seam_pts.append((p.x, p.y))

                if len(seam_pts) != 2:
                    continue  # complex multi-seam case left for future work

                # route the cut through any shared geometry between the
                # un-buffered originals (e.g. the apex point in c-T)
                if shared_pts:
                    s_pt0 = shapely.Point(seam_pts[0])
                    ordered = sorted(shared_pts, key=lambda c: s_pt0.distance(shapely.Point(c)))
                    cut_path = [seam_pts[0]] + ordered + [seam_pts[1]]
                else:
                    cut_path = list(seam_pts)
                cut = shapely.LineString(cut_path)

                try:
                    split_result = shapely.ops.split(comp, cut)
                except Exception:
                    continue
                pieces = list(split_result.geoms) if hasattr(split_result, "geoms") else [split_result]
                for piece in pieces:
                    if piece.is_empty or not piece.is_valid or piece.area == 0:
                        continue
                    rp = piece.representative_point()
                    if oa.distance(rp) < ob.distance(rp):
                        out[kj] = out[kj].difference(piece)
                    else:
                        out[ki] = out[ki].difference(piece)
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


def render(scene_name, originals, buffered, opt1, opt3, axis_labels, fname):
    """Three-panel comparison: buffered (problem) | option 1 | option 3.

    Image size is kept modest (figsize 12x4 at dpi 100 ~ 1200x400 px) so
    rendered figures stay under typical upload-size limits.
    """
    xs = [p.bounds for p in buffered.values()]
    xmin = min(b[0] for b in xs) - 0.02
    xmax = max(b[2] for b in xs) + 0.02
    ymin = min(b[1] for b in xs) - 20
    ymax = max(b[3] for b in xs) + 20

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["buffered (overlap)",
              "option 1: subtract neighbour + rasterized cleanup",
              "option 3: boundary-Voronoi cut"]
    sets = [buffered, opt1, opt3]
    for ax, title, s in zip(axes, titles, sets):
        for k, geom in s.items():
            add_shape(ax, geom, COLORS[k], alpha=0.85)
        for k, o in originals.items():
            x, y = o.exterior.xy
            ax.plot(x, y, color="k", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(title, fontsize=9)
        ax.set_aspect("auto")
    fig.suptitle(f"{scene_name}  (dashed = original un-buffered shapes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=100, bbox_inches="tight")
    print(f"wrote {fname}")


def render_apex_zoom(originals, opt1, opt3, fname):
    """Zoom on the c-T eutectic apex: option 1 vs option 3 side by side.

    Kept small (figsize 8x4 at dpi 100 ~ 800x400 px).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    titles = ["option 1", "option 3"]
    sets = [opt1, opt3]
    for ax, title, s in zip(axes, titles, sets):
        for k, geom in s.items():
            add_shape(ax, geom, COLORS[k], alpha=0.85)
        for k, o in originals.items():
            x, y = o.exterior.xy
            ax.plot(x, y, color="k", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_xlim(0.42, 0.58)
        ax.set_ylim(770, 830)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("auto")
    fig.suptitle("apex zoom: option 1 vs option 3", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=100, bbox_inches="tight")
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
        opt3, t3 = time_it(option3, buffered, originals)
        timings[name] = (t1, t2, t2l, t3)
        slug = "cT" if "c-T" in name else "Tmu"
        render(name, originals, buffered, opt1, opt3, axis_labels,
               f"docs/issue_125/{slug}.png")
        if slug == "cT":
            render_apex_zoom(originals, opt1, opt3, "docs/issue_125/cT_apex_zoom.png")

    print("\nTimings (mean over 5 runs, milliseconds):")
    print(f"{'scene':<55} {'opt1':>10} {'opt2':>10} {'opt2_local':>12} {'opt3':>10}")
    for name, (t1, t2, t2l, t3) in timings.items():
        print(f"{name:<55} {t1*1000:>8.2f}ms {t2*1000:>8.2f}ms "
              f"{t2l*1000:>10.2f}ms {t3*1000:>8.2f}ms")


if __name__ == "__main__":
    run()
