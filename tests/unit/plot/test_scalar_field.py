"""Tests for colouring 2d phase-diagram fields by a scalar column (``color_by=``).

``_region_interpolator`` is pinned directly on synthetic point sets. The
end-to-end tests colour by a column that is linear in the axis coordinates
(``T`` or ``mu``), where the interpolation is exact inside the convex hull of a
region's points, so every drawn node must carry either its own coordinate or,
on the buffered rim, the value of one of its phase's data points.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm
from scipy.spatial import Delaunay

import landau.calculate as ldc
from landau.phases import LinePhase, kB
from landau.plot import _region_interpolator, plot_mu_phase_diagram, plot_phase_diagram

_ATOL = 1e-9


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture(scope="module")
def line_phase_diagram(eutectic_phases):
    """Eutectic system plus a line compound at c = 0.4, stable at low T."""
    line = LinePhase("AB", fixed_concentration=0.4, line_energy=-2.68, line_entropy=0.5 * kB)
    return ldc.calc_phase_diagram([*eutectic_phases, line], np.linspace(200.0, 1000.0, 20), mu=40, refine=True)


def _meshes(ax):
    return [c for c in ax.collections if isinstance(c, QuadMesh)]


def _nodes(mesh):
    """(n, 2) node coordinates and (n,) values of a gouraud mesh."""
    return mesh.get_coordinates().reshape(-1, 2), np.asarray(mesh.get_array()).ravel()


def _region(mesh):
    """The phase polygon a mesh is clipped to, in data coordinates."""
    return shapely.Polygon(mesh.get_clip_path()._patch.get_xy())


def _inside(mesh, xy):
    """Nodes inside the polygon the mesh is clipped to."""
    return shapely.contains_xy(_region(mesh), xy[:, 0], xy[:, 1])


# --- _region_interpolator ----------------------------------------------------


def test_region_interpolator_exact_on_linear_data_inside_hull():
    rng = np.random.default_rng(0)
    xy = np.vstack([[[0, 0], [1, 0], [0, 1], [1, 1]], rng.random((40, 2))])
    z = 2 * xy[:, 0] - 3 * xy[:, 1] + 1
    q = rng.random((500, 2))
    np.testing.assert_allclose(_region_interpolator(xy, z)(q), 2 * q[:, 0] - 3 * q[:, 1] + 1, atol=1e-12)


def test_region_interpolator_nearest_outside_hull():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    z = np.array([5.0, 7.0, 11.0])
    q = np.array([[3.0, -0.1], [-0.2, 4.0], [0.9, 0.9]])
    np.testing.assert_array_equal(_region_interpolator(xy, z)(q), [7.0, 11.0, 7.0])


def test_region_interpolator_collinear_interpolates_along_line_and_clamps():
    # a c-T line phase: one concentration, many temperatures
    xy = np.column_stack([np.full(6, 0.5), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    z = 100 * xy[:, 1]
    q = np.array([[0.4, 0.3], [0.6, 0.7], [0.5, -1.0], [0.5, 2.0]])
    np.testing.assert_allclose(_region_interpolator(xy, z)(q), [30.0, 70.0, 0.0, 100.0], atol=1e-12)


@pytest.mark.parametrize("n", [1, 2])
def test_region_interpolator_too_few_points_is_nearest(n):
    xy = np.array([[0.0, 0.0], [1.0, 1.0]])[:n]
    z = np.array([3.0, 4.0])[:n]
    q = np.array([[0.1, 0.0], [0.9, 1.2]])
    expected = [3.0, 3.0] if n == 1 else [3.0, 4.0]
    np.testing.assert_array_equal(_region_interpolator(xy, z)(q), expected)


# --- end to end ---------------------------------------------------------------


@pytest.mark.parametrize(
    "plot, column, axis",
    [(plot_mu_phase_diagram, "T", 1), (plot_mu_phase_diagram, "mu", 0), (plot_phase_diagram, "T", 1)],
)
def test_field_reproduces_linear_column(eutectic_diagram, ax, plot, column, axis):
    plot(eutectic_diagram, ax=ax, color_by=column, poly_method="concave")
    stable = eutectic_diagram.query("stable")
    x = "mu" if plot is plot_mu_phase_diagram else "c"
    meshes = _meshes(ax)
    assert meshes
    n_exact = 0
    for mesh in meshes:
        xy, z = _nodes(mesh)
        inside = _inside(mesh, xy)
        xy, z = xy[inside], z[inside]
        exact = np.isclose(z, xy[:, axis], rtol=0, atol=_ATOL)
        # off the exact branch the value is a nearest data point's -- never a blend
        data = stable[column].to_numpy()
        assert np.isin(z[~exact], data[np.isfinite(data)]).all()
        # and every node inside the hull of the region's points is on the exact branch
        region = _region(mesh)
        phase = mesh.get_clip_path()._patch.get_label().rstrip("'")
        pts = stable.loc[stable["phase"] == phase, [x, "T"]].to_numpy()
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = pts[shapely.intersects_xy(region, pts[:, 0], pts[:, 1])]
        span = pts.max(axis=0) - pts.min(axis=0)
        hull = Delaunay((pts - pts.min(axis=0)) / span)
        in_hull = hull.find_simplex((xy - pts.min(axis=0)) / span) >= 0
        assert exact[in_hull].all()
        n_exact += exact.sum()
    assert n_exact > 0.8 * sum(_inside(m, _nodes(m)[0]).sum() for m in meshes)


def test_field_does_not_blend_across_boundaries(eutectic_diagram, ax):
    # c jumps across every mu-T boundary; a region drawn from its neighbour's
    # points too would leave its own phase's concentration range near the seam
    plot_mu_phase_diagram(eutectic_diagram, ax=ax, color_by="c", poly_method="concave")
    stable = eutectic_diagram.query("stable")
    for mesh in _meshes(ax):
        phase = mesh.get_clip_path()._patch.get_label().rstrip("'")
        c = stable.loc[stable["phase"] == phase, "c"]
        xy, z = _nodes(mesh)
        z = z[_inside(mesh, xy)]
        assert z.min() >= c.min() - _ATOL
        assert z.max() <= c.max() + _ATOL


def test_line_phase_field_follows_temperature(line_phase_diagram, ax):
    plot_phase_diagram(line_phase_diagram, ax=ax, color_by="T", poly_method="concave")
    T = line_phase_diagram.query("stable and phase == 'AB'")["T"]
    T = T[np.isfinite(T)]
    (mesh,) = [m for m in _meshes(ax) if m.get_clip_path()._patch.get_label() == "AB"]
    assert np.ptp(mesh.get_coordinates()[..., 0]) < 0.05  # the strip a line phase is buffered into
    xy, z = _nodes(mesh)
    np.testing.assert_allclose(z, np.clip(xy[:, 1], T.min(), T.max()), atol=_ATOL)


def test_field_artists(eutectic_diagram, ax):
    plot_mu_phase_diagram(eutectic_diagram, ax=ax, color_by="c", poly_method="concave")
    patches = [p for p in ax.patches]
    meshes = _meshes(ax)
    assert len(meshes) == len(patches)
    assert {id(m.get_clip_path()._patch) for m in meshes} == {id(p) for p in patches}
    assert all(p.get_facecolor()[3] == 0 for p in patches)
    (cbar_ax,) = [a for a in ax.figure.axes if a is not ax]
    assert cbar_ax.get_ylabel() == "c"
    c = eutectic_diagram.query("stable")["c"]
    c = c[np.isfinite(c) & np.isfinite(eutectic_diagram.query("stable")["mu"])]
    assert meshes[0].norm.vmin == c.min()
    assert meshes[0].norm.vmax == c.max()


def test_field_norm_cmap_and_no_colorbar(eutectic_diagram, ax):
    norm = LogNorm(200, 1000)
    plot_mu_phase_diagram(
        eutectic_diagram, ax=ax, color_by="T", norm=norm, cmap="magma", colorbar=False, poly_method="concave"
    )
    assert ax.figure.axes == [ax]
    assert all(m.norm is norm and m.cmap.name == "magma" for m in _meshes(ax))


def test_no_color_by_draws_no_field(eutectic_diagram, ax):
    plot_mu_phase_diagram(eutectic_diagram, ax=ax, poly_method="concave")
    assert _meshes(ax) == []
    assert all(p.get_facecolor()[3] > 0 for p in ax.patches)


@pytest.mark.parametrize("plot", [plot_phase_diagram, plot_mu_phase_diagram])
def test_field_labels_phases_inline(eutectic_diagram, ax, plot):
    plot(eutectic_diagram, ax=ax, color_by="T", poly_method="concave")
    assert ax.get_legend() is None
    labels = {t.get_text() for t in ax.texts}
    stable = set(eutectic_diagram.query("stable")["phase"])
    assert {lbl.rstrip("'") for lbl in labels} == stable


@pytest.mark.parametrize("plot", [plot_phase_diagram, plot_mu_phase_diagram])
def test_field_refuses_legend_box(eutectic_diagram, ax, plot):
    with pytest.raises(ValueError, match="inline_legend=True"):
        plot(eutectic_diagram, ax=ax, color_by="T", inline_legend=False, poly_method="concave")
    assert len(ax.patches) == 0 and len(ax.collections) == 0  # nothing drawn before refusing


def test_field_without_any_legend(eutectic_diagram, ax):
    plot_mu_phase_diagram(eutectic_diagram, ax=ax, color_by="T", inline_legend=False, legend=False, poly_method="concave")
    assert ax.get_legend() is None and len(ax.texts) == 0
    assert _meshes(ax)


def test_unknown_column_raises(eutectic_diagram, ax):
    with pytest.raises(KeyError, match="no_such_column"):
        plot_mu_phase_diagram(eutectic_diagram, ax=ax, color_by="no_such_column", poly_method="concave")
