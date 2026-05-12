"""Unit tests for the public polygon-plotting plumbing in `landau.plot`.

These tests cover `get_polygons` and `plot_polygons` directly (issue #88).
They avoid going through `calc_phase_diagram` so each test is fast and
exercises one behaviour at a time.
"""
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Polygon

from landau import plot as plot_mod
from landau.plot import get_polygons, plot_polygons
from landau.poly import AbstractPolyMethod, Concave


# --- shared fixtures ---------------------------------------------------------


def _stable_df():
    """Two well-separated single-phase blobs in (c, T) space.

    Has a couple of unstable rows so the `query("stable")` filter has something
    to drop.  No `mu` column — `cluster_phase` calls `cluster(..., use_mu=False)`.
    """
    rng = np.random.default_rng(0)
    n_per_phase = 20
    blob_a = rng.uniform([0.0, 200.0], [0.2, 800.0], size=(n_per_phase, 2))
    blob_b = rng.uniform([0.8, 200.0], [1.0, 800.0], size=(n_per_phase, 2))
    df = pd.DataFrame(
        {
            "phase": ["A"] * n_per_phase + ["B"] * n_per_phase,
            "c": np.concatenate([blob_a[:, 0], blob_b[:, 0]]),
            "T": np.concatenate([blob_a[:, 1], blob_b[:, 1]]),
            "stable": True,
        }
    )
    unstable = pd.DataFrame(
        {
            "phase": ["A", "B"],
            "T": [500.0, 500.0],
            "c": [0.5, 0.6],
            "stable": [False, False],
        }
    )
    return pd.concat([df, unstable], ignore_index=True)


@dataclass
class _CapturingPoly(AbstractPolyMethod):
    """An `AbstractPolyMethod` that records the arguments passed to `apply`."""

    captured: list = field(default_factory=list)

    def apply(self, df, variables=["c", "T"]):
        self.captured.append({"df": df.copy(), "variables": list(variables)})
        return pd.Series([], dtype=object)

    def _make(self, pp, border, segment_label):  # pragma: no cover - not exercised
        return None


# --- get_polygons ------------------------------------------------------------


def test_get_polygons_filters_unstable_rows():
    cap = _CapturingPoly()
    get_polygons(_stable_df(), poly_method=cap)
    seen = cap.captured[0]["df"]
    assert seen["stable"].all()


def test_get_polygons_does_not_mutate_input():
    df = _stable_df()
    before = df.copy()
    get_polygons(df, poly_method=_CapturingPoly())
    pd.testing.assert_frame_equal(df, before)


def test_get_polygons_defaults_to_c_T():
    cap = _CapturingPoly()
    get_polygons(_stable_df(), poly_method=cap)
    assert cap.captured[0]["variables"] == ["c", "T"]


def test_get_polygons_forwards_variables():
    df = _stable_df().assign(mu=0.1)
    cap = _CapturingPoly()
    get_polygons(df, poly_method=cap, variables=["mu", "T"])
    assert cap.captured[0]["variables"] == ["mu", "T"]


def test_get_polygons_forwards_kwargs_to_handle_poly_method(monkeypatch):
    """`alpha` / `min_c_width` and the method name itself reach
    `handle_poly_method` unchanged."""
    seen = {}
    cap = _CapturingPoly()

    def fake_handle(poly_method, **kwargs):
        seen["poly_method"] = poly_method
        seen["kwargs"] = kwargs
        return cap

    monkeypatch.setattr(plot_mod.poly, "handle_poly_method", fake_handle)
    get_polygons(_stable_df(), poly_method="concave", alpha=0.37, min_c_width=0.05)

    assert seen["poly_method"] == "concave"
    assert seen["kwargs"] == {"alpha": 0.37, "min_c_width": 0.05}


def test_get_polygons_drops_failed_clusters_and_warns(monkeypatch):
    """Rows where `cluster_phase` produces phase_unit == -1 are dropped and
    a warning is emitted."""

    def fake_cluster_phase(df):
        df = df.copy()
        df["phase_unit"] = [-1 if i == 0 else 0 for i in range(len(df))]
        df["phase_id"] = df["phase"].astype(str) + "_" + df["phase_unit"].astype(str)
        return df

    monkeypatch.setattr(plot_mod, "cluster_phase", fake_cluster_phase)

    cap = _CapturingPoly()
    with pytest.warns(UserWarning, match="Clustering of phase points failed"):
        get_polygons(_stable_df(), poly_method=cap)

    seen = cap.captured[0]["df"]
    assert (seen["phase_unit"] >= 0).all()


def test_get_polygons_returns_series_of_polygons():
    """End-to-end smoke test on a small synthetic frame: the returned Series
    is indexed by (phase, phase_unit) and its values are matplotlib Polygons.

    `drop_interior=False` is used because the synthetic frame has no `border`
    column; the default `drop_interior=True` would then return nothing.
    """
    result = get_polygons(_stable_df(), poly_method=Concave(drop_interior=False))
    assert isinstance(result, pd.Series)
    assert isinstance(result.index, pd.MultiIndex)
    assert list(result.index.names) == ["phase", "phase_unit"]
    assert len(result) > 0
    for poly in result:
        assert isinstance(poly, Polygon)


# --- plot_polygons -----------------------------------------------------------


def _square_polygon(xy=(0.0, 0.0), size=1.0):
    s = size
    x, y = xy
    return Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])


def _polys_series(items):
    """Build a `pd.Series` of Polygons indexed by `items` (list of keys)."""
    polys = [_square_polygon((i, 0.0), 1.0) for i, _ in enumerate(items)]
    if items and isinstance(items[0], tuple):
        index = pd.MultiIndex.from_tuples(items, names=["phase", "phase_unit"])
    else:
        index = pd.Index(items, name="phase")
    return pd.Series(polys, index=index)


def test_plot_polygons_adds_one_patch_per_item():
    fig, ax = plt.subplots()
    polys = _polys_series([("A", 0), ("B", 0), ("A", 1)])
    color_map = {"A": "red", "B": "blue"}
    plot_polygons(polys, color_map, ax=ax)
    assert len(ax.patches) == 3
    plt.close(fig)


def test_plot_polygons_uses_provided_ax():
    fig_a, ax_a = plt.subplots()
    fig_b, ax_b = plt.subplots()
    plt.sca(ax_b)  # make ax_b the current axes
    polys = _polys_series([("A", 0)])
    plot_polygons(polys, {"A": "red"}, ax=ax_a)
    assert len(ax_a.patches) == 1
    assert len(ax_b.patches) == 0
    plt.close(fig_a)
    plt.close(fig_b)


def test_plot_polygons_defaults_to_plt_gca():
    fig, ax = plt.subplots()
    plt.sca(ax)
    polys = _polys_series([("A", 0)])
    plot_polygons(polys, {"A": "red"})
    assert len(ax.patches) == 1
    plt.close(fig)


def test_plot_polygons_label_and_color_for_scalar_key():
    fig, ax = plt.subplots()
    polys = _polys_series(["A"])
    plot_polygons(polys, {"A": "red"}, ax=ax)
    patch = ax.patches[0]
    assert patch.get_label() == "A"
    # set_color makes both facecolor and edgecolor the same — then edgecolor
    # gets overwritten to "k"; check the facecolor side.
    assert tuple(patch.get_facecolor()) == matplotlib.colors.to_rgba("red")
    plt.close(fig)


def test_plot_polygons_label_appends_apostrophes_for_phase_unit():
    fig, ax = plt.subplots()
    polys = _polys_series([("A", 0), ("A", 1), ("A", 2)])
    plot_polygons(polys, {"A": "red"}, ax=ax)
    labels = [p.get_label() for p in ax.patches]
    assert labels == ["A", "A'", "A''"]
    plt.close(fig)


def test_plot_polygons_edgecolor_is_black():
    fig, ax = plt.subplots()
    polys = _polys_series([("A", 0), ("B", 0)])
    plot_polygons(polys, {"A": "red", "B": "blue"}, ax=ax)
    for patch in ax.patches:
        assert tuple(patch.get_edgecolor()) == matplotlib.colors.to_rgba("k")
    plt.close(fig)


def test_plot_polygons_zorder_is_inverse_to_bbox_area():
    """Bigger polygons should sit behind smaller ones."""
    fig, ax = plt.subplots()
    big = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    small = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    polys = pd.Series(
        [big, small],
        index=pd.MultiIndex.from_tuples([("A", 0), ("B", 0)], names=["phase", "phase_unit"]),
    )
    plot_polygons(polys, {"A": "red", "B": "blue"}, ax=ax)
    z_big, z_small = (p.zorder for p in ax.patches)
    assert z_big < z_small
    plt.close(fig)
