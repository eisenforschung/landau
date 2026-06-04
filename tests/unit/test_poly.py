import numpy as np
import pandas as pd
import shapely
from hypothesis import given, strategies as st, settings
from landau.poly import (
    Concave,
    Segments,
    _greedy_stitch,
    _pca_sort_segment,
    _segment_tsp_polygon,
    _segments_from_labels,
    handle_poly_method,
)
import pytest
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA as SkPCA

# Check if optional dependencies are available
try:
    import python_tsp
    from landau.poly import PythonTsp, SegmentPythonTsp

    HAS_PYTHON_TSP = True
except ImportError:
    HAS_PYTHON_TSP = False

try:
    import fast_tsp
    from landau.poly import FastTsp, SegmentFastTsp

    HAS_FAST_TSP = True
except ImportError:
    HAS_FAST_TSP = False


@st.composite
def poly_dataframe(draw):
    # Strategy to generate DataFrames suitable for testing poly methods
    n_points = draw(st.integers(min_value=5, max_value=20))
    data = {
        "c": draw(
            st.lists(
                st.floats(
                    min_value=0, max_value=1, allow_nan=False, allow_infinity=False
                ),
                min_size=n_points,
                max_size=n_points,
            )
        ),
        "T": draw(
            st.lists(
                st.floats(
                    min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
                ),
                min_size=n_points,
                max_size=n_points,
            )
        ),
        "mu": draw(
            st.lists(
                st.floats(
                    min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
                ),
                min_size=n_points,
                max_size=n_points,
            )
        ),
        "border": draw(st.lists(st.booleans(), min_size=n_points, max_size=n_points)),
        "phase": draw(
            st.lists(st.sampled_from(["A", "B"]), min_size=n_points, max_size=n_points)
        ),
        "phase_unit": draw(
            st.lists(
                st.integers(min_value=0, max_value=2),
                min_size=n_points,
                max_size=n_points,
            )
        ),
        "refined": draw(st.lists(st.booleans(), min_size=n_points, max_size=n_points)),
        "phase_id": draw(
            st.lists(
                st.sampled_from(["A_0", "A_1", "B_0"]),
                min_size=n_points,
                max_size=n_points,
            )
        ),
    }
    # Ensure some border points
    data["border"][0] = True
    data["border"][1] = True

    # For Segments, we need some structure to not have it completely empty
    data["mu"][1] = data["mu"][0]
    data["T"][1] = data["T"][0]

    return pd.DataFrame(data)


@settings(deadline=None)
@given(df=poly_dataframe())
def test_concave(df):
    method = Concave()
    # Test apply (which calls prepare and make)
    res = method.apply(df)
    assert isinstance(res, (pd.Series, pd.DataFrame))
    if isinstance(res, pd.Series):
        for p in res:
            assert isinstance(p, Polygon)


@settings(deadline=None)
@given(df=poly_dataframe())
def test_segments(df):
    method = Segments()
    # Segments.prepare requires 'refined' and 'phase_id'
    res = method.apply(df)
    assert isinstance(res, (pd.Series, pd.DataFrame))
    if isinstance(res, pd.Series):
        for p in res:
            assert isinstance(p, Polygon)


@pytest.mark.skipif(not HAS_PYTHON_TSP, reason="python-tsp not installed")
@settings(deadline=None)
@given(df=poly_dataframe())
def test_python_tsp(df):
    method = PythonTsp()
    res = method.apply(df)
    assert isinstance(res, (pd.Series, pd.DataFrame))
    if isinstance(res, pd.Series):
        for p in res:
            assert isinstance(p, Polygon)


@pytest.mark.skipif(not HAS_FAST_TSP, reason="fast-tsp not installed")
@settings(deadline=None)
@given(df=poly_dataframe())
def test_fast_tsp(df):
    method = FastTsp()
    res = method.apply(df)
    assert isinstance(res, (pd.Series, pd.DataFrame))
    if isinstance(res, pd.Series):
        for p in res:
            assert isinstance(p, Polygon)


@pytest.mark.skipif(not HAS_PYTHON_TSP, reason="python-tsp not installed")
@settings(deadline=None)
@given(df=poly_dataframe())
def test_segment_python_tsp(df):
    method = SegmentPythonTsp()
    res = method.apply(df)
    assert isinstance(res, (pd.Series, pd.DataFrame))
    if isinstance(res, pd.Series):
        for p in res:
            assert isinstance(p, Polygon)


@pytest.mark.skipif(not HAS_FAST_TSP, reason="fast-tsp not installed")
@settings(deadline=None)
@given(df=poly_dataframe())
def test_segment_fast_tsp(df):
    method = SegmentFastTsp()
    res = method.apply(df)
    assert isinstance(res, (pd.Series, pd.DataFrame))
    if isinstance(res, pd.Series):
        for p in res:
            assert isinstance(p, Polygon)


@pytest.mark.skipif(not HAS_PYTHON_TSP, reason="python-tsp not installed")
def test_python_tsp_clustered_points():
    # Regression for GH-38: clustered border points can produce a
    # self-intersecting polygon that causes GEOSException in _trim_overlaps.
    n = 10
    # Two phases, each with many border points clustered near a single spot
    # plus one outlier so the convex hull is not just a point.
    data = {
        "c":         [0.0] * n + [0.05] + [1.0] * n + [0.95],
        "T":         [500.0] * n + [600.0] + [500.0] * n + [600.0],
        "mu":        [0.0] * (n + 1) + [0.0] * (n + 1),
        "border":    [True] * (n + 1) + [True] * (n + 1),
        "phase":     ["A"] * (n + 1) + ["B"] * (n + 1),
        "phase_unit":[0] * (n + 1) + [0] * (n + 1),
        "refined":   [True] * (n + 1) + [True] * (n + 1),
        "phase_id":  ["A_0"] * (n + 1) + ["B_0"] * (n + 1),
    }
    df = pd.DataFrame(data)
    result = PythonTsp().apply(df)
    assert isinstance(result, pd.Series)


def test_handle_poly_method():
    assert isinstance(handle_poly_method("concave"), Concave)
    assert isinstance(handle_poly_method("segments"), Segments)
    if HAS_PYTHON_TSP:
        assert isinstance(handle_poly_method("tsp"), PythonTsp)
        assert isinstance(handle_poly_method("segment-tsp"), SegmentPythonTsp)
    if HAS_FAST_TSP:
        assert isinstance(handle_poly_method("fasttsp"), FastTsp)
        assert isinstance(handle_poly_method("segment-fasttsp"), SegmentFastTsp)

    # Test with custom arguments
    res = handle_poly_method("concave", alpha=0.2, min_c_width=0.02)
    assert isinstance(res, Concave)
    assert res.ratio == 0.2
    assert res.min_c_width == 0.02

    # ratio= keyword (canonical spelling) must not raise TypeError (regression for #172)
    res_ratio = handle_poly_method("concave", ratio=0.3)
    assert isinstance(res_ratio, Concave)
    assert res_ratio.ratio == 0.3

    # deprecated alpha= spelling still works
    res_alpha = handle_poly_method("concave", alpha=0.4)
    assert isinstance(res_alpha, Concave)
    assert res_alpha.ratio == 0.4

    # canonical ratio= beats deprecated alpha= when both are supplied
    res_both = handle_poly_method("concave", ratio=0.3, alpha=0.9)
    assert res_both.ratio == 0.3

    with pytest.raises(ValueError):
        handle_poly_method("invalid_method")

    with pytest.raises(TypeError):
        handle_poly_method(123)


# --- _greedy_stitch tests ---


def _segment(xs, ys):
    return pd.DataFrame({"c": xs, "T": ys})


@pytest.fixture
def unit_norm():
    return np.array([1.0, 1.0])


def test_greedy_stitch_empty(unit_norm):
    assert _greedy_stitch([], unit_norm, "c", "T") == []


def test_greedy_stitch_single_segment_returns_unchanged(unit_norm):
    seg = _segment([0.3, 0.4, 0.5], [10.0, 20.0, 30.0])
    out = _greedy_stitch([seg], unit_norm, "c", "T")
    assert len(out) == 1
    pd.testing.assert_frame_equal(out[0], seg)


def test_greedy_stitch_picks_min_x_as_head(unit_norm):
    left = _segment([0.0, 0.1], [0.0, 0.0])
    right = _segment([0.5, 0.6], [0.0, 0.0])
    # right is passed first, but left has smaller min(x) and should lead
    out = _greedy_stitch([right, left], unit_norm, "c", "T")
    assert out[0] is left
    assert out[1] is right


def test_greedy_stitch_flips_segment_when_end_is_closer(unit_norm):
    head = _segment([0.0, 0.1], [0.0, 0.0])  # head ends at (0.1, 0)
    # Other goes from x=0.5 down to x=0.2. Its end (x=0.2) is closer to
    # head's tail than its start (x=0.5) — should be flipped in place.
    other = _segment([0.5, 0.4, 0.3, 0.2], [0.0, 0.0, 0.0, 0.0])
    out = _greedy_stitch([head, other], unit_norm, "c", "T")
    assert out[0] is head
    assert out[1] is other
    # other was flipped: now starts at the previously-final row
    assert out[1].iloc[0]["c"] == pytest.approx(0.2)
    assert out[1].iloc[-1]["c"] == pytest.approx(0.5)


def test_greedy_stitch_no_flip_when_start_is_closer(unit_norm):
    head = _segment([0.0, 0.1], [0.0, 0.0])
    # Other starts at x=0.2 (close to head's tail) and ends at x=0.5 (far)
    other = _segment([0.2, 0.3, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0])
    pre_flip = other.copy()
    out = _greedy_stitch([head, other], unit_norm, "c", "T")
    pd.testing.assert_frame_equal(out[1], pre_flip)


def test_greedy_stitch_three_segments_orders_by_distance():
    # head at x=[0,1], y=0; near segment touching head's tail; far segment
    # further out. Expected order: head, near, far.
    head = _segment([0.0, 1.0], [0.0, 0.0])
    near = _segment([1.1, 2.0], [0.0, 0.0])
    far = _segment([5.0, 6.0], [0.0, 0.0])
    norm = np.array([1.0, 1.0])
    out = _greedy_stitch([far, head, near], norm, "c", "T")
    assert [s.iloc[0]["c"] for s in out] == pytest.approx([0.0, 1.1, 5.0])


def test_greedy_stitch_norm_scales_axes():
    # y dominates raw distance, but a large y-norm should make x dominate.
    head = _segment([0.0, 0.1], [0.0, 0.0])  # head tail at (0.1, 0)
    seg_a = _segment([0.2, 0.3], [100.0, 100.0])  # close in x, far in raw y
    seg_b = _segment([5.0, 5.1], [0.5, 0.5])     # far in x, close in raw y
    # With unit norm, seg_b wins (y matters more)
    out_unit = _greedy_stitch(
        [head, seg_a.copy(), seg_b.copy()], np.array([1.0, 1.0]), "c", "T"
    )
    assert out_unit[1].iloc[0]["c"] == pytest.approx(5.0)
    # With y-norm large, y becomes irrelevant and seg_a (closer in x) wins
    out_scaled = _greedy_stitch(
        [head, seg_a.copy(), seg_b.copy()], np.array([1.0, 1000.0]), "c", "T"
    )
    assert out_scaled[1].iloc[0]["c"] == pytest.approx(0.2)


def test_greedy_stitch_custom_columns(unit_norm):
    # Sanity that the column names are honoured rather than hard-coded to c/T.
    head = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]})
    other = pd.DataFrame({"x": [3.0, 2.0], "y": [0.0, 0.0]})  # ends at x=2, closer
    out = _greedy_stitch([head, other], unit_norm, "x", "y")
    assert out[0] is head
    # other should have been flipped to start at x=2
    assert out[1].iloc[0]["x"] == pytest.approx(2.0)


# --- _pca_sort_segment tests ---


def test_pca_sort_segment_empty():
    pts = np.zeros((0, 2))
    result = _pca_sort_segment(pts)
    assert result.shape == (0, 2)


def test_pca_sort_segment_single_point():
    pts = np.array([[3.0, 4.0]])
    result = _pca_sort_segment(pts)
    np.testing.assert_array_equal(result, pts)


def test_pca_sort_segment_collinear_x():
    # Collinear along x-axis in scrambled order; result must be monotone in x.
    pts = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    result = _pca_sort_segment(pts)
    assert set(map(tuple, result.tolist())) == {(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)}
    xs = result[:, 0]
    assert np.all(np.diff(xs) >= 0) or np.all(np.diff(xs) <= 0)


def test_pca_sort_segment_diagonal_matches_argsort():
    # Points on y=x diagonal. PCA principal axis is [1,1]/sqrt(2).
    # Assert the returned order equals pts[argsort(PCA projection)].
    pts = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]])
    result = _pca_sort_segment(pts)
    pca = SkPCA(n_components=1)
    proj = pca.fit_transform(pts).ravel()
    expected = pts[np.argsort(proj)]
    np.testing.assert_array_equal(result, expected)


# --- _segments_from_labels tests ---


def test_segments_from_labels_empty():
    pp = np.zeros((0, 2))
    labels = np.array([], dtype=int)
    result = _segments_from_labels(pp, labels)
    assert result == []


def test_segments_from_labels_single_label():
    pts = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    labels = np.array([7, 7, 7])
    result = _segments_from_labels(pts, labels)
    assert len(result) == 1
    assert set(map(tuple, result[0].tolist())) == {(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)}


def test_segments_from_labels_two_labels():
    pts = np.array([[1.0, 0.0], [0.0, 0.0], [3.0, 1.0], [2.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    result = _segments_from_labels(pts, labels)
    assert len(result) == 2
    assert set(map(tuple, result[0].tolist())) == {(0.0, 0.0), (1.0, 0.0)}
    assert set(map(tuple, result[1].tolist())) == {(2.0, 1.0), (3.0, 1.0)}


def test_segments_from_labels_each_group_pca_sorted():
    # Two groups of 3 collinear points; each must come out monotone along its axis.
    pts = np.array([
        [2.0, 0.0], [0.0, 0.0], [1.0, 0.0],   # label 0, collinear along x
        [0.0, 3.0], [0.0, 1.0], [0.0, 2.0],   # label 1, collinear along y
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = _segments_from_labels(pts, labels)
    assert len(result) == 2
    xs = result[0][:, 0]
    ys = result[1][:, 1]
    assert np.all(np.diff(xs) >= 0) or np.all(np.diff(xs) <= 0)
    assert np.all(np.diff(ys) >= 0) or np.all(np.diff(ys) <= 0)


# --- _segment_tsp_polygon tests ---


def _identity_tour(dm):
    return list(range(len(dm)))


def test_segment_tsp_polygon_no_segments():
    assert _segment_tsp_polygon([], _identity_tour) is None


def test_segment_tsp_polygon_single_segment_two_points():
    seg = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert _segment_tsp_polygon([seg], _identity_tour) is None


def test_segment_tsp_polygon_single_segment_triangle():
    seg = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    result = _segment_tsp_polygon([seg], _identity_tour)
    assert isinstance(result, shapely.Polygon)
    assert not result.is_empty


def test_segment_tsp_polygon_all_zero_distances():
    # All endpoints at the same location → pairwise distance matrix is all-zero
    # → solve_tour must not be called; convex-hull fallback returned instead.
    seg = np.array([[1.0, 1.0], [1.0, 1.0]])
    called = []

    def tracking_tour(dm):
        called.append(dm)
        return list(range(len(dm)))

    result = _segment_tsp_polygon([seg, seg.copy()], tracking_tour)
    assert not called
    assert result is not None


def test_segment_tsp_polygon_forward_traversal():
    # Even-numbered tour nodes → segment traversed in its stored order.
    seg0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    seg1 = np.array([[1.0, 1.0], [0.0, 1.0]])
    # Tour [0,1,2,3]: node 0 (even) → seg0 forward; node 2 (even) → seg1 forward.
    result = _segment_tsp_polygon([seg0, seg1], lambda dm: [0, 1, 2, 3])
    assert isinstance(result, shapely.Polygon)
    coords = np.array(result.exterior.coords)[:-1]  # drop closing vertex
    np.testing.assert_allclose(coords, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_segment_tsp_polygon_backward_traversal():
    # Odd-numbered tour nodes → segment traversed in reverse.
    seg0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    seg1 = np.array([[1.0, 1.0], [0.0, 1.0]])
    # Tour [1,0,3,2]: node 1 (odd) → seg0 backward; node 3 (odd) → seg1 backward.
    result = _segment_tsp_polygon([seg0, seg1], lambda dm: [1, 0, 3, 2])
    assert isinstance(result, shapely.Polygon)
    coords = np.array(result.exterior.coords)[:-1]
    np.testing.assert_allclose(coords, [[1, 0], [0, 0], [0, 1], [1, 1]])
