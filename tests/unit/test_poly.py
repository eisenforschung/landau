import pandas as pd
import shapely
from hypothesis import given, strategies as st, settings
from landau.poly import Concave, Segments, AbstractPolyMethod, handle_poly_method
import pytest
from matplotlib.patches import Polygon


class _StubPoly(AbstractPolyMethod):
    """Minimal :class:`AbstractPolyMethod` subclass for exercising
    :meth:`_trim_overlaps` directly."""

    def _make(self, pp, border, segment_label):
        return None

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


def test_trim_overlaps_point_tangent_apex():
    """Two solid-solution phases tangent at a eutectic apex: the
    buffered shapes overlap right at the tip, and trimming should land
    the seam through the apex with both phases ending up equal in area.
    """
    r = 0.02
    eutectic = (0.5, 800.0)
    alpha = shapely.Polygon([(0.0, 200), eutectic, (0.0, 900)]).buffer(r)
    beta = shapely.Polygon([(1.0, 200), (1.0, 900), eutectic]).buffer(r)
    shapes = pd.Series({"alpha": alpha, "beta": beta})

    trimmed = _StubPoly(min_c_width=2 * r)._trim_overlaps(shapes)

    overlap = trimmed["alpha"].intersection(trimmed["beta"]).area
    assert overlap < 1e-9, f"overlap area {overlap} should be ~0"
    # symmetric scene -> symmetric trimmed areas
    assert abs(trimmed["alpha"].area - trimmed["beta"].area) < 1e-3
    # the apex point should sit on the seam (distance ~0 to both)
    apex = shapely.Point(eutectic)
    assert trimmed["alpha"].distance(apex) < 1e-3
    assert trimmed["beta"].distance(apex) < 1e-3


def test_trim_overlaps_shared_edge():
    """Three rectangles tiling a T-mu plane with no gaps: after buffer
    each pair overlaps along the shared edge; trimming should remove
    every pairwise overlap without losing meaningful area."""
    r = 0.02
    a = shapely.Polygon([(-0.5, 200), (-0.1, 200), (-0.1, 900), (-0.5, 900)]).buffer(r)
    b = shapely.Polygon([(-0.1, 200), (0.2, 200), (0.2, 900), (-0.1, 900)]).buffer(r)
    c = shapely.Polygon([(0.2, 200), (0.6, 200), (0.6, 900), (0.2, 900)]).buffer(r)
    shapes = pd.Series({"a": a, "b": b, "c": c})

    trimmed = _StubPoly(min_c_width=2 * r)._trim_overlaps(shapes)

    for ki, kj in [("a", "b"), ("a", "c"), ("b", "c")]:
        ov = trimmed[ki].intersection(trimmed[kj]).area
        assert ov < 1e-9, f"{ki} ∩ {kj} overlap {ov} should be ~0"
    for k in ("a", "b", "c"):
        assert trimmed[k].area > 0


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

    with pytest.raises(ValueError):
        handle_poly_method("invalid_method")

    with pytest.raises(TypeError):
        handle_poly_method(123)
