import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from landau.poly import Concave, Segments, PythonTsp, FastTsp, handle_poly_method
import pytest
import shapely

# Check if optional dependencies are available
try:
    import python_tsp
    HAS_PYTHON_TSP = True
except ImportError:
    HAS_PYTHON_TSP = False

try:
    import fast_tsp
    HAS_FAST_TSP = True
except ImportError:
    HAS_FAST_TSP = False

@st.composite
def poly_dataframe(draw):
    # Strategy to generate DataFrames suitable for testing poly methods
    n_points = draw(st.integers(min_value=5, max_value=20))
    data = {
        'c': draw(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False), min_size=n_points, max_size=n_points)),
        'T': draw(st.lists(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False), min_size=n_points, max_size=n_points)),
        'mu': draw(st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=n_points, max_size=n_points)),
        'border': draw(st.lists(st.booleans(), min_size=n_points, max_size=n_points)),
        'phase': draw(st.lists(st.sampled_from(['A', 'B']), min_size=n_points, max_size=n_points)),
        'phase_unit': draw(st.lists(st.integers(min_value=0, max_value=2), min_size=n_points, max_size=n_points)),
        'refined': draw(st.lists(st.booleans(), min_size=n_points, max_size=n_points)),
        'phase_id': draw(st.lists(st.sampled_from(['A_0', 'A_1', 'B_0']), min_size=n_points, max_size=n_points)),
    }
    # Ensure some border points
    data['border'][0] = True
    data['border'][1] = True

    # For Segments, we need some structure to not have it completely empty
    data['mu'][1] = data['mu'][0]
    data['T'][1] = data['T'][0]

    return pd.DataFrame(data)

@settings(deadline=None)
@given(df=poly_dataframe())
def test_concave(df):
    method = Concave()
    # Test apply (which calls prepare and make)
    method.apply(df)

@settings(deadline=None)
@given(df=poly_dataframe())
def test_segments(df):
    method = Segments()
    # Segments.prepare requires 'refined' and 'phase_id'
    method.apply(df)

@pytest.mark.skipif(not HAS_PYTHON_TSP, reason="python-tsp not installed")
@settings(deadline=None)
@given(df=poly_dataframe())
def test_python_tsp(df):
    method = PythonTsp()
    method.apply(df)

@pytest.mark.skipif(not HAS_FAST_TSP, reason="fast-tsp not installed")
@settings(deadline=None)
@given(df=poly_dataframe())
def test_fast_tsp(df):
    method = FastTsp()
    method.apply(df)

def test_handle_poly_method():
    handle_poly_method('concave')
    handle_poly_method('segments')
    if HAS_PYTHON_TSP:
        handle_poly_method('tsp')
    if HAS_FAST_TSP:
        handle_poly_method('fasttsp')

    # Test with custom arguments
    handle_poly_method('concave', alpha=0.2, min_c_width=0.02)

    with pytest.raises(ValueError):
        handle_poly_method('invalid_method')

    with pytest.raises(TypeError):
        handle_poly_method(123)

def test_segments_sort_segments():
    # Test _sort_segments with simple data
    df = pd.DataFrame({
        'c': [0.1, 0.2, 0.3],
        'T': [100, 110, 120],
        'border_segment': ['s1', 's1', 's1']
    })
    sorted_df = Segments._sort_segments(df)
    assert len(sorted_df) == 3
