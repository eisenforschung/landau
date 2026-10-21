"""Unit tests for ``_region_interpolator``, the scattered-data interpolation behind
``color_by=`` on the 2d phase-diagram plotters.

End-to-end tests of the coloured fields live in ``tests/integration/test_scalar_field.py``.
"""
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from landau.plot import _region_interpolator

_ATOL = 1e-9


_unit = st.floats(0, 1)
_coeff = st.floats(-10, 10)


@given(
    inner=arrays(float, st.tuples(st.integers(0, 40), st.just(2)), elements=_unit),
    # strictly inside: a query on the hull edge can miss every simplex once Qhull drops a
    # point collinear with that edge, and falls back to nearest like the rim does
    q=arrays(float, st.tuples(st.integers(1, 50), st.just(2)), elements=st.floats(1e-6, 1 - 1e-6)),
    a=_coeff,
    b=_coeff,
    d=_coeff,
)
def test_region_interpolator_exact_on_linear_data_inside_hull(inner, q, a, b, d):
    # the corners make the hull the unit square
    xy = np.vstack([[[0, 0], [1, 0], [0, 1], [1, 1]], inner])
    z = a * xy[:, 0] + b * xy[:, 1] + d
    np.testing.assert_allclose(_region_interpolator(xy, z)(q), a * q[:, 0] + b * q[:, 1] + d, atol=_ATOL)


def test_region_interpolator_collinear_is_nearest():
    # a c-T line phase: one concentration, many temperatures
    xy = np.column_stack([np.full(6, 0.5), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
    z = 100 * xy[:, 1]
    q = np.array([[0.4, 0.29], [0.6, 0.71], [0.5, -1.0], [0.5, 2.0]])
    np.testing.assert_array_equal(_region_interpolator(xy, z)(q), [20.0, 80.0, 0.0, 100.0])


@pytest.mark.parametrize("n", [1, 2])
def test_region_interpolator_too_few_points_is_nearest(n):
    xy = np.array([[0.0, 0.0], [1.0, 1.0]])[:n]
    z = np.array([3.0, 4.0])[:n]
    q = np.array([[0.1, 0.0], [0.9, 1.2]])
    expected = [3.0, 3.0] if n == 1 else [3.0, 4.0]
    np.testing.assert_array_equal(_region_interpolator(xy, z)(q), expected)
