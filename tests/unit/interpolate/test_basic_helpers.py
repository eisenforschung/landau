"""Direct unit tests for _scalarize, the 0-d numpy -> Python scalar collapse.

_scalarize is the single site that upholds the scalar-in/scalar-out contract
across AbstractLinePhase.concentration, IdealSolution.semigrand_potential,
RegularSolution.semigrand_potential, InterpolatingPhase._find_phi_c, and the
closure-based Interpolations in basic.py. Only exercised transitively before
this file; none of those call sites can pin what _scalarize alone must do.
"""

import numpy as np

from landau.interpolate.basic import _scalarize


def test_0d_ndarray_collapses_to_python_scalar():
    out = _scalarize(np.array(1.5))
    assert out == 1.5
    assert out.__class__ is float


def test_numpy_scalar_type_is_not_an_ndarray_and_passes_through():
    # np.float64 is a numpy scalar type, not an np.ndarray instance, so the
    # isinstance(x, np.ndarray) guard does not catch it.
    x = np.float64(1.5)
    assert not isinstance(x, np.ndarray)
    assert _scalarize(x) is x


def test_1d_array_passes_through_unchanged():
    x = np.array([1.0, 2.0, 3.0])
    out = _scalarize(x)
    assert out is x
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_python_scalar_passes_through_unchanged():
    x = 1.5
    assert _scalarize(x) is x
    n = 1
    assert _scalarize(n) is n


def test_higher_dimensional_array_passes_through_unchanged():
    x = np.arange(6.0).reshape(2, 3)
    out = _scalarize(x)
    assert out is x
    assert out.shape == (2, 3)
