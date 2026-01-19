from landau.phases import RegularSolution, LinePhase
import numpy as np
import pytest

def test_regular_solution_concentration_array_len_1():
    p1 = LinePhase("A", 0, 0)
    p2 = LinePhase("B", 1, 0)
    p3 = LinePhase("C", 0.5, 0)
    # With 3 phases, num_coeffs = min(3-2, 4) = 1.
    r = RegularSolution(name="sol", phases=[p1, p2, p3])

    # Test with 1-element array
    res = r.concentration(333, np.array([0]))
    assert isinstance(res, np.ndarray)
    assert res.shape == (1,)
    assert np.isclose(res[0], 0.5)

def test_regular_solution_concentration_scalar():
    p1 = LinePhase("A", 0, 0)
    p2 = LinePhase("B", 1, 0)
    p3 = LinePhase("C", 0.5, 0)
    r = RegularSolution(name="sol", phases=[p1, p2, p3])

    # Test with scalar
    res = r.concentration(333, 0)
    assert isinstance(res, (float, np.float64, np.ndarray))
    if isinstance(res, np.ndarray):
        assert res.shape == () or res.size == 1
    assert np.isclose(res, 0.5)

def test_regular_solution_concentration_large_array():
    p1 = LinePhase("A", 0, 0)
    p2 = LinePhase("B", 1, 0)
    p3 = LinePhase("C", 0.5, 0)
    r = RegularSolution(name="sol", phases=[p1, p2, p3])

    # Test with multi-element array
    dmus = np.linspace(-0.1, 0.1, 10)
    res = r.concentration(333, dmus)
    assert isinstance(res, np.ndarray)
    assert res.shape == (10,)
