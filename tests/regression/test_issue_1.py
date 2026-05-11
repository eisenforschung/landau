# https://github.com/eisenforschung/landau/issues/1
import numpy as np
import pytest


def test_regular_solution_concentration_array_len_1(three_phase_regular_solution):
    r = three_phase_regular_solution
    res = r.concentration(333, np.array([0]))
    assert isinstance(res, np.ndarray)
    assert res.shape == (1,)
    assert np.isclose(res[0], 0.5)


def test_regular_solution_concentration_scalar(three_phase_regular_solution):
    r = three_phase_regular_solution
    res = r.concentration(333, 0)
    assert isinstance(res, (float, np.float64, np.ndarray))
    if isinstance(res, np.ndarray):
        assert res.shape == () or res.size == 1
    assert np.isclose(res, 0.5)


def test_regular_solution_concentration_large_array(three_phase_regular_solution):
    r = three_phase_regular_solution
    dmus = np.linspace(-0.1, 0.1, 10)
    res = r.concentration(333, dmus)
    assert isinstance(res, np.ndarray)
    assert res.shape == (10,)
