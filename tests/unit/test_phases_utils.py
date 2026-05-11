import pytest
import numpy as np
from numpy.testing import assert_allclose
from landau.phases import S, Sprime, c_from_dmu

def test_Sprime_scalar():
    # Sprime(c) = -kB * np.log(c / (1 - c))
    # where c is between 0 and 1
    # Edge cases
    assert Sprime(0.0) == np.inf
    assert Sprime(1.0) == -np.inf
    assert Sprime(0.5) == 0.0

def test_Sprime_array():
    # Array inputs including edge cases
    c_arr = np.array([0.0, 0.5, 1.0])
    expected = np.array([np.inf, 0.0, -np.inf])
    res = Sprime(c_arr)
    assert isinstance(res, np.ndarray)
    assert_allclose(res, expected, atol=1e-10)

def test_Sprime_list():
    # List input should be handled correctly
    c_list = [0.0, 0.5, 1.0]
    expected = np.array([np.inf, 0.0, -np.inf])
    res = Sprime(c_list)
    assert isinstance(res, np.ndarray)
    assert_allclose(res, expected, atol=1e-10)

def test_Sprime_close_to_edges():
    # Values close to 0 and 1 should still return +/- inf
    # based on np.isclose defaults (rtol=1e-05, atol=1e-08)

    # Very close to 0
    assert Sprime(1e-10) == np.inf

    # Very close to 1
    assert Sprime(1.0 - 1e-10) == -np.inf

    # Not quite close enough to 0 (e.g., 1e-7 which > atol)
    # Should calculate a valid negative number, not infinity
    # But np.isclose with default atol=1e-8 might trigger if we are very close

    # For a generic value
    res = Sprime(0.1)
    # log(0.1 / 0.9) = log(1/9) < 0. So Sprime(0.1) > 0.
    assert res > 0
    assert np.isfinite(res)


def test_S_interior():
    # S(c) = kB * (entr(c) + entr(1-c)), positive for 0 < c < 1
    assert S(0.5) > 0
    assert np.isfinite(S(0.5))


def test_S_edge_cases():
    # scipy.special.entr returns 0 for x=0, so S(0) and S(1) must be 0 not NaN
    assert S(0.0) == 0.0
    assert S(1.0) == 0.0


def test_c_from_dmu_symmetry():
    # At dmu == e_defect the sigmoid equals exactly 0.5
    assert c_from_dmu(dmu=1.0, T=300.0, e_defect=1.0) == pytest.approx(0.5)


def test_c_from_dmu_large_positive_dmu():
    # dmu >> e_defect → exponent → -∞ → c → 1
    assert c_from_dmu(dmu=1e10, T=300.0, e_defect=0.0) == pytest.approx(1.0)


def test_c_from_dmu_large_negative_dmu():
    # dmu << e_defect → exponent → +∞ → c → 0
    assert c_from_dmu(dmu=-1e10, T=300.0, e_defect=0.0) == pytest.approx(0.0)
