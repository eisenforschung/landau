"""Direct tests for the pure helpers of ``landau.phases.quasiharmonic``.

The module imports without the ``phonopy`` extra -- only the functions that touch a
``ThermalProperties`` are decorated with its import alarm -- so these need no skip
marker, unlike the phase-level tests in ``test_quasiharmonic.py``.
"""

import warnings

import numpy as np
import pytest

from landau.interpolate import PolyFit
from landau.interpolate.basic import PolynomialInterpolation
from landau.phases.quasiharmonic import DynamicalInstabilityWarning, _select_stable_volumes, _stationary_points

# four stable volumes is the minimum the equation of state needs, so every case here that
# expects a selection starts from five and drops at most one
VOLUMES = (11.0, 12.0, 13.0, 14.0, 15.0)
STABLE = (1.0, 2.0, 3.0, 4.0, 5.0)


def select(volumes=VOLUMES, lowest=STABLE, min_frequency=-0.05):
    return _select_stable_volumes(volumes, lowest, min_frequency, "phase", PolyFit(4))


def test_all_stable_keeps_every_volume():
    assert select() == (0, 1, 2, 3, 4)


def test_unstable_volume_is_dropped():
    with pytest.warns(DynamicalInstabilityWarning):
        assert select(lowest=(1.0, 2.0, -0.3, 4.0, 5.0)) == (0, 1, 3, 4)


def test_warning_names_the_dropped_volume_and_its_frequency():
    with pytest.warns(DynamicalInstabilityWarning) as record:
        select(lowest=(1.0, 2.0, -0.3, 4.0, 5.0))
    message = str(record[0].message)
    assert "(13.0, -0.3)" in message
    assert "phase" in message


def test_frequency_exactly_at_the_cut_is_kept():
    """The cut is strict, and the default ``min_frequency`` is deliberately negative."""
    assert select(lowest=(-0.05, 2.0, 3.0, 4.0, 5.0)) == (0, 1, 2, 3, 4)


def test_order_is_by_volume_per_atom_not_by_input_position():
    assert select(volumes=(13.0, 11.0, 15.0, 12.0, 14.0)) == (1, 3, 0, 4, 2)


def test_selection_is_independent_of_input_order():
    """Shuffling the parallel inputs permutes the indices but not the volumes selected."""
    volumes = (13.0, 11.0, 15.0, 12.0, 14.0)
    lowest = (3.0, 1.0, -0.3, 2.0, 5.0)
    with pytest.warns(DynamicalInstabilityWarning):
        stable = select(volumes=volumes, lowest=lowest)
    assert tuple(volumes[i] for i in stable) == (11.0, 12.0, 13.0, 14.0)


def test_three_survivors_raise():
    with pytest.raises(ValueError, match="3 dynamically stable volume"):
        select(lowest=(1.0, -0.3, -0.3, 4.0, 5.0))


def test_error_names_the_cut_and_the_eos():
    with pytest.raises(ValueError) as excinfo:
        _select_stable_volumes(VOLUMES, (1.0, -0.3, -0.3, 4.0, 5.0), -0.05, "phase", "vinet")
    message = str(excinfo.value)
    assert "fitting vinet needs at least four" in message
    assert "-0.05 THz is the cut" in message


def test_min_frequency_decides_what_counts_as_unstable():
    """Same frequencies, two cuts: a lower one keeps the slightly imaginary volume."""
    lowest = (-0.2, 2.0, 3.0, 4.0, 5.0)
    with pytest.warns(DynamicalInstabilityWarning):
        assert select(lowest=lowest) == (1, 2, 3, 4)
    assert select(lowest=lowest, min_frequency=-0.5) == (0, 1, 2, 3, 4)


def test_no_warning_when_nothing_is_dropped():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DynamicalInstabilityWarning)
        assert select() == (0, 1, 2, 3, 4)


# _stationary_points works in the centred, scaled coordinate u = (V - mid) / span of the
# fit, on [-0.5, 0.5], and answers in volumes; two frames tell the rescale from a no-op
FRAMES = pytest.mark.parametrize("mid, span", [(0.0, 1.0), (12.0, 4.0)])
# round-off of the root finding and of the rescale, for results that are exact otherwise
ATOL = 1e-12
# spacing of the sign-change scan in u
SCAN_STEP = 1 / 256


def polynomial(*roots):
    """A derivative with the polynomial branch's ``.poly``, zero at ``roots``."""
    return PolynomialInterpolation(np.poly1d(roots, r=True))


@FRAMES
def test_polynomial_roots_are_recovered_in_volume_units(mid, span):
    roots = np.array([-0.3, 0.05, 0.35])
    found = np.sort(_stationary_points(polynomial(*roots), mid, span))
    np.testing.assert_allclose(found, roots * span + mid, rtol=0, atol=ATOL)


@FRAMES
def test_polynomial_roots_outside_the_fitted_range_are_returned(mid, span):
    """Restricting to the sampled volumes is the caller's job, not the root finder's."""
    roots = np.array([-2.0, 0.1, 3.0])
    found = np.sort(_stationary_points(polynomial(*roots), mid, span))
    np.testing.assert_allclose(found, roots * span + mid, rtol=0, atol=ATOL)


@FRAMES
def test_complex_roots_are_dropped_and_real_ones_kept(mid, span):
    """(u - 0.2) (u**2 + 1) has one real root and the pair +-i."""
    found = _stationary_points(polynomial(0.2, 1j, -1j), mid, span)
    assert len(found) == 1
    assert np.isrealobj(found[0])
    assert found[0] == pytest.approx(0.2 * span + mid, rel=0, abs=ATOL)


def test_polynomial_without_real_roots_has_no_stationary_point():
    assert _stationary_points(polynomial(1j, -1j), 12.0, 4.0) == []


@FRAMES
def test_scan_locates_a_linear_crossing_exactly(mid, span):
    """Between two samples the scan interpolates linearly, so a straight line is exact.

    0.123 is not a scan node, so returning either bracketing sample would be off by a
    sizeable fraction of ``SCAN_STEP``.
    """
    (found,) = _stationary_points(lambda u: u - 0.123, mid, span)
    assert found == pytest.approx(0.123 * span + mid, rel=0, abs=ATOL)


@FRAMES
def test_scan_locates_a_curved_crossing_within_the_interpolation_error(mid, span):
    """Linear interpolation of a smooth curve is second order in the scan spacing."""
    (found,) = _stationary_points(lambda u: np.tanh(3 * (u - 0.123)), mid, span)
    assert found == pytest.approx(0.123 * span + mid, rel=0, abs=span * SCAN_STEP**2)


@FRAMES
def test_scan_returns_every_crossing_in_scan_order(mid, span):
    """| |u| - 0.25 | - 0.1 changes sign at |u| = 0.15 and 0.35.

    Its kinks sit on scan nodes (0 and +-0.25), so it is linear in every interval that
    holds a crossing and all four come back exactly.
    """
    found = _stationary_points(lambda u: np.abs(np.abs(u) - 0.25) - 0.1, mid, span)
    expected = np.array([-0.35, -0.15, 0.15, 0.35]) * span + mid
    assert len(found) == 4
    np.testing.assert_allclose(found, expected, rtol=0, atol=ATOL)


def test_scan_ignores_crossings_outside_the_fitted_range():
    """Unlike the polynomial branch the scan only sees [-0.5, 0.5]."""
    assert _stationary_points(lambda u: u - 0.75, 12.0, 4.0) == []


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_scan_of_a_derivative_of_constant_sign_finds_nothing(sign):
    assert _stationary_points(lambda u: sign * (u**2 + 1), 12.0, 4.0) == []


def test_scan_returns_python_floats():
    (found,) = _stationary_points(lambda u: u - 0.123, 12.0, 4.0)
    assert type(found) is float
