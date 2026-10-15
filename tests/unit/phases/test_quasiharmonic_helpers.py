"""Direct tests for the pure helpers of ``landau.phases.quasiharmonic``.

The module imports without the ``phonopy`` extra -- only the functions that touch a
``ThermalProperties`` are decorated with its import alarm -- so these need no skip
marker, unlike the phase-level tests in ``test_quasiharmonic.py``.
"""

import warnings

import pytest

from landau.interpolate import PolyFit
from landau.phases.quasiharmonic import DynamicalInstabilityWarning, _select_stable_volumes

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
