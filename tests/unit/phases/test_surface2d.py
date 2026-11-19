"""Direct tests for ``Surface2DInterpolatingPhase._gather_training_data``.

The training triple ``(T, c, H)`` is checked against the line phases themselves,
independent of any surface fit; the interpolator is never called.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from landau.interpolate import PolyFit, SoftplusSurface2DInterpolator
from landau.phases import LinePhase, S, Surface2DInterpolatingPhase, TemperatureDependentLinePhase


def _line(c, Tlo, Thi):
    T = np.linspace(Tlo, Thi, 11)
    return TemperatureDependentLinePhase("s", c, T, -0.1 - 1e-4 * T + 0.2 * (c - 0.5) ** 2, interpolator=PolyFit(2))


def _phase(lines, **kwargs):
    return Surface2DInterpolatingPhase(
        "s", lines, surface_interpolator=SoftplusSurface2DInterpolator(), **kwargs
    )


# Two interior concentrations, where S(c) != 0, on deliberately different T windows.
LINES = [_line(0.3, 300.0, 600.0), _line(0.6, 400.0, 900.0)]


def test_explicit_temperature_range_wins():
    n = 7
    TT, CC, FF = _phase(LINES, temperature_range=(500, 550), num_temperature_samples=n)._gather_training_data()
    assert len(TT) == len(CC) == len(FF) == len(LINES) * n
    assert_array_equal(TT, np.tile(np.linspace(500.0, 550.0, n), len(LINES)))


def test_default_range_is_union_of_line_phase_windows():
    n = 5
    TT, _, _ = _phase(LINES, num_temperature_samples=n)._gather_training_data()
    assert_array_equal(TT, np.tile(np.linspace(300.0, 900.0, n), len(LINES)))


def test_no_range_and_no_sampled_temperatures_raises():
    lines = [LinePhase("a", 0.3, -0.1), LinePhase("b", 0.6, -0.2)]
    with pytest.raises(ValueError, match="temperature_range"):
        _phase(lines)._gather_training_data()


@pytest.mark.parametrize("add_entropy", [False, True])
def test_concentrations_and_free_energies_come_from_the_line_phases(add_entropy):
    n = 6
    _, CC, FF = _phase(LINES, num_temperature_samples=n, add_entropy=add_entropy)._gather_training_data()
    tg = np.linspace(300.0, 900.0, n)
    for i, p in enumerate(LINES):
        block = slice(i * n, (i + 1) * n)
        assert_array_equal(CC[block], np.full(n, p.line_concentration))
        f = np.asarray(p.line_free_energy(tg), float)
        expected = f if add_entropy else f + tg * S(np.full(n, p.line_concentration))
        assert_array_equal(FF[block], expected)


def test_add_entropy_branches_differ_by_T_S():
    n = 6
    _, CC, F_raw = _phase(LINES, num_temperature_samples=n, add_entropy=True)._gather_training_data()
    TT, _, H = _phase(LINES, num_temperature_samples=n, add_entropy=False)._gather_training_data()
    diff = H - F_raw
    assert (diff > 0).all()
    np.testing.assert_allclose(diff, TT * S(CC), rtol=1e-12, atol=0)
