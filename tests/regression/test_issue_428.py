# https://github.com/eisenforschung/landau/issues/428
"""Regression tests for #428: WhitneyTemperatureInterpolator's scalar contract.

``WhitneyTemperatureInterpolator.fit`` promoted scalar ``T`` with ``np.atleast_1d``
and never demoted the result, so ``line_free_energy(500.0)`` returned a
shape-``(1,)`` array where every other ``TemperatureInterpolator`` returns a
scalar.  Beyond ``float()``/``math.exp()`` failing on the value, one such phase
among ordinary phases made
``np.array([p.semigrand_potential(T, mu) for p in phases])`` in
:func:`~landau.calculate._semigrand_average_concentration` ragged, so
``calc_phase_diagram`` raised an "inhomogeneous shape" ``ValueError`` naming
neither Whitney nor the phase.

The phase-diagram test is the one that could not be written against the
interpolator alone: it needs a Whitney-backed phase sitting next to phases that
return plain scalars.
"""

import math

import numpy as np

from landau.calculate import calc_phase_diagram
from landau.interpolate import WhitneyTemperatureInterpolator
from landau.phases import TemperatureDependentLinePhase

TEMPERATURES = np.linspace(100.0, 1500.0, 40)
FREE_ENERGIES = -0.05 - 3e-4 * TEMPERATURES - 1e-7 * TEMPERATURES**2


def whitney_line_phase():
    """An intermetallic at c=0.5 whose F(T) is interpolated by Whitney."""
    return TemperatureDependentLinePhase(
        name="AB",
        fixed_concentration=0.5,
        temperatures=TEMPERATURES,
        free_energies=FREE_ENERGIES,
        interpolator=WhitneyTemperatureInterpolator(),
    )


def test_scalar_temperature_gives_python_scalar():
    """Scalar T in, 0-d value out — usable by ``float()`` and ``math.exp()``."""
    f = whitney_line_phase().line_free_energy(500.0)

    assert np.ndim(f) == 0, f"expected a scalar, got shape {np.shape(f)}"
    assert math.isfinite(float(f))
    math.exp(f)  # raises TypeError on a shape-(1,) array


def test_array_temperature_preserves_shape():
    """Array T in, same-shape array out, and it agrees with the scalar call."""
    phase = whitney_line_phase()
    Ts = np.array([400.0, 500.0, 600.0])
    f = phase.line_free_energy(Ts)

    assert np.shape(f) == Ts.shape
    np.testing.assert_allclose(f[1], phase.line_free_energy(500.0), rtol=1e-12, atol=0)


def test_calc_phase_diagram_runs_with_a_whitney_phase(two_phase_ideal):
    """A Whitney-backed phase alongside phases that return plain scalars.

    ``mu`` as an int routes through ``guess_mu_range``, which evaluates every
    phase at a scalar ``mu`` — the call that went ragged before the fix.
    """
    df = calc_phase_diagram(
        two_phase_ideal + [whitney_line_phase()],
        Ts=np.linspace(200.0, 1200.0, 6),
        mu=21,
        refine=False,
        keep_unstable=True,
    )

    assert set(df["phase"].unique()) == {"A", "B", "sol", "AB"}
    # the intermetallic is the deepest phase at c=0.5; it must win somewhere
    assert not df.query("stable and phase == 'AB'").empty
    assert np.isfinite(df["phi"].astype(float)).all()
