# https://github.com/eisenforschung/landau/issues/428
"""Regression tests for #428: the scalar-in/scalar-out contract of every
:class:`~landau.interpolate.basic.TemperatureInterpolator`.

``WhitneyTemperatureInterpolator.fit`` promoted scalar ``T`` with
``np.atleast_1d`` and never demoted the result, so ``line_free_energy(500.0)``
returned a shape-``(1,)`` array where ``PolyFit``/``SGTE`` return a scalar.
Beyond ``float()``/``math.exp()`` failing on the value, one such phase among
ordinary phases made ``np.array([p.semigrand_potential(T, mu) for p in phases])``
in :func:`~landau.calculate._semigrand_average_concentration` ragged, so
``calc_phase_diagram`` raised an "inhomogeneous shape" ``ValueError`` naming
neither Whitney nor the phase.

The parametrisation covers every concrete ``TemperatureInterpolator``;
``test_interpolator_list_is_exhaustive`` fails if a new one is added without
being pinned here, because this class of breakage only shows up once an
interpolator is driven through a phase.
"""

import inspect
import math

import numpy as np
import pytest

import landau.interpolate as interpolate_pkg
from landau.calculate import calc_phase_diagram
from landau.interpolate import (
    PolyFit,
    SGTE,
    SoftplusFit,
    StitchedFit,
    TemperatureInterpolator,
    WhitneyTemperatureInterpolator,
)
from landau.phases import IdealSolution, LinePhase, TemperatureDependentLinePhase

# every concrete TemperatureInterpolator, with arguments that fit the data below
INTERPOLATORS = {
    "PolyFit": PolyFit(4),
    "SGTE": SGTE(3),
    "StitchedFit": StitchedFit(),
    "SoftplusFit": SoftplusFit(),
    "WhitneyTemperatureInterpolator": WhitneyTemperatureInterpolator(),
}

TEMPERATURES = np.linspace(100.0, 1500.0, 40)
FREE_ENERGIES = -3.0 - 3e-4 * TEMPERATURES - 1e-7 * TEMPERATURES**2


def make_phase(interpolator):
    """The intermetallic AB, free energies sampled on :data:`TEMPERATURES`."""
    return TemperatureDependentLinePhase(
        name="AB",
        fixed_concentration=0.5,
        temperatures=TEMPERATURES,
        free_energies=FREE_ENERGIES,
        interpolator=interpolator,
    )


def make_phases(interpolator):
    """A/B terminals + ideal solution + the interpolated intermetallic.

    The terminals return plain scalars, so a shape-``(1,)`` element from the
    intermetallic is what makes the collected array ragged.
    """
    a = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=0)
    b = LinePhase("B", fixed_concentration=1, line_energy=-2.5, line_entropy=0)
    return [a, b, IdealSolution("sol", a, b), make_phase(interpolator)]


def test_interpolator_list_is_exhaustive():
    """Every concrete TemperatureInterpolator exported from landau.interpolate is pinned."""
    exported = {
        name
        for name in interpolate_pkg.__all__
        if inspect.isclass(cls := getattr(interpolate_pkg, name))
        and issubclass(cls, TemperatureInterpolator)
        and not inspect.isabstract(cls)
    }
    assert exported == set(INTERPOLATORS), (
        f"unpinned TemperatureInterpolator(s): {sorted(exported - set(INTERPOLATORS))}; "
        f"stale entries: {sorted(set(INTERPOLATORS) - exported)}"
    )


@pytest.mark.parametrize("interpolator", INTERPOLATORS.values(), ids=list(INTERPOLATORS))
def test_scalar_temperature_gives_python_scalar(interpolator):
    """Scalar T in, 0-d value out — usable by ``float()`` and ``math.exp()``."""
    f = make_phase(interpolator).line_free_energy(500.0)

    assert np.ndim(f) == 0, f"expected a scalar, got shape {np.shape(f)}"
    assert math.isfinite(float(f))
    math.exp(f)  # raises TypeError on a shape-(1,) array


@pytest.mark.parametrize("interpolator", INTERPOLATORS.values(), ids=list(INTERPOLATORS))
def test_array_temperature_preserves_shape(interpolator):
    """Array T in, same-shape array out, and it agrees with the scalar call."""
    phase = make_phase(interpolator)
    Ts = np.array([400.0, 500.0, 600.0])
    f = phase.line_free_energy(Ts)

    assert np.shape(f) == Ts.shape
    np.testing.assert_allclose(f[1], phase.line_free_energy(500.0), rtol=1e-12, atol=0)


@pytest.mark.parametrize("interpolator", INTERPOLATORS.values(), ids=list(INTERPOLATORS))
def test_calc_phase_diagram_runs_with_every_interpolator(interpolator):
    """``mu`` as an int routes through ``guess_mu_range``, which evaluates every
    phase at a scalar ``mu`` — the call that went ragged before the fix."""
    df = calc_phase_diagram(
        make_phases(interpolator),
        Ts=np.linspace(200.0, 1200.0, 6),
        mu=21,
        refine=False,
        keep_unstable=True,
    )

    assert not df.empty
    assert set(df["phase"].unique()) == {"A", "B", "sol", "AB"}
    # the intermetallic is the deepest phase at c=0.5; it must win somewhere
    assert df.query("stable and phase == 'AB'").shape[0] > 0
    assert np.isfinite(df["phi"].astype(float)).all()
