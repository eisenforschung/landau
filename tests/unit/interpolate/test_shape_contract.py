"""The shape contract every :class:`~landau.interpolate.basic.Interpolation` owes.

Scalar in, 0-d out; array in, array out of the same shape; and values laid out by
position, so ``ravel(f(x)) == f(ravel(x))``.

Ported from the Whitney shape test added with #428, where the contract broke:
``WhitneyTemperatureInterpolator`` returned a shape-``(1,)`` array for scalar input,
and one phase backed by it among phases returning plain scalars made
``calc_phase_diagram`` raise an "inhomogeneous shape" ``ValueError``.  A test that
exercises one interpolator cannot see that class of breakage, so the property runs
against all of them here, and against each :meth:`~.Interpolation.deriv` too.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from landau.interpolate import (
    PolyFit,
    RedlichKister,
    SGTE,
    SoftplusFit,
    SplineFit,
    StitchedFit,
)

# concentration axis: RedlichKister needs the terminals, the rest do not mind them
C = np.linspace(0.0, 1.0, 30)
F_C = 0.4 * (C - 0.5) ** 2
# temperature axis
T = np.linspace(300.0, 1000.0, 30)
F_T = -T * np.log(T) + 0.01 * T

CASES = {
    "PolyFit(4) in T": (PolyFit(4), T, F_T),
    "PolyFit(4) in c": (PolyFit(4), C, F_C),
    "SplineFit in c": (SplineFit(), C, F_C),
    "SGTE(3) in T": (SGTE(3), T, F_T),
    "StitchedFit in T": (StitchedFit(), T, F_T),
    "SoftplusFit in c": (SoftplusFit(), C, F_C),
    "RedlichKister(3) in c": (RedlichKister(3), C, F_C),
}

@pytest.fixture(scope="module")
def fits():
    """Fit each interpolator once; ``@given`` re-runs its body ~100 times."""
    return {name: (itp.fit(x, y), x.min(), x.max()) for name, (itp, x, y) in CASES.items()}


def _branch(interpolation, derivative):
    return interpolation.deriv() if derivative else interpolation


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("derivative", [False, True], ids=["f", "deriv"])
@given(
    unit=arrays(
        dtype=float,
        shape=array_shapes(min_dims=0, max_dims=3, min_side=1, max_side=4),
        elements=st.floats(min_value=0.0, max_value=1.0),
    )
)
def test_output_shape_matches_input_shape(fits, name, derivative, unit):
    """Any input shape in, the same shape out — 0-d included."""
    interpolation, lo, hi = fits[name]
    f = _branch(interpolation, derivative)
    x = lo + unit * (hi - lo)

    out = f(x)

    assert np.shape(out) == x.shape
    if x.ndim == 0:
        # a 0-d value, not a shape-(1,) array: float() must work on it
        float(out)
    else:
        # ravelling the input ravels the output identically, so values are laid
        # out by position rather than merely counted
        np.testing.assert_array_equal(np.ravel(out), f(x.ravel()))


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("derivative", [False, True], ids=["f", "deriv"])
def test_python_scalar_in_scalar_out(fits, name, derivative):
    """A plain Python float in gives a 0-d value out, usable by ``float()``."""
    interpolation, lo, hi = fits[name]
    f = _branch(interpolation, derivative)

    out = f(float(0.5 * (lo + hi)))

    assert np.ndim(out) == 0
    assert np.isfinite(float(out))
