"""The broadcast contract every :class:`~landau.phases.Phase` owes.

``semigrand_potential(T, dmu)``, ``concentration(T, dmu)`` and, where a phase
has one, ``free_energy(T, c)`` broadcast their two arguments like a numpy ufunc:
the result has the broadcast shape, a 0-d result is a scalar rather than an
array, and values are laid out by position, so each element equals the call on
that element's own scalar arguments.  Line phases owe the same for
``line_free_energy(T)``.

``calc_phase_diagram`` stacks the outputs of all phases in one frame, so a single
phase that breaks this (a shape-``(1,)`` array where the others return a float, a
dropped axis) breaks every diagram it takes part in.  A test that exercises one
phase class cannot see that class of breakage, so the property runs against all of
them here.  Every new phase gets an entry in ``CASES``.
"""

import warnings
from dataclasses import dataclass
from functools import cache, partial

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import BroadcastableShapes, mutually_broadcastable_shapes
from pyiron_snippets.import_alarm import ImportAlarm
from scipy.constants import Boltzmann, eV

from landau.interpolate import SGTE, CalphadSurface2DInterpolator
from landau.interpolate.basic import G_calphad
from landau.phases import (
    FastInterpolatingPhase,
    IdealSolution,
    InterpolatingPhase,
    LinePhase,
    RegularSolution,
    SlowInterpolatingPhase,
    Surface2DInterpolatingPhase,
    TemperatureDependentLinePhase,
)
from landau.phases.pointdefects import (
    CappedPointDefectSublattice,
    ConstantPointDefect,
    LowTemperatureExpansionSublattice,
    PointDefectedPhase,
    PointDefectSublattice,
)

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import HarmonicThermo

    from landau.phases.asewrapper import AsePhase

with ImportAlarm() as phonopy_alarm:
    import phonopy  # noqa: F401

kB = Boltzmann / eV

# Each element is checked against a call on its own scalar (T, dmu); the solvers are
# elementwise, so the two agree to round-off.
_PARITY_ATOL = 1e-10


def _solution_line_phases():
    return [LinePhase("A", 0.0, 0.0), LinePhase("mid", 0.5, -0.3), LinePhase("B", 1.0, 0.0)]


def _tdlp():
    T = np.linspace(300.0, 1200.0, 20)
    return TemperatureDependentLinePhase("tdlp", 0.5, T, G_calphad(T, 1e-4, -2.0, 5e-4), interpolator=SGTE(3))


def _b2(sublattice_class):
    host = LinePhase("AB", 0.5, -0.40, 1.0 * kB)
    alpha = sublattice_class(name="alpha", sublattice=0, sublattice_fraction=0.5, defects=[
        ConstantPointDefect("B_a", excess_energy=0.30, excess_entropy=0.0, excess_solutes=+1),
        ConstantPointDefect("V_a", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0),
    ])
    beta = sublattice_class(name="beta", sublattice=1, sublattice_fraction=0.5, defects=[
        ConstantPointDefect("A_b", excess_energy=0.45, excess_entropy=0.0, excess_solutes=-1),
        ConstantPointDefect("V_b", excess_energy=0.50, excess_entropy=0.0, excess_solutes=0),
    ])
    return PointDefectedPhase("AB", line_phase=host, sublattices=[alpha, beta])


def _qha():
    # the phonopy fixture lives with the rest of the phonopy tests
    from test_quasiharmonic import planted_phase

    return planted_phase()


@dataclass(frozen=True)
class Case:
    make: callable
    T: tuple
    dmu: tuple
    c: tuple = (0.05, 0.95)
    free_energy: bool = True
    """False for a phase without ``free_energy(T, c)``."""
    restrict: callable = None
    """Maps a drawn pair of argument shapes onto the shapes a phase supports, for the deprecated ones."""
    parity_atol: float | None = _PARITY_ATOL
    """None skips the elementwise check, for a phase whose elements depend on each other."""
    marks: tuple = ()


CASES = {
    "LinePhase": Case(lambda: LinePhase("line", 0.3, -1.0, 1e-4), (100.0, 1500.0), (-1.0, 1.0)),
    "TemperatureDependentLinePhase": Case(_tdlp, (300.0, 1200.0), (-1.0, 1.0)),
    "IdealSolution": Case(
        lambda: IdealSolution("ideal", LinePhase("A", 0, 0.05, 5e-4), LinePhase("B", 1, -0.2, 2e-4)),
        (100.0, 1500.0), (-1.0, 1.0),
    ),
    # The two deprecated phases are pinned only as far as they go.  RegularSolution
    # caches its fit per T (T must be a hashable scalar), and its concentration is a
    # finite difference across the dmu array it is handed: 1-D, at least three
    # points, and each element depends on its neighbours.
    "RegularSolution": Case(
        lambda: RegularSolution("regular", _solution_line_phases()), (400.0, 1500.0), (-0.5, 0.5),
        restrict=lambda T_shape, dmu_shape: ((), dmu_shape and (max(3, int(np.prod(dmu_shape))),)),
        parity_atol=None,
    ),
    # InterpolatingPhase needs T to have at least as many dimensions as dmu.
    "InterpolatingPhase": Case(
        lambda: InterpolatingPhase("interp", _solution_line_phases()), (400.0, 1500.0), (-0.5, 0.5),
        restrict=lambda T_shape, dmu_shape: ((1,) * (len(dmu_shape) - len(T_shape)) + T_shape, dmu_shape),
    ),
    "SlowInterpolatingPhase": Case(
        lambda: SlowInterpolatingPhase("slow", _solution_line_phases()), (400.0, 1500.0), (-0.5, 0.5)
    ),
    "FastInterpolatingPhase": Case(
        lambda: FastInterpolatingPhase("fast", _solution_line_phases()), (400.0, 1500.0), (-0.5, 0.5)
    ),
    "Surface2DInterpolatingPhase": Case(
        lambda: Surface2DInterpolatingPhase(
            "surface", _solution_line_phases(), temperature_range=(300.0, 1200.0),
            surface_interpolator=CalphadSurface2DInterpolator(num_coeffs=1),
        ),
        (400.0, 1100.0), (-0.5, 0.5),
    ),
    # dmu reaches past both saturation points of the unbounded LTE model
    "PointDefectedPhase[lte]": Case(
        partial(_b2, LowTemperatureExpansionSublattice), (100.0, 2000.0), (-0.6, 0.6), free_energy=False
    ),
    "PointDefectedPhase[exact]": Case(
        partial(_b2, PointDefectSublattice), (100.0, 2000.0), (-0.6, 0.6), free_energy=False
    ),
    "PointDefectedPhase[capped]": Case(
        partial(_b2, partial(CappedPointDefectSublattice, max_site_fraction=0.2)), (100.0, 2000.0), (-0.6, 0.6),
        free_energy=False,
    ),
    "AsePhase": Case(
        lambda: AsePhase("ase", 0.5, thermochem=HarmonicThermo(vib_energies=[0.1]), pressure=None),
        (100.0, 1000.0), (-1.0, 1.0),
        marks=(pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed"),),
    ),
    "PhonopyQuasiHarmonicPhase": Case(
        _qha, (0.0, 1000.0), (-1.0, 1.0),
        marks=(pytest.mark.skipif(phonopy_alarm.message is not None, reason="phonopy is not installed"),),
    ),
}

LINE_PHASES = ("LinePhase", "TemperatureDependentLinePhase", "AsePhase", "PhonopyQuasiHarmonicPhase")


# method -> the CASES attribute bounding its second argument
METHODS = {"semigrand_potential": "dmu", "concentration": "dmu", "free_energy": "c"}


def _params(names):
    return [pytest.param(name, marks=CASES[name].marks, id=name) for name in names]


def _method_params():
    return [
        pytest.param(name, method, marks=case.marks, id=f"{name}-{method}")
        for name, case in CASES.items()
        for method in METHODS
        if method != "free_energy" or case.free_energy
    ]


@cache
def _phase(name):
    """Build each phase once; ``@given`` re-runs its body ~100 times."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return CASES[name].make()


def _draw(shape, bounds, rng):
    lo, hi = bounds
    return lo + (hi - lo) * rng.random(shape)


def _assert_contract(out, expected_shape):
    assert np.shape(out) == expected_shape
    if expected_shape == ():
        # a scalar, not a 0-d or shape-(1,) array: float() and math work on it
        assert not isinstance(out, np.ndarray), f"got a 0-d {type(out).__name__}"
        float(out)
    else:
        assert out.dtype == float
    assert np.all(np.isfinite(out))


# edge shapes hypothesis need not hit on every run: size-1 axes (#1), 0-d arrays,
# and an outer product
_EDGE_SHAPES = [
    BroadcastableShapes(input_shapes=((), ()), result_shape=()),
    BroadcastableShapes(input_shapes=((1,), ()), result_shape=(1,)),
    BroadcastableShapes(input_shapes=((), (1,)), result_shape=(1,)),
    BroadcastableShapes(input_shapes=((1, 1), (1,)), result_shape=(1, 1)),
    BroadcastableShapes(input_shapes=((2, 1), (3,)), result_shape=(2, 3)),
]


def _with_edge_shapes(test):
    for shapes in _EDGE_SHAPES:
        test = example(shapes=shapes, seed=0)(test)
    return test


@pytest.mark.parametrize("name, method", _method_params())
@settings(deadline=None, max_examples=50)
@_with_edge_shapes
@given(shapes=mutually_broadcastable_shapes(num_shapes=2, max_dims=3, max_side=3), seed=st.integers(0, 2**32 - 1))
def test_two_argument_broadcast(name, method, shapes, seed):
    """Any pair of broadcastable (T, dmu) or (T, c) shapes in, the broadcast shape out."""
    case = CASES[name]
    f = getattr(_phase(name), method)
    T_shape, x_shape = shapes.input_shapes
    if case.restrict is not None:
        T_shape, x_shape = case.restrict(T_shape, x_shape)
    rng = np.random.default_rng(seed)
    T = _draw(T_shape, case.T, rng)
    x = _draw(x_shape, getattr(case, METHODS[method]), rng)

    out = f(T, x)

    _assert_contract(out, np.broadcast_shapes(T_shape, x_shape))
    if case.parity_atol is None:
        return
    # values sit at the position of their own arguments, checked against Python-float calls
    Tb, xb = np.broadcast_arrays(T, x)
    expected = [f(float(t), float(v)) for t, v in zip(Tb.ravel(), xb.ravel())]
    np.testing.assert_allclose(np.ravel(out), expected, rtol=0, atol=case.parity_atol)


@pytest.mark.parametrize("name", _params(LINE_PHASES))
@settings(deadline=None, max_examples=50)
@_with_edge_shapes
@given(shapes=mutually_broadcastable_shapes(num_shapes=2, max_dims=3, max_side=3), seed=st.integers(0, 2**32 - 1))
def test_line_free_energy_follows_T(name, shapes, seed):
    """``line_free_energy`` keeps the shape of ``T``, 0-d included."""
    case = CASES[name]
    f = _phase(name).line_free_energy
    T = _draw(shapes.input_shapes[0], case.T, np.random.default_rng(seed))

    out = f(T)

    _assert_contract(out, T.shape)
    np.testing.assert_allclose(np.ravel(out), [f(float(t)) for t in T.ravel()], rtol=0, atol=case.parity_atol)
