"""
Broadcasting tests for Phase.semigrand_potential.

Tests verify that all phase objects handle various input shapes (scalars,
0-dim arrays, 1-dim arrays, multi-dim arrays) correctly and follow NumPy
broadcasting rules.
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from landau.interpolate import SGTE
from landau.interpolate.basic import G_calphad
from landau.phases import (
    ConstantPointDefect,
    IdealSolution,
    InterpolatingPhase,
    LinePhase,
    PointDefectSublattice,
    PointDefectedPhase,
    RegularSolution,
    TemperatureDependentLinePhase,
)


# ---------------------------------------------------------------------------
# Phase factories
# ---------------------------------------------------------------------------

def _lp(c=0.0, energy=-1.0):
    return LinePhase(name=f"lp_{c}", fixed_concentration=c, line_energy=energy, line_entropy=5e-5)


def _tdlp(c=0.5):
    T = np.linspace(300, 1200, 20)
    f = G_calphad(T, 1e-4, -2.0, 5e-4)
    return TemperatureDependentLinePhase(
        name="tdlp", fixed_concentration=c, temperatures=T, free_energies=f,
        interpolator=SGTE(3),
    )


def make_line_phase():
    return _lp(c=0.5)


def make_tdlp():
    return _tdlp()


def make_ideal_solution():
    return IdealSolution(name="ideal", phase1=_lp(c=0.0), phase2=_lp(c=1.0))


def make_regular_solution():
    return RegularSolution(
        name="regular",
        phases=[_lp(c=0.0), _lp(c=0.5), _lp(c=1.0)],
    )


def make_interpolating_phase():
    return InterpolatingPhase(
        name="interp",
        phases=[_lp(c=0.0), _lp(c=0.5), _lp(c=1.0)],
    )


def make_point_defected_phase():
    host = _lp(c=0.0)
    defect = ConstantPointDefect(
        name="vac", excess_energy=0.5, excess_entropy=1e-4, excess_solutes=0.0
    )
    sub = PointDefectSublattice(
        name="A", sublattice=0, sublattice_fraction=1.0, defects=[defect]
    )
    return PointDefectedPhase(name="pdp", line_phase=host, sublattices=[sub])


# Phases that should support full N-dim broadcasting.
BROADCAST_PHASES = [
    make_line_phase,
    make_tdlp,
    make_ideal_solution,
    make_interpolating_phase,
    make_point_defected_phase,
]

# All phases, including RegularSolution which only supports scalar T.
ALL_PHASES = BROADCAST_PHASES + [make_regular_solution]


# ---------------------------------------------------------------------------
# Hypothesis strategy: two shapes that broadcast to a common output shape.
# Axes are only toggled to 1; no leading ones are added.
# ---------------------------------------------------------------------------

@st.composite
def broadcastable_pair(draw):
    ndim = draw(st.integers(min_value=1, max_value=5))
    output_shape = draw(
        st.lists(st.integers(min_value=1, max_value=4), min_size=ndim, max_size=ndim)
    )

    T_shape = list(output_shape)
    dmu_shape = list(output_shape)

    for i in range(ndim):
        # 0 → toggle T axis to 1, 1 → toggle dmu axis to 1, 2 → keep both
        choice = draw(st.integers(min_value=0, max_value=2))
        if choice == 0:
            T_shape[i] = 1
        elif choice == 1:
            dmu_shape[i] = 1

    return tuple(T_shape), tuple(dmu_shape), tuple(output_shape)


# ---------------------------------------------------------------------------
# Broadcasting shape test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_phase", BROADCAST_PHASES)
@given(
    shapes=broadcastable_pair(),
    T_val=st.floats(min_value=300.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
    dmu_val=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20)
def test_semigrand_potential_broadcasting(make_phase, shapes, T_val, dmu_val):
    """Output shape matches the broadcast of T and dmu input shapes."""
    T_shape, dmu_shape, expected_shape = shapes
    phase = make_phase()

    T = np.full(T_shape, T_val)
    dmu = np.full(dmu_shape, dmu_val)

    result = phase.semigrand_potential(T, dmu)
    result_arr = np.asarray(result)

    assert result_arr.shape == expected_shape, (
        f"{type(phase).__name__}: expected shape {expected_shape}, got {result_arr.shape} "
        f"(T_shape={T_shape}, dmu_shape={dmu_shape})"
    )


# ---------------------------------------------------------------------------
# List input test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_phase", BROADCAST_PHASES)
@given(
    T_vals=st.lists(
        st.floats(min_value=300.0, max_value=1200.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    ),
    dmu_vals=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    ),
)
@settings(max_examples=20)
def test_semigrand_potential_list_inputs(make_phase, T_vals, dmu_vals):
    """semigrand_potential accepts same-length Python lists for T and dmu."""
    n = min(len(T_vals), len(dmu_vals))
    T_vals = T_vals[:n]
    dmu_vals = dmu_vals[:n]

    phase = make_phase()
    result = phase.semigrand_potential(T_vals, dmu_vals)
    result_arr = np.asarray(result)

    assert result_arr.ndim == 1 and len(result_arr) == n, (
        f"{type(phase).__name__}: expected 1-d array of length {n}, got shape {result_arr.shape}"
    )


# ---------------------------------------------------------------------------
# Scalar / 0-dim / 1-dim type combination tests.
#
# Allowed return types: Python scalar (not np.ndarray) or 1-d ndarray.
# 0-dim ndarrays are NOT acceptable returns.
# 1-d/1-d combinations are excluded (tested via broadcasting test above).
# ---------------------------------------------------------------------------

_T_SCALAR = 700.0
_DMU_SCALAR = 0.05
_T_0D = np.array(700.0)
_DMU_0D = np.array(0.05)
_T_1D = np.array([500.0, 700.0, 900.0])
_DMU_1D = np.array([-0.1, 0.0, 0.1])


def _assert_scalar(result):
    assert not isinstance(result, np.ndarray), (
        f"Expected Python scalar, got {type(result).__name__} "
        f"(ndim={np.ndim(result)})"
    )


def _assert_1d(result, expected_length):
    assert isinstance(result, np.ndarray) and result.ndim == 1, (
        f"Expected 1-d ndarray, got {type(result).__name__} "
        f"(ndim={np.ndim(result)})"
    )
    assert len(result) == expected_length


@pytest.mark.parametrize("make_phase", ALL_PHASES)
def test_scalar_scalar_returns_scalar(make_phase):
    result = make_phase().semigrand_potential(_T_SCALAR, _DMU_SCALAR)
    _assert_scalar(result)


@pytest.mark.parametrize("make_phase", ALL_PHASES)
def test_scalar_0dim_returns_scalar(make_phase):
    result = make_phase().semigrand_potential(_T_SCALAR, _DMU_0D)
    _assert_scalar(result)


@pytest.mark.parametrize("make_phase", ALL_PHASES)
def test_0dim_scalar_returns_scalar(make_phase):
    result = make_phase().semigrand_potential(_T_0D, _DMU_SCALAR)
    _assert_scalar(result)


@pytest.mark.parametrize("make_phase", ALL_PHASES)
def test_0dim_0dim_returns_scalar(make_phase):
    result = make_phase().semigrand_potential(_T_0D, _DMU_0D)
    _assert_scalar(result)


@pytest.mark.parametrize("make_phase", ALL_PHASES)
def test_scalar_1dim_returns_1dim(make_phase):
    result = make_phase().semigrand_potential(_T_SCALAR, _DMU_1D)
    _assert_1d(result, len(_DMU_1D))


@pytest.mark.parametrize("make_phase", ALL_PHASES)
def test_0dim_1dim_returns_1dim(make_phase):
    result = make_phase().semigrand_potential(_T_0D, _DMU_1D)
    _assert_1d(result, len(_DMU_1D))


@pytest.mark.parametrize("make_phase", ALL_PHASES)
def test_1dim_scalar_returns_1dim(make_phase):
    result = make_phase().semigrand_potential(_T_1D, _DMU_SCALAR)
    _assert_1d(result, len(_T_1D))


@pytest.mark.parametrize("make_phase", ALL_PHASES)
def test_1dim_0dim_returns_1dim(make_phase):
    result = make_phase().semigrand_potential(_T_1D, _DMU_0D)
    _assert_1d(result, len(_T_1D))
