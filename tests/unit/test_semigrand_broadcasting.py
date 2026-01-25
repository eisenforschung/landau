import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from landau.phases import (
    LinePhase,
    TemperatureDependentLinePhase,
    IdealSolution,
    RegularSolution,
    InterpolatingPhase,
    SlowInterpolatingPhase,
    PointDefectedPhase,
    PointDefectSublattice,
    ConstantPointDefect
)

def get_phases():
    p0 = LinePhase(name="A", fixed_concentration=0, line_energy=0, line_entropy=0)
    p1 = LinePhase(name="B", fixed_concentration=1, line_energy=1, line_entropy=0)

    tp = TemperatureDependentLinePhase(
        name="C",
        fixed_concentration=0.5,
        temperatures=[500, 1000, 1500],
        free_energies=[0.1, 0.2, 0.3]
    )

    ideal = IdealSolution(name="Ideal", phase1=p0, phase2=p1)
    regular = RegularSolution(name="Regular", phases=[p0, p1])
    interp = InterpolatingPhase(name="Interp", phases=[p0, p1])
    slow = SlowInterpolatingPhase(name="Slow", phases=[p0, p1])

    defect = ConstantPointDefect(name="vac", excess_energy=1.0, excess_entropy=0, excess_solutes=1)
    sublattice = PointDefectSublattice(name="sub", sublattice=0, sublattice_fraction=1.0, defects=[defect])
    point_defected = PointDefectedPhase(name="Defected", line_phase=p0, sublattices=[sublattice])

    return [p0, tp, ideal, regular, interp, slow, point_defected]

@st.composite
def broadcastable_pair(draw):
    base_shape = draw(st.lists(st.integers(1, 3), min_size=0, max_size=3).map(tuple))

    def get_broadcastable(shape):
        new_shape = list(shape)
        # Randomly change some dimensions to 1
        for i in range(len(new_shape)):
            if draw(st.booleans()):
                new_shape[i] = 1
        # Randomly add leading 1s
        while len(new_shape) < 3 and draw(st.booleans()):
            new_shape.insert(0, 1)
        return tuple(new_shape)

    return get_broadcastable(base_shape), get_broadcastable(base_shape)

@pytest.mark.parametrize("phase", get_phases(), ids=lambda p: p.name)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=10)
@given(st.data())
def test_semigrand_potential_broadcasting(phase, data):
    # Generate broadcastable shapes
    shape_T, shape_dmu = data.draw(broadcastable_pair())

    # Generate random values for T and dmu
    size_T = int(np.prod(shape_T, dtype=int))
    T_vals = data.draw(st.lists(st.floats(min_value=100, max_value=2000, allow_nan=False, allow_infinity=False),
                                min_size=size_T, max_size=size_T).map(sorted))
    T = np.array(T_vals).reshape(shape_T)

    size_dmu = int(np.prod(shape_dmu, dtype=int))
    dmu_vals = data.draw(st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False),
                                  min_size=size_dmu, max_size=size_dmu).map(sorted))
    dmu = np.array(dmu_vals).reshape(shape_dmu)

    expected_shape = np.broadcast_shapes(T.shape, dmu.shape)

    res = phase.semigrand_potential(T, dmu)

    assert np.shape(res) == expected_shape

@pytest.mark.parametrize("phase", get_phases(), ids=lambda p: p.name)
@settings(deadline=None, max_examples=10)
@given(
    T=st.lists(st.floats(min_value=100, max_value=2000), min_size=1, max_size=5).map(sorted),
    dmu=st.lists(st.floats(min_value=-1, max_value=1), min_size=1, max_size=5).map(sorted)
)
def test_semigrand_potential_list_input(phase, T, dmu):
    # Ensure T and dmu have the same length for this specific test case
    min_len = min(len(T), len(dmu))
    T = T[:min_len]
    dmu = dmu[:min_len]

    res = phase.semigrand_potential(T, dmu)

    assert np.shape(res) == (min_len,)
