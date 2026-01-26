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
    # Output shape: up to five random integers
    output_shape = draw(st.lists(st.integers(1, 5), min_size=0, max_size=5).map(tuple))

    def get_input_shape(base_shape):
        shape = list(base_shape)
        # Toggle random axes to 1
        for i in range(len(shape)):
            if draw(st.booleans()):
                shape[i] = 1
        # Randomly add leading 1s
        while len(shape) < 5 and draw(st.booleans()):
            shape.insert(0, 1)
        return tuple(shape)

    return get_input_shape(output_shape), get_input_shape(output_shape)

@pytest.mark.parametrize("phase", get_phases(), ids=lambda p: p.name)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], max_examples=10)
@given(st.data())
def test_semigrand_potential_broadcasting(phase, data):
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

    # Randomly convert some 0-dim arrays to scalars
    if T.shape == () and data.draw(st.booleans()):
        T = T.item()
    if dmu.shape == () and data.draw(st.booleans()):
        dmu = dmu.item()

    expected_shape = np.broadcast_shapes(np.shape(T), np.shape(dmu))

    res = phase.semigrand_potential(T, dmu)

    assert np.shape(res) == expected_shape

@pytest.mark.parametrize("phase", get_phases(), ids=lambda p: p.name)
@settings(deadline=None, max_examples=10)
@given(st.data())
def test_semigrand_potential_list_input(phase, data):
    # Two random lists of the same length, in the range 10-100
    length = data.draw(st.integers(10, 100))
    T = data.draw(st.lists(st.floats(min_value=100, max_value=2000), min_size=length, max_size=length).map(sorted))
    dmu = data.draw(st.lists(st.floats(min_value=-1, max_value=1), min_size=length, max_size=length).map(sorted))

    res = phase.semigrand_potential(T, dmu)

    assert np.shape(res) == (length,)
