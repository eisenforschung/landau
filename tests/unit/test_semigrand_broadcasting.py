import numpy as np
import pytest
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

@pytest.mark.parametrize("phase", get_phases(), ids=lambda p: p.name)
@pytest.mark.parametrize("T_shape, dmu_shape", [
    ((), ()),           # Scalar, Scalar
    ((5,), ()),         # 1D Array, Scalar
    ((), (5,)),         # Scalar, 1D Array
    ((5,), (5,)),       # 1D Array, 1D Array
    ((3, 1), (1, 4)),   # Broadcasting (3,1) and (1,4) -> (3,4)
])
def test_semigrand_potential_broadcasting(phase, T_shape, dmu_shape):
    T_val = 1000
    dmu_val = 0.1

    if T_shape == ():
        T = float(T_val)
    else:
        T = np.full(T_shape, float(T_val))

    if dmu_shape == ():
        dmu = float(dmu_val)
    else:
        dmu = np.full(dmu_shape, float(dmu_val))

    expected_shape = np.broadcast_shapes(T_shape, dmu_shape)

    res = phase.semigrand_potential(T, dmu)

    assert np.shape(res) == expected_shape

@pytest.mark.parametrize("phase", get_phases(), ids=lambda p: p.name)
def test_semigrand_potential_list_input(phase):
    T = [500, 1000, 1500]
    dmu = [0.1, 0.2, 0.3]

    res = phase.semigrand_potential(T, dmu)

    assert np.shape(res) == (3,)
