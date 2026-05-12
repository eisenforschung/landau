# https://github.com/eisenforschung/landau/issues/51
import numpy as np
import pytest
from scipy.constants import Boltzmann, eV

kB = Boltzmann / eV


def test_ideal_solution_semigrand_potential_scalar_low_T():
    """IdealSolution.semigrand_potential must return a scalar when called with scalar T and dmu,
    even when phi overflows (e.g. at very low T) and the non-finite fallback branch is hit."""
    from landau.phases import LinePhase, IdealSolution

    solid_a = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB)
    solid_b = LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB)
    solid = IdealSolution("solid", solid_a, solid_b)

    result = solid.semigrand_potential(np.float64(1.0), np.float64(-0.6))
    assert np.ndim(result) == 0, f"Expected scalar, got shape {np.shape(result)}"
    assert np.isfinite(result)


def test_intermetallics_calc_phase_diagram():
    """calc_phase_diagram must not raise TypeError for a system with a line phase
    intermetallic and a low starting temperature (T=1 K)."""
    from landau.phases import LinePhase, IdealSolution
    from landau.calculate import calc_phase_diagram

    solid_a = LinePhase("A", fixed_concentration=0, line_energy=-2.0, line_entropy=1.0 * kB)
    solid_b = LinePhase("B", fixed_concentration=1, line_energy=-3.0, line_entropy=1.5 * kB)
    solid = IdealSolution("solid", solid_a, solid_b)

    liquid_a = LinePhase("A(l)", fixed_concentration=0, line_energy=-1.9, line_entropy=2.5 * kB)
    liquid_b = LinePhase("B(l)", fixed_concentration=1, line_energy=-2.9, line_entropy=2.2 * kB)
    liquid = IdealSolution("liquid", liquid_a, liquid_b)

    inter = LinePhase("AB2", fixed_concentration=2 / 3, line_energy=-2.8, line_entropy=1.3 * kB)

    df = calc_phase_diagram([solid, liquid, inter], Ts=np.linspace(1, 2000, 100), mu=200, keep_unstable=True)
    assert not df.empty
    assert {"solid", "liquid", "AB2"}.issubset(set(df.phase.unique()))
