import numpy as np
import pytest
from landau.phases import LinePhase, SlowInterpolatingPhase
from landau.calculate import calc_phase_diagram, refine_concentration_jumps

def test_refine_concentration_jumps_demixing():
    # Define a model with demixing below Tc
    # Enthalpy H = W * c * (1-c)
    # Entropy S = -kB * (c*ln(c) + (1-c)*ln(1-c))
    # Tc = W / (2*kB)
    # Let W = 0.1 eV, Tc = 0.1 / (2 * 8.617e-5) approx 580 K

    p0 = LinePhase("A", 0, 0, 0)
    p1 = LinePhase("B", 1, 0, 0)
    # We provide a middle point to help with the interpolation
    p_mid = LinePhase("Mid", 0.5, 0.025, 0)

    phase = SlowInterpolatingPhase(
        name="MiscibilityGapPhase",
        phases=[p0, p_mid, p1],
        add_entropy=True, # This adds the -TS part
    )

    # Tc approx 580 K.
    # Test at T = 300 (below Tc) and T = 700 (above Tc)
    mus = np.linspace(-0.05, 0.05, 100)

    # T = 300 (should have a jump)
    df_300 = calc_phase_diagram([phase], 300, mus)
    c_left, c_right, mu_j, T_j = refine_concentration_jumps(
        df_300["T"].values, df_300["mu"].values, df_300["c"].values, phase
    )

    assert len(mu_j) == 1
    assert abs(mu_j[0]) < 1e-3
    assert c_left[0] < 0.1
    assert c_right[0] > 0.9

    # T = 700 (should NOT have a jump)
    df_700 = calc_phase_diagram([phase], 700, mus)
    c_left, c_right, mu_j, T_j = refine_concentration_jumps(
        df_700["T"].values, df_700["mu"].values, df_700["c"].values, phase
    )

    assert len(mu_j) == 0

def test_refine_concentration_jumps_no_data():
    p0 = LinePhase("A", 0, 0, 0)
    cl, cr, mu, T = refine_concentration_jumps([], [], [], p0)
    assert len(mu) == 0
    assert isinstance(mu, np.ndarray)
