"""Unit tests for landau.resample."""
import numpy as np
import pandas as pd
import pytest

from landau.phases import LinePhase, Phase
from landau.resample import RandomlyShiftedPhase, _resample_borders_once


@pytest.fixture
def base_phase():
    # fixed_concentration=0.5, line_energy=-1.0, line_entropy=0.0
    return LinePhase("base", 0.5, -1.0, 0.0)


def test_zero_noise_produces_zero_shift(base_phase):
    """With noise=0 the deterministic shift must be exactly 0."""
    r = RandomlyShiftedPhase("a", base_phase, noise=0.0)
    assert r.shift == 0.0


def test_semigrand_potential_adds_shift(base_phase):
    """semigrand_potential should equal the underlying potential plus the stored shift."""
    np.random.seed(0)
    r = RandomlyShiftedPhase("a", base_phase, noise=0.1)
    assert r.shift != 0.0  # guard against an accidentally zero draw obscuring the test
    assert r.semigrand_potential(300.0, 0.05) == base_phase.semigrand_potential(300.0, 0.05) + r.shift


def test_semigrand_potential_shift_is_constant_across_states(base_phase):
    """The same instance must apply the same shift at all (T, mu)."""
    np.random.seed(123)
    r = RandomlyShiftedPhase("a", base_phase, noise=0.1)
    deltas = [
        r.semigrand_potential(T, mu) - base_phase.semigrand_potential(T, mu)
        for T, mu in [(100.0, 0.0), (500.0, -0.1), (1000.0, 0.3)]
    ]
    assert np.allclose(deltas, r.shift)


def test_concentration_is_passthrough(base_phase):
    """Adding a constant noise to the potential must not change concentration."""
    r = RandomlyShiftedPhase("a", base_phase, noise=1.0)  # large noise on purpose
    assert r.concentration(300.0, 0.0) == base_phase.concentration(300.0, 0.0)


def test_is_subclass_of_phase(base_phase):
    """The wrapper must remain a Phase so it composes with calc_phase_diagram et al."""
    r = RandomlyShiftedPhase("a", base_phase, noise=0.0)
    assert isinstance(r, Phase)


def test_shift_scale_matches_noise():
    """Empirical std of shifts over many instances should approximate the noise level."""
    np.random.seed(42)
    base = LinePhase("base", 0.5, 0.0, 0.0)
    noise = 0.05
    shifts = np.array([RandomlyShiftedPhase("a", base, noise=noise).shift for _ in range(2000)])
    # Standard error of std for N=2000 samples of N(0, noise) is noise/sqrt(2*(N-1)).
    # Allow ~15% tolerance to keep the test stable across platforms.
    assert np.isclose(shifts.std(), noise, rtol=0.15)
    assert abs(shifts.mean()) < 0.01


# ---- _resample_borders_once ----

# Two terminal line phases: semigrand potential phi_A = 0, phi_B = -mu,
# so the A-B boundary falls at exactly mu = 0 for any T.
_TWO_PHASES = [LinePhase("A", 0, 0, 0), LinePhase("B", 1, 0, 0)]
_TS = np.linspace(300, 800, 5)
_DMUS = np.linspace(-0.3, 0.3, 30)


def test_resample_borders_once_all_rows_are_border():
    """All returned rows must have border=True and finite mu."""
    df = _resample_borders_once(_TWO_PHASES, _TS, _DMUS, noise_norm=0.0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df["border"].all()
    assert np.isfinite(df["mu"]).all()


def test_resample_borders_once_run_column_value():
    """The 'run' column must equal the provided run argument on every row."""
    df = _resample_borders_once(_TWO_PHASES, _TS, _DMUS, noise_norm=0.0, run=5)
    assert (df["run"] == 5).all()


def test_resample_borders_once_default_run_is_zero():
    df = _resample_borders_once(_TWO_PHASES, _TS, _DMUS, noise_norm=0.0)
    assert (df["run"] == 0).all()


def test_resample_borders_once_noise_shifts_refined_boundary():
    """With nonzero noise the refined A-B boundary must shift from the noiseless mu=0 location."""
    df_zero = _resample_borders_once(_TWO_PHASES, _TS, _DMUS, noise_norm=0.0)
    np.random.seed(0)
    df_noisy = _resample_borders_once(_TWO_PHASES, _TS, _DMUS, noise_norm=0.05)
    refined_zero = df_zero[df_zero["refined"] != "no"]["mu"]
    refined_noisy = df_noisy[df_noisy["refined"] != "no"]["mu"]
    assert len(refined_zero) > 0
    assert len(refined_noisy) > 0
    # seed=0, noise_norm=0.05 shifts A and B by different amounts (~0.088 and ~0.020),
    # moving the boundary from mu=0 to mu≈-0.068; mean must differ by more than 1e-8.
    assert not np.isclose(refined_noisy.mean(), refined_zero.mean(), atol=1e-8)
