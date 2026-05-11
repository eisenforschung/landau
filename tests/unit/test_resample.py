"""Unit tests for landau.resample.RandomlyShiftedPhase."""
import numpy as np
import pytest

from landau.phases import LinePhase, Phase
from landau.resample import RandomlyShiftedPhase


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
    r = RandomlyShiftedPhase("a", base_phase, noise=0.0)
    # noise=0 makes the assertion exact and independent of RNG state
    assert r.semigrand_potential(300.0, 0.05) == base_phase.semigrand_potential(300.0, 0.05)


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
