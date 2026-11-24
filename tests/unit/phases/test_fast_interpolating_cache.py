"""Direct tests for the solve cache of :class:`FastInterpolatingPhase`.

``_find_phi_c_cached`` keys an ``lru_cache`` on the ``(shape, bytes)`` of ``T`` and ``dmu`` so that
``semigrand_potential`` and ``concentration`` at the same point share one ``_solve_fixed_T`` call;
``_find_phi_c`` copies the cached arrays out so callers cannot mutate them.
Value tests against ``SlowInterpolatingPhase`` pass with or without either property, so these count solves and
mutate results instead.
"""

import numpy as np
import pytest

from landau.phases import FastInterpolatingPhase, LinePhase

# the unpatched solver, captured before any test counts its calls, for reference values
_solve = FastInterpolatingPhase._solve_fixed_T

T = 800.0
DMU = np.linspace(-0.5, 0.5, 11)


@pytest.fixture
def phase():
    # the lru_cache lives on the class and is keyed on the (structurally hashed) phase, so equal phases built by
    # other tests would hit entries left behind by them
    FastInterpolatingPhase._find_phi_c_cached.cache_clear()
    yield FastInterpolatingPhase(
        "sol",
        [
            LinePhase("A", 0.0, 0.0),
            LinePhase("AB", 0.5, -0.1),
            LinePhase("B", 1.0, 0.05),
        ],
    )
    FastInterpolatingPhase._find_phi_c_cached.cache_clear()


@pytest.fixture
def solves(monkeypatch):
    """Record the ``(T, dmu)`` of every ``_solve_fixed_T`` call."""
    calls = []

    def counting(self, T, dmu):
        calls.append((T, np.array(dmu)))
        return _solve(self, T, dmu)

    monkeypatch.setattr(FastInterpolatingPhase, "_solve_fixed_T", counting)
    return calls


def test_semigrand_potential_then_concentration_solves_once(phase, solves):
    phase.semigrand_potential(T, DMU)
    phase.concentration(T, DMU)
    assert len(solves) == 1


def test_repeated_call_solves_once(phase, solves):
    phase.concentration(T, DMU)
    phase.concentration(T, DMU)
    assert len(solves) == 1


def test_other_dmu_at_same_T_solves_again_and_is_correct(phase, solves):
    other = DMU + 0.05
    phase.semigrand_potential(T, DMU)
    phi = phase.semigrand_potential(T, other)
    c = phase.concentration(T, other)
    assert len(solves) == 2
    np.testing.assert_array_equal(solves[1][1], other)
    phi_ref, c_ref = _solve(phase, T, other)
    np.testing.assert_array_equal(phi, phi_ref)
    np.testing.assert_array_equal(c, c_ref)
    # a stale hit on the T-only part of the key would hand back the first dmu's results
    assert not np.allclose(c, phase.concentration(T, DMU))


def test_other_T_at_same_dmu_solves_again(phase, solves):
    phase.concentration(T, DMU)
    phase.concentration(T + 100.0, DMU)
    assert [t for t, _ in solves] == [T, T + 100.0]


def test_equal_arrays_built_separately_share_the_key(phase, solves):
    first = np.linspace(-0.5, 0.5, 11)
    second = np.array(first.tolist())
    assert first is not second
    phase.semigrand_potential(np.float64(T), first)
    phase.concentration(float(T), second)
    assert len(solves) == 1


def test_non_contiguous_view_shares_the_key_of_its_copy(phase, solves):
    wide = np.linspace(-0.5, 0.5, 21)
    view = wide[::2]
    assert not view.flags.c_contiguous
    phase.concentration(T, view)
    phase.concentration(T, np.ascontiguousarray(view))
    assert len(solves) == 1


def test_same_bytes_different_shape_solves_again(phase, solves):
    phase.concentration(T, DMU)
    c = phase.concentration(T, DMU.reshape(1, -1))
    assert len(solves) == 2
    assert np.shape(c) == (1, DMU.size)


@pytest.mark.parametrize("temperature", [T, np.full(DMU.shape, T)], ids=["scalar-T", "array-T"])
@pytest.mark.parametrize("method", ["semigrand_potential", "concentration"])
def test_mutating_a_result_leaves_the_cache_intact(phase, solves, temperature, method):
    get = getattr(phase, method)
    first = get(temperature, DMU)
    expected = first.copy()
    first[:] = np.nan
    np.testing.assert_array_equal(get(temperature, DMU), expected)
    assert len(solves) == 1


def test_mutating_phi_leaves_the_shared_concentration_intact(phase, solves):
    c_expected = _solve(phase, T, DMU)[1]
    phi = phase.semigrand_potential(T, DMU)
    phi[:] = np.nan
    np.testing.assert_array_equal(phase.concentration(T, DMU), c_expected)
    assert len(solves) == 1
