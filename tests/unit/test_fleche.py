"""Tests for the fleche digest hooks registered by ``landau.fleche``.

The two properties that make the hooks worth anything -- a digest that survives
a new interpreter, and one that does not depend on when fleche got round to
loading its entry points -- can only be observed from a fresh process, so they
are checked by running a probe under two ``PYTHONHASHSEED`` values.
"""

import importlib.metadata
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from landau import PolyFit, SGTE, SplineFit, TemperatureDependentLinePhase
from landau.phases.asewrapper import AsePhase  # importable without ASE

pytest.importorskip("fleche")

from fleche import cache, fleche
from fleche.caches import Cache
from fleche.digest import Hook, Indigestible, digest, load_entry_points
from fleche.storage import CallMemory, ValueMemory

from landau.fleche import digest_hooks

try:
    from ase.thermochemistry import HarmonicThermo

    HAS_ASE = True
except ImportError:
    HAS_ASE = False

needs_ase = pytest.mark.skipif(not HAS_ASE, reason="ASE is not installed")

#: The quasi-harmonic phase is built through the Einstein-solid helper the
#: quasi-harmonic unit tests use, so the planted spectrum lives in one place.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "phases"))
try:
    from test_quasiharmonic import einstein_phase

    HAS_PHONOPY = True
except ImportError:
    HAS_PHONOPY = False

needs_phonopy = pytest.mark.skipif(not HAS_PHONOPY, reason="phonopy is not installed")

#: These run in their own CI env, installed with the test-fleche extra; the rest
#: of the suite deselects them with -m "not fleche".
pytestmark = pytest.mark.fleche


# ---------------------------------------------------------------------------
# Cross-process / load-order probe
# ---------------------------------------------------------------------------

#: The probe runs as a script, not through a process pool: a pool cannot
#: guarantee two different hash seeds, which is the whole point of the check.
#: ``ProcessPoolExecutor``'s default context on Linux for Python < 3.14 is
#: ``fork``, whose children inherit the parent's salt outright; ``forkserver``
#: children all share the forkserver's one salt.  Only ``spawn`` gives each
#: worker its own, and even that collapses to a single salt when
#: ``PYTHONHASHSEED`` is set in the environment, as a reproducible test run may
#: well do.  Every one of those cases makes the assertions below vacuously true,
#: so the seeds are set explicitly here and checked in
#: :func:`test_probe_runs_really_had_different_hash_seeds`.
_PROBE = os.path.join(os.path.dirname(__file__), "flecheprobe.py")


def _run_probe(hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run(
        [sys.executable, _PROBE], env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def probes():
    """The probe run under two different hash seeds, i.e. two separate interpreters."""
    return _run_probe(0), _run_probe(1)


def test_digests_are_stable_across_processes(probes):
    """The point of the hooks: a phase-derived cache key must survive a new interpreter.

    ``hash(bytes)`` is salted per process, so anything folding it in silently
    misses on every fresh run (issue #438).
    """
    first, second = probes
    assert first["warm"] == second["warm"]
    assert first["cold"] == second["cold"]


def test_digests_do_not_depend_on_entry_point_load_order(probes):
    """fleche loads entry points lazily, on the first value it cannot digest.

    A type it can already digest would therefore get one digest before that
    happens and another after, making the cache key depend on call order.  Hooks
    are registered only for types fleche rejects outright, so every object has
    exactly one possible digest.
    """
    for probe in probes:
        assert probe["cold"] == probe["warm"]


def test_digests_do_not_change_when_objects_are_used(probes):
    """Evaluating must not move a digest: ASE, for one, writes onto its thermochem."""
    for probe in probes:
        assert probe["evaluated"] == probe["warm"]


def test_probe_runs_really_had_different_hash_seeds(probes):
    """Guard against a vacuous pass.

    Both properties below compare digests from two interpreters.  If the two
    runs shared a hash seed, a salted digest would agree trivially and the
    checks would pass with the bug present.
    """
    first, second = probes
    assert first["salt"] != second["salt"]


def test_probe_covers_the_public_object_model(probes):
    """Guard: the probe must actually digest something for every family of type."""
    digests = probes[0]["warm"]
    assert not [k for k, v in digests.items() if v == "REFUSED"]
    # a hook returning a constant would pass every "it digests" assertion above
    assert len(set(digests.values())) == len(digests)


# ---------------------------------------------------------------------------
# Entry point registration
# ---------------------------------------------------------------------------


def test_entry_point_is_registered():
    """Installing landau is all a fleche user should have to do."""
    eps = list(importlib.metadata.entry_points(group="fleche", name="digest"))
    assert "landau.fleche:digest_hooks" in [ep.value for ep in eps]


def test_entry_point_loads_to_hooks():
    (ep,) = [
        ep
        for ep in importlib.metadata.entry_points(group="fleche", name="digest")
        if ep.value == "landau.fleche:digest_hooks"
    ]
    hooks = ep.load()
    assert hooks is digest_hooks
    assert all(isinstance(h, Hook) for h in hooks)


def test_no_hook_targets_a_dataclass():
    """fleche digests a dataclass by its fields, and its value storages rely on that.

    A value store rebuilds a stored dataclass field by field and files it under the
    field-walk digest, so a dataclass digested any other way would be stored under a
    key no lookup computes.  Load order makes it worse: entry points load lazily, so
    a hooked dataclass fleche can already walk gets one digest before and another after.
    """
    import dataclasses

    hooked = [h.type for h in digest_hooks if dataclasses.is_dataclass(h.type)]
    # the one exception: its ThermalProperties are indigestible, so it never reaches a value store
    assert [t.__name__ for t in hooked] == ["PhonopyQuasiHarmonicPhase"]


# ---------------------------------------------------------------------------
# Digest sensitivity: a digest that ignores a field is worse than no digest
# ---------------------------------------------------------------------------


def _phase(**kwargs):
    T = np.linspace(100, 1000, 20)
    defaults = dict(
        name="A", fixed_concentration=0.25, temperatures=T, free_energies=-T * 1e-3, interpolator=SGTE(3)
    )
    return TemperatureDependentLinePhase(**{**defaults, **kwargs})


def test_equal_phases_share_a_digest():
    assert digest(_phase()) == digest(_phase())


@pytest.mark.parametrize(
    "changed",
    [
        dict(name="B"),
        dict(fixed_concentration=0.26),
        dict(temperatures=np.linspace(100, 1001, 20)),
        dict(free_energies=-np.linspace(100, 1000, 20) * 1.001e-3),
        dict(interpolator=SGTE(4)),
        dict(interpolator=PolyFit(3)),
    ],
)
def test_every_field_of_a_phase_moves_its_digest(changed):
    assert digest(_phase(**changed)) != digest(_phase())


def test_derived_hash_does_not_enter_the_digest():
    """``_hash`` is derived state, and salted; it must not reach the digest.

    Pinned by overwriting it with nonsense -- the digest must not notice.
    """
    p = _phase()
    before = digest(p)
    object.__setattr__(p, "_hash", 12345)
    assert digest(p) == before


@pytest.mark.parametrize(
    "left,right",
    [
        (lambda R: R.ClausiusClapeyronRefiner(dT_max=5.0), lambda R: R.ClausiusClapeyronRefiner(dT_max=4.0)),
        (lambda R: R.MiscibilityGapRefiner(dc_max=0.01), lambda R: R.MiscibilityGapRefiner(dc_max=0.02)),
        (lambda R: R.ScanRefiner("mu"), lambda R: R.ScanRefiner("T")),
        (lambda R: R.DelaunayLineRefiner(), lambda R: R.DelaunayTripleRefiner()),
    ],
)
def test_refiner_settings_move_the_digest(left, right):
    """Refiners are plain classes; their configuration is the whole identity."""
    import landau.refine as R

    load_entry_points()
    assert digest(left(R)) != digest(right(R))
    assert digest(left(R)) == digest(left(R))


# ---------------------------------------------------------------------------
# Fitted interpolations digest by their state; bare closures are refused
# ---------------------------------------------------------------------------


def test_fleche_cannot_key_on_a_closure():
    """Why a closure is refused, stated as an executable fact.

    Two closures out of one factory share a code object, and since 0.22.1 fleche
    falls back to digesting that alone whenever what a closure captured is itself
    indigestible -- so a raw closure is not a usable key.
    """
    from scipy.interpolate import InterpolatedUnivariateSpline

    x = np.linspace(0, 1, 10)

    def closure(power):
        spline = InterpolatedUnivariateSpline(x, x**power)
        return lambda c: spline(c)

    quadratic, cubic = closure(2), closure(3)
    assert quadratic(0.5) != cubic(0.5)
    assert digest(quadratic.__code__) == digest(cubic.__code__)


def test_wrapped_closures_are_refused():
    from landau.interpolate.basic import _CallableInterpolation

    load_entry_points()
    with pytest.raises(Indigestible, match="code object"):
        digest(_CallableInterpolation(lambda x: x**2))


#: Each fitted interpolation class, fitted twice to the same data and once to different data.
_T = np.linspace(100.0, 1000.0, 30)
_C = np.linspace(0.0, 1.0, 9)
_SURFACE_T, _SURFACE_C = np.repeat(_T[::5], 9), np.tile(_C, 6)


def _fits(scale):
    from landau.interpolate import (
        RedlichKister,
        SoftplusFit,
        SoftplusSurface2DInterpolator,
        StitchedFit,
        WhitneySurface2DInterpolator,
        WhitneyTemperatureInterpolator,
    )

    f_T = -_T * 1e-3 * scale
    f_C = 0.1 * scale * np.sin(3 * _C)
    H = -1e-3 * _SURFACE_T + 0.2 * scale * _SURFACE_C * (1 - _SURFACE_C)
    return {
        "SplineInterpolation": SplineFit().fit(_C, f_C),
        "StitchedInterpolation": StitchedFit().fit(_T, f_T),
        "WhitneyInterpolation": WhitneyTemperatureInterpolator().fit(_T, f_T),
        "SoftplusInterpolation": SoftplusFit().fit(_T, f_T),
        "SGTEDerivative": SGTE(3).fit(_T, f_T).deriv(),
        "RedlichKisterDerivative": RedlichKister(3).fit(_C, f_C).deriv(),
        "WhitneySlice": WhitneySurface2DInterpolator().fit(_SURFACE_T, _SURFACE_C, H).slice_at(500.0),
        "SoftplusSliceDerivative": SoftplusSurface2DInterpolator()
        .fit(_SURFACE_T, _SURFACE_C, H)
        .slice_at(500.0)
        .deriv(),
    }


@pytest.fixture(scope="module")
def fits():
    return _fits(1.0), _fits(1.0), _fits(2.0)


@pytest.mark.parametrize("name", list(_fits(1.0)))
def test_fitted_interpolations_digest_by_their_state(fits, name):
    load_entry_points()
    one, same, other = (f[name] for f in fits)
    assert type(one).__name__ == name
    assert digest(one) == digest(same)
    assert digest(one) != digest(other)


# ---------------------------------------------------------------------------
# Fitted scipy splines (stopgap hook, see #449)
# ---------------------------------------------------------------------------


def test_fitted_splines_digest_by_knots_and_coefficients():
    """fleche refuses a scipy spline outright; the hook keys it on ``(t, c, k)``.

    Goes away with the hook once fleche digests scipy objects itself (#449).
    """
    from scipy.interpolate import InterpolatedUnivariateSpline

    load_entry_points()
    x = np.linspace(0, 1, 10)
    quadratic = InterpolatedUnivariateSpline(x, x**2)
    cubic = InterpolatedUnivariateSpline(x, x**3)

    assert digest(quadratic) == digest(InterpolatedUnivariateSpline(x, x**2))
    assert digest(quadratic) != digest(cubic)
    assert digest(quadratic) != digest(InterpolatedUnivariateSpline(x, x**2, k=2))
    assert digest(quadratic) != digest(InterpolatedUnivariateSpline(x, x**2, ext=1))


def test_digestible_interpolations_are_not_refused():
    """The refusal must be aimed at closures only, not at fitted curves generally."""
    T = np.linspace(100, 1000, 20)
    load_entry_points()
    assert digest(PolyFit(3).fit(T, T * 2.0)) != digest(PolyFit(3).fit(T, T * 3.0))
    assert digest(SGTE(3).fit(T, -T * 1e-3)) == digest(SGTE(3).fit(T, -T * 1e-3))


# ---------------------------------------------------------------------------
# Opaque third-party state: phonopy and the Whitney RBF
# ---------------------------------------------------------------------------


@needs_phonopy
def test_quasiharmonic_phase_digest_follows_equality():
    """Its phonopy ``ThermalProperties`` are opaque, so the digest rides the
    identity key the class already compares and hashes by."""
    load_entry_points()
    a = einstein_phase(fresh=True)
    same = einstein_phase(fresh=True)
    other = einstein_phase(omegas=6.0, fresh=True)

    assert a == same and digest(a) == digest(same)
    assert a != other and digest(a) != digest(other)


def test_whitney_rbf_digest_tracks_data_and_hyperparameters():
    """A fitted ``WhitneyRBFInterpolator`` holds a scipy ``RBFInterpolator`` and a
    ``ConvexHull``; both follow from the training data and the hyperparameters."""
    from landau.interpolate.whitney import WhitneyRBFInterpolator

    load_entry_points()
    T = np.linspace(100, 500, 5)
    c = np.linspace(0.0, 1.0, 7)
    X = np.column_stack([np.repeat(T, 7), np.tile(c, 5)])
    y = -1e-3 * X[:, 0] + X[:, 1] * (1 - X[:, 1])

    fitted = WhitneyRBFInterpolator().fit(X, y)
    assert digest(fitted) == digest(WhitneyRBFInterpolator().fit(X, y))
    assert digest(fitted) != digest(WhitneyRBFInterpolator().fit(X, 2 * y))
    assert digest(fitted) != digest(WhitneyRBFInterpolator(degree=1).fit(X, y))
    assert digest(fitted) != digest(WhitneyRBFInterpolator().fit(X * 1.01, y))
    assert digest(WhitneyRBFInterpolator()) != digest(WhitneyRBFInterpolator(smoothing=1.0))


def test_whitney_fitted_surface_is_digestible():
    """What the estimator hook buys: the surface's whole state is that estimator."""
    from landau.interpolate import WhitneySurface2DInterpolator

    load_entry_points()
    T = np.linspace(100, 500, 5)
    c = np.linspace(0.0, 1.0, 7)
    Tg, cg = np.repeat(T, 7), np.tile(c, 5)
    H = -1e-3 * Tg + cg * (1 - cg)

    surface = WhitneySurface2DInterpolator().fit(Tg, cg, H)
    assert digest(surface) == digest(WhitneySurface2DInterpolator().fit(Tg, cg, H))
    assert digest(surface) != digest(WhitneySurface2DInterpolator().fit(Tg, cg, 2 * H))


# ---------------------------------------------------------------------------
# AsePhase
# ---------------------------------------------------------------------------


@needs_ase
def test_ase_phase_digest_follows_equality():
    load_entry_points()
    a = AsePhase("p", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.03])))
    same = AsePhase("p", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.03])))
    other = AsePhase("p", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.04])))

    assert a == same and digest(a) == digest(same)
    assert a != other and digest(a) != digest(other)


@needs_ase
def test_ase_phase_digest_survives_evaluation():
    """ASE writes the ``verbose`` flag it was last called with onto the thermochem."""
    load_entry_points()
    a = AsePhase("p", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.03])))
    before = digest(a)
    a.line_free_energy(np.array([300.0, 600.0]))
    assert "verbose" in vars(a.thermochem)
    assert digest(a) == before
    assert digest(a.thermochem) == digest(HarmonicThermo(np.array([0.01, 0.02, 0.03])))


@needs_ase
def test_ase_phase_is_stored_under_its_digest():
    """What a value store files a stored argument under must be what a lookup computes."""
    load_entry_points()
    a = AsePhase("p", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.03])))
    assert ValueMemory({}).save(a) == digest(a)


@needs_ase
def test_ase_phase_digest_tracks_the_wrapper_fields():
    load_entry_points()
    vib = np.array([0.01, 0.02, 0.03])
    base = AsePhase("p", 0.0, HarmonicThermo(vib))
    assert digest(AsePhase("q", 0.0, HarmonicThermo(vib))) != digest(base)
    assert digest(AsePhase("p", 0.5, HarmonicThermo(vib))) != digest(base)
    assert digest(AsePhase("p", 0.0, HarmonicThermo(vib), atoms_per_formula=2)) != digest(base)


# ---------------------------------------------------------------------------
# End to end: the promise the entry point actually makes
# ---------------------------------------------------------------------------


def test_cached_function_hits_on_an_equal_but_distinct_phase():
    calls = []

    @fleche
    def melting_point(phase):
        calls.append(phase.name)
        return float(np.min(phase.free_energies))

    with cache(Cache(ValueMemory({}), CallMemory({}))):
        first = melting_point(_phase())
        second = melting_point(_phase())

    assert first == second
    assert len(calls) == 1


def test_cached_function_misses_on_a_different_phase():
    calls = []

    @fleche
    def melting_point(phase):
        calls.append(phase.name)
        return float(np.min(phase.free_energies))

    with cache(Cache(ValueMemory({}), CallMemory({}))):
        melting_point(_phase())
        melting_point(_phase(fixed_concentration=0.75))

    assert len(calls) == 2


def test_cached_function_accepts_refiners_and_phases_together():
    """``calc_phase_diagram``'s own signature: phases plus a refiner sequence."""
    import landau.refine as R

    calls = []

    @fleche
    def run(phases, refiners):
        calls.append(len(phases))
        return len(phases) + len(refiners)

    def args():
        return [_phase(), _phase(name="B", fixed_concentration=0.75)], [
            R.ClausiusClapeyronRefiner(),
            R.DelaunayTripleRefiner(),
        ]

    with cache(Cache(ValueMemory({}), CallMemory({}))):
        assert run(*args()) == 4
        assert run(*args()) == 4

    assert len(calls) == 1


@needs_ase
def test_deprecated_ase_phase_digest_is_the_field_digest():
    from landau.fleche import ase_phase_digest

    load_entry_points()
    a = AsePhase("p", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.03])))
    with pytest.warns(DeprecationWarning):
        assert ase_phase_digest(a) == digest(a)


# ---------------------------------------------------------------------------
# Round trip through a fleche cache, written and read by different interpreters
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
import pickleprobe

_PICKLEPROBE = os.path.join(os.path.dirname(__file__), "pickleprobe.py")
#: Shared with test_pickle.py: a refit Whitney RBF reproduces the original to round-off.
_ATOL = 1e-12


def _run_pickleprobe(mode, root, serializer, inline, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run(
        [sys.executable, _PICKLEPROBE, mode, str(root), serializer, str(int(inline))],
        env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module", params=[False, True], ids=["arrays-stored", "arrays-inline"])
def fleche_round_trip(request, tmp_path_factory):
    """Every case cached by one interpreter and read back by another, through a dill value store.

    ``remaining_depth=2`` destructures a phase into its fields, so the reading side
    rebuilds it with ``object.__new__`` and never runs ``__post_init__`` or
    ``__setstate__``.  ``arrays-inline`` keeps numpy arrays inside their parent entry,
    as graceland configures it, so fewer fields come back as separate entries.
    """
    pytest.importorskip("dill")
    root = tmp_path_factory.mktemp("cache")
    stored = _run_pickleprobe("store", root, "dill", request.param, hashseed=0)
    loaded = _run_pickleprobe("load", root, "dill", request.param, hashseed=1)
    assert stored["salt"] != loaded["salt"]
    return stored["cases"], loaded["cases"]


@pytest.mark.parametrize("name", [n for n in pickleprobe.available() if n != "PhonopyQuasiHarmonicPhase"])
def test_fleche_round_trip(fleche_round_trip, name):
    stored, loaded = (cases[name] for cases in fleche_round_trip)
    assert stored["made"] and stored["hit"]
    np.testing.assert_allclose(stored["used"], stored["values"], rtol=0, atol=_ATOL)
    # a new interpreter finds both entries for an equal, freshly built object ...
    assert loaded["made"] and loaded["hit"]
    assert loaded["digest"] == stored["digest"]
    # ... and what it reads back evaluates to what was stored
    np.testing.assert_allclose(loaded["values"], stored["values"], rtol=0, atol=_ATOL)


#: Digests of fresh phases computed by landau 1.19.0 (6e1d3ae) with fleche 0.22.4.
#: Existing cache entries are keyed on these, so they must not move.  A fleche release
#: that changes its digest wire format moves them legitimately; that arrives with a
#: bump of the test-fleche pin.  AsePhase is left out: it digests by its fields now,
#: and no value store could hold one before (its ThermoChem was indigestible).
_PINNED = {
    "LinePhase": "0c80b32f5fe3502dfcf30a3937fd25e5881ecc624a1ef0ed9cd0e217808ec4bf",
    "TDLP-SGTE": "1a0ffe4c62382c0e97419b38c4d3fd387e47b084d7a4cd7a964e32515872d5ec",
    "TDLP-Whitney": "b2b921583210d3d6684ce7496fa8222e23ded4ee8263b29a70c0ceebec19effb",
    "TDLP-Stitched": "ed89e6cedf99174db4c8474480ea06be25d309b7a8eb6535998c4ed2235823bd",
    "TDLP-Softplus": "8a1b87a7b9d9fc17dd3e9fbb0bae32a6ec41037ce2d8cfc4fd7d3c6b5608386a",
    "IdealSolution-Whitney": "27a810563d32acd649f13a5c18ecf6e48f933ba499bddc14106cba2029b25140",
    "Fast-RK-Whitney": "7f646ac7b460db20adc8112feb154c0763652aee220238e72ad80415f0bc60e7",
    "Fast-Spline-Whitney": "796d3577ff8688a2c5f3be88ec5b6c064f69721766c1b58f6f87785167701027",
    "Surface-Calphad-Whitney": "e74ff84d870f25c19c6beca5c7fdb2cdc0a3a810180688cedc294ee760823768",
    "Surface-Whitney2D": "031b2bdbd8d373a6c322bffee4228f44ee05243c670b1fb7054144ee429bbc9b",
    "Surface-Softplus": "8991993479a8e75c177cc23606cbc28959b93f69278c55bd586136f13fdc6855",
}


@pytest.mark.parametrize("name", list(_PINNED))
def test_phase_digests_did_not_move(name):
    load_entry_points()
    phase = pickleprobe.build(name)
    assert str(digest(phase)) == _PINNED[name]
    pickleprobe.values(name, phase)
    assert str(digest(phase)) == _PINNED[name]


# ---------------------------------------------------------------------------
# landau_digest as reusable API (exported for custom types via add_hook)
# ---------------------------------------------------------------------------


def test_landau_digest_skips_derived_fields_of_a_dataclass():
    """The dataclass branch is what makes ``landau_digest`` safe to reuse.

    None of the registered hooks target a dataclass, so this contract is only
    exercised through the exported helper -- but it is the branch that keeps a
    derived, per-interpreter-salted field out of a custom type's digest.
    """
    import dataclasses

    from landau.fleche import landau_digest

    @dataclasses.dataclass(frozen=True)
    class Custom:
        value: float
        derived: int = dataclasses.field(default=0, init=False)

        def __post_init__(self):
            object.__setattr__(self, "derived", hash(str(self.value)))

    one, two = Custom(1.0), Custom(1.0)
    object.__setattr__(two, "derived", 999)
    assert landau_digest(one) == landau_digest(two)
    assert landau_digest(Custom(2.0)) != landau_digest(one)


def test_landau_digest_uses_private_attributes_of_a_plain_class():
    """For the fitted surfaces the private attributes *are* the state."""
    from landau.fleche import landau_digest

    class Fitted:
        def __init__(self, weights):
            self._weights = np.asarray(weights)

    assert landau_digest(Fitted([1.0, 2.0])) == landau_digest(Fitted([1.0, 2.0]))
    assert landau_digest(Fitted([1.0, 2.0])) != landau_digest(Fitted([1.0, 3.0]))


def test_landau_digest_refuses_a_slotted_object_it_cannot_read():
    from landau.fleche import landau_digest

    class Slotted:
        __slots__ = ("a",)

        def __init__(self):
            self.a = 1

    with pytest.raises(Indigestible, match="neither a dataclass nor has a __dict__"):
        landau_digest(Slotted())
