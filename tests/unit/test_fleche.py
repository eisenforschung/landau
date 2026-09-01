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

from pyiron_snippets.import_alarm import ImportAlarm

from landau import PolyFit, SGTE, SplineFit, TemperatureDependentLinePhase

with ImportAlarm() as fleche_alarm:
    from fleche import cache, fleche
    from fleche.caches import Cache
    from fleche.digest import Hook, Indigestible, digest, load_entry_points
    from fleche.storage import CallMemory, ValueMemory

    from landau.fleche import digest_hooks

pytestmark = pytest.mark.skipif(fleche_alarm.message is not None, reason="fleche is not installed")

with ImportAlarm() as ase_alarm:
    from ase.thermochemistry import HarmonicThermo

    from landau.phases.asewrapper import AsePhase

needs_ase = pytest.mark.skipif(ase_alarm.message is not None, reason="ASE is not installed")


# ---------------------------------------------------------------------------
# Cross-process / load-order probe
# ---------------------------------------------------------------------------

#: Digests every representative object twice: once before fleche has loaded its
#: entry points (so the built-in dataclass path is all that is available) and
#: once after (so the hooks are live).  Reported as JSON on stdout.
_PROBE = r"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from fleche.digest import digest, Indigestible, load_entry_points
from landau import (LinePhase, TemperatureDependentLinePhase, IdealSolution, RegularSolution,
                    FastInterpolatingPhase, PolyFit, SGTE, RedlichKister, SplineFit)
from landau.interpolate import NumericalDerivative
from landau.phases import pointdefects as pdf
import landau.refine as R
import landau.poly as P

T = np.linspace(100, 1000, 20)
A = TemperatureDependentLinePhase("A", 0.0, T, -T * 1e-3, SGTE(3))
B = TemperatureDependentLinePhase("B", 1.0, T, -T * 1.1e-3, SGTE(3))
M = TemperatureDependentLinePhase("M", 0.5, T, -T * 1.2e-3 - 0.05, SGTE(3))
L = LinePhase("L", 0.3, -0.1, 1e-4)
d = pdf.ConstantPointDefect("d", 0.1, 1e-4, 0.5)
sl = pdf.PointDefectSublattice("s", 0, 1.0, [d])

def build():
    o = {
        "LinePhase": L,
        "TemperatureDependentLinePhase": A,
        "IdealSolution": IdealSolution("I", A, B),
        "RegularSolution": RegularSolution("R", [A, M, B]),
        "FastInterpolatingPhase": FastInterpolatingPhase("F", [A, M, B]),
        "PolyFit": PolyFit(3),
        "SGTE": SGTE(3),
        "RedlichKister": RedlichKister(3),
        "SplineFit": SplineFit(),
        "PolynomialInterpolation": PolyFit(3).fit(T, T * 2.0),
        "SGTEInterpolation": SGTE(3).fit(T, -T * 1e-3),
        "NumericalDerivative": NumericalDerivative(PolyFit(3).fit(T, T * 2.0)),
        "ConstantPointDefect": d,
        "PointDefectSublattice": sl,
        "PointDefectedPhase": pdf.PointDefectedPhase("PD", L, [sl]),
        "ScanRefiner": R.ScanRefiner("mu"),
        "DelaunayTripleRefiner": R.DelaunayTripleRefiner(),
        "ClausiusClapeyronRefiner": R.ClausiusClapeyronRefiner(),
        "MiscibilityGapRefiner": R.MiscibilityGapRefiner(),
        "Concave": P.Concave(),
        "Segments": P.Segments(),
    }
    try:
        from ase.thermochemistry import HarmonicThermo
        from landau import AsePhase
        o["AsePhase"] = AsePhase("ase", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.03])))
    except Exception:
        pass
    return o

def one(v):
    try:
        return str(digest(v))
    except Indigestible:
        return "REFUSED"

cold = {k: one(v) for k, v in build().items()}
load_entry_points()
warm = {k: one(v) for k, v in build().items()}
print(json.dumps({"cold": cold, "warm": warm}))
"""


def _run_probe(hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True
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


def test_ase_phase_hook_precedes_any_base_class_hook():
    """Hooks match by ``isinstance`` and the first match wins.

    ``AsePhase`` needs its pickle-based digest rather than a field walk, so any
    hook for one of its base classes must come after it.
    """
    types_ = [h.type for h in digest_hooks]
    ase_at = types_.index(AsePhase) if AsePhase in types_ else None
    assert ase_at is not None
    later = types_[ase_at + 1 :]
    assert not [t for t in later if issubclass(AsePhase, t)]


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
# Closure-backed fits are refused rather than silently collided
# ---------------------------------------------------------------------------


def test_fleche_collides_closures_over_different_data():
    """The reason the refusal below exists, stated as an executable fact.

    fleche digests a function from its code object; two fits of one interpolator
    share that code object, so their raw closures are indistinguishable to it.
    """
    x = np.linspace(0, 1, 10)
    quadratic = SplineFit().fit(x, x**2)
    cubic = SplineFit().fit(x, x**3)
    assert quadratic(0.5) != cubic(0.5)
    assert digest(quadratic.func) == digest(cubic.func)


@pytest.mark.parametrize("interpolator", ["spline", "stitched"])
def test_closure_backed_interpolations_are_refused(interpolator):
    from landau.interpolate import StitchedFit

    load_entry_points()
    if interpolator == "spline":
        x = np.linspace(0, 1, 10)
        fitted = SplineFit().fit(x, x**2)
    else:
        T = np.linspace(100, 1000, 30)
        fitted = StitchedFit().fit(T, -T * 1e-3)

    with pytest.raises(Indigestible, match="code object"):
        digest(fitted)


def test_digestible_interpolations_are_not_refused():
    """The refusal must be aimed at closures only, not at fitted curves generally."""
    T = np.linspace(100, 1000, 20)
    load_entry_points()
    assert digest(PolyFit(3).fit(T, T * 2.0)) != digest(PolyFit(3).fit(T, T * 3.0))
    assert digest(SGTE(3).fit(T, -T * 1e-3)) == digest(SGTE(3).fit(T, -T * 1e-3))


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
