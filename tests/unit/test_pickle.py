"""Phases survive pickle, dill and cloudpickle; fitted interpolations survive dill and cloudpickle.

A phase pickles its dataclass fields only, so a hash taken in the writing interpreter
(salted per process) never reaches the reading one, and a cached fit, a closure that
plain pickle refuses, is recomputed on first use.  The cases live in ``pickleprobe.py``,
shared with the fleche round trip in ``test_fleche.py``.
"""

import dataclasses
import json
import os
import pickle
import subprocess
import sys

import cloudpickle
import dill
import numpy as np
import pytest

from landau.phases import Phase, TemperatureDependentLinePhase

sys.path.insert(0, os.path.dirname(__file__))
import pickleprobe

SERIALIZERS = {"pickle": pickle, "dill": dill, "cloudpickle": cloudpickle}
CASES = pickleprobe.available()
PHASES = [n for n in CASES if pickleprobe.CASES[n][1] == "phase"]
#: Bare fits are closures, which plain pickle refuses; a phase pickles without its cached fits.
ROUND_TRIPS = [(n, s) for n in CASES for s in SERIALIZERS if n in PHASES or s != "pickle"]

#: A refit Whitney RBF and a re-solved phase reproduce the original to round-off.
ATOL = 1e-12


def _clear_landau_caches():
    """Empty every ``lru_cache``/``cache`` on the phase classes.

    They live on the class and are keyed by equality, so a loaded phase equal to one
    evaluated before would otherwise be answered from the original's cache, and the
    round trip would compare the original with itself.
    """
    stack = [Phase]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        for attr in vars(cls).values():
            fn = attr.fget if isinstance(attr, property) else attr
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()


@pytest.mark.parametrize("evaluated", [False, True], ids=["fresh", "evaluated"])
@pytest.mark.parametrize(("name", "serializer"), ROUND_TRIPS)
def test_round_trip_reproduces_the_values(name, serializer, evaluated):
    obj = pickleprobe.build(name)
    expected = pickleprobe.values(name, pickleprobe.build(name))
    if evaluated:
        pickleprobe.values(name, obj)
    blob = SERIALIZERS[serializer].dumps(obj)
    _clear_landau_caches()
    loaded = SERIALIZERS[serializer].loads(blob)
    assert type(loaded) is type(obj)
    np.testing.assert_allclose(pickleprobe.values(name, loaded), expected, rtol=0, atol=ATOL)


@pytest.mark.parametrize("name", PHASES)
def test_phase_state_is_its_fields(name):
    """Derived attributes -- hashes, ``cached_property`` values -- are not pickled."""
    phase = pickleprobe.build(name)
    pickleprobe.values(name, phase)
    hash(phase)
    assert list(phase.__getstate__()) == [f.name for f in dataclasses.fields(phase)]


# ---------------------------------------------------------------------------
# The hash is computed on first use, from the fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["TDLP-SGTE", "TDLP-Whitney", "PhonopyQuasiHarmonicPhase"])
def test_loaded_phase_hashes_like_a_fresh_one(name):
    if name not in CASES:
        pytest.skip("phonopy is not installed")
    phase = pickleprobe.build(name)
    hash(phase)
    loaded = pickle.loads(pickle.dumps(phase))
    fresh = pickleprobe.build(name)
    assert loaded == fresh and hash(loaded) == hash(fresh)
    assert loaded in {fresh}
    assert fresh in {loaded}


@pytest.mark.parametrize("name", ["TDLP-SGTE", "TDLP-Whitney", "PhonopyQuasiHarmonicPhase"])
def test_phase_rebuilt_without_post_init_still_works(name):
    """fleche rebuilds a stored dataclass with ``object.__new__`` and ``setattr`` per field.

    Neither ``__post_init__`` nor ``__setstate__`` runs, so whatever those derive
    must be derivable on first use.
    """
    if name not in CASES:
        pytest.skip("phonopy is not installed")
    phase = pickleprobe.build(name)
    rebuilt = object.__new__(type(phase))
    for f in dataclasses.fields(phase):
        object.__setattr__(rebuilt, f.name, getattr(phase, f.name))
    assert hash(rebuilt) == hash(phase)
    assert rebuilt in {phase}
    _clear_landau_caches()
    np.testing.assert_allclose(
        pickleprobe.values(name, rebuilt), pickleprobe.values(name, phase), rtol=0, atol=ATOL
    )


def test_hash_follows_the_fields_not_construction():
    T = pickleprobe.T
    a = TemperatureDependentLinePhase("A", 0.0, T, -T * 1e-3)
    assert hash(a) == hash(TemperatureDependentLinePhase("A", 0.0, T, -T * 1e-3))
    assert hash(a) != hash(TemperatureDependentLinePhase("A", 0.0, T, -T * 1.1e-3))


# ---------------------------------------------------------------------------
# Across interpreters
# ---------------------------------------------------------------------------

_WRITER = r"""
import json, pickle, sys
import dill
sys.path.insert(0, sys.argv[1])
import pickleprobe
out = {}
for name in pickleprobe.available():
    obj = pickleprobe.build(name)
    out[name] = {"values": pickleprobe.values(name, obj).tolist(), "hash": hash(obj)}
    with open(f"{sys.argv[2]}/{name}.pkl", "wb") as f:
        (pickle if pickleprobe.CASES[name][1] == "phase" else dill).dump(obj, f)
print(json.dumps({"salt": hash("x"), "cases": out}))
"""


@pytest.fixture(scope="module")
def written_elsewhere(tmp_path_factory):
    """Every case evaluated, hashed and pickled by another interpreter with another hash seed."""
    root = tmp_path_factory.mktemp("pickles")
    for seed in (0, 1):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        out = subprocess.run(
            [sys.executable, "-c", _WRITER, os.path.dirname(__file__), str(root)],
            env=env, capture_output=True, text=True, check=True,
        )
        written = json.loads(out.stdout)
        if written["salt"] != hash("x"):
            return root, written
    pytest.fail("could not get a subprocess with a different hash seed")


@pytest.mark.parametrize("name", CASES)
def test_pickle_from_another_interpreter(written_elsewhere, name):
    root, written = written_elsewhere
    with open(root / f"{name}.pkl", "rb") as f:
        loaded = (pickle if name in PHASES else dill).load(f)
    _clear_landau_caches()
    np.testing.assert_allclose(pickleprobe.values(name, loaded), written["cases"][name]["values"], rtol=0, atol=ATOL)
    if name in PHASES:
        fresh = pickleprobe.build(name)
        assert loaded == fresh and hash(loaded) == hash(fresh) and loaded in {fresh}
