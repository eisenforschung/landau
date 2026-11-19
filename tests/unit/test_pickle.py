"""Phases and fitted interpolations survive pickle, dill and cloudpickle.

Every fit is an instance of a module-level class rather than a closure, so plain
pickle handles it; a phase pickles its dataclass fields only, so a hash taken in the
writing interpreter (salted per process) never reaches the reading one.  The cases
live in ``pickleprobe.py``, shared with the fleche round trip in ``test_fleche.py``.
"""

import copyreg
import dataclasses
import io
import json
import os
import pickle
import subprocess
import sys

import cloudpickle
import dill
import numpy as np
import pytest

from landau.interpolate.whitney import WhitneyRBFInterpolator
from landau.phases import STATE_VERSION, Phase, TemperatureDependentLinePhase

sys.path.insert(0, os.path.dirname(__file__))
import pickleprobe

SERIALIZERS = {"pickle": pickle, "dill": dill, "cloudpickle": cloudpickle}
CASES = pickleprobe.available()
PHASES = [n for n in CASES if pickleprobe.CASES[n][1] == "phase"]

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
@pytest.mark.parametrize("serializer", list(SERIALIZERS))
@pytest.mark.parametrize("name", CASES)
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
def test_phase_state_is_its_fields_and_a_version_tag(name):
    """Derived attributes -- hashes, ``cached_property`` values -- are not pickled."""
    phase = pickleprobe.build(name)
    pickleprobe.values(name, phase)
    hash(phase)
    state = phase.__getstate__()
    assert state.pop("__landau__")[0] == STATE_VERSION
    assert list(state) == [f.name for f in dataclasses.fields(phase)]


def test_state_from_a_newer_layout_is_refused():
    phase = pickleprobe.build("TDLP-SGTE")
    state = phase.__getstate__()
    state["__landau__"] = (STATE_VERSION + 1, "99.0")
    with pytest.raises(ValueError, match="newer than this landau"):
        object.__new__(type(phase)).__setstate__(state)


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
# Pickles written by landau <= 1.19
# ---------------------------------------------------------------------------


def _legacy_dumps(obj):
    """Pickle *obj* the way landau <= 1.19 did, without the ``__getstate__`` methods added since.

    Objects then reduced to their whole ``__dict__``: a phase with the ``_hash`` its
    ``__post_init__`` set (salted in the writing interpreter, so stale in any other)
    and any ``cached_property`` values, and a Whitney estimator with its fitted scipy
    objects and scikit-learn's version tag.
    """

    class Legacy(pickle.Pickler):
        def reducer_override(self, o):
            if isinstance(o, Phase):
                state = dict(vars(o))
                if "_hash" in state:
                    state["_hash"] = 12345  # what another interpreter's salt made of it
                return copyreg.__newobj__, (type(o),), state
            if isinstance(o, WhitneyRBFInterpolator):
                return copyreg.__newobj__, (type(o),), {**vars(o), "_sklearn_version": "1.5.0"}
            return NotImplemented

    buf = io.BytesIO()
    Legacy(buf).dump(obj)
    return buf.getvalue()


@pytest.mark.parametrize("evaluated", [False, True], ids=["fresh", "evaluated"])
@pytest.mark.parametrize("name", PHASES)
def test_pickles_from_before_getstate_still_load(name, evaluated):
    phase = pickleprobe.build(name)
    expected = pickleprobe.values(name, pickleprobe.build(name))
    hash(phase)
    if evaluated:
        pickleprobe.values(name, phase)
    blob = _legacy_dumps(phase)
    _clear_landau_caches()
    loaded = pickle.loads(blob)
    assert "_hash" not in vars(loaded) or vars(loaded)["_hash"] != 12345
    np.testing.assert_allclose(pickleprobe.values(name, loaded), expected, rtol=0, atol=ATOL)
    fresh = pickleprobe.build(name)
    assert loaded == fresh and hash(loaded) == hash(fresh)


def test_legacy_pickle_keeps_a_fitted_whitney_surface():
    """An evaluated surface phase pickled its fitted estimator, ``hull_`` and all."""
    phase = pickleprobe.build("Surface-Whitney2D")
    pickleprobe.values("Surface-Whitney2D", phase)
    assert "_fitted_surface" in vars(phase)
    surface = pickle.loads(_legacy_dumps(phase._fitted_surface))
    assert surface._rbf.hull_ is not None
    np.testing.assert_allclose(
        surface.slice_at(1000.0)(pickleprobe.GRID_C),
        phase._fitted_surface.slice_at(1000.0)(pickleprobe.GRID_C),
        rtol=0,
        atol=ATOL,
    )


# ---------------------------------------------------------------------------
# Whitney: refit on load instead of unpickling scipy internals
# ---------------------------------------------------------------------------


def test_whitney_estimator_pickles_its_data_not_its_fit():
    X, y = np.c_[pickleprobe.T], pickleprobe.T * 1e-3
    est = WhitneyRBFInterpolator(degree=1).fit(X, y)
    state = est.__getstate__()
    assert set(state) == {"params", "X", "y"}
    assert state["params"] == est.get_params()
    np.testing.assert_array_equal(state["X"], X)
    np.testing.assert_array_equal(state["y"], y)


def test_unfitted_whitney_estimator_round_trips():
    est = pickle.loads(pickle.dumps(WhitneyRBFInterpolator(smoothing=1.0)))
    assert est.get_params() == WhitneyRBFInterpolator(smoothing=1.0).get_params()
    assert not hasattr(est, "rbf_")


# ---------------------------------------------------------------------------
# Across interpreters
# ---------------------------------------------------------------------------

_WRITER = r"""
import json, pickle, sys
sys.path.insert(0, sys.argv[1])
import pickleprobe
out = {}
for name in pickleprobe.available():
    obj = pickleprobe.build(name)
    out[name] = {"values": pickleprobe.values(name, obj).tolist(), "hash": hash(obj)}
    with open(f"{sys.argv[2]}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
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
        loaded = pickle.load(f)
    _clear_landau_caches()
    np.testing.assert_allclose(pickleprobe.values(name, loaded), written["cases"][name]["values"], rtol=0, atol=ATOL)
    if name in PHASES:
        fresh = pickleprobe.build(name)
        assert loaded == fresh and hash(loaded) == hash(fresh) and loaded in {fresh}
