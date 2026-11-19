"""One instance of every phase and fitted interpolation landau builds, plus a fleche round trip.

Imported by ``tests/unit/test_pickle.py`` for the case list, and run as a script by
it and by ``tests/unit/test_fleche.py`` in a subprocess, because a pickle or a cache
entry written in one interpreter and read in another is the only way to see a stale
per-process ``hash`` or a missing ``__post_init__``-derived attribute.

Usage::

    python pickleprobe.py store <root> <serializer> <inline>   # cache every case, print values and digests
    python pickleprobe.py load <root> <serializer> <inline>    # read them back in a new interpreter

The cache is a dill/pickle/cloudpickle value store destructuring two levels deep
(``remaining_depth=2``) with a pickled call index, both under ``root``.

``inline`` 1 registers a fleche destructurer that keeps numpy arrays inside their
parent entry instead of storing each one separately, as graceland does.
"""

import json
import os
import sys
import warnings

import numpy as np

from landau.interpolate import (
    SGTE,
    CalphadSurface2DInterpolator,
    RedlichKister,
    SoftplusFit,
    SoftplusSurface2DInterpolator,
    SplineFit,
    StitchedFit,
    WhitneySurface2DInterpolator,
    WhitneyTemperatureInterpolator,
)
from landau.phases import (
    FastInterpolatingPhase,
    IdealSolution,
    LinePhase,
    Surface2DInterpolatingPhase,
    TemperatureDependentLinePhase,
    kB,
)

#: Sampled temperatures of every line phase, and the grids values are compared on.
#: The grids reach past the samples on both sides, into each fit's extrapolation.
T = np.linspace(600.0, 1400.0, 41)
GRID_T = np.array([500.0, 1000.0, 1500.0])
GRID_MU = np.array([-0.1, 0.05, 0.2])
GRID_C = np.array([0.05, 0.3, 0.5, 0.8])
#: The quasi-harmonic phase is computed, not fitted, at every temperature, so it is also
#: compared on a finer grid from 0 K, through its thermal expansion.
GRID_T_QHA = np.linspace(0.0, 1500.0, 16)


def _free_energy(x, s):
    return -2.0 + 0.1 * x - T * s * kB - 1e-5 * T * np.log(T) * (1 + x) - 0.05 * x * (1 - x)


def _line(name, x, interpolator, s=3.0):
    return TemperatureDependentLinePhase(name, x, T, _free_energy(x, s + x), interpolator)


def _liquid_points():
    W = WhitneyTemperatureInterpolator()
    return [_line(f"l{x}", x, W, s=4.0) for x in (0.0, 0.3, 0.6, 1.0)]


def _surface_samples():
    Tg, cg = np.repeat(T[::8], 7), np.tile(np.linspace(0.0, 1.0, 7), len(T[::8]))
    return Tg, cg, -1e-3 * Tg + 0.2 * cg * (1 - cg) + 0.01 * cg


def _fit_in_T(interpolator):
    return interpolator.fit(T, _free_energy(0.0, 3.0))


def _ase_phase():
    from ase.thermochemistry import HarmonicThermo

    from landau.phases import AsePhase

    return AsePhase("ase", 0.0, HarmonicThermo(np.array([0.01, 0.02, 0.03])))


def _quasiharmonic_phase():
    """Einstein solids through real phonopy objects, softening with volume so the phase expands."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "phases"))
    from test_quasiharmonic import grueneisen_phase

    return grueneisen_phase(fresh=True)


#: name -> (builder, what to evaluate on): "phase" on GRID_T x GRID_MU, "T" or "c" on that grid.
CASES = {
    "LinePhase": (lambda: LinePhase("A", 0.0, -2.0, kB), "phase"),
    "TDLP-SGTE": (lambda: _line("A", 0.0, SGTE(3)), "phase"),
    "TDLP-Whitney": (lambda: _line("A", 0.0, WhitneyTemperatureInterpolator()), "phase"),
    "TDLP-Stitched": (lambda: _line("A", 0.0, StitchedFit()), "phase"),
    "TDLP-Softplus": (lambda: _line("A", 0.0, SoftplusFit()), "phase"),
    "IdealSolution-Whitney": (lambda: IdealSolution("I", *_liquid_points()[::3]), "phase"),
    "Fast-RK-Whitney": (lambda: FastInterpolatingPhase("F", _liquid_points(), interpolator=RedlichKister(3)), "phase"),
    "Fast-Spline-Whitney": (lambda: FastInterpolatingPhase("F", _liquid_points(), interpolator=SplineFit()), "phase"),
    "Surface-Calphad-Whitney": (
        lambda: Surface2DInterpolatingPhase(
            "S",
            _liquid_points(),
            surface_interpolator=CalphadSurface2DInterpolator(terminal_interpolator=WhitneyTemperatureInterpolator()),
        ),
        "phase",
    ),
    "Surface-Whitney2D": (
        lambda: Surface2DInterpolatingPhase("S", _liquid_points(), surface_interpolator=WhitneySurface2DInterpolator()),
        "phase",
    ),
    "Surface-Softplus": (
        lambda: Surface2DInterpolatingPhase(
            "S", _liquid_points(), surface_interpolator=SoftplusSurface2DInterpolator(), num_temperature_samples=8
        ),
        "phase",
    ),
    "AsePhase": (_ase_phase, "phase"),
    "PhonopyQuasiHarmonicPhase": (_quasiharmonic_phase, "phase"),
    "WhitneyInterpolation": (lambda: _fit_in_T(WhitneyTemperatureInterpolator()), "T"),
    "StitchedInterpolation": (lambda: _fit_in_T(StitchedFit()), "T"),
    "SoftplusInterpolation": (lambda: _fit_in_T(SoftplusFit()), "T"),
    "SGTEDerivative": (lambda: _fit_in_T(SGTE(3)).deriv(), "T"),
    "SplineInterpolation": (lambda: SplineFit().fit(np.linspace(0, 1, 9), np.linspace(0, 1, 9) ** 3), "c"),
    "RedlichKisterDerivative": (
        lambda: RedlichKister(3).fit(np.linspace(0, 1, 6), 0.1 * np.sin(3 * np.linspace(0, 1, 6))).deriv(),
        "c",
    ),
    "WhitneySlice": (lambda: WhitneySurface2DInterpolator().fit(*_surface_samples()).slice_at(1000.0), "c"),
    "SoftplusSliceDerivative": (
        lambda: SoftplusSurface2DInterpolator().fit(*_surface_samples()).slice_at(1000.0).deriv(),
        "c",
    ),
}


def available():
    """The cases whose optional dependencies are installed."""
    names = list(CASES)
    try:
        import ase  # noqa: F401
    except ImportError:
        names.remove("AsePhase")
    try:
        _quasiharmonic_phase()
    except ImportError:
        names.remove("PhonopyQuasiHarmonicPhase")
    return names


def build(name):
    return CASES[name][0]()


def values(name, obj):
    """``obj`` evaluated on its grid, as a flat float array."""
    kind = CASES[name][1]
    if kind == "phase":
        phi = np.array([[float(obj.semigrand_potential(t, mu)) for mu in GRID_MU] for t in GRID_T]).ravel()
        if name == "PhonopyQuasiHarmonicPhase":
            return np.concatenate([phi, obj.line_free_energy(GRID_T_QHA)])
        return phi
    return np.asarray(obj(GRID_T if kind == "T" else GRID_C), dtype=float).ravel()


def _activate_cache(root, serializer, inline):
    import fleche as fl
    from fleche.caches import Cache
    from fleche.storage.pickle_file import CallPickleFile, ValuePickleFile

    if inline:
        from fleche.storage.destructuring import register_destructurer

        register_destructurer(lambda v: isinstance(v, np.ndarray) and v.dtype != object, lambda intern, v: (v, 0))
    fl.cache(
        Cache(
            values=ValuePickleFile.with_serializer(
                serializer, root=os.path.join(root, "values"), remaining_depth=2, compress=True
            ),
            calls=CallPickleFile.with_pickle(root=os.path.join(root, "calls")),
        )
    )


def main(mode, root, serializer, inline):
    """Cache every case as a result and as an argument, or read both back; print what was seen as JSON.

    ``make`` returns a case, so its result is stored; ``use`` takes one, so it is
    stored as an argument and the call is keyed on it.  Both are evaluated first,
    so whatever evaluating caches on an object is in the store too.
    """
    import fleche as fl
    from fleche.digest import digest

    warnings.filterwarnings("ignore")
    _activate_cache(root, serializer, inline == "1")

    @fl.fleche(version=1)
    def make(name):
        obj = build(name)
        values(name, obj)
        return obj

    @fl.fleche(version=1)
    def use(name, obj):
        values(name, obj)
        return values(name, obj).tolist()

    out = {}
    for name in available():
        if mode == "store":
            obj = make(name)
            out[name] = {"values": values(name, obj).tolist(), "used": use(name, build(name))}
        else:
            loaded = make.fleche.load(name)
            out[name] = {"values": values(name, loaded).tolist(), "type": type(loaded).__name__}
        fresh = build(name)
        out[name]["digest"] = str(digest(fresh))
        out[name]["made"] = make.fleche.contains(name)
        out[name]["hit"] = use.fleche.contains(name, fresh)
    print(json.dumps({"salt": hash("x"), "cases": out}))


if __name__ == "__main__":
    main(*sys.argv[1:])
