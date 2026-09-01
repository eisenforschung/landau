"""Digest hooks that make landau objects usable as arguments to cached functions.

`fleche <https://github.com/pmrv/fleche>`_ keys its cache on a content digest of
every argument.  This module registers digest hooks for the landau types fleche
cannot digest on its own, under the ``fleche`` entry point group, so installing
landau is all it takes -- no imports, no :func:`fleche.digest.add_hook` calls::

    from fleche import fleche
    from landau.calculate import calc_phase_diagram

    @fleche
    def diagram(phases, Ts, mu, refiners):
        return calc_phase_diagram(phases, Ts, mu, refine=refiners)

landau itself never imports fleche; this module is loaded only by fleche's entry
point loader.

:func:`landau_digest` is exported for reuse: a custom phase or refiner that
fleche cannot digest -- one that is not a dataclass, or one whose fields it does
not understand -- can be covered with
``add_hook((MyPhase, landau_digest))``.

What is hooked, and what deliberately is not
--------------------------------------------

fleche already digests a frozen dataclass by walking its declared fields, which
covers every :class:`~landau.phases.Phase`,
:class:`~landau.interpolate.Interpolator`,
:class:`~landau.interpolate.SurfaceInterpolator`, point-defect class and
:class:`~landau.poly.AbstractPolyMethod` in landau.  Those get **no hook**, on
purpose: fleche loads entry points lazily, only after it meets a value it cannot
digest, so a type it can already handle would be digested one way before the
first :exc:`~fleche.digest.Indigestible` in a process and another way after.
The resulting cache key would depend on call order.  Hooks are therefore
registered only for types fleche rejects outright, which leaves exactly one
possible digest per object no matter when the hooks load.

That leaves:

* :class:`~landau.refine.Refiner` and :class:`~landau.interpolate.FittedSurface`
  -- plain classes rather than dataclasses (no subclass of either is a
  dataclass, so hooking the base class introduces no such split).
* The three plain :class:`~landau.interpolate.Interpolation` classes.  The ABC
  itself is not hooked because ``SGTEInterpolation`` and
  ``RedlichKisterInterpolation`` *are* dataclasses.
* :class:`~landau.phases.asewrapper.AsePhase`, which is a dataclass but holds an
  opaque ASE ``ThermoChem``.

For phases the hooks are only half the story: a digest is worth nothing if it is
not reproducible in the next interpreter.
:class:`~landau.phases.TemperatureDependentLinePhase` keeps its precomputed
``_hash`` out of the dataclass fields for exactly that reason -- see the comment
on its ``__post_init__``.

Fitted curves built from closures
---------------------------------

:class:`~landau.interpolate.Interpolation` objects that wrap a closure -- what
``SplineFit``, ``StitchedFit``, ``SoftplusFit`` and the Whitney interpolators
return -- are refused with :exc:`fleche.digest.Indigestible` rather than
digested.  fleche digests a function from its code object alone, and two fits of
one interpolator share that code object while closing over different data, so
digesting them would hand out a single cache entry for two different curves.
Pass the :class:`~landau.interpolate.Interpolator` (which digests exactly) and
its samples into the cached function and fit there instead.

``WhitneyFittedSurface`` stays undigestible for the neighbouring reason: its
state is a fitted scipy ``RBFInterpolator`` and ``ConvexHull``, which are not
landau's to describe.  It raises naming those objects rather than silently
digesting around them.  Fitting from a
:class:`~landau.interpolate.SurfaceInterpolator` inside the cached function has
the same answer here as it does for the closures above.
"""

import dataclasses
import types
from collections.abc import Mapping

from fleche.digest import Digest, Hook, Indigestible, digest

from .interpolate.basic import (
    FittedSurface,
    NumericalDerivative,
    PolynomialInterpolation,
    _CallableInterpolation,
)
from .phases import AsePhase
from .refine import Refiner

__all__ = ["landau_digest", "ase_phase_digest", "digest_hooks"]


def _declared_state(obj) -> dict:
    """The attributes that make up *obj*'s identity, by name.

    Dataclasses report the fields that take part in construction -- ``init=False``
    fields are derived from those in ``__post_init__`` and add no identity of
    their own.  Plain classes report ``vars``, private names included: for the
    fitted surfaces and interpolations those *are* the state.
    """
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init}
    state = getattr(obj, "__dict__", None)
    if state is None:
        raise Indigestible(
            f"{type(obj).__name__} is neither a dataclass nor has a __dict__, so landau "
            "cannot read its state; give it a __digest__ method or register a hook for it."
        )
    return dict(state)


def _opaque_callables(value, path: str = "") -> list[str]:
    """Paths of plain functions reachable from *value* through containers.

    fleche digests a :class:`types.FunctionType` from its code object, ignoring
    the cells it closes over, so every fit of one interpolator would share a
    digest.  Only containers cheap to walk are descended into; numpy arrays hold
    no functions and are skipped.
    """
    if isinstance(value, types.FunctionType):
        return [path or "the value itself"]
    if isinstance(value, Mapping):
        return [p for k, v in value.items() for p in _opaque_callables(v, f"{path}[{k!r}]")]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [p for i, v in enumerate(value) for p in _opaque_callables(v, f"{path}[{i}]")]
    return []


def landau_digest(obj) -> Digest:
    """Digest a landau object from its declared state.

    Raises:
        Indigestible: if that state holds a plain function, whose digest would
            not tell it apart from any other closure over the same code.
    """
    state = _declared_state(obj)
    opaque = _opaque_callables(state)
    if opaque:
        raise Indigestible(
            f"{type(obj).__name__} holds a plain function at {', '.join(sorted(opaque))}. "
            "fleche digests a function from its code object alone, so two closures over "
            "different data share a digest and this would return one cached result for two "
            "different fits. Pass the Interpolator and its samples into the cached function "
            "and fit there, instead of passing the fitted curve."
        )
    return digest((type(obj).__name__, state))


def ase_phase_digest(phase: AsePhase) -> Digest:
    """Digest an :class:`~landau.phases.asewrapper.AsePhase`.

    ``thermochem`` is an opaque ASE object, so the digest reuses the phase's own
    identity key -- which pickles it -- keeping the digest in step with the
    class's ``__eq__``/``__hash__`` by construction.
    """
    return digest((type(phase).__name__, phase._key()))


#: Hooks fleche loads from the ``fleche`` entry point group (see ``pyproject.toml``).
#: Only types fleche cannot digest by itself appear here; see the module docstring
#: for why covering the dataclasses too would make their digests order-dependent.
digest_hooks = [
    Hook(AsePhase, ase_phase_digest),
    Hook(Refiner, landau_digest),
    Hook(FittedSurface, landau_digest),
    Hook(NumericalDerivative, landau_digest),
    Hook(PolynomialInterpolation, landau_digest),
    Hook(_CallableInterpolation, landau_digest),
]
