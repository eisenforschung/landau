"""Digest hooks that let landau objects be passed to fleche-cached functions.

`fleche <https://github.com/pmrv/fleche>`_ keys its cache on a content digest of
every argument.  Installing landau registers these hooks under the ``fleche``
entry point group, so no imports and no :func:`fleche.digest.add_hook` calls are
needed::

    from fleche import fleche
    from landau.calculate import calc_phase_diagram

    @fleche
    def diagram(phases, Ts, mu, refiners):
        return calc_phase_diagram(phases, Ts, mu, refine=refiners)

landau never imports fleche; this module is loaded only by fleche's entry point
loader.  :func:`landau_digest` is exported for reuse on a custom type, via
``add_hook((MyPhase, landau_digest))``.

Only types fleche cannot digest itself are hooked: :class:`~landau.refine.Refiner`
and :class:`~landau.interpolate.FittedSurface` (plain classes, and no subclass of
either is a dataclass), the three plain
:class:`~landau.interpolate.Interpolation` classes, and
:class:`~landau.phases.asewrapper.AsePhase` (a dataclass, but holding an opaque
ASE ``ThermoChem``).  Everything fleche already handles -- every
:class:`~landau.phases.Phase`, ``Interpolator``, ``SurfaceInterpolator``,
point-defect class and ``AbstractPolyMethod`` -- is left to its dataclass walk
deliberately: fleche loads entry points lazily, on the first value it cannot
digest, so hooking one of those would give it one digest before that happens and
another after, making the cache key depend on call order.

That covers phases only because
:class:`~landau.phases.TemperatureDependentLinePhase` keeps its precomputed
``_hash`` out of the dataclass fields; see the comment on its ``__post_init__``.

Interpolations built from a closure (``SplineFit``, ``StitchedFit``,
``SoftplusFit``, Whitney) are refused rather than digested -- two fits of one
interpolator share a code object, and fleche falls back to digesting that alone
whenever the state a closure captured is itself indigestible.
``WhitneyFittedSurface`` likewise raises, naming the fitted scipy objects it
holds.  Fit from an ``Interpolator`` inside the cached function instead.
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

    fleche digests a :class:`types.FunctionType` from what it captured where it
    can, and from its code object alone where it cannot -- and fits of one
    interpolator share that code object.  Only containers cheap to walk are
    descended into; numpy arrays hold no functions and are skipped.
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
            f"{type(obj).__name__} holds a plain function at {', '.join(sorted(opaque))}; "
            "fleche digests a function by its code object alone once what it captured "
            "is indigestible, so two fits would collide. "
            "Pass the Interpolator and its samples instead."
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
