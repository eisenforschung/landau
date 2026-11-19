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
either is a dataclass), the plain :class:`~landau.interpolate.Interpolation`
classes the fits return, and
:class:`~landau.phases.quasiharmonic.PhonopyQuasiHarmonicPhase` (a dataclass, but
holding opaque phonopy ``ThermalProperties``), and
:class:`~landau.interpolate.whitney.WhitneyRBFInterpolator` (an sklearn-style
estimator whose fitted state is a scipy ``RBFInterpolator`` and a ``ConvexHull``).
Everything fleche already handles -- every
:class:`~landau.phases.Phase`, ``Interpolator``, ``SurfaceInterpolator``,
point-defect class and ``AbstractPolyMethod`` -- is left to its dataclass walk
deliberately: fleche loads entry points lazily, on the first value it cannot
digest, so hooking one of those would give it one digest before that happens and
another after, making the cache key depend on call order.

That covers phases only because
:class:`~landau.phases.TemperatureDependentLinePhase` and
``PhonopyQuasiHarmonicPhase`` keep their hash out of the dataclass fields, as a
``cached_property`` computed on first use.

Every fitted interpolation is an instance of a module-level class holding its
fitted state (``WhitneyInterpolation``, ``SplineInterpolation``,
``StitchedInterpolation``, ...), digested here by that state.
:class:`~landau.interpolate.basic._CallableInterpolation` around a user's own
closure is still refused -- two closures over one code object can be told apart
only by what they captured, and fleche falls back to digesting the code object
alone whenever that is itself indigestible (#449).

:func:`spline_digest` and :func:`thermochem_digest` are the hooks here for types
landau does not own: fleche refuses a fitted scipy spline and an ASE ``ThermoChem``.
Both are stopgaps; hooks are global to every fleche user with landau installed, so
they belong upstream (#449 for the spline, fleche-ase for the thermochem).
:class:`~landau.phases.asewrapper.AsePhase` itself is left to the dataclass walk,
which reaches its ``thermochem`` through the latter.
"""

import dataclasses
import types
from collections.abc import Mapping

from fleche.digest import Digest, Hook, Indigestible, digest
from pyiron_snippets.deprecate import deprecate
from scipy.interpolate import UnivariateSpline

from .interpolate.basic import (
    FittedSurface,
    NumericalDerivative,
    PolynomialInterpolation,
    RedlichKisterDerivative,
    SGTEDerivative,
    SplineInterpolation,
    StitchedInterpolation,
    _CallableInterpolation,
)
from .interpolate.softplus import SoftplusInterpolation, SoftplusSliceDerivative
from .interpolate.whitney import WhitneyInterpolation, WhitneyRBFInterpolator, WhitneySlice
from .phases import AsePhase, asewrapper
from .phases.quasiharmonic import PhonopyQuasiHarmonicPhase  # importable without phonopy
from .refine import Refiner

__all__ = [
    "landau_digest",
    "ase_phase_digest",
    "quasiharmonic_phase_digest",
    "spline_digest",
    "thermochem_digest",
    "whitney_rbf_digest",
    "digest_hooks",
]


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


@deprecate("fleche digests an AsePhase by its fields; call fleche.digest.digest instead", version="2.0")
def ase_phase_digest(phase: AsePhase) -> Digest:
    """Digest an :class:`~landau.phases.asewrapper.AsePhase`: the same as ``fleche.digest.digest(phase)``.

    No longer registered as a hook.  fleche's value storages rebuild a stored
    dataclass field by field and key it by the dataclass digest, so a phase
    digested any other way is stored under a key no lookup computes.  With
    :func:`thermochem_digest` registered, the field walk covers ``thermochem`` too.
    """
    return digest(phase)


def quasiharmonic_phase_digest(phase: PhonopyQuasiHarmonicPhase) -> Digest:
    """Digest a :class:`~landau.phases.quasiharmonic.PhonopyQuasiHarmonicPhase`.

    ``thermal_properties`` holds opaque phonopy objects, so the digest reuses the
    phase's own identity key -- pickled bytes of each, taken at 0 K so the
    temperature the phase last ran at cannot drift them -- keeping the digest in
    step with the class's ``__eq__``/``__hash__`` by construction.
    """
    return digest((type(phase).__name__, phase._key))


def whitney_rbf_digest(estimator: WhitneyRBFInterpolator) -> Digest:
    """Digest a :class:`~landau.interpolate.whitney.WhitneyRBFInterpolator`.

    Its fitted state is a scipy ``RBFInterpolator`` and a ``ConvexHull``, neither
    of which fleche digests -- and both of which are deterministic functions of the
    training data and the hyperparameters, so those are what the digest keys on.
    ``rbf_.y``/``rbf_.d`` are the points and values the RBF was built from; an
    unfitted estimator has neither and digests by hyperparameters alone.
    """
    rbf = getattr(estimator, "rbf_", None)
    return digest(
        (
            type(estimator).__name__,
            estimator.kernel,
            estimator.smoothing,
            estimator.degree,
            estimator.epsilon,
            estimator.grad_eps,
            None if rbf is None else rbf.y,
            None if rbf is None else rbf.d,
            getattr(estimator, "x_min_", None),
            getattr(estimator, "x_max_", None),
        )
    )


def spline_digest(spline: UnivariateSpline) -> Digest:
    """Digest a fitted scipy spline by the arguments it is evaluated with.

    ``_eval_args`` is the ``(t, c, k)`` tuple handed to ``splev`` -- knots,
    coefficients, degree -- so with ``ext`` it is the whole of what the spline
    computes.  It is private, but the public accessors do not cover it:
    ``get_knots`` returns interior knots only and ``UnivariateSpline`` exposes no
    degree.

    Stopgap.  A scipy spline is not a landau type, and hooks are global to every
    fleche user with landau installed, so this belongs upstream; see #449, to be
    dropped as soon as fleche digests scipy objects itself.
    """
    return digest((type(spline).__name__, spline._eval_args, spline.ext))


def thermochem_digest(thermochem) -> Digest:
    """Digest an ASE ``ThermoChem`` by its pickled bytes, leaving out ``verbose``.

    fleche refuses the object, so an :class:`~landau.phases.asewrapper.AsePhase`
    stored with its fields destructured had no key for its ``thermochem``.  The
    bytes are the ones :class:`~landau.phases.asewrapper.AsePhase` compares by:
    ASE writes ``verbose`` on every evaluation, so a digest including it would move
    the first time the phase is used.

    Stopgap.  ``ThermoChem`` is an ASE type, and hooks are global to every fleche
    user with landau installed; its home is fleche-ase.
    """
    return digest((type(thermochem).__name__, asewrapper._thermochem_bytes(thermochem)))


#: Hooks fleche loads from the ``fleche`` entry point group (see ``pyproject.toml``).
#: Only types fleche cannot digest by itself appear here; see the module docstring
#: for why covering the dataclasses too would make their digests order-dependent.
digest_hooks = [
    Hook(Refiner, landau_digest),
    Hook(FittedSurface, landau_digest),
    Hook(NumericalDerivative, landau_digest),
    Hook(PolynomialInterpolation, landau_digest),
    Hook(SGTEDerivative, landau_digest),
    Hook(RedlichKisterDerivative, landau_digest),
    Hook(SplineInterpolation, landau_digest),
    Hook(StitchedInterpolation, landau_digest),
    Hook(SoftplusInterpolation, landau_digest),
    Hook(SoftplusSliceDerivative, landau_digest),
    Hook(WhitneyInterpolation, landau_digest),
    Hook(WhitneySlice, landau_digest),
    Hook(_CallableInterpolation, landau_digest),
    Hook(UnivariateSpline, spline_digest),
    Hook(PhonopyQuasiHarmonicPhase, quasiharmonic_phase_digest),
    Hook(WhitneyRBFInterpolator, whitney_rbf_digest),
]
if asewrapper.ase_alarm.message is None:
    digest_hooks.append(Hook(asewrapper.ThermoChem, thermochem_digest))
