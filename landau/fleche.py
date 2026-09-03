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
:class:`~landau.phases.asewrapper.AsePhase` and
:class:`~landau.phases.quasiharmonic.PhonopyQuasiHarmonicPhase` (dataclasses, but
holding an opaque ASE ``ThermoChem`` and phonopy ``ThermalProperties``), and
:class:`~landau.interpolate.whitney.WhitneyRBFInterpolator` (an sklearn-style
estimator whose fitted state is a scipy ``RBFInterpolator`` and a ``ConvexHull``).
Everything fleche already handles -- every
:class:`~landau.phases.Phase`, ``Interpolator``, ``SurfaceInterpolator``,
point-defect class and ``AbstractPolyMethod`` -- is left to its dataclass walk
deliberately: fleche loads entry points lazily, on the first value it cannot
digest, so hooking one of those would give it one digest before that happens and
another after, making the cache key depend on call order.

That covers phases only because
:class:`~landau.phases.TemperatureDependentLinePhase` keeps its precomputed
``_hash`` out of the dataclass fields; see the comment on its ``__post_init__``.
``PhonopyQuasiHarmonicPhase`` keeps its own out for the same reason, though its
hook means no digest reaches those fields anyway.

Interpolations built from a closure (``SplineFit``, ``StitchedFit``,
``SoftplusFit``, Whitney) are refused rather than digested -- two fits of one
interpolator share a code object, and fleche falls back to digesting that alone
whenever the state a closure captured is itself indigestible.  Narrowing the
refusal to that case is open (#449).
Fit from an ``Interpolator`` inside the cached function instead.
``WhitneyFittedSurface`` is not among them: its whole state is one
``WhitneyRBFInterpolator``, which the hook above digests.

:func:`spline_digest` is the one hook here for a type landau does not own: fleche
refuses a fitted scipy spline, so a function closing over one -- or taking one --
cannot be cached at all.  It is a stopgap for #449.
"""

import dataclasses
import types
from collections.abc import Mapping

from fleche.digest import Digest, Hook, Indigestible, digest
from scipy.interpolate import UnivariateSpline

from .interpolate.basic import (
    FittedSurface,
    NumericalDerivative,
    PolynomialInterpolation,
    _CallableInterpolation,
)
from .interpolate.whitney import WhitneyRBFInterpolator
from .phases import AsePhase
from .phases.quasiharmonic import PhonopyQuasiHarmonicPhase  # importable without phonopy
from .refine import Refiner

__all__ = [
    "landau_digest",
    "ase_phase_digest",
    "quasiharmonic_phase_digest",
    "spline_digest",
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


def ase_phase_digest(phase: AsePhase) -> Digest:
    """Digest an :class:`~landau.phases.asewrapper.AsePhase`.

    ``thermochem`` is an opaque ASE object, so the digest reuses the phase's own
    identity key -- which pickles it -- keeping the digest in step with the
    class's ``__eq__``/``__hash__`` by construction.
    """
    return digest((type(phase).__name__, phase._key()))


def quasiharmonic_phase_digest(phase: PhonopyQuasiHarmonicPhase) -> Digest:
    """Digest a :class:`~landau.phases.quasiharmonic.PhonopyQuasiHarmonicPhase`.

    ``thermal_properties`` holds opaque phonopy objects, so the digest reuses the
    phase's own identity key -- pickled bytes of each, taken at 0 K so the
    temperature the phase last ran at cannot drift them -- keeping the digest in
    step with the class's ``__eq__``/``__hash__`` by construction.  The same route
    :func:`ase_phase_digest` takes for its ``ThermoChem``.
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
    Hook(UnivariateSpline, spline_digest),
    Hook(PhonopyQuasiHarmonicPhase, quasiharmonic_phase_digest),
    Hook(WhitneyRBFInterpolator, whitney_rbf_digest),
]
