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
classes, and
:class:`~landau.interpolate.whitney.WhitneyRBFInterpolator` (an sklearn-style
estimator whose fitted state is a scipy ``RBFInterpolator`` and a ``ConvexHull``).
Everything fleche already handles -- every
:class:`~landau.phases.Phase`, ``Interpolator``, ``SurfaceInterpolator``,
point-defect class and ``AbstractPolyMethod`` -- is left to its dataclass walk
deliberately: fleche loads entry points lazily, on the first value it cannot
digest, so hooking one of those would give it one digest before that happens and
another after, making the cache key depend on call order.  A second reason holds
for every dataclass: fleche's value stores file a destructured dataclass under its
field-walk digest, so a hooked one would be stored under a key no lookup computes.

That covers phases only because
:class:`~landau.phases.TemperatureDependentLinePhase` and
:class:`~landau.phases.quasiharmonic.PhonopyQuasiHarmonicPhase` keep their hash
(and the latter its comparison key) out of the dataclass fields, as a
``cached_property`` computed on first use.

Most fits return a :class:`~landau.interpolate.basic._CallableInterpolation`
around a closure.  fleche digests a closure by its code object and the values it
captured, so two fits of one interpolator differ by their captures, and a capture
fleche cannot digest makes the whole closure ``Indigestible``.  The fitted scipy
spline and the Whitney estimator such closures capture are covered by the hooks
below.  A closure's code object is part of its digest, so a key built on a fit
moves with the Python version and with any edit to the closure.

:func:`spline_digest` and :func:`thermal_properties_digest` are the hooks here for
types landau does not own: fleche refuses a fitted scipy spline and a phonopy
``ThermalProperties``.  The spline hook is a stopgap; hooks are global to every
fleche user with landau installed, so it belongs upstream (#449).
``PhonopyQuasiHarmonicPhase`` is left to the dataclass walk, which reaches its
``thermal_properties`` through the latter.
:class:`~landau.phases.asewrapper.AsePhase` is left to it too; its ``thermochem``
is digested by `fleche-ase <https://pypi.org/project/fleche-ase/>`_, without which
fleche refuses it.
"""

import dataclasses

from fleche.digest import Digest, Hook, Indigestible, digest
from pyiron_snippets.deprecate import deprecate
from scipy.interpolate import UnivariateSpline

from .interpolate.basic import FittedSurface, NumericalDerivative, PolynomialInterpolation, _CallableInterpolation
from .interpolate.whitney import WhitneyRBFInterpolator
from .phases import AsePhase
from .phases.quasiharmonic import PhonopyQuasiHarmonicPhase, _thermal_properties_state  # importable without phonopy
from .refine import Refiner

__all__ = [
    "landau_digest",
    "ase_phase_digest",
    "quasiharmonic_phase_digest",
    "spline_digest",
    "thermal_properties_digest",
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


def landau_digest(obj) -> Digest:
    """Digest a landau object from its declared state."""
    return digest((type(obj).__name__, _declared_state(obj)))


@deprecate("fleche digests an AsePhase by its fields; call fleche.digest.digest instead", version="2.0")
def ase_phase_digest(phase: AsePhase) -> Digest:
    """Digest an :class:`~landau.phases.asewrapper.AsePhase`: the same as ``fleche.digest.digest(phase)``.

    No longer registered as a hook.  fleche's value storages rebuild a stored
    dataclass field by field and key it by the dataclass digest, so a phase
    digested any other way is stored under a key no lookup computes.  With
    fleche-ase installed, the field walk covers ``thermochem`` too.
    """
    return digest(phase)


@deprecate(
    "fleche digests a PhonopyQuasiHarmonicPhase by its fields; call fleche.digest.digest instead", version="2.0"
)
def quasiharmonic_phase_digest(phase: PhonopyQuasiHarmonicPhase) -> Digest:
    """Digest a :class:`~landau.phases.quasiharmonic.PhonopyQuasiHarmonicPhase`, as ``fleche.digest.digest`` does.

    No longer registered as a hook, for the reason :func:`ase_phase_digest` is not:
    a value store files a destructured dataclass under its field-walk digest.  With
    :func:`thermal_properties_digest` registered, the field walk covers
    ``thermal_properties`` too.
    """
    return digest(phase)


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


def thermal_properties_digest(thermal_properties) -> Digest:
    """Digest a phonopy ``ThermalProperties`` by the spectrum it sums over.

    That is its ``__dict__`` less what a run overwrites and what ``__init__`` derives
    (:func:`~landau.phases.quasiharmonic._thermal_properties_state`), passed through
    fleche's own digest: mode frequencies and mesh weights as arrays, the cutoff,
    classical or quantum statistics, the band selection, and the formula units.  The
    temperatures and results of the last run are left out, so evaluating a phase does
    not move its key, and so is the backend (``_lang``).  Not the pickled bytes, which
    carry those and differ between phonopy releases in layout as well as content.

    Holds no q-points because the object holds none: frequencies and weights are all
    the mode sum reads.  The state is phonopy's private layout, so a phonopy release
    that renames an attribute re-keys entries (they miss once) rather than colliding.
    """
    return digest((type(thermal_properties).__name__, _thermal_properties_state(thermal_properties)))


#: Hooks fleche loads from the ``fleche`` entry point group (see ``pyproject.toml``).
#: Only types fleche cannot digest by itself appear here; see the module docstring
#: for why covering the dataclasses too would make their digests order-dependent.
digest_hooks = [
    Hook(Refiner, landau_digest),
    Hook(FittedSurface, landau_digest),
    Hook(NumericalDerivative, landau_digest),
    Hook(PolynomialInterpolation, landau_digest),
    Hook(_CallableInterpolation, landau_digest),
    Hook(UnivariateSpline, spline_digest),
    Hook(WhitneyRBFInterpolator, whitney_rbf_digest),
]
try:
    from phonopy.phonon.thermal_properties import ThermalProperties
except ImportError:
    pass
else:
    digest_hooks.append(Hook(ThermalProperties, thermal_properties_digest))
