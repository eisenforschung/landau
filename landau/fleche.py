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

Only types fleche refuses are hooked: :class:`~landau.refine.Refiner`,
:class:`~landau.interpolate.FittedSurface`, the plain
:class:`~landau.interpolate.Interpolation` classes,
:class:`~landau.interpolate.whitney.WhitneyRBFInterpolator`, and, for types landau
does not own, a fitted scipy spline and phonopy's ``ThermalProperties``.
Dataclasses -- every phase, interpolator and poly method -- are left to fleche's
field walk: a hooked type would digest one way before fleche loads its entry points
and another after, and a value store files a destructured dataclass under its
field-walk digest, not a hook's.  So phases keep anything derived from their fields
(hashes, comparison keys) out of them, as a ``cached_property``.

Fits are mostly closures in a :class:`~landau.interpolate.basic._CallableInterpolation`;
fleche digests a closure by its code object and its captured values, so a key built
on a bare fit moves with the Python version and with any edit to the closure.
An :class:`~landau.phases.asewrapper.AsePhase` needs
`fleche-ase <https://pypi.org/project/fleche-ase/>`_ for its ``thermochem``.
"""

import dataclasses

from fleche.digest import Digest, Hook, Indigestible, digest
from scipy.interpolate import UnivariateSpline

from .interpolate.basic import FittedSurface, NumericalDerivative, PolynomialInterpolation, _CallableInterpolation
from .interpolate.whitney import WhitneyRBFInterpolator
from .phases.quasiharmonic import _thermal_properties_state  # importable without phonopy
from .refine import Refiner

__all__ = [
    "landau_digest",
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

    :func:`~landau.phases.quasiharmonic._thermal_properties_state`: its ``__dict__``
    less the last run's temperatures and results, the backend and what ``__init__``
    derives, so evaluating a phase does not move its key.  Not the pickled bytes,
    which carry the last run and change layout between phonopy releases.
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
