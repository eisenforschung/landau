from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
import scipy
from scipy.optimize import least_squares

from .basic import (
    ConcentrationInterpolator,
    TemperatureInterpolator,
    Interpolation,
    SurfaceInterpolator,
    FittedSurface,
    _CallableInterpolation,
    _scalarize,
)


def _softplus(t):
    t = np.asarray(t, float)
    return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)


def _sigmoid(t):
    # Numerically stable: avoid overflow in exp for large |t|.
    t = np.asarray(t, float)
    out = np.empty_like(t)
    pos = t >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    ex = np.exp(t[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _softplus_inv(a):
    """Inverse of softplus: ``alpha`` such that ``softplus(alpha) = a`` (``a > 0``).

    Uses ``alpha = a + log1p(-exp(-a))`` rather than ``log(expm1(a))`` so it stays
    finite for large ``a`` (where ``softplus`` is ~identity) instead of
    overflowing ``expm1``.
    """
    a = np.maximum(np.asarray(a, float), 1e-12)
    return a + np.log1p(-np.exp(-a))


# --------------------------------------------------------------------------- #
# Fixed-T softplus model:  offset + sum_i a_i * softplus(b_i * (cn + c_i)).
# The single shared evaluation used by SoftplusFit, the per-slice seed fit, and
# the fitted surface slice -- value, c-derivative, and parameter Jacobian.
# --------------------------------------------------------------------------- #
def _terms(cn, a, b, c, offset):
    """Evaluate the softplus model on the normalised grid ``cn``."""
    out = np.full(np.shape(cn), offset, float)
    for ai, bi, ci in zip(a, b, c):
        out = out + ai * _softplus(bi * (cn + ci))
    return out


def _terms_dc(cn, a, b, c):
    """``d/dcn`` of the softplus model (the offset drops out)."""
    out = np.zeros(np.shape(cn), float)
    for ai, bi, ci in zip(a, b, c):
        out = out + ai * _sigmoid(bi * (cn + ci)) * bi
    return out


def _terms_jac(cn, a, b, c):
    """Jacobian wrt the flat ``[a_i, b_i, c_i]*n + [offset]`` parameter vector.

    For one term ``a*softplus(b*(cn+c))`` with ``t = b*(cn+c)``:
    ``d/da = softplus(t)``, ``d/db = a*sigmoid(t)*(cn+c)``, ``d/dc = a*sigmoid(t)*b``;
    the offset contributes a column of ones.
    """
    n = len(a)
    J = np.empty((cn.size, 3 * n + 1))
    for i in range(n):
        u = cn + c[i]
        t = b[i] * u
        sig = _sigmoid(t)
        J[:, 3 * i] = _softplus(t)
        J[:, 3 * i + 1] = a[i] * sig * u
        J[:, 3 * i + 2] = a[i] * sig * b[i]
    J[:, -1] = 1.0
    return J


def _flat(a, b, c, offset):
    """Pack per-term ``(a, b, c)`` plus the offset into ``[a_0,b_0,c_0, ..., offset]``."""
    return np.concatenate([np.ravel(np.column_stack([a, b, c])), [offset]])


def _split(p, n):
    """Inverse of :func:`_flat`: the ``(a, b, c, offset)`` of a flat slice vector."""
    return p[0:3 * n:3], p[1:3 * n:3], p[2:3 * n:3], p[-1]


def _fit_softplus(cn, y, p0, bounds, *, loss="linear", f_scale=1.0, max_nfev, x_scale=1.0):
    """Bounded trust-region least-squares fit of the softplus sum (:func:`_terms`) to
    ``(cn, y)`` from seed ``p0`` (clipped into ``bounds``).  The single solver shared
    by :meth:`SoftplusFit.fit` and the surface seed fit :func:`_fit_slice`; they
    differ only in how they build the seed and bounds.  Returns the scipy result,
    whose ``x`` is the flat ``[a, b, c]*n + [offset]`` vector."""
    n = (len(p0) - 1) // 3
    return least_squares(
        lambda p: _terms(cn, *_split(p, n)) - y, np.clip(p0, *bounds),
        jac=lambda p: _terms_jac(cn, *_split(p, n)[:3]),
        bounds=bounds, loss=loss, f_scale=f_scale, max_nfev=max_nfev, x_scale=x_scale,
    )


@dataclass(frozen=True, eq=True)
class SoftplusFit(ConcentrationInterpolator, TemperatureInterpolator):
    """
    Fits data using a sum of softplus functions for smooth, flexible interpolation.

    The softplus function provides numerically stable, smooth approximations suitable
    for thermodynamic data. This interpolator uses multiple softplus terms to capture
    complex behavior.

    :meth:`.fit` runs a local Levenberg–Marquardt / trust-region fit with
    analytical Jacobian.
    """

    n_softplus: int = 2
    """Number of softplus terms to fit."""
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    """Loss function for robust fitting."""
    max_nfev: int = 100
    """Maximum number of function evaluations for the local fit."""
    f_scale: float = 1.0
    """Soft margin between inlier and outlier residuals for the robust loss
    (see :func:`scipy.optimize.least_squares`).  Larger values weaken the
    influence of the robust loss — at the limit it behaves quadratically over
    the full residual range, equivalent to ``loss='linear'``.  Has no effect
    when ``loss='linear'``."""

    def _seed(self, xn, y):
        """Initial parameter vector ``p0`` and bounds ``(lo, hi)`` for the local fit.

        ``a_guess`` is the full peak-to-peak range of ``y`` rather than
        ``ptp/(2*n)``: amplitudes large enough to actually reach the data keep the
        optimizer out of the wide near-constant basin that ``soft_l1`` opens up
        around small ``a`` (PR #82).  The bound on ``a`` is open above, so
        over-estimating is harmless — the optimizer trims it.
        """
        n = self.n_softplus
        ptp = float(np.ptp(y))
        a_guess = ptp if ptp > 0 else 1.0
        knees = [float(np.quantile(xn, (i + 0.5) / n)) for i in range(n)]
        slopes = [3.0 if i % 2 == 0 else -3.0 for i in range(n)]
        p0 = _flat([a_guess] * n, slopes, [-k for k in knees], float(np.median(y)))
        lo = _flat(np.zeros(n), np.full(n, -20.0), np.full(n, -10.0), -np.inf)
        hi = _flat(np.full(n, np.inf), np.full(n, 20.0), np.full(n, 10.0), np.inf)
        return p0, lo, hi

    def fit(self, x, y) -> Interpolation:
        """Local nonlinear least-squares fit with analytical Jacobian."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        shift = float(x.mean())
        scale = float(x.std()) if x.std() > 0 else 1.0
        xn = (x - shift) / scale
        n = self.n_softplus

        p0, lo, hi = self._seed(xn, y)
        popt = _fit_softplus(xn, y, p0, (lo, hi),
                             loss=self.loss, f_scale=self.f_scale, max_nfev=self.max_nfev).x
        return _CallableInterpolation(
            lambda x_new: _terms((np.asarray(x_new, float) - shift) / scale, *_split(popt, n))
        )


def _standardize(x):
    """Centre ``x`` on its mean and scale by its std (1 if constant, so a
    single-valued input maps to ``0`` rather than ``nan``).  Returns the
    normalised array plus the ``(shift, scale)`` needed to reproduce the
    transform on a later input.  Centring and scaling conditions the basis the
    coefficient polynomials are solved against -- ``np.vander``/``polyval`` raise
    the variable to powers as given, so raw ``T`` (``T**4 ~ 1e12`` over
    400-1600 K) or raw ``1/T`` (``[6e-4, 2.5e-3]``) span many decades."""
    x = np.asarray(x, float)
    shift = float(x.mean())
    scale = float(x.std() or 1.0)
    return (x - shift) / scale, shift, scale


@dataclass(frozen=True)
class _SoftplusSlice(Interpolation):
    """A fixed-T softplus curve ``off + sum_i a_i*softplus(b_i*(cn + c_i))``.

    ``cn = (c - cm)/cs`` is the concentration normalisation used during the
    surface fit.  Carries an analytic c-derivative.  Produced by
    :meth:`SoftplusFittedSurface.slice_at`; not constructed directly.
    """

    offset: float
    a: np.ndarray  # (n,) non-negative amplitudes
    b: np.ndarray  # (n,) slopes
    c: np.ndarray  # (n,) knee positions in normalised coords
    cm: float
    cs: float

    def __call__(self, c):
        cn = (np.asarray(c, float) - self.cm) / self.cs
        return _scalarize(_terms(cn, self.a, self.b, self.c, self.offset))

    def deriv(self) -> Interpolation:
        # d/dc [ a*softplus(b*(cn + c)) ] = a*sigmoid(b*(cn+c))*b * d(cn)/dc,
        # with d(cn)/dc = 1/cs
        a, b, c, cm, cs = self.a, self.b, self.c, self.cm, self.cs

        def dfdc(cc):
            cn = (np.asarray(cc, float) - cm) / cs
            return _scalarize(_terms_dc(cn, a, b, c) / cs)

        return _CallableInterpolation(dfdc)


def _scipy_at_least(major: int, minor: int) -> bool:
    try:
        return tuple(int(p) for p in scipy.__version__.split(".")[:2]) >= (major, minor)
    except (ValueError, AttributeError):  # unexpected version string -> be conservative
        return False


def _knee_position(cn, H):
    """Well-bottom (knee) estimate: the ``argmin`` of the data with its best-fit
    line in ``c`` removed.  A softplus term's knee is where its curvature
    concentrates; for a convex well that is the curvature peak, which the
    detrended ``argmin`` recovers.  The raw ``argmin`` instead returns the
    function *minimum*, which for an asymmetric well sits off the knee and, when
    the entropy-removed free energy is a small dimple riding a chemical-potential
    tilt, collapses onto the downhill window edge.  Removing the linear trend
    exposes the curvature and leaves the knee where it belongs."""
    return cn[np.argmin(H - np.polyval(np.polyfit(cn, H, 1), cn))]


def _smoothv_seed(cn, H, n, bb=12.0):
    """Data-driven initial guess for one fixed-T slice of ``n`` softplus terms: an
    opposite-slope softplus pair at the knee (the smooth-V identity
    ``softplus(b u)+softplus(-b u) ~ |u|``); any further terms start inert (tiny
    amplitude) and are activated by the polish only if the data needs them.
    Returns flat ``[a, b, c]*n + [offset]`` in the normalised concentration ``cn``.

    Knee and branch slopes are seeded from *different* features.  The knee is the
    curvature centre (:func:`_knee_position`).  The two branch amplitudes come
    from one-sided line fits, but the data is split at the function *minimum* (its
    turning point) rather than at the knee: the two softplus arms are monotone
    only on either side of where ``H`` actually turns around, so splitting at the
    minimum keeps each line fit on a monotone arm even when the minimum sits off
    the knee (an asymmetric well)."""
    c0 = _knee_position(cn, H)
    split = cn[np.argmin(H)]
    left, right = cn <= split, cn >= split
    sL = abs(np.polyfit(cn[left], H[left], 1)[0]) if left.sum() >= 2 else 1.0
    sR = abs(np.polyfit(cn[right], H[right], 1)[0]) if right.sum() >= 2 else 1.0
    a = [max(sR / bb, 1e-6), max(sL / bb, 1e-6)] + [1e-4] * (n - 2)
    b = [bb, -bb] + [bb] * (n - 2)
    c = [-c0] * n
    return _flat(a, b, c, float(np.median(H)))


def _fit_slice(cn, H, n, max_nfev, bmax=200.0):
    """Robust single-slice fit: data-driven smooth-V seed, then a bounded,
    graduated-sharpness polish via :func:`_fit_softplus` (``x_scale='jac'``).
    Returns the per-term ``(a, b, c)`` arrays and the offset.  Used only to seed the
    coupled surface fit, so two slices (coldest, central) are enough regardless of
    how many temperatures are sampled.

    The convex-well-aware seed (:func:`_smoothv_seed`) and the data-driven knee
    bounds are what distinguish this from the generic :meth:`SoftplusFit.fit`; both
    share the same underlying solver.  ``bmax`` bounds the (normalised) slope: a
    redundant softplus term is unconstrained in a flat direction and would otherwise
    drift to infinity; the bound keeps every seed finite without touching the
    sharpness range the data needs."""
    pad = 0.5 * (cn.max() - cn.min())
    klo, khi = -(cn.max() + pad), -(cn.min() - pad)
    bounds = (_flat(np.zeros(n), np.full(n, -bmax), np.full(n, klo), -np.inf),
              _flat(np.full(n, np.inf), np.full(n, bmax), np.full(n, khi), np.inf))
    seed = _smoothv_seed(cn, H, n)

    best = None
    for scale in (0.4, 1.0):  # graduated sharpness: blunt corners first, then full
        seed = seed.copy()
        seed[1:3 * n:3] *= scale
        res = _fit_softplus(cn, H, seed, bounds, max_nfev=max_nfev, x_scale="jac")
        best = res if best is None or res.cost < best.cost else best
        seed = res.x  # warm-start the sharper stage

    a, b, c, off = _split(best.x, n)
    return a, b, c, float(off)


class _CachedResidual:
    """Residual and Jacobian of one coupled surface solve, sharing a single
    ``_model_and_jac`` pass per point.  ``least_squares`` calls ``fun(p)`` then
    ``jac(p)`` at the same ``p`` each iteration, so the per-term
    ``softplus``/``sigmoid`` (and the ridge toward the seed) are evaluated once.
    The augmented system stacks a tiny ridge ``ridge*(p - seed)`` under the data
    residual to pin flat directions; ``solve`` starts from the seed itself.
    """

    def __init__(self, model_and_jac, cn, vt, f, seed, ridge, eye, solver_kwargs):
        self._model_and_jac = model_and_jac
        self._cn, self._vt, self._f = cn, vt, f
        self._seed, self._ridge, self._eye = seed, ridge, eye
        self._kwargs = solver_kwargs
        self._p = None

    def _refresh(self, p):
        if self._p is None or not np.array_equal(p, self._p):
            model, J = self._model_and_jac(p, self._cn, self._vt)
            self._p = p.copy()
            self._res = np.concatenate([model - self._f, self._ridge * (p - self._seed)])
            self._jac = np.vstack([J, self._eye])

    def fun(self, p):
        self._refresh(p)
        return self._res

    def jac(self, p):
        self._refresh(p)
        return self._jac

    def solve(self, max_nfev):
        return least_squares(self.fun, self._seed, jac=self.jac, max_nfev=max_nfev, **self._kwargs)


class SoftplusFittedSurface(FittedSurface):
    """Fitted surface for :class:`SoftplusSurface2DInterpolator`.

    Holds the per-term coefficient polynomials ``A`` / ``B`` / ``C`` (amplitude
    pre-activation / slope / knee) and the offset polynomial ``O``, plus the T,
    1/T and c normalisations.  The amplitude, knee and offset polynomials are in
    normalised T; the **slope ``B`` is a polynomial in normalised 1/T** (well
    sharpness scales with inverse temperature).  :meth:`slice_at` evaluates them
    at T to produce a :class:`_SoftplusSlice` with an analytic c-derivative; the
    amplitude passes through the softplus link ``a_i(T) = softplus(polyval(Tn,
    A[i]))`` so it is non-negative and the slice is convex in c.
    """

    def __init__(self, A, B, C, O, Tm, Ts, wm, ws, cm, cs):
        self._A, self._B, self._C, self._O = A, B, C, O
        self._Tm, self._Ts, self._wm, self._ws = Tm, Ts, wm, ws
        self._cm, self._cs = cm, cs

    def slice_at(self, T: float) -> _SoftplusSlice:
        Tn = (float(T) - self._Tm) / self._Ts
        Wn = (1.0 / float(T) - self._wm) / self._ws
        pv = np.polynomial.polynomial.polyval
        a = _softplus(np.array([pv(Tn, row) for row in self._A]))  # non-negative amplitude
        b = np.array([pv(Wn, row) for row in self._B])             # slope: polynomial in 1/T
        c = np.array([pv(Tn, row) for row in self._C])
        return _SoftplusSlice(float(pv(Tn, self._O)), a, b, c, self._cm, self._cs)


@dataclass(frozen=True, eq=True)
class SoftplusSurface2DInterpolator(SurfaceInterpolator):
    """Temperature-aware 2-D softplus surface ``f(T, c)``.

    Fits the entropy-removed free energy

        H(T, c) = off(T) + sum_i a_i(T) softplus( b_i(T) (cn + c_i(T)) )

    as a single nonlinear least-squares problem with an analytic Jacobian.  ``c``
    carries the shape; ``T`` only modulates the coefficients.  The amplitude
    ``a_i``, knee ``c_i`` and offset polynomials are in normalised ``T``; the
    **slope ``b_i`` is a polynomial in normalised 1/T** -- the well sharpens as the
    temperature drops, so its inverse-temperature dependence is far better
    approximated by a low-order polynomial in 1/T than in T (a polynomial in T
    cannot track the strong sharpening at the cold end over a wide range).

    **Convexity.** The amplitudes are reparametrised through a non-negative link,
    ``a_i(T) = softplus(alpha_i(T))`` with ``alpha_i(T)`` the fitted polynomial,
    so ``a_i(T) >= 0`` at *every* temperature (training and extrapolated).  Each
    term ``a_i * softplus(b_i*(cn + c_i))`` is then convex in ``c`` (softplus is
    convex, affine argument, non-negative weight), so the fitted ``H(c)`` is
    convex by construction -- no spurious miscibility gaps, matching the guarantee
    the bounded 1-D :class:`SoftplusFit` gives.  A plain unconstrained polynomial
    amplitude could dip negative between knots and break this.

    **Shared knee.**  Set :attr:`shared_knee` to tie every term to one knee
    polynomial ``c(T)`` instead of fitting ``n_softplus`` independent ones --
    the knee still moves with T, it's just the same function for every term.
    This matches a single well built from an opposite-slope softplus pair,
    which is already what the per-slice seed assumes (see :func:`_smoothv_seed`);
    turning it on removes degrees of freedom the seed was not using anyway.

    **Avoiding bad minima.**  The coupled fit is multimodal -- the sharp V-wells
    that motivate softplus sit in a flat-gradient landscape where a single
    optimisation easily stalls with the knee or branch slopes badly placed.  Two
    slices -- the coldest (sharpest, most structure) and the central -- are each
    given a *convex-well-aware* per-slice seed (the knee taken from the data
    minimum, an opposite-slope softplus pair seeding the V; see
    :func:`_fit_slice`) rather than a generic quantile guess.  The two seeds are
    then *raced*: each gets a short coupled polish and only the better-converging
    one is finished (a full solve restarted from that seed), so a bad basin is
    avoided without paying to converge both.  Seeding cost is independent of how
    many temperatures are sampled (hundreds in practice).

    **Solver.**  By default the coupled solve picks its least-squares method by
    scipy version (see :attr:`_LM_HAS_INTERNAL_SCALING`); set :attr:`method` to
    ``'lm'`` or ``'trf'`` to override.  scipy 1.16 switched the default variable
    scaling of ``least_squares(method='lm')`` to MINPACK's internal scaling
    (``mode=1``), matching what plain ``leastsq`` always did -- see gh-22790 and
    the bug it fixed, gh-19459 over on gh/scipy.  Before 1.16, ``lm`` silently disabled that
    scaling (``mode=2`` with an all-ones ``diag``) and converged poorly on the
    badly conditioned sharp-knee Jacobian.  So on scipy >= 1.16 the solve defaults
    to ``lm`` (no per-iteration SVD, faster); on older scipy it defaults to the
    trust-region solver with an explicit ``x_scale='jac'``.  A robust ``loss``
    always uses ``trf`` (``lm`` supports only the linear loss).  The residual and
    Jacobian are evaluated together (:meth:`_model_and_jac`) so the per-term
    ``softplus``/``sigmoid`` are computed once per point.  Most of the speed-up
    is from doing less redundant work -- racing the seeds instead of converging
    both, capping the per-slice seed, and the shared evaluation.

    Unlike :class:`CalphadSurface2DInterpolator` there is no terminal-phase
    requirement, so this works for narrow intermetallics as well as solution
    phases; and unlike :class:`WhitneySurface2DInterpolator` the fixed-T slice
    carries an analytic c-derivative, so the parent phase's logit-Newton solver
    uses exact gradients.
    """

    n_softplus: int = 2
    """Number of softplus terms."""
    a_order: int = 1
    """Polynomial order in T for the amplitudes a_i(T)."""
    b_order: int = 2
    """Polynomial order in 1/T for the slopes b_i(T) (the strongest T-dependence)."""
    c_order: int = 1
    """Polynomial order in T for the knee positions c_i(T)."""
    shared_knee: bool = False
    """When True, every softplus term shares a single knee polynomial c(T)
    instead of fitting one independently per term -- the knee can still vary
    with T, it is just the same function for all ``n_softplus`` terms.  This is
    the constraint the smooth-V seed already assumes for an opposite-slope
    term pair (:func:`_smoothv_seed` seeds identical knees), so turning it on
    only removes degrees of freedom the seed was not using anyway.  Reduces the
    knee parameter count from ``n_softplus * (c_order + 1)`` to ``c_order + 1``."""
    offset_order: int = 2
    """Polynomial order in T for the vertical offset off(T)."""
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "linear"
    """Loss for the surface fit (see :func:`scipy.optimize.least_squares`).  The
    default is ``linear`` -- the well bottom is the signal, not an outlier, so a
    robust loss that down-weights large residuals is counter-productive here.  Set
    a robust loss only for data with genuine outlier points."""
    f_scale: float = 1.0
    """Soft margin between inlier and outlier residuals for a robust ``loss``;
    ignored when ``loss='linear'``."""
    max_nfev: int = 4000
    """Maximum function evaluations per start of the surface fit."""
    method: Literal["lm", "trf"] | None = None
    """Least-squares solver for the coupled fit.  ``None`` (default) picks by
    scipy version: ``'lm'`` on scipy >= 1.16 (faster, no per-iteration SVD) and
    ``'trf'`` on older scipy, where ``'lm'`` lacks variable scaling and converges
    poorly on stiff data (see :attr:`_LM_HAS_INTERNAL_SCALING`).  Force ``'lm'``
    or ``'trf'`` to override the default; ``'lm'`` requires ``loss='linear'`` and,
    on scipy < 1.16, still converges poorly on sharp-knee data."""

    # --- internal solver tuning (not dataclass fields) ---
    _SEED_MAX_NFEV: ClassVar[int] = 300
    """Iteration cap for the per-slice seed fits (:func:`_fit_slice`).  The seed
    only has to start the coupled fit in the right basin, not converge deeply; past
    a few hundred iterations the extra polish does not improve (and on a sharp
    slice can mildly worsen) the final coupled fit, while a hard slice would
    otherwise grind a single seed all the way to ``max_nfev``."""
    _SEED_RACE_NFEV: ClassVar[int] = 80
    """Length of the short coupled polish each seed gets before the better one is
    kept and finished (see :meth:`fit`).  Long enough that the faster-converging
    seed is clearly ahead, short enough that the loser's polish is a small fraction
    of the winner's full solve."""
    _LM_HAS_INTERNAL_SCALING: ClassVar[bool] = _scipy_at_least(1, 16)
    """Whether ``least_squares(method='lm')`` applies MINPACK's internal variable
    scaling.  scipy 1.16 (gh-22790, fixing gh-19459) corrected ``lm`` to default to
    MINPACK's ``mode=1`` scaling, as plain ``leastsq`` always did; before 1.16
    ``lm`` silently disabled it (``mode=2``, all-ones ``diag``) and converged poorly
    on a badly scaled Jacobian -- such as the stiff sharp-knee fit here."""

    def __post_init__(self):
        if self.method == "lm" and self.loss != "linear":
            raise ValueError(
                "method='lm' supports only loss='linear'; use method='trf' for a robust loss"
            )

    @property
    def _orders(self):
        return self.a_order + 1, self.b_order + 1, self.c_order + 1, self.offset_order + 1

    @property
    def _n_params(self):
        na, nb, nc, no = self._orders
        n = self.n_softplus
        if self.shared_knee:
            return n * (na + nb) + nc + no
        return n * (na + nb + nc) + no

    def _unpack(self, p):
        """Split the flat vector into A (n,na), B (n,nb), C (n,nc), O (no,).

        ``A`` holds the amplitude *pre-activation* polynomial coefficients:
        ``a_i(T) = softplus(polyval(Tn, A[i]))``.  ``B`` is in 1/T, the rest in T.
        With :attr:`shared_knee` off, terms are laid out contiguously
        ``[A_i | B_i | C_i]`` then the offset.  With it on, only ``[A_i | B_i]``
        is per-term; one shared knee block follows all terms and is broadcast
        into ``C`` so :meth:`_model_and_jac` and :class:`SoftplusFittedSurface`
        see the usual per-term ``(n, nc)`` shape either way.
        """
        na, nb, nc, no = self._orders
        n = self.n_softplus
        if self.shared_knee:
            per = na + nb
            terms = p[:n * per].reshape(n, per)
            c_shared = p[n * per:n * per + nc]
            C = np.tile(c_shared, (n, 1))
            offset = p[n * per + nc:]
            return terms[:, :na], terms[:, na:na + nb], C, offset
        per = na + nb + nc
        terms = p[:n * per].reshape(n, per)
        return terms[:, :na], terms[:, na:na + nb], terms[:, na + nb:], p[n * per:]

    def _const_init(self, a, b, c, off):
        """Coefficient vector for a constant-in-T surface from one slice's
        ``(a, b, c, offset)`` -- order-0 terms set, higher orders zero.  The
        amplitude is mapped through the inverse link (``alpha0 = softplus_inv(a)``)
        since ``A`` parametrises ``alpha``, not ``a``.  With :attr:`shared_knee`
        the per-slice seed already has identical knees across terms
        (:func:`_smoothv_seed`), so ``c[0]`` is the shared value."""
        na, nb, nc, no = self._orders
        n = self.n_softplus
        offset = np.zeros(no)
        offset[0] = off
        if self.shared_knee:
            terms = np.zeros((n, na + nb))
            terms[:, 0] = _softplus_inv(a)
            terms[:, na] = b
            c_shared = np.zeros(nc)
            c_shared[0] = c[0]
            return np.concatenate([terms.ravel(), c_shared, offset])
        terms = np.zeros((n, na + nb + nc))
        terms[:, 0] = _softplus_inv(a)
        terms[:, na] = b
        terms[:, na + nb] = c
        return np.concatenate([terms.ravel(), offset])

    def _model_and_jac(self, p, cn, vt):
        """Model values and analytic Jacobian ``d(model)/d(params)`` in one pass.

        The coupled fit evaluates the residual and Jacobian at the same point on
        every iteration, so the per-term ``softplus``/``sigmoid`` are computed
        once here and reused for both the value and its derivatives.  ``vt`` is the
        Vandermonde of normalised T for a/c/offset and of normalised 1/T for b;
        the Jacobian columns follow the ``[A_i | B_i | C_i]`` term layout, offset
        last -- or, with :attr:`shared_knee`, ``[A_i | B_i]`` per term, one shared
        knee block, then offset, matching :meth:`_unpack`.  In the shared case
        every term evaluates the same knee polynomial (``C`` is already broadcast
        by :meth:`_unpack`), and by the chain rule the derivative wrt a shared
        coefficient is the *sum* of each term's contribution, so the per-term
        knee blocks are accumulated rather than each appended separately.
        """
        VTa, VWb, VTc, VTo = vt
        A, B, C, O = self._unpack(p)
        out = VTo @ O
        blocks = []
        dC_shared = np.zeros((cn.size, VTc.shape[1])) if self.shared_knee else None
        for Ai, Bi, Ci in zip(A, B, C):
            alpha = VTa @ Ai
            a = _softplus(alpha)        # amplitude link -> convexity; da/dalpha = sigmoid(alpha)
            b = VWb @ Bi                # slope: polynomial in 1/T
            u = cn + VTc @ Ci
            t = b * u
            sig = _sigmoid(t)
            spt = _softplus(t)
            out = out + a * spt
            blocks.append((spt * _sigmoid(alpha))[:, None] * VTa)  # d/dA via the link
            blocks.append((a * sig * u)[:, None] * VWb)
            dC = (a * sig * b)[:, None] * VTc
            if self.shared_knee:
                dC_shared += dC
            else:
                blocks.append(dC)
        if self.shared_knee:
            blocks.append(dC_shared)
        blocks.append(VTo)
        return out, np.hstack(blocks)

    def _solver_kwargs(self) -> dict:
        """``least_squares`` keyword arguments for the coupled solve.

        Uses ``method='lm'`` where scipy applies MINPACK's internal variable
        scaling (:attr:`_LM_HAS_INTERNAL_SCALING`) and the loss is linear,
        otherwise the trust-region solver with ``x_scale='jac'``.  An explicit
        :attr:`method` overrides the version-based choice.
        """
        method = self.method
        if method is None:
            method = "lm" if (self._LM_HAS_INTERNAL_SCALING and self.loss == "linear") else "trf"
        if method == "lm":
            return dict(method="lm")  # __post_init__ guarantees loss == "linear"
        return dict(method="trf", loss=self.loss, f_scale=self.f_scale, x_scale="jac")

    def _vandermonde(self, Tn, Wn):
        """Vandermonde bases: normalised T for amplitude/knee/offset, 1/T for the slope."""
        na, nb, nc, no = self._orders
        return (
            np.vander(Tn, na, increasing=True),
            np.vander(Wn, nb, increasing=True),
            np.vander(Tn, nc, increasing=True),
            np.vander(Tn, no, increasing=True),
        )

    def _seed_starts(self, T, cn, f):
        """Constant-in-T seed vectors from the coldest and central temperature slices.

        Trying both guards against a bad basin: the central slice alone misplaces
        the knee on a strongly T-sharpening well, while the cold slice alone misses
        a second term that only the warm data resolves.  The per-slice fit is
        capped (:attr:`_SEED_MAX_NFEV`), so seeding cost is independent of how many
        temperatures are sampled.
        """
        uT = np.unique(T)
        max_nfev = min(self.max_nfev, self._SEED_MAX_NFEV)
        return [
            self._const_init(*_fit_slice(cn[np.isclose(T, Tseed)], f[np.isclose(T, Tseed)],
                                         self.n_softplus, max_nfev))
            for Tseed in (uT[0], uT[len(uT) // 2])
        ]

    def fit(self, T, c, f) -> SoftplusFittedSurface:
        T = np.asarray(T, float)
        c = np.asarray(c, float)
        f = np.asarray(f, float)
        if np.unique(c).size < 2:
            raise ValueError(
                "SoftplusSurface2DInterpolator requires at least two distinct concentrations"
            )

        Tn, Tm, Ts = _standardize(T)        # amplitude / knee / offset polynomials are in T
        Wn, wm, ws = _standardize(1.0 / T)  # slope polynomial is in 1/T
        cn, cm, cs = _standardize(c)
        vt = self._vandermonde(Tn, Wn)

        ridge = 1e-7 * (float(np.abs(f).max()) or 1.0)
        eye = ridge * np.eye(self._n_params)
        kwargs = self._solver_kwargs()

        def coupled(seed):
            return _CachedResidual(self._model_and_jac, cn, vt, f, seed, ridge, eye, kwargs)

        # Race the two seeds: a short polish each, finish only the better one.  The
        # faster-converging seed is *not* the one with the lower constant-in-T start
        # cost (a sharp well's cold seed starts further off yet relaxes far quicker
        # than the central seed has to sharpen), so the short race is what tells them
        # apart.  Restart the winner from its seed -- not the drifted trial point --
        # so structureless data can't wander down a flat valley to a degenerate
        # huge-amplitude basin; if the short polish already converged it *is* the answer.
        race_nfev = min(self.max_nfev, self._SEED_RACE_NFEV)
        trials = [(seed, coupled(seed).solve(race_nfev)) for seed in self._seed_starts(T, cn, f)]
        seed, trial = min(trials, key=lambda st: st[1].cost)
        res = trial if trial.status > 0 else coupled(seed).solve(self.max_nfev)

        A, B, C, O = self._unpack(res.x)
        return SoftplusFittedSurface(A, B, C, O, Tm, Ts, wm, ws, cm, cs)
