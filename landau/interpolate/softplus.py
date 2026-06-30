from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

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

    def _prepare(self, x, y):
        """Set up normalization, model, Jacobian, bounds and initial guess.

        Returns ``(xn, y, model, jac, p0, lb, ub, xm, xs)``.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        xm = x.mean()
        xs = x.std() if x.std() > 0 else 1.0
        xn = (x - xm) / xs

        n = self.n_softplus

        def model(xn_, p):
            out = np.full_like(xn_, p[-1], dtype=float)
            for i in range(n):
                a = p[3 * i]
                b = p[3 * i + 1]
                c = p[3 * i + 2]
                out = out + a * _softplus(b * (xn_ + c))
            return out

        # Analytical Jacobian of residuals w.r.t. parameters.
        # For one term  a * softplus(b*(xn+c)), with t = b*(xn+c):
        #   d/da = softplus(t)
        #   d/db = a * sigmoid(t) * (xn + c)
        #   d/dc = a * sigmoid(t) * b
        # The vertical offset contributes a column of ones.
        def jac(p):
            J = np.empty((xn.size, 3 * n + 1), dtype=float)
            for i in range(n):
                a = p[3 * i]
                b = p[3 * i + 1]
                c = p[3 * i + 2]
                u = xn + c
                t = b * u
                sig = _sigmoid(t)
                J[:, 3 * i] = _softplus(t)
                J[:, 3 * i + 1] = a * sig * u
                J[:, 3 * i + 2] = a * sig * b
            J[:, -1] = 1.0
            return J

        # Initial guess.  ``a_guess`` is set to the full peak-to-peak range of
        # ``y`` rather than ``ptp/(2*n)``: starting with amplitudes that are
        # large enough to actually reach the data prevents the optimizer from
        # collapsing into the wide near-constant basin that ``soft_l1`` opens
        # up around small ``a`` (see PR #82).  The bound on ``a`` is open
        # above, so over-estimating is harmless — the optimizer trims it.
        ptp = float(np.ptp(y))
        offset = float(np.median(y))
        a_guess = ptp if ptp > 0 else 1.0
        p0 = []
        for i in range(n):
            frac = (i + 0.5) / n
            knee = float(np.quantile(xn, frac))
            b_guess = 3.0 if (i % 2 == 0) else -3.0
            p0 += [a_guess, b_guess, -knee]
        p0 += [offset]
        p0 = np.array(p0, dtype=float)

        lb, ub = [], []
        for _ in range(n):
            lb += [0.0, -20.0, -10.0]
            ub += [np.inf, 20.0, 10.0]
        lb += [-np.inf]
        ub += [np.inf]
        return xn, y, model, jac, p0, lb, ub, xm, xs

    def _least_squares(self, xn, y, model, jac, p0, lb, ub):
        def resid(p):
            return model(xn, p) - y

        return least_squares(
            resid,
            p0,
            jac=jac,
            bounds=(lb, ub),
            loss=self.loss,
            f_scale=self.f_scale,
            max_nfev=self.max_nfev,
        )

    @staticmethod
    def _make_predictor(model, popt, xm, xs):
        def predictor(x_new):
            x_new = np.asarray(x_new, float)
            xn_new = (x_new - xm) / xs
            return model(xn_new, popt)

        return predictor

    def fit(self, x, y) -> Interpolation:
        """Local nonlinear least-squares fit with analytical Jacobian."""
        xn, y, model, jac, p0, lb, ub, xm, xs = self._prepare(x, y)
        res = self._least_squares(xn, y, model, jac, p0, lb, ub)
        return _CallableInterpolation(self._make_predictor(model, res.x, xm, xs))


def _softplus_inv(a):
    """Inverse of softplus: ``alpha`` such that ``softplus(alpha) = a`` (``a > 0``).

    Uses ``alpha = a + log1p(-exp(-a))`` rather than ``log(expm1(a))`` so it stays
    finite for large ``a`` (where ``softplus`` is ~identity) instead of
    overflowing ``expm1``.
    """
    a = np.maximum(np.asarray(a, float), 1e-12)
    return a + np.log1p(-np.exp(-a))


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
        c_arr = np.asarray(c, float)
        cn = (c_arr - self.cm) / self.cs
        out = np.full(c_arr.shape, self.offset, float)
        for ai, bi, ci in zip(self.a, self.b, self.c):
            out = out + ai * _softplus(bi * (cn + ci))
        return _scalarize(out)

    def deriv(self) -> Interpolation:
        # d/dc [ a*softplus(b*(cn + c)) ] = a*sigmoid(b*(cn+c))*b * d(cn)/dc,
        # with d(cn)/dc = 1/cs
        a, b, c, cm, cs = self.a, self.b, self.c, self.cm, self.cs

        def dfdc(cc):
            cc_arr = np.asarray(cc, float)
            cn = (cc_arr - cm) / cs
            out = np.zeros(cc_arr.shape, float)
            for ai, bi, ci in zip(a, b, c):
                out = out + ai * _sigmoid(bi * (cn + ci)) * bi / cs
            return _scalarize(out)

        return _CallableInterpolation(dfdc)


_BMAX = 200.0
"""Magnitude bound on the (normalised) slope during the per-slice seed fit.

A redundant softplus term is unconstrained in a flat direction and would otherwise
drift to infinity; the bound keeps every seed finite without touching the genuine
sharpness range the data needs."""

_SEED_MAX_NFEV = 300
"""Iteration cap for the per-slice seed fits (:func:`_fit_slice`).

The seed only has to start the coupled fit in the right basin, not converge
deeply.  Past a few hundred iterations the extra polish does not improve the
final coupled fit -- and on a sharp slice can mildly worsen it, by handing the
coupled solve an extreme constant-in-T start it then has to spread back out --
while a hard slice would otherwise grind a single seed all the way to
``max_nfev``.  Capping the seed keeps it cheap without affecting (often slightly
improving) the fitted surface."""

_SEED_RACE_NFEV = 80
"""Length of the short coupled polish each seed gets before the better one is
kept and finished (see :meth:`SoftplusSurface2DInterpolator.fit`).  Long enough
that the faster-converging seed is clearly ahead, short enough that the loser's
polish is a small fraction of the winner's full solve."""


def _scipy_at_least(major: int, minor: int) -> bool:
    try:
        return tuple(int(p) for p in scipy.__version__.split(".")[:2]) >= (major, minor)
    except (ValueError, AttributeError):  # unexpected version string -> be conservative
        return False


_LM_HAS_INTERNAL_SCALING = _scipy_at_least(1, 16)
"""Whether ``scipy.optimize.least_squares(method='lm')`` applies MINPACK's
internal variable scaling.

scipy 1.16 (gh-22790, fixing gh-19459) corrected ``lm`` to default to MINPACK's
``mode=1`` internal scaling, as plain ``leastsq`` always did; before 1.16 ``lm``
silently disabled it (``mode=2`` with an all-ones ``diag``) and converged poorly
on a badly scaled Jacobian -- such as the stiff sharp-knee fit here.  When the
fix is present the coupled solve uses ``lm`` (no per-iteration SVD, so faster);
otherwise it uses the trust-region solver with an explicit ``x_scale='jac'``,
which is robust on every supported scipy but does an SVD each step."""


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
    return np.concatenate([np.ravel(np.column_stack([a, b, c])), [float(np.median(H))]])


def _slice_model(p, cn, n):
    a, b, c, off = p[0:3 * n:3], p[1:3 * n:3], p[2:3 * n:3], p[-1]
    out = np.full(cn.shape, off, float)
    for ai, bi, ci in zip(a, b, c):
        out = out + ai * _softplus(bi * (cn + ci))
    return out


def _slice_jac(p, cn, n):
    a, b, c = p[0:3 * n:3], p[1:3 * n:3], p[2:3 * n:3]
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


def _fit_slice(cn, H, n, max_nfev):
    """Robust single-slice fit: data-driven smooth-V seed, then a bounded,
    graduated-sharpness trust-region polish (``x_scale='jac'``).  Returns the
    per-term ``(a, b, c)`` arrays and the offset.  Used only to seed the coupled
    surface fit, so two slices (coldest, central) are enough regardless of how
    many temperatures are sampled."""
    pad = 0.5 * (cn.max() - cn.min())
    klo, khi = -(cn.max() + pad), -(cn.min() - pad)
    lo = np.array([v for _ in range(n) for v in (0.0, -_BMAX, klo)] + [-np.inf])
    hi = np.array([v for _ in range(n) for v in (np.inf, _BMAX, khi)] + [np.inf])
    p0 = np.clip(_smoothv_seed(cn, H, n), lo, hi)
    best = None
    for scale in (0.4, 1.0):  # graduated sharpness: blunt corners first, then full
        p = p0.copy()
        p[1:3 * n:3] = np.clip(p[1:3 * n:3] * scale, lo[1:3 * n:3], hi[1:3 * n:3])
        res = least_squares(lambda q: _slice_model(q, cn, n) - H, p,
                            jac=lambda q: _slice_jac(q, cn, n), bounds=(lo, hi),
                            method="trf", x_scale="jac", max_nfev=max_nfev)
        if best is None or res.cost < best.cost:
            best = res
        p0 = res.x  # warm-start the sharper stage
    s = best.x
    return s[0:3 * n:3], s[1:3 * n:3], s[2:3 * n:3], float(s[-1])


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
        offset = float(pv(Tn, self._O))
        return _SoftplusSlice(offset, a, b, c, self._cm, self._cs)


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
    scipy version (see :data:`_LM_HAS_INTERNAL_SCALING`); set :attr:`method` to
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
    poorly on stiff data (see :data:`_LM_HAS_INTERNAL_SCALING`).  Force ``'lm'``
    or ``'trf'`` to override the default; ``'lm'`` requires ``loss='linear'`` and,
    on scipy < 1.16, still converges poorly on sharp-knee data."""

    def __post_init__(self):
        if self.method == "lm" and self.loss != "linear":
            raise ValueError(
                "method='lm' supports only loss='linear'; use method='trf' for a robust loss"
            )

    @property
    def _orders(self):
        return self.a_order + 1, self.b_order + 1, self.c_order + 1, self.offset_order + 1

    def _unpack(self, p):
        """Split the flat vector into A (n,na), B (n,nb), C (n,nc), O (no,).

        ``A`` holds the amplitude *pre-activation* polynomial coefficients:
        ``a_i(T) = softplus(polyval(Tn, A[i]))``.  ``B`` is in 1/T, the rest in T.
        """
        na, nb, nc, no = self._orders
        n = self.n_softplus
        per = na + nb + nc
        A = np.empty((n, na))
        B = np.empty((n, nb))
        C = np.empty((n, nc))
        for i in range(n):
            base = i * per
            A[i] = p[base:base + na]
            B[i] = p[base + na:base + na + nb]
            C[i] = p[base + na + nb:base + per]
        O = p[n * per:]
        return A, B, C, O

    def _model_and_jac(self, p, cn, vt):
        """Model values and analytic Jacobian ``d(model)/d(params)`` in one pass.

        The coupled fit evaluates the residual and Jacobian at the same point on
        every iteration, so the per-term ``softplus``/``sigmoid`` are computed
        once here and reused for both the value and its derivatives rather than
        twice by an evaluation that returns only one of them.  ``vt`` carries the
        Vandermonde of normalised T for a/c/offset and of normalised 1/T for b.
        """
        na, nb, nc, no = self._orders
        n = self.n_softplus
        VTa, VWb, VTc, VTo = vt
        A, B, C, O = self._unpack(p)
        per = na + nb + nc
        out = VTo @ O
        J = np.empty((cn.size, n * per + no))
        for i in range(n):
            alpha = VTa @ A[i]
            a = _softplus(alpha)        # amplitude link -> convexity; da/dalpha = sigmoid(alpha)
            sig_a = _sigmoid(alpha)
            b = VWb @ B[i]              # slope: polynomial in 1/T
            cc = VTc @ C[i]
            u = cn + cc
            t = b * u
            spt = _softplus(t)
            sig = _sigmoid(t)
            out = out + a * spt
            base = i * per
            # d/dA = d/da * da/dalpha * dalpha/dA = softplus(t) * sigmoid(alpha) * Tn^k
            J[:, base:base + na] = (spt * sig_a)[:, None] * VTa
            J[:, base + na:base + na + nb] = (a * sig * u)[:, None] * VWb
            J[:, base + na + nb:base + per] = (a * sig * b)[:, None] * VTc
        J[:, n * per:] = VTo
        return out, J

    def _const_init(self, a, b, c, off):
        """Coefficient vector for a constant-in-T surface from one slice's
        ``(a, b, c, offset)`` -- order-0 terms set, higher orders zero.  The
        amplitude is mapped through the inverse link (``alpha0 = softplus_inv(a)``)
        since ``A`` parametrises ``alpha``, not ``a``."""
        na, nb, nc, no = self._orders
        n = self.n_softplus
        per = na + nb + nc
        p = np.zeros(n * per + no)
        for i in range(n):
            base = i * per
            p[base] = _softplus_inv(a[i])
            p[base + na] = b[i]
            p[base + na + nb] = c[i]
        p[n * per] = off
        return p

    def _solver_kwargs(self) -> dict:
        """``least_squares`` keyword arguments for the coupled solve.

        Uses ``method='lm'`` where scipy applies MINPACK's internal variable
        scaling (:data:`_LM_HAS_INTERNAL_SCALING`) and the loss is linear,
        otherwise the trust-region solver with ``x_scale='jac'``.  An explicit
        :attr:`method` overrides the version-based choice.
        """
        method = self.method
        if method is None:
            method = "lm" if (_LM_HAS_INTERNAL_SCALING and self.loss == "linear") else "trf"
        if method == "lm":
            return dict(method="lm")  # __post_init__ guarantees loss == "linear"
        return dict(method="trf", loss=self.loss, f_scale=self.f_scale, x_scale="jac")

    def fit(self, T, c, f) -> SoftplusFittedSurface:
        T = np.asarray(T, float)
        c = np.asarray(c, float)
        f = np.asarray(f, float)
        if np.unique(c).size < 2:
            raise ValueError(
                "SoftplusSurface2DInterpolator requires at least two distinct concentrations"
            )

        Tm, Ts = float(T.mean()), float(T.std() or 1.0)
        cm, cs = float(c.mean()), float(c.std() or 1.0)
        w = 1.0 / T
        wm, ws = float(w.mean()), float(w.std() or 1.0)
        Tn = (T - Tm) / Ts
        Wn = (w - wm) / ws
        cn = (c - cm) / cs
        na, nb, nc, no = self._orders
        vt = (
            np.vander(Tn, na, increasing=True),
            np.vander(Wn, nb, increasing=True),   # slope basis is 1/T
            np.vander(Tn, nc, increasing=True),
            np.vander(Tn, no, increasing=True),
        )

        # Seed from the coldest (sharpest) and central slices.  Trying both guards
        # against a bad basin (the central slice alone misplaces the knee on a
        # strongly T-sharpening well; the cold slice alone misses a second term
        # that only the warm data resolves).  Rather than run both coupled fits to
        # completion and keep the cheaper -- the worse seed reaches the same (or a
        # higher-cost) minimum, so finishing it was wasted work -- race them: give
        # each a short polish, keep whichever is converging better, and only finish
        # that one.  Which seed converges fastest is *not* the one with the lower
        # cost at the constant-in-T start (a sharp well's cold seed starts further
        # off yet relaxes far quicker than the central seed has to sharpen), so the
        # short race is what tells them apart.
        #
        # Solver: on scipy >= 1.16 the unbounded solve uses Levenberg-Marquardt
        # (``method='lm'``), whose MINPACK step has no per-iteration SVD and --
        # since gh-22790 -- applies internal variable scaling, so it copes with
        # the badly conditioned sharp-knee Jacobian and is faster than the SVD
        # trust region.  On older scipy ``lm`` lacks that scaling and converges
        # poorly here, so the solve falls back to the trust-region solver with an
        # explicit ``x_scale='jac'`` (see ``_LM_HAS_INTERNAL_SCALING``).  ``lm``
        # only supports the linear loss, so a robust loss always takes ``trf``.
        # Either way a tiny ridge toward the seed pins flat directions; the weight
        # is negligible against the data residual, so it does not bias an active
        # term.
        uT = np.unique(T)
        n = self.n_softplus
        ridge = 1e-7 * (float(np.abs(f).max()) or 1.0)
        eye = ridge * np.eye(self.n_softplus * (na + nb + nc) + no)
        seed_nfev = min(self.max_nfev, _SEED_MAX_NFEV)
        seeds = []
        for Tseed in (uT[0], uT[len(uT) // 2]):
            m = np.isclose(T, Tseed)
            a0, b0, c0, off0 = _fit_slice(cn[m], f[m], n, seed_nfev)
            seeds.append(self._const_init(a0, b0, c0, off0))

        def coupled(seed, x0, max_nfev):
            # Residual and Jacobian share one ``_model_and_jac`` pass per point via
            # a one-slot cache: ``least_squares`` calls ``fun(p)`` then ``jac(p)``
            # at the same ``p`` each iteration, so the transcendentals (and the
            # ridge toward ``seed``) are evaluated once.
            cache = {}

            def evaluate(p):
                if cache.get("p") is None or not np.array_equal(p, cache["p"]):
                    model, J = self._model_and_jac(p, cn, vt)
                    cache["p"] = p.copy()
                    cache["res"] = np.concatenate([model - f, ridge * (p - seed)])
                    cache["jac"] = np.vstack([J, eye])
                return cache

            return least_squares(
                lambda p: evaluate(p)["res"], x0, jac=lambda p: evaluate(p)["jac"],
                max_nfev=max_nfev, **self._solver_kwargs(),
            )

        race_nfev = min(self.max_nfev, _SEED_RACE_NFEV)
        trials = [(seed, coupled(seed, seed, race_nfev)) for seed in seeds]
        seed, trial = min(trials, key=lambda st: st[1].cost)
        # Finish the winning seed with a full solve *from its constant-in-T start*,
        # not from the drifted trial point: restarting from the seed re-treads the
        # same bounded trajectory a single-seed fit takes, so structureless data
        # (pure noise) can't wander down a flat valley to a degenerate
        # huge-amplitude basin the way continuing from the trial point can.  If the
        # short polish already converged (an easy in-family surface needs only a
        # few dozen steps) the trial *is* the answer.
        res = trial if trial.status > 0 else coupled(seed, seed, self.max_nfev)

        A, B, C, O = self._unpack(res.x)
        return SoftplusFittedSurface(A, B, C, O, Tm, Ts, wm, ws, cm, cs)
