from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
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


class SoftplusFittedSurface(FittedSurface):
    """Fitted surface for :class:`SoftplusSurface2DInterpolator`.

    Holds the per-term coefficient polynomials ``A`` / ``B`` / ``C`` (amplitude
    pre-activation / slope / knee, each a row of ascending-power coefficients in
    normalised T) and the offset polynomial ``O``, plus the T and c
    normalisations.  :meth:`slice_at` evaluates them at T to produce a
    :class:`_SoftplusSlice` with an analytic c-derivative; the amplitude passes
    through the softplus link ``a_i(T) = softplus(polyval(Tn, A[i]))`` so it is
    non-negative and the slice is convex in c.
    """

    def __init__(self, A, B, C, O, Tm, Ts, cm, cs):
        self._A, self._B, self._C, self._O = A, B, C, O
        self._Tm, self._Ts, self._cm, self._cs = Tm, Ts, cm, cs

    def slice_at(self, T: float) -> _SoftplusSlice:
        Tn = (float(T) - self._Tm) / self._Ts
        pv = np.polynomial.polynomial.polyval
        a = _softplus(np.array([pv(Tn, row) for row in self._A]))  # non-negative amplitude
        b = np.array([pv(Tn, row) for row in self._B])
        c = np.array([pv(Tn, row) for row in self._C])
        offset = float(pv(Tn, self._O))
        return _SoftplusSlice(offset, a, b, c, self._cm, self._cs)


@dataclass(frozen=True, eq=True)
class SoftplusSurface2DInterpolator(SurfaceInterpolator):
    """Temperature-aware 2-D softplus surface ``f(T, c)``.

    Fits the entropy-removed free energy

        H(T, c) = off(T) + sum_i a_i(T) softplus( b_i(T) (cn + c_i(T)) )

    as a single nonlinear least-squares problem with an analytic Jacobian, where
    each softplus coefficient is a polynomial in normalised T.  ``c`` carries the
    shape; ``T`` only modulates the coefficients, with the slope ``b`` expected to
    move the most (hence ``b_order`` defaults higher than ``a_order`` /
    ``c_order``).

    **Convexity.** The amplitudes are reparametrised through a non-negative link,
    ``a_i(T) = softplus(alpha_i(T))`` with ``alpha_i(T)`` the fitted polynomial,
    so ``a_i(T) >= 0`` at *every* temperature (training and extrapolated).  Each
    term ``a_i * softplus(b_i*(cn + c_i))`` is then convex in ``c`` (softplus is
    convex, affine argument, non-negative weight), so the fitted ``H(c)`` is
    convex by construction -- no spurious miscibility gaps, matching the guarantee
    the bounded 1-D :class:`SoftplusFit` gives.  A plain unconstrained polynomial
    amplitude could dip negative between knots and break this.

    The fit is multi-started from both the *fitted* and the *large-amplitude
    heuristic* 1-D :class:`SoftplusFit` on the central-T slice (the latter rescues
    slices where the per-T fit collapses into the flat near-zero-amplitude basin),
    keeping the lower-cost result.

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
    """Polynomial order in T for the slopes b_i(T) (the strongest T-dependence)."""
    c_order: int = 1
    """Polynomial order in T for the knee positions c_i(T)."""
    offset_order: int = 2
    """Polynomial order in T for the vertical offset off(T)."""
    loss: Literal["linear", "soft_l1", "huber", "cauchy", "arctan"] = "soft_l1"
    """Robust loss for the surface fit (see :func:`scipy.optimize.least_squares`)."""
    f_scale: float = 1.0
    """Soft margin between inlier and outlier residuals for the robust loss."""
    max_nfev: int = 4000
    """Maximum function evaluations per start of the surface fit."""

    @property
    def _orders(self):
        return self.a_order + 1, self.b_order + 1, self.c_order + 1, self.offset_order + 1

    def _unpack(self, p):
        """Split the flat vector into A (n,na), B (n,nb), C (n,nc), O (no,).

        ``A`` holds the amplitude *pre-activation* polynomial coefficients:
        ``a_i(T) = softplus(polyval(Tn, A[i]))``.
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

    def _model(self, p, cn, vt):
        """Model values at the normalised points (no Jacobian)."""
        VTa, VTb, VTc, VTo = vt
        A, B, C, O = self._unpack(p)
        out = VTo @ O
        for i in range(self.n_softplus):
            a = _softplus(VTa @ A[i])   # non-negative amplitude link -> convexity
            b = VTb @ B[i]
            cc = VTc @ C[i]
            out = out + a * _softplus(b * (cn + cc))
        return out

    def _jac(self, p, cn, vt):
        """Analytic Jacobian d(model)/d(params) at the normalised points."""
        na, nb, nc, no = self._orders
        n = self.n_softplus
        VTa, VTb, VTc, VTo = vt
        A, B, C, O = self._unpack(p)
        per = na + nb + nc
        J = np.empty((cn.size, n * per + no))
        for i in range(n):
            alpha = VTa @ A[i]
            a = _softplus(alpha)        # amplitude; da/dalpha = sigmoid(alpha)
            sig_a = _sigmoid(alpha)
            b = VTb @ B[i]
            cc = VTc @ C[i]
            u = cn + cc
            t = b * u
            sig = _sigmoid(t)
            base = i * per
            # d/dA = d/da * da/dalpha * dalpha/dA = softplus(t) * sigmoid(alpha) * Tn^k
            J[:, base:base + na] = (_softplus(t) * sig_a)[:, None] * VTa
            J[:, base + na:base + na + nb] = (a * sig * u)[:, None] * VTb
            J[:, base + na + nb:base + per] = (a * sig * b)[:, None] * VTc
        J[:, n * per:] = VTo
        return J

    def _pack_order0(self, slice_params, xm, xs, cm, cs):
        """Pack a 1-D softplus vector ``[a,b,coff]*n + [offset]`` (slice
        normalisation ``xm, xs``) into a 2-D coefficient vector: order-0 terms are
        those values transformed to the surface normalisation ``cm, cs``, higher
        orders zero.  The amplitude is mapped through the inverse link
        (``alpha0 = softplus_inv(a)``) since ``A`` parametrises ``alpha``, not
        ``a``; ``b' = b*cs/xs``, ``c' = (xs*coff + cm - xm)/cs``.
        """
        na, nb, nc, no = self._orders
        n = self.n_softplus
        per = na + nb + nc
        p = np.zeros(n * per + no)
        for i in range(n):
            a, b, coff = slice_params[3 * i], slice_params[3 * i + 1], slice_params[3 * i + 2]
            base = i * per
            p[base] = _softplus_inv(a)
            p[base + na] = b * cs / xs
            p[base + na + nb] = (xs * coff + cm - xm) / cs
        p[n * per] = slice_params[-1]
        return p

    def _candidate_starts(self, Tn, cn, f, cm, cs):
        """Both 1-D softplus starts on the central-T slice: the large-amplitude
        heuristic (avoids the flat near-zero-a basin) and the fitted per-T
        solution (fast when good)."""
        n = self.n_softplus
        mask = np.isclose(Tn, Tn[np.argmin(np.abs(Tn))])
        c_slice = cn[mask] * cs + cm
        f_slice = f[mask]
        sf = SoftplusFit(n_softplus=n, loss=self.loss, f_scale=self.f_scale, max_nfev=self.max_nfev)
        xn, yv, model, jac, p0, lb, ub, xm, xs = sf._prepare(c_slice, f_slice)
        res = sf._least_squares(xn, yv, model, jac, p0, lb, ub)
        return [
            self._pack_order0(p0, xm, xs, cm, cs),     # heuristic (large a)
            self._pack_order0(res.x, xm, xs, cm, cs),  # fitted per-T
        ]

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
        Tn = (T - Tm) / Ts
        cn = (c - cm) / cs
        na, nb, nc, no = self._orders
        vt = (
            np.vander(Tn, na, increasing=True),
            np.vander(Tn, nb, increasing=True),
            np.vander(Tn, nc, increasing=True),
            np.vander(Tn, no, increasing=True),
        )

        # multi-start; keep the lower-cost fit
        best = None
        for p0 in self._candidate_starts(Tn, cn, f, cm, cs):
            res = least_squares(
                lambda p: self._model(p, cn, vt) - f, p0,
                jac=lambda p: self._jac(p, cn, vt),
                loss=self.loss, f_scale=self.f_scale, max_nfev=self.max_nfev,
            )
            if best is None or res.cost < best.cost:
                best = res

        A, B, C, O = self._unpack(best.x)
        return SoftplusFittedSurface(A, B, C, O, Tm, Ts, cm, cs)
