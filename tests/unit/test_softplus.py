import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from landau.interpolate import SoftplusFit
from landau.interpolate.basic import ConcentrationInterpolator, TemperatureInterpolator
from landau.interpolate.softplus import (
    _flat,
    _sigmoid,
    _softplus,
    _softplus_inv,
    _split,
    _terms,
    _terms_dc,
    _terms_jac,
)


# Maximum allowed deviation between fit and truth, as a fraction of ``ptp(y)``.
# Generous enough to cope with scipy patch-version differences in the
# trust-region step but tight enough to assert a real fit.
RECOVERY_TOL = 0.03


def _truth(x, params):
    """Evaluate a sum of softplus terms plus offset on x directly (no normalization)."""
    n = (len(params) - 1) // 3
    out = np.full_like(x, params[-1], dtype=float)
    for i in range(n):
        a, b, c = params[3 * i : 3 * i + 3]
        out = out + a * _softplus(b * (x + c))
    return out


def _assert_fit_recovers(predict, x, y):
    scale = max(1e-3, float(np.ptp(y)))
    rms = float(np.sqrt(np.mean((predict(x) - y) ** 2)))
    assert rms < RECOVERY_TOL * scale, (
        f"rms={rms:.4f} exceeds {RECOVERY_TOL:.0%} of scale={scale:.4f}"
    )


class TestSoftplusFit:
    def test_is_dual_interpolator(self):
        f = SoftplusFit()
        assert isinstance(f, ConcentrationInterpolator)
        assert isinstance(f, TemperatureInterpolator)

    def test_exported_from_package(self):
        from landau.interpolate import SoftplusFit  # noqa: F401
        import landau.interpolate as pkg
        assert "SoftplusFit" in pkg.__all__

    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    @given(
        a=st.floats(min_value=0.5, max_value=5.0),
        b=st.floats(min_value=1.0, max_value=10.0),
        c=st.floats(min_value=-1.5, max_value=1.5),
        offset=st.floats(min_value=-3.0, max_value=3.0),
    )
    def test_recovery_single_term(self, a, b, c, offset):
        """n_softplus=1 should reproduce data generated from a single softplus term."""
        x = np.linspace(-1.0, 1.0, 201)
        params = [a, b, c, offset]
        y = _truth(x, params)

        fn = SoftplusFit(n_softplus=1, max_nfev=2000).fit(x, y)
        _assert_fit_recovers(fn, x, y)

    # Well-conditioned two-softplus targets that the local fit reliably
    # recovers from its default initial guess.  The 9-D residual surface has
    # occasional local minima the local optimizer cannot escape, so we use a
    # curated parametrize list here rather than a hypothesis sweep.
    @pytest.mark.parametrize("params", [
        [2.0, 2.25, 0.0, 2.0, -2.0, 1.0, 0.0],   # PR #82 historical case
        [1.5, 3.0, -0.5, 1.5, -3.0, 0.5, 0.0],
        [1.0, 5.0, 0.3, 0.5, -4.0, -0.3, 0.2],
        [2.0, 2.0, -1.0, 1.0, -3.0, 0.5, 1.0],
        [0.8, 4.0, 0.0, 1.2, -2.5, 0.8, -0.5],
    ])
    def test_recovery_two_terms(self, params):
        x = np.linspace(-1.0, 1.0, 201)
        y = _truth(x, params)

        fn = SoftplusFit(n_softplus=2, max_nfev=10000).fit(x, y)
        _assert_fit_recovers(fn, x, y)

    def test_f_scale_widens_soft_l1_quadratic_region(self):
        """At sufficiently large ``f_scale`` the soft_l1 loss is quadratic over
        the full residual range, so the resulting fit must coincide with the
        linear-loss fit on the same data.
        """
        params = [1.5, 4.0, -0.3, 1.0, -3.0, 0.5, 0.2]
        x = np.linspace(-1.0, 1.0, 121)
        y = _truth(x, params)

        y_linear = SoftplusFit(n_softplus=2, loss="linear", max_nfev=2000).fit(x, y)(x)
        y_weak = SoftplusFit(
            n_softplus=2, loss="soft_l1", f_scale=1e4, max_nfev=2000
        ).fit(x, y)(x)
        scale = max(1e-3, float(np.ptp(y)))
        np.testing.assert_allclose(y_weak, y_linear, atol=RECOVERY_TOL * scale)

    @pytest.mark.parametrize("x0", [-0.31, 0.0, 0.37])
    def test_recover_relu_kink_location(self, x0):
        """A ReLU-like kink ``f(x) = max(0, x - x0)`` is the b -> oo limit of a
        single softplus term.  The fit should put its inflection point near
        x0 (within ~10% of the input range).
        """
        x = np.linspace(-1.0, 1.0, 201)
        y = np.maximum(0.0, x - x0)

        fn = SoftplusFit(n_softplus=1, max_nfev=2000).fit(x, y)

        x_dense = np.linspace(-1.0, 1.0, 4001)
        y_pred = fn(x_dense)
        # Inflection of the softplus is the point of maximum second derivative.
        d2 = np.diff(y_pred, n=2)
        kink_idx = int(np.argmax(d2)) + 1
        x_kink = x_dense[kink_idx]
        assert abs(x_kink - x0) < 0.1, (
            f"recovered kink {x_kink:.3f} too far from true {x0:.3f}"
        )


class TestSoftplusPrimitive:
    def test_large_positive_linear_asymptote(self):
        t = np.array([100.0, 500.0, 1000.0])
        np.testing.assert_allclose(_softplus(t), t, atol=1e-12)

    def test_large_negative_zero_asymptote(self):
        t = np.array([-100.0, -500.0, -1000.0])
        np.testing.assert_allclose(_softplus(t), 0.0, atol=1e-12)

    def test_at_zero(self):
        assert abs(_softplus(0.0) - np.log(2.0)) < 1e-15

    def test_matches_naive_formula_on_moderate_range(self):
        t = np.linspace(-30.0, 30.0, 601)
        naive = np.log1p(np.exp(t))
        np.testing.assert_allclose(_softplus(t), naive, atol=1e-12)

    def test_sigmoid_bounded(self):
        t = np.linspace(-1000.0, 1000.0, 2001)
        s = _sigmoid(t)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)

    def test_sigmoid_at_zero(self):
        assert abs(_sigmoid(np.array([0.0]))[0] - 0.5) < 1e-15

    def test_sigmoid_is_derivative_of_softplus(self):
        t = np.linspace(-10.0, 10.0, 201)
        h = 1e-5
        fd = (_softplus(t + h) - _softplus(t - h)) / (2 * h)
        np.testing.assert_allclose(_sigmoid(t), fd, atol=1e-6)


# Shared tolerance for finite-difference derivative checks against the analytic
# helpers.  ``h = 1e-6`` gives a central-difference truncation error of order
# ``h**2 = 1e-12`` on smooth softplus; the residual is dominated by float64
# round-off, which sits well under 1e-7 on the moderate-range parameter grids
# used here.
_FD_ATOL = 1e-7
_FD_STEP = 1e-6


class TestFlatSplit:
    """Pack/unpack round-trip between per-term arrays and the flat parameter vector."""

    def test_flat_layout_interleaves_terms_then_offset(self):
        p = _flat([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], 7.0)
        # Layout is [a_0, b_0, c_0, a_1, b_1, c_1, offset].
        np.testing.assert_array_equal(p, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0])

    def test_split_inverts_flat(self):
        a_in, b_in, c_in = [1.5, -2.5], [3.0, 4.5], [-0.3, 0.7]
        off_in = 0.25
        a, b, c, off = _split(_flat(a_in, b_in, c_in, off_in), 2)
        np.testing.assert_array_equal(a, a_in)
        np.testing.assert_array_equal(b, b_in)
        np.testing.assert_array_equal(c, c_in)
        assert off == off_in

    def test_flat_zero_terms_is_offset_only(self):
        p = _flat([], [], [], 3.14)
        np.testing.assert_array_equal(p, [3.14])
        a, b, c, off = _split(p, 0)
        assert a.size == 0 and b.size == 0 and c.size == 0
        assert off == 3.14


class TestTerms:
    """Value and c-derivative of the sum-of-softplus model."""

    def test_n_one_matches_scalar_formula(self):
        cn = np.linspace(-1.0, 1.0, 11)
        a, b, c, off = 1.5, 2.0, 0.3, -0.25
        expected = off + a * _softplus(b * (cn + c))
        np.testing.assert_allclose(_terms(cn, [a], [b], [c], off), expected, atol=1e-14)

    def test_n_two_sums_contributions(self):
        cn = np.linspace(-1.0, 1.0, 11)
        a = [1.5, 0.8]
        b = [2.0, -3.0]
        c = [0.3, -0.1]
        off = 0.5
        expected = off + sum(ai * _softplus(bi * (cn + ci)) for ai, bi, ci in zip(a, b, c))
        np.testing.assert_allclose(_terms(cn, a, b, c, off), expected, atol=1e-14)

    def test_zero_softplus_terms_returns_constant_offset(self):
        cn = np.linspace(-1.0, 1.0, 5)
        np.testing.assert_array_equal(
            _terms(cn, [], [], [], 4.2),
            np.full_like(cn, 4.2),
        )

    def test_scalar_input_produces_scalar_shape(self):
        # ``_terms`` uses ``np.shape(cn)`` for the output shape, so a scalar
        # input must produce a 0-d array (the surface slice call site relies on
        # this via ``_scalarize``).
        out = _terms(0.0, [1.0], [2.0], [0.0], 0.0)
        assert np.ndim(out) == 0

    def test_dc_matches_central_difference(self):
        cn = np.linspace(-1.0, 1.0, 21)
        a, b, c = [1.5, 0.8], [2.0, -3.0], [0.3, -0.1]
        fd = (_terms(cn + _FD_STEP, a, b, c, 0.0) - _terms(cn - _FD_STEP, a, b, c, 0.0)) / (
            2 * _FD_STEP
        )
        np.testing.assert_allclose(_terms_dc(cn, a, b, c), fd, atol=_FD_ATOL)


class TestTermsJac:
    """Analytic parameter Jacobian of the sum-of-softplus model."""

    def _params(self):
        return dict(
            a=np.array([1.5, 0.8]),
            b=np.array([2.0, -3.0]),
            c=np.array([0.3, -0.1]),
            off=0.5,
        )

    def test_shape_is_len_cn_by_three_n_plus_one(self):
        cn = np.linspace(-1.0, 1.0, 7)
        p = self._params()
        J = _terms_jac(cn, p["a"], p["b"], p["c"])
        assert J.shape == (cn.size, 3 * len(p["a"]) + 1)

    def test_offset_column_is_ones(self):
        cn = np.linspace(-1.0, 1.0, 7)
        p = self._params()
        J = _terms_jac(cn, p["a"], p["b"], p["c"])
        np.testing.assert_array_equal(J[:, -1], np.ones(cn.size))

    def test_columns_match_finite_diff_of_terms(self):
        # Central-difference every parameter (``a``, ``b``, ``c`` per term, then
        # the offset) and check the corresponding Jacobian column.  This catches
        # any chain-rule mistake in the analytic construction.
        cn = np.linspace(-1.0, 1.0, 21)
        p = self._params()
        J = _terms_jac(cn, p["a"], p["b"], p["c"])
        flat = _flat(p["a"], p["b"], p["c"], p["off"])
        n = len(p["a"])
        h = _FD_STEP
        for k in range(flat.size):
            plus = flat.copy()
            plus[k] += h
            minus = flat.copy()
            minus[k] -= h
            fd = (_terms(cn, *_split(plus, n)) - _terms(cn, *_split(minus, n))) / (2 * h)
            np.testing.assert_allclose(
                J[:, k], fd, atol=_FD_ATOL, err_msg=f"column {k} mismatch"
            )


class TestSoftplusInv:
    """Inverse of the softplus link used to constrain amplitudes non-negative."""

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(a=st.floats(min_value=1e-3, max_value=5.0))
    def test_round_trip_on_moderate_range(self, a):
        assert _softplus(_softplus_inv(a)) == pytest.approx(a, abs=1e-12)

    def test_finite_for_large_input(self):
        # ``log(expm1(a))`` overflows for ``a`` past ~700; the ``log1p``-based
        # form documented in the ``_softplus_inv`` docstring must stay finite
        # and round-trip to ``a`` (softplus is ~identity there).
        a = 1000.0
        alpha = _softplus_inv(a)
        assert np.isfinite(alpha)
        assert _softplus(alpha) == pytest.approx(a, abs=1e-9)

    def test_clamps_nonpositive_input(self):
        # Softplus is strictly positive so its inverse is undefined at ``a<=0``.
        # The helper clamps to ``1e-12`` rather than returning ``-inf`` / ``nan``,
        # keeping downstream ``_bseed`` seeds finite when the fitted amplitude
        # is numerically zero.
        assert np.isfinite(_softplus_inv(0.0))
        assert np.isfinite(_softplus_inv(-1.0))
        assert _softplus_inv(-1.0) == _softplus_inv(0.0)
