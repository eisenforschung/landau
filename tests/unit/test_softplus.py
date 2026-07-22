import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from landau.interpolate import SoftplusFit
from landau.interpolate.basic import ConcentrationInterpolator, TemperatureInterpolator
from landau.interpolate.softplus import _knee_position, _sigmoid, _smoothv_seed, _softplus, _split


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


class TestSmoothVSeed:
    """Convex-well-aware per-slice seed against a synthetic V-well."""

    def _v_well(self, c0=0.2):
        cn = np.linspace(-1.0, 1.0, 41)
        return cn, np.abs(cn - c0)

    def test_opposite_slope_pair(self):
        cn, H = self._v_well()
        _, b, _, _ = _split(_smoothv_seed(cn, H, 2), 2)
        np.testing.assert_array_equal(b, [12.0, -12.0])

    def test_knee_ties_to_negative_knee_position(self):
        cn, H = self._v_well()
        c0 = _knee_position(cn, H)
        _, _, c, _ = _split(_smoothv_seed(cn, H, 2), 2)
        np.testing.assert_array_equal(c, [-c0, -c0])

    def test_offset_is_median(self):
        cn, H = self._v_well()
        _, _, _, off = _split(_smoothv_seed(cn, H, 2), 2)
        assert off == np.median(H)

    def test_extra_terms_start_inert(self):
        cn, H = self._v_well()
        a, b, _, _ = _split(_smoothv_seed(cn, H, 4), 4)
        np.testing.assert_array_equal(a[2:], [1e-4, 1e-4])
        np.testing.assert_array_equal(b[2:], [12.0, 12.0])
