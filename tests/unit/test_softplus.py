import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from landau.interpolate import SoftplusFit
from landau.interpolate.basic import ConcentrationInterpolator, TemperatureInterpolator


def _softplus(t):
    t = np.asarray(t, float)
    return np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0.0)


def _truth(x, params):
    """Evaluate a sum of softplus terms plus offset on x directly (no normalization)."""
    n = (len(params) - 1) // 3
    out = np.full_like(x, params[-1], dtype=float)
    for i in range(n):
        a, b, c = params[3 * i : 3 * i + 3]
        out = out + a * _softplus(b * (x + c))
    return out


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

        scale = max(1e-3, float(np.ptp(y)))
        np.testing.assert_allclose(fn(x), y, atol=0.02 * scale)

    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    )
    @given(
        a1=st.floats(min_value=0.5, max_value=3.0),
        b1=st.floats(min_value=2.0, max_value=8.0),
        c1=st.floats(min_value=-1.5, max_value=1.5),
        a2=st.floats(min_value=0.5, max_value=3.0),
        b2=st.floats(min_value=-8.0, max_value=-2.0),
        c2=st.floats(min_value=-1.5, max_value=1.5),
        offset=st.floats(min_value=-2.0, max_value=2.0),
    )
    def test_recovery_random_coeffs(self, a1, b1, c1, a2, b2, c2, offset):
        """The fitter should restore a random sum-of-softplus function (in
        function-value sense, i.e. modulo permutation of the terms) given
        dense, noise-free samples and matching ``n_softplus``.

        Uses ``loss="linear"`` — the default ``soft_l1`` reduces residual
        weight on the steep parts of the curve and frequently gets stuck in
        a near-constant local minimum on noise-free data (see PR #82).
        """
        params = [a1, b1, c1, a2, b2, c2, offset]
        x = np.linspace(-1.0, 1.0, 201)
        y = _truth(x, params)

        fn = SoftplusFit(n_softplus=2, loss="linear", max_nfev=2000).fit(x, y)

        x_dense = np.linspace(-1.0, 1.0, 401)
        y_pred = fn(x_dense)
        y_true_dense = _truth(x_dense, params)
        scale = max(1e-3, float(np.ptp(y_true_dense)))
        np.testing.assert_allclose(y_pred, y_true_dense, atol=0.02 * scale)

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
