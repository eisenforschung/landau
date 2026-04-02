import numpy as np
import pytest

from landau.interpolate.whitney import WhitneyRBFInterpolator, WhitneyTemperatureInterpolator


class TestWhitneyRBFInterpolator:
    def test_1d_interior_exact(self):
        """Interior points should be reproduced exactly with smoothing=0."""
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.sin(2 * np.pi * X[:, 0])
        model = WhitneyRBFInterpolator(smoothing=0.0).fit(X, y)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-6)

    def test_2d_interior_exact(self):
        """Interior points in 2D should be reproduced exactly with smoothing=0."""
        rng = np.random.default_rng(42)
        X = rng.uniform(-1, 1, (50, 2))
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        model = WhitneyRBFInterpolator(smoothing=0.0).fit(X, y)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-5)

    def test_1d_exterior_finite(self):
        """Exterior points in 1D should produce finite values."""
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = X[:, 0] ** 2
        model = WhitneyRBFInterpolator(smoothing=0.0).fit(X, y)
        x_ext = np.array([[2.0], [-1.0]])
        pred = model.predict(x_ext)
        assert np.all(np.isfinite(pred))

    def test_2d_exterior_finite(self):
        """Exterior points in 2D should produce finite values."""
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (40, 2))
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        model = WhitneyRBFInterpolator(smoothing=0.0).fit(X, y)
        x_ext = np.array([[5.0, 5.0], [-5.0, -5.0]])
        pred = model.predict(x_ext)
        assert np.all(np.isfinite(pred))

    def test_1d_c0_continuity(self):
        """Values just inside and just outside the 1D boundary should match closely."""
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        y = np.sin(2 * np.pi * X[:, 0])
        model = WhitneyRBFInterpolator(smoothing=0.0).fit(X, y)
        eps = 1e-5
        x_in = np.array([[1.0 - eps]])
        x_out = np.array([[1.0 + eps]])
        assert abs(model.predict(x_in)[0] - model.predict(x_out)[0]) < 1e-3

    @pytest.mark.parametrize("kernel", ["quintic", "gaussian", "multiquadric"])
    def test_1d_c1_continuity(self, kernel):
        """C1: derivative should be continuous at the 1D boundary for smooth kernels."""
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        y = X[:, 0] ** 2
        model = WhitneyRBFInterpolator(smoothing=0.0, kernel=kernel).fit(X, y)
        eps = 1e-5
        # One-sided derivatives at x=1 (right boundary)
        d_in = (model.predict([[1.0]]) - model.predict([[1.0 - eps]]))[0] / eps
        d_out = (model.predict([[1.0 + eps]]) - model.predict([[1.0]]))[0] / eps
        np.testing.assert_allclose(d_in, d_out, atol=0.01)

    @pytest.mark.parametrize("kernel", ["quintic", "gaussian", "multiquadric"])
    def test_2d_c1_continuity(self, kernel):
        """C1: gradient should be continuous at the hull boundary for smooth kernels."""
        xs = np.linspace(-1, 1, 7)
        X = np.array([[x, z] for x in xs for z in xs])
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        model = WhitneyRBFInterpolator(smoothing=0.0, kernel=kernel).fit(X, y)
        eps = 1e-4
        # At (1.0, 0.0): x=1 is the right edge of the hull; outward normal is (1, 0)
        d_in = (model.predict([[1.0, 0.0]]) - model.predict([[1.0 - eps, 0.0]]))[0] / eps
        d_out = (model.predict([[1.0 + eps, 0.0]]) - model.predict([[1.0, 0.0]]))[0] / eps
        np.testing.assert_allclose(d_in, d_out, atol=0.1)

    @pytest.mark.parametrize("degree", [2, 3, 4])
    def test_1d_c1_continuity_degree(self, degree):
        """C1: derivative should be continuous at the 1D boundary for degree >= 2."""
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        y = X[:, 0] ** 2
        model = WhitneyRBFInterpolator(smoothing=0.0, degree=degree).fit(X, y)
        eps = 1e-5
        # One-sided derivatives at x=1 (right boundary)
        d_in = (model.predict([[1.0]]) - model.predict([[1.0 - eps]]))[0] / eps
        d_out = (model.predict([[1.0 + eps]]) - model.predict([[1.0]]))[0] / eps
        np.testing.assert_allclose(d_in, d_out, atol=0.01)

    @pytest.mark.parametrize("degree", [2, 3, 4])
    def test_2d_c1_continuity_degree(self, degree):
        """C1: gradient should be continuous at the hull boundary for degree >= 2."""
        xs = np.linspace(-1, 1, 7)
        X = np.array([[x, z] for x in xs for z in xs])
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        model = WhitneyRBFInterpolator(smoothing=0.0, degree=degree).fit(X, y)
        eps = 1e-4
        # At (1.0, 0.0): x=1 is the right edge of the hull; outward normal is (1, 0)
        d_in = (model.predict([[1.0, 0.0]]) - model.predict([[1.0 - eps, 0.0]]))[0] / eps
        d_out = (model.predict([[1.0 + eps, 0.0]]) - model.predict([[1.0, 0.0]]))[0] / eps
        np.testing.assert_allclose(d_in, d_out, atol=0.1)

    def test_feature_mismatch_raises(self):
        """predict() should raise when feature count doesn't match fit."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = X[:, 0]
        model = WhitneyRBFInterpolator().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.ones((3, 2)))

    def test_input_must_be_2d(self):
        """predict() should raise when input is not 2D."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = X[:, 0]
        model = WhitneyRBFInterpolator().fit(X, y)
        with pytest.raises(ValueError, match="2D"):
            model.predict(np.array([0.5, 1.5]))

    def test_sklearn_r2_score(self):
        """score() should return a high R² on training data."""
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        y = np.sin(2 * np.pi * X[:, 0])
        model = WhitneyRBFInterpolator(smoothing=0.0).fit(X, y)
        assert model.score(X, y) > 0.99

    def test_not_fitted_raises(self):
        """predict() before fit() should raise NotFittedError."""
        from sklearn.exceptions import NotFittedError
        model = WhitneyRBFInterpolator()
        with pytest.raises(NotFittedError):
            model.predict(np.array([[0.5]]))

    def test_fit_returns_self(self):
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = X[:, 0]
        model = WhitneyRBFInterpolator()
        assert model.fit(X, y) is model


class TestWhitneyTemperatureInterpolator:
    def _make_data(self, n=25):
        T = np.linspace(300, 1000, n)
        y = -T * np.log(T) + 0.01 * T
        return T, y

    def test_fit_returns_callable(self):
        T, y = self._make_data()
        fn = WhitneyTemperatureInterpolator().fit(T, y)
        assert callable(fn)

    def test_interpolation_interior(self):
        """Predictions at training temperatures should be close to targets."""
        T, y = self._make_data()
        fn = WhitneyTemperatureInterpolator(smoothing=0.0).fit(T, y)
        pred = fn(T)
        np.testing.assert_allclose(pred, y, atol=1e-4)

    def test_extrapolation_finite(self):
        """Predictions outside training range should be finite."""
        T, y = self._make_data()
        fn = WhitneyTemperatureInterpolator().fit(T, y)
        t_ext = np.array([100.0, 2000.0])
        pred = fn(t_ext)
        assert np.all(np.isfinite(pred))

    def test_is_temperature_interpolator(self):
        """WhitneyTemperatureInterpolator should be a TemperatureInterpolator."""
        from landau.interpolate.basic import TemperatureInterpolator
        assert isinstance(WhitneyTemperatureInterpolator(), TemperatureInterpolator)

    def test_exported_from_package(self):
        """WhitneyTemperatureInterpolator is importable from landau.interpolate; WhitneyRBFInterpolator is not part of the public API."""
        from landau.interpolate import WhitneyTemperatureInterpolator  # noqa: F401
        import landau.interpolate as pkg
        assert "WhitneyRBFInterpolator" not in pkg.__all__
