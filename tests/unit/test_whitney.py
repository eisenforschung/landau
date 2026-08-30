import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from landau.interpolate.whitney import WhitneyRBFInterpolator, WhitneyTemperatureInterpolator


@pytest.fixture(scope="module")
def temperature_fit():
    """One Whitney fit shared by the shape-contract property test.

    ``@given`` re-runs its body ~100 times and fitting is the slow part; the
    interpolation is immutable, so one fit serves every draw.  Module scope also
    keeps hypothesis' function-scoped-fixture health check quiet.
    """
    T = np.linspace(300, 1000, 25)
    return WhitneyTemperatureInterpolator(smoothing=0.0).fit(T, -T * np.log(T) + 0.01 * T)


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
        """``smoothing=0`` interpolates, so the fit reproduces its own training data
        to machine precision (measured 1e-16 relative).

        That is why a residual against the training data cannot gate fit quality
        the way it can for ``PolyFit`` — it is ~0 by construction (#428).
        """
        T, y = self._make_data()
        fn = WhitneyTemperatureInterpolator(smoothing=0.0).fit(T, y)
        rms = np.sqrt(np.mean((fn(T) - y) ** 2))
        assert rms < 1e-13 * np.abs(y).max()

    def test_scalar_input_returns_python_scalar(self):
        """Scalar T in, Python scalar out (#428): ``atleast_1d`` used to leak a shape-(1,) array."""
        T, y = self._make_data()
        fn = WhitneyTemperatureInterpolator(smoothing=0.0).fit(T, y)
        pred = fn(500.0)
        assert isinstance(pred, float)
        np.testing.assert_allclose(pred, fn(np.array([500.0]))[0], rtol=1e-12, atol=0)

    @given(
        t=arrays(
            dtype=float,
            shape=array_shapes(min_dims=0, max_dims=3, min_side=1, max_side=4),
            elements=st.floats(min_value=50.0, max_value=2000.0),
        )
    )
    def test_output_shape_follows_input_shape(self, temperature_fit, t):
        """The closure preserves any input shape, 0-d included, in and outside the data."""
        out = temperature_fit(t)

        assert np.shape(out) == t.shape
        if t.ndim == 0:
            assert isinstance(out, float)
        else:
            # ravelling the input ravels the output identically: the values are
            # laid out by position, not merely counted
            np.testing.assert_array_equal(np.ravel(out), temperature_fit(t.ravel()))
    @staticmethod
    def _einstein(T, theta=350.0, n=3, E0=-3.0):
        """Einstein-oscillator free energy per atom: concave in T, S -> 0 as T -> 0."""
        from scipy.constants import Boltzmann, eV

        kB = Boltzmann / eV
        T = np.asarray(T, float)
        return E0 + 0.5 * n * kB * theta + n * kB * T * np.log1p(-np.exp(-theta / T))

    def test_extension_freezes_the_entropy_at_each_boundary(self):
        """Outside the data the extension is exactly linear, so ``S = -f'(T_b)`` is
        frozen at the *nearest* boundary — a different slope below the data than above.

        Subsumes a plain finiteness check on the extrapolated range: a non-finite
        prediction makes the slope spread ``nan`` and fails the linearity assertion.
        """
        T = np.linspace(100.0, 600.0, 60)
        fn = WhitneyTemperatureInterpolator(smoothing=0.0).fit(T, self._einstein(T))

        below = np.linspace(10.0, 90.0, 20)
        above = np.linspace(650.0, 2000.0, 28)
        slope_below = np.diff(fn(below)) / np.diff(below)
        slope_above = np.diff(fn(above)) / np.diff(above)

        # a curved extension, or one that reverted to the RBF, spreads these
        assert np.ptp(slope_below) < 1e-12
        assert np.ptp(slope_above) < 1e-12
        # each side freezes its own boundary slope: S(100 K) is far below S(600 K)
        # for a third-law free energy, so one global slope would fail this
        assert abs(slope_below.mean()) < 0.2 * abs(slope_above.mean())

        # against the exact dF/dT(T_b). The RBF's edge error is 0.9% at the upper
        # boundary; it is larger at the lower one, where the data curves hardest
        T_b = T.max()
        true_slope = (self._einstein(T_b + 1e-3) - self._einstein(T_b - 1e-3)) / 2e-3
        np.testing.assert_allclose(slope_above.mean(), true_slope, rtol=2e-2)

    def test_extension_bounds_a_concave_free_energy_from_above(self):
        """F(T) is concave, so the tangent extension over-estimates it (#428 docs note)."""
        T = np.linspace(100.0, 600.0, 60)
        F = self._einstein(T)
        whitney = WhitneyTemperatureInterpolator(smoothing=0.0).fit(T, F)

        T_ext = np.linspace(650.0, 2000.0, 28)
        F_true = self._einstein(T_ext)

        assert np.all(whitney(T_ext) >= F_true)
        # the gap is the entropy the real system keeps gaining, and it grows
        assert np.all(np.diff(whitney(T_ext) - F_true) > 0)

    def test_is_temperature_interpolator(self):
        """WhitneyTemperatureInterpolator should be a TemperatureInterpolator."""
        from landau.interpolate.basic import TemperatureInterpolator
        assert isinstance(WhitneyTemperatureInterpolator(), TemperatureInterpolator)

    def test_exported_from_package(self):
        """WhitneyTemperatureInterpolator is importable from landau.interpolate; WhitneyRBFInterpolator is not part of the public API."""
        from landau.interpolate import WhitneyTemperatureInterpolator  # noqa: F401
        import landau.interpolate as pkg
        assert "WhitneyRBFInterpolator" not in pkg.__all__


class TestWhitneySurface2DInterpolator:
    """The 2-D surface interpolator backed by a Whitney-extended RBF.

    Unlike ``CalphadSurface2DInterpolator`` there is no terminal-phase
    requirement, so it works for narrow intermetallic windows as well as full
    solution phases; the trade-off is a numerical (rather than analytic)
    c-derivative on each slice.
    """

    RECOVER_ATOL = 1e-6   # training-point recovery with smoothing=0 (measured ~1e-11)
    GIBBS_ATOL = 1e-4     # c = -dphi/ddmu via the numerical slice derivative (~3e-6)
    CONVEX_ATOL = 1e-9

    @staticmethod
    def _grid(Tlo=400.0, Thi=800.0, nT=20, clo=0.05, chi=0.95, nc=15):
        Tg = np.linspace(Tlo, Thi, nT)
        cg = np.linspace(clo, chi, nc)
        return np.repeat(Tg, nc), np.tile(cg, nT), Tg, cg

    def test_recovers_training_data(self):
        """With ``smoothing=0`` the RBF interpolates, so each fixed-T slice
        reproduces its own data slice at the interior training temperatures."""
        from landau.interpolate import WhitneySurface2DInterpolator
        from landau.interpolate.basic import FittedSurface

        T, c, Tg, cg = self._grid()
        H = 0.4 * (c - 0.5) ** 2 + 2e-5 * (T - 600) * (c - 0.5) ** 2
        surface = WhitneySurface2DInterpolator(smoothing=0.0).fit(T, c, H)
        assert isinstance(surface, FittedSurface)
        for Tq in Tg[1:-1]:  # interior training temperatures
            expect = 0.4 * (cg - 0.5) ** 2 + 2e-5 * (Tq - 600) * (cg - 0.5) ** 2
            np.testing.assert_allclose(surface.slice_at(float(Tq))(cg), expect, atol=self.RECOVER_ATOL)

    def test_slice_derivative_recovers_analytic_slope(self):
        """The numerical slice derivative recovers the true ``dH/dc`` of a known
        quadratic surface at an interior temperature (a constant/degenerate fit
        would miss the linear slope)."""
        from landau.interpolate import WhitneySurface2DInterpolator

        T, c, Tg, cg = self._grid()
        H = 0.4 * (c - 0.5) ** 2 + 2e-5 * (T - 600) * (c - 0.5) ** 2
        surface = WhitneySurface2DInterpolator(smoothing=0.0).fit(T, c, H)
        cc = np.linspace(0.2, 0.8, 30)
        d = np.asarray(surface.slice_at(600.0).deriv()(cc))
        np.testing.assert_allclose(d, 0.8 * (cc - 0.5), atol=1e-4)

    def test_slice_scalar_and_array_shapes(self):
        from landau.interpolate import WhitneySurface2DInterpolator

        T, c, Tg, cg = self._grid()
        H = 0.4 * (c - 0.5) ** 2
        sl = WhitneySurface2DInterpolator().fit(T, c, H).slice_at(600.0)
        assert np.ndim(sl(0.5)) == 0
        arr = sl(np.linspace(0.3, 0.7, 11))
        assert isinstance(arr, np.ndarray) and arr.shape == (11,)
        assert np.ndim(sl.deriv()(0.5)) == 0

    def test_fit_requires_two_distinct_concentrations(self):
        from landau.interpolate import WhitneySurface2DInterpolator

        T = np.linspace(400, 800, 12)
        c = np.full_like(T, 0.5)
        f = np.zeros_like(T)
        with pytest.raises(ValueError, match="two distinct concentrations"):
            WhitneySurface2DInterpolator().fit(T, c, f)

    def test_is_frozen_hashable_and_exported(self):
        from landau.interpolate import WhitneySurface2DInterpolator
        from landau.interpolate.basic import SurfaceInterpolator
        import landau.interpolate as pkg

        a = WhitneySurface2DInterpolator()
        b = WhitneySurface2DInterpolator()
        assert a == b and hash(a) == hash(b) and len({a, b}) == 1
        assert WhitneySurface2DInterpolator(degree=3) != a
        assert WhitneySurface2DInterpolator(smoothing=1.0) != a
        assert isinstance(a, SurfaceInterpolator)
        assert "WhitneySurface2DInterpolator" in pkg.__all__
        with pytest.raises(Exception):
            a.degree = 5

    def test_surface_phase_narrow_window_gibbs_duhem(self):
        """No terminal requirement -- a narrow intermetallic window (never
        reaching c=0/c=1) fits, and ``c = -dphi/ddmu`` holds at interior c."""
        from landau.interpolate import WhitneySurface2DInterpolator, SGTE
        from landau.phases import (
            Surface2DInterpolatingPhase,
            TemperatureDependentLinePhase,
            S,
        )

        clo, chi, c0 = 0.30, 0.40, 0.35
        Tsweep = np.linspace(400.0, 800.0, 60)
        lines = []
        for ci in np.linspace(clo, chi, 7):
            H = 1.0 * (1 + 1e-4 * Tsweep) * (ci - c0) ** 2
            f = H - Tsweep * S(np.array(ci))
            lines.append(TemperatureDependentLinePhase("im", float(ci), Tsweep, f, interpolator=SGTE(3)))

        phase = Surface2DInterpolatingPhase(
            "im", lines, concentration_range=(clo, chi), add_entropy=False,
            temperature_range=(400.0, 800.0),
            surface_interpolator=WhitneySurface2DInterpolator(smoothing=1e-4))

        T = 500.0
        dmu = np.linspace(-0.2, 0.2, 41)
        c = phase.concentration(np.full_like(dmu, T), dmu)
        phi = phase.semigrand_potential(np.full_like(dmu, T), dmu)
        assert np.isfinite(c).all() and np.isfinite(phi).all()
        assert (c >= clo - 1e-6).all() and (c <= chi + 1e-6).all()

        h = 1e-4
        cphi = -(phase.semigrand_potential(np.full_like(dmu, T), dmu + h)
                 - phase.semigrand_potential(np.full_like(dmu, T), dmu - h)) / (2 * h)
        interior = (c > clo + 1e-3) & (c < chi - 1e-3)
        assert interior.sum() > 5
        np.testing.assert_allclose(c[interior], cphi[interior], atol=self.GIBBS_ATOL)
