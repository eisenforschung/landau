import numpy as np
import pytest
from scipy.spatial import ConvexHull

from landau.interpolate.whitney import (
    _in_hull,
    _in_interval,
    _project_to_hull,
    _project_to_interval,
    _rbf_gradient,
)

ATOL = 1e-9


@pytest.fixture
def unit_square_hull():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return ConvexHull(pts)


class TestInInterval:
    def test_interior_point(self):
        assert _in_interval(np.array([[0.5]]), 0.0, 1.0)[0]

    def test_endpoints_are_closed(self):
        points = np.array([[0.0], [1.0]])
        np.testing.assert_array_equal(_in_interval(points, 0.0, 1.0), [True, True])

    def test_exterior_below_and_above(self):
        points = np.array([[-0.1], [1.1]])
        np.testing.assert_array_equal(_in_interval(points, 0.0, 1.0), [False, False])

    def test_empty_input(self):
        result = _in_interval(np.empty((0, 1)), 0.0, 1.0)
        assert result.dtype == bool
        assert result.shape == (0,)


class TestInHull:
    def test_interior_point(self, unit_square_hull):
        assert _in_hull(np.array([[0.5, 0.5]]), unit_square_hull)[0]

    def test_face_points(self, unit_square_hull):
        faces = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]])
        assert _in_hull(faces, unit_square_hull).all()

    def test_corner_points(self, unit_square_hull):
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert _in_hull(corners, unit_square_hull).all()

    def test_exterior_points(self, unit_square_hull):
        exterior = np.array([[2.0, 0.5], [-1.0, -1.0], [0.5, 2.0]])
        result = _in_hull(exterior, unit_square_hull)
        assert not result.any()

    def test_output_shape(self, unit_square_hull):
        points = np.array([[0.5, 0.5], [2.0, 2.0], [0.0, 0.0]])
        result = _in_hull(points, unit_square_hull)
        assert result.shape == (3,)


class TestProjectToInterval:
    def test_interior_unchanged(self):
        np.testing.assert_allclose(_project_to_interval(np.array([0.5]), 0.0, 1.0), [0.5])

    def test_clips_below(self):
        np.testing.assert_allclose(_project_to_interval(np.array([-2.0]), 0.0, 1.0), [0.0])

    def test_clips_above(self):
        np.testing.assert_allclose(_project_to_interval(np.array([3.0]), 0.0, 1.0), [1.0])

    def test_shape_preserved(self):
        x = np.array([-2.0, 0.5, 3.0])
        result = _project_to_interval(x, 0.0, 1.0)
        assert result.shape == x.shape
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


class TestProjectToHull:
    def test_projects_to_nearest_face(self, unit_square_hull):
        x_b = _project_to_hull(np.array([2.0, 0.5]), unit_square_hull)
        np.testing.assert_allclose(x_b, [1.0, 0.5], atol=1e-6)

    def test_projects_to_nearest_corner(self, unit_square_hull):
        x_b = _project_to_hull(np.array([2.0, 2.0]), unit_square_hull)
        np.testing.assert_allclose(x_b, [1.0, 1.0], atol=1e-6)

    def test_output_shape_matches_input(self, unit_square_hull):
        x_b = _project_to_hull(np.array([-3.0, 0.5]), unit_square_hull)
        assert x_b.shape == (2,)
        np.testing.assert_allclose(x_b, [0.0, 0.5], atol=1e-6)


class TestRbfGradient:
    """Central finite-difference gradient of any callable ``(N, D) -> (N,)``.

    Downstream ``_rbf_gradient`` is called on a fitted ``RBFInterpolator``, but
    the helper only relies on that interface, so the tests use plain lambdas
    with a known analytic gradient.
    """

    def test_constant_function_has_zero_gradient(self):
        # f(x + eps) == f(x - eps) exactly, so the numerator vanishes on
        # every axis regardless of eps or dimension.
        f = lambda X: np.full(X.shape[0], 3.14)  # noqa: E731
        np.testing.assert_allclose(
            _rbf_gradient(f, np.array([0.5, -1.0, 2.0]), eps=0.1),
            np.zeros(3),
            atol=ATOL,
        )

    def test_linear_function_recovers_coefficients(self):
        # Central difference is exact for polynomials of degree <= 2, so
        # a linear f(x) = a . x returns its coefficient vector to machine
        # precision independent of the step size.
        a = np.array([2.0, -3.0, 0.5])
        f = lambda X: X @ a  # noqa: E731
        np.testing.assert_allclose(
            _rbf_gradient(f, np.array([0.7, 1.3, -0.4]), eps=1e-2),
            a,
            atol=ATOL,
        )

    def test_quadratic_isotropic_gradient_is_exact(self):
        # grad(0.5 * x . x) = x, again exact under central difference.
        f = lambda X: 0.5 * np.einsum("ni,ni->n", X, X)  # noqa: E731
        x = np.array([0.3, -0.6, 1.1, 2.0])
        np.testing.assert_allclose(_rbf_gradient(f, x, eps=1e-3), x, atol=ATOL)

    @pytest.mark.parametrize("D", [1, 2, 4])
    def test_output_shape_matches_input_dim(self, D):
        f = lambda X: np.zeros(X.shape[0])  # noqa: E731
        grad = _rbf_gradient(f, np.zeros(D), eps=0.1)
        assert grad.shape == (D,)

    def test_partial_derivative_only_steps_along_its_own_axis(self):
        # Each partial must probe ``rbf`` at (x + eps * e_d) and
        # (x - eps * e_d) — the loop nulls all other axes to zero.  A
        # recording callable exposes the six sample points the helper
        # visits, so the axis isolation is checked directly.
        seen = []

        def f(X):
            seen.append(X.copy())
            return np.zeros(X.shape[0])

        x = np.array([10.0, 20.0, 30.0])
        _rbf_gradient(f, x, eps=0.5)
        assert len(seen) == 2 * x.size
        for row in seen:
            diff = row[0] - x
            nonzero = np.flatnonzero(np.abs(diff) > 1e-12)
            assert nonzero.size == 1
            assert np.isclose(abs(diff[nonzero[0]]), 0.5)
