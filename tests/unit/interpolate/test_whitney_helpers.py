import numpy as np
import pytest
from scipy.spatial import ConvexHull

from landau.interpolate.whitney import (
    _in_hull,
    _in_interval,
    _project_to_hull,
    _project_to_interval,
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
