import unittest
from unittest.mock import MagicMock

import numpy as np
import scipy.stats
from Altendorf_Jeulin_Model.utils.utils import (
    angle_between,
    cut_border,
    discretize_lines,
    discretize_spheres_nonperiodic,
    discretize_spheres_periodic,
    is_in_image,
    normalized,
    set_value,
    spherical_to_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fiber(coords, radius=1.0):
    """Build a minimal Fiber-like object from a list of (x,y,z) tuples."""
    import Altendorf_Jeulin_Model.Fiber as Fiber
    balls = []
    for coord in coords:
        b = Fiber.Ball(np.array(coord, dtype=float), radius, 0.0)
        balls.append(b)
    fiber = MagicMock()
    fiber.balls = balls
    return fiber


# ---------------------------------------------------------------------------
# angle_between
# ---------------------------------------------------------------------------

class TestAngleBetween(unittest.TestCase):

    def test_orthogonal(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(angle_between(v1, v2), np.pi / 2.0)

    def test_parallel(self):
        v1 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(angle_between(v1, 200 * v1), 0.0)

    def test_antiparallel(self):
        v1 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(angle_between(v1, -v1), np.pi)

    def test_near_parallel(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v5 = np.array([1.0, 0.0, 1e-12])
        self.assertAlmostEqual(angle_between(v1, v5), 0.0, places=6)

    def test_all_axes(self):
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(angle_between(v2, v3), np.pi / 2.0)

    def test_zero_v1_raises(self):
        with self.assertRaises(ValueError):
            angle_between(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    def test_zero_v2_raises(self):
        with self.assertRaises(ValueError):
            angle_between(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))

    def test_symmetry(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        self.assertAlmostEqual(angle_between(v1, v2), angle_between(v2, v1))

    def test_known_45_degrees(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(angle_between(v1, v2), np.pi / 4.0)


# ---------------------------------------------------------------------------
# normalized
# ---------------------------------------------------------------------------

class TestNormalized(unittest.TestCase):

    def test_unit_vectors_unchanged(self):
        for v in [np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])]:
            length, vn = normalized(v)
            self.assertAlmostEqual(length, 1.0)
            np.testing.assert_allclose(vn, v, atol=1e-15)

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        length, vn = normalized(v)
        self.assertEqual(length, 0)
        np.testing.assert_array_equal(vn, v)

    def test_scaled_vector(self):
        v = np.array([3.0, 0.0, 0.0])
        length, vn = normalized(v)
        self.assertAlmostEqual(length, 3.0)
        np.testing.assert_allclose(vn, [1.0, 0.0, 0.0], atol=1e-15)

    def test_result_is_unit_length(self):
        v = np.array([1.0, 2.0, 3.0])
        _, vn = normalized(v)
        self.assertAlmostEqual(np.linalg.norm(vn), 1.0, places=14)

    def test_length_matches_norm(self):
        v = np.array([3.0, 4.0, 0.0])
        length, _ = normalized(v)
        self.assertAlmostEqual(length, 5.0)


# ---------------------------------------------------------------------------
# spherical_to_matrix
# ---------------------------------------------------------------------------

class TestSphericalToMatrix(unittest.TestCase):

    def test_identity_at_origin(self):
        # theta=0, phi=0 → z-axis direction → rotation matrix should be identity
        mat = spherical_to_matrix(0, 0)
        np.testing.assert_array_almost_equal(mat, np.eye(3))

    def test_shape(self):
        mat = spherical_to_matrix(np.pi / 4, np.pi / 3)
        self.assertEqual(mat.shape, (3, 3))

    def test_orthogonality(self):
        # columns of the rotation matrix should be orthonormal
        for theta, phi in [(0.3, 0.7), (np.pi / 2, np.pi / 4), (np.pi / 3, np.pi)]:
            mat = spherical_to_matrix(theta, phi)
            np.testing.assert_array_almost_equal(mat.T @ mat, np.eye(3), decimal=10)

    def test_z_direction_column(self):
        # third column should equal the unit vector in direction (theta, phi)
        theta, phi = np.pi / 3, np.pi / 4
        mat = spherical_to_matrix(theta, phi)
        expected_z_col = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])
        np.testing.assert_array_almost_equal(mat[:, 2], expected_z_col)


# ---------------------------------------------------------------------------
# is_in_image
# ---------------------------------------------------------------------------

class TestIsInImage(unittest.TestCase):

    def _size(self):
        return (100, 100, 100)

    def test_interior(self):
        self.assertTrue(is_in_image(np.array([50.0, 50.0, 50.0]), self._size()))

    def test_at_zero(self):
        self.assertTrue(is_in_image(np.array([0.0, 0.0, 0.0]), self._size()))

    def test_at_max(self):
        self.assertTrue(is_in_image(np.array([100.0, 100.0, 100.0]), self._size()))

    def test_just_outside_negative(self):
        # default buffer=100, so -101 should be outside
        self.assertFalse(is_in_image(np.array([-101.0, 50.0, 50.0]), self._size(), buffer=100))

    def test_within_buffer(self):
        self.assertTrue(is_in_image(np.array([-50.0, 50.0, 50.0]), self._size(), buffer=100))

    def test_outside_positive(self):
        self.assertFalse(is_in_image(np.array([201.0, 50.0, 50.0]), self._size(), buffer=100))

    def test_zero_buffer_at_boundary(self):
        self.assertFalse(is_in_image(np.array([-1.0, 50.0, 50.0]), self._size(), buffer=0))
        self.assertTrue(is_in_image(np.array([0.0, 50.0, 50.0]), self._size(), buffer=0))

    def test_each_axis_independently(self):
        size = (10, 20, 30)
        self.assertFalse(is_in_image(np.array([-200.0, 10.0, 15.0]), size, buffer=100))
        self.assertFalse(is_in_image(np.array([5.0, -200.0, 15.0]), size, buffer=100))
        self.assertFalse(is_in_image(np.array([5.0, 10.0, -200.0]), size, buffer=100))


# ---------------------------------------------------------------------------
# set_value
# ---------------------------------------------------------------------------

class TestSetValue(unittest.TestCase):

    def test_scalar_int(self):
        self.assertEqual(set_value(5, None), 5)

    def test_scalar_float(self):
        self.assertAlmostEqual(set_value(3.14, None), 3.14)

    def test_scipy_rv(self):
        rng = np.random.default_rng(0)
        rv = scipy.stats.uniform(loc=2.0, scale=0.0)  # always returns 2.0
        self.assertAlmostEqual(set_value(rv, rng), 2.0)

    def test_scipy_rv_range(self):
        rng = np.random.default_rng(42)
        rv = scipy.stats.uniform(loc=0.0, scale=10.0)
        for _ in range(20):
            val = set_value(rv, rng)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 10.0)

    def test_invalid_raises(self):
        with self.assertRaises((ValueError, AttributeError)):
            set_value("not_a_value", None)

    def test_zero(self):
        self.assertEqual(set_value(0, None), 0)

    def test_negative(self):
        self.assertAlmostEqual(set_value(-7.5, None), -7.5)


# ---------------------------------------------------------------------------
# cut_border
# ---------------------------------------------------------------------------

class TestCutBorder(unittest.TestCase):

    def _fiber_along_x(self, x_start, x_end, n=5, y=50.0, z=50.0, radius=1.0):
        import Altendorf_Jeulin_Model.Fiber as Fiber
        balls = []
        for x in np.linspace(x_start, x_end, n):
            b = Fiber.Ball(np.array([x, y, z], dtype=float), radius, 0.0)
            balls.append(b)
        fiber = MagicMock()
        fiber.balls = balls
        return fiber

    def test_fiber_entirely_inside_survives(self):
        image_size = np.array([100, 100, 100])
        boundary = 10
        # fiber spans x=15..85 (inside [0,100] after subtracting boundary of 10)
        fiber = self._fiber_along_x(x_start=25, x_end=95, n=7)
        result = cut_border([fiber], image_size, boundary)
        self.assertEqual(len(result), 1)
        self.assertGreater(len(result[0].balls), 0)

    def test_coordinates_shifted_by_boundary(self):
        image_size = np.array([100, 100, 100])
        boundary = 10
        fiber = self._fiber_along_x(x_start=20, x_end=80, n=5)
        result = cut_border([fiber], image_size, boundary)
        # all surviving ball x-coordinates should be in [0, 100]
        for ball in result[0].balls:
            self.assertGreaterEqual(ball.coordinate[0], 0)
            self.assertLessEqual(ball.coordinate[0], 100)

    def test_empty_fiber_system(self):
        result = cut_border([], np.array([100, 100, 100]), 10)
        self.assertEqual(result, [])

    def test_original_fiber_system_not_mutated(self):
        image_size = np.array([100, 100, 100])
        boundary = 10
        fiber = self._fiber_along_x(x_start=20, x_end=80, n=5)
        original_coords = [b.coordinate.copy() for b in fiber.balls]
        cut_border([fiber], image_size, boundary)
        for orig, ball in zip(original_coords, fiber.balls):
            np.testing.assert_array_equal(orig, ball.coordinate)


# ---------------------------------------------------------------------------
# discretize_spheres_periodic
# ---------------------------------------------------------------------------

class TestDiscretizeSpheresPeriodic(unittest.TestCase):

    def _single_fiber(self, coord, radius):
        import Altendorf_Jeulin_Model.Fiber as Fiber
        ball = Fiber.Ball(np.array(coord, dtype=float), radius, 0.0)
        fiber = MagicMock()
        fiber.balls = [ball]
        return [fiber]

    def test_output_shape(self):
        fs = self._single_fiber([50, 50, 50], 5)
        img = discretize_spheres_periodic(fs, 100, 100, 100)
        self.assertEqual(img.shape, (100, 100, 100))

    def test_output_dtype(self):
        fs = self._single_fiber([50, 50, 50], 5)
        img = discretize_spheres_periodic(fs, 100, 100, 100)
        self.assertEqual(img.dtype, np.uint8)

    def test_center_voxel_is_set(self):
        fs = self._single_fiber([50, 50, 50], 5)
        img = discretize_spheres_periodic(fs, 100, 100, 100)
        self.assertEqual(img[50, 50, 50], 1)

    def test_far_voxel_is_zero(self):
        fs = self._single_fiber([50, 50, 50], 3)
        img = discretize_spheres_periodic(fs, 100, 100, 100)
        self.assertEqual(img[0, 0, 0], 0)

    def test_ball_near_negative_boundary_wraps(self):
        # ball at (1,50,50) with radius 5: voxels at x<0 should wrap to positive side
        fs = self._single_fiber([1, 50, 50], 5)
        img = discretize_spheres_periodic(fs, 100, 100, 100)
        # voxels that wrapped should be set (e.g. x=98 wraps from x=-2)
        self.assertEqual(img[98, 50, 50], 1)

    def test_ball_near_positive_boundary_wraps(self):
        fs = self._single_fiber([98, 50, 50], 5)
        img = discretize_spheres_periodic(fs, 100, 100, 100)
        self.assertEqual(img[1, 50, 50], 1)

    def test_empty_fiber_system_gives_zeros(self):
        img = discretize_spheres_periodic([], 50, 50, 50)
        self.assertEqual(img.sum(), 0)

    def test_approximate_volume(self):
        # sphere of radius r should fill roughly (4/3)*pi*r^3 voxels
        r = 10
        fs = self._single_fiber([50, 50, 50], r)
        img = discretize_spheres_periodic(fs, 100, 100, 100)
        expected = (4 / 3) * np.pi * r ** 3
        self.assertAlmostEqual(img.sum(), expected, delta=0.05 * expected)


# ---------------------------------------------------------------------------
# discretize_spheres_nonperiodic
# ---------------------------------------------------------------------------

class TestDiscretizeSpheresNonperiodic(unittest.TestCase):

    def _single_fiber(self, coord, radius):
        import Altendorf_Jeulin_Model.Fiber as Fiber
        ball = Fiber.Ball(np.array(coord, dtype=float), radius, 0.0)
        fiber = MagicMock()
        fiber.balls = [ball]
        return [fiber]

    def test_output_shape(self):
        fs = self._single_fiber([50, 50, 50], 5)
        img = discretize_spheres_nonperiodic(fs, 100, 100, 100)
        self.assertEqual(img.shape, (100, 100, 100))

    def test_center_voxel_is_set(self):
        fs = self._single_fiber([50, 50, 50], 5)
        img = discretize_spheres_nonperiodic(fs, 100, 100, 100)
        self.assertEqual(img[50, 50, 50], 1)

    def test_out_of_bounds_voxels_not_written(self):
        # ball right at the edge: out-of-bounds voxels must be clipped, not wrapped
        fs = self._single_fiber([0, 50, 50], 5)
        img = discretize_spheres_nonperiodic(fs, 100, 100, 100)
        self.assertEqual(img.shape, (100, 100, 100))  # no index error
        # the negative-x side is clipped; image boundary voxel should still be set
        self.assertEqual(img[0, 50, 50], 1)

    def test_nonperiodic_does_not_wrap(self):
        # ball at x=1 with radius 5: unlike periodic, x=97..99 must NOT be set
        fs = self._single_fiber([1, 50, 50], 5)
        img = discretize_spheres_nonperiodic(fs, 100, 100, 100)
        self.assertEqual(img[98, 50, 50], 0)
        self.assertEqual(img[99, 50, 50], 0)

    def test_empty_gives_zeros(self):
        img = discretize_spheres_nonperiodic([], 50, 50, 50)
        self.assertEqual(img.sum(), 0)

    def test_approximate_volume(self):
        r = 10
        fs = self._single_fiber([50, 50, 50], r)
        img = discretize_spheres_nonperiodic(fs, 100, 100, 100)
        expected = (4 / 3) * np.pi * r ** 3
        self.assertAlmostEqual(img.sum(), expected, delta=0.05 * expected)

    def test_periodic_and_nonperiodic_agree_for_interior_ball(self):
        # for a ball well away from boundaries both should give the same result
        fs = self._single_fiber([50, 50, 50], 5)
        img_p = discretize_spheres_periodic(fs, 100, 100, 100)
        img_np = discretize_spheres_nonperiodic(fs, 100, 100, 100)
        np.testing.assert_array_equal(img_p, img_np)


# ---------------------------------------------------------------------------
# discretize_lines
# ---------------------------------------------------------------------------

class TestDiscretizeLines(unittest.TestCase):

    def _straight_line(self, radius=2):
        # a horizontal line along x at y=50, z=50
        start = np.array([10, 50, 50])
        end = np.array([90, 50, 50])
        return [(start, end, radius)]

    def test_output_shape(self):
        img = discretize_lines(self._straight_line(), (100, 100, 100))
        self.assertEqual(img.shape, (100, 100, 100))

    def test_output_dtype(self):
        img = discretize_lines(self._straight_line(), (100, 100, 100))
        self.assertEqual(img.dtype, np.uint16)

    def test_line_centre_is_set(self):
        img = discretize_lines(self._straight_line(radius=2), (100, 100, 100))
        self.assertGreater(img[50, 50, 50], 0)

    def test_empty_line_system(self):
        img = discretize_lines([], (50, 50, 50))
        self.assertEqual(img.sum(), 0)

    def test_multiple_lines_accumulate(self):
        line1 = (np.array([10, 25, 50]), np.array([90, 25, 50]), 2)
        line2 = (np.array([10, 75, 50]), np.array([90, 75, 50]), 2)
        img = discretize_lines([line1, line2], (100, 100, 100))
        self.assertGreater(img[50, 25, 50], 0)
        self.assertGreater(img[50, 75, 50], 0)

    def test_far_from_line_is_zero(self):
        img = discretize_lines(self._straight_line(radius=2), (100, 100, 100))
        # corner voxel should be untouched
        self.assertEqual(img[0, 0, 0], 0)


if __name__ == "__main__":
    unittest.main()
