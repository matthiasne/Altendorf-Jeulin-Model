"""
tests/test_fiber.py – unit tests for Fiber.pyx (Ball and Fiber classes)
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from Altendorf_Jeulin_Model.Fiber import Ball, Fiber


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_ball(coord, radius=1.0, fiber_label=-1, ball_label=-1, angle=np.pi):
    return Ball(np.array(coord, dtype=np.float64), radius,
                fiber_label=fiber_label, ball_label=ball_label, angle=angle)


def straight_fiber(coords, radius=1.0):
    """Build a Fiber from a list of coordinate triples."""
    balls = [make_ball(c, radius) for c in coords]
    f = Fiber(balls[0])
    for b in balls[1:]:
        f.add_ball(b)
    return f


# ---------------------------------------------------------------------------
# Ball.__init__
# ---------------------------------------------------------------------------

class TestBallInit(unittest.TestCase):

    def test_coordinate_stored(self):
        b = make_ball([1.0, 2.0, 3.0])
        assert_allclose([b.coordinate[0], b.coordinate[1], b.coordinate[2]],
                        [1.0, 2.0, 3.0])

    def test_radius_stored(self):
        b = make_ball([0, 0, 0], radius=2.5)
        self.assertAlmostEqual(b.radius, 2.5)

    def test_fiber_label_default(self):
        b = make_ball([0, 0, 0])
        self.assertEqual(b.fiber_label, -1)

    def test_ball_label_default(self):
        b = make_ball([0, 0, 0])
        self.assertEqual(b.ball_label, -1)

    def test_angle_default(self):
        b = make_ball([0, 0, 0])
        self.assertAlmostEqual(b.angle, np.pi)

    def test_angle_custom(self):
        b = make_ball([0, 0, 0], angle=np.pi / 2)
        self.assertAlmostEqual(b.angle, np.pi / 2)

    def test_force_initialised_to_zero(self):
        b = make_ball([1, 2, 3])
        assert_allclose([b.force[0], b.force[1], b.force[2]], [0.0, 0.0, 0.0])

    def test_overlap_initialised_to_zero(self):
        b = make_ball([0, 0, 0])
        self.assertAlmostEqual(b.overlap, 0.0)

    def test_neighbor_dist_is_half_radius(self):
        b = make_ball([0, 0, 0], radius=4.0)
        self.assertAlmostEqual(b.neighbor_dist, 2.0)

    def test_angle_diff_initialised_to_zero(self):
        b = make_ball([0, 0, 0])
        self.assertAlmostEqual(b.angle_diff, 0.0)

    def test_optim_sum_initialised_to_zero(self):
        # Regression for Bug 2: optim_sum declared in .pxd but was not
        # initialised in __init__ → should be 0, not garbage.
        b = make_ball([0, 0, 0])
        self.assertAlmostEqual(b.optim_sum, 0.0)

    def test_explicit_labels(self):
        b = Ball(np.array([0, 0, 0], dtype=np.float64), 1.0,
                 fiber_label=3, ball_label=7)
        self.assertEqual(b.fiber_label, 3)
        self.assertEqual(b.ball_label, 7)


# ---------------------------------------------------------------------------
# Fiber.__init__ and add_ball
# ---------------------------------------------------------------------------

class TestFiberConstruction(unittest.TestCase):

    def test_single_ball_fiber(self):
        f = Fiber(make_ball([0, 0, 0]))
        self.assertEqual(len(f.balls), 1)

    def test_add_ball_increases_count(self):
        f = Fiber(make_ball([0, 0, 0]))
        f.add_ball(make_ball([1, 0, 0]))
        f.add_ball(make_ball([2, 0, 0]))
        self.assertEqual(len(f.balls), 3)

    def test_get_number_of_balls(self):
        f = straight_fiber([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        self.assertEqual(f.get_number_of_balls(), 3)

    def test_balls_are_in_insertion_order(self):
        coords = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        f = straight_fiber(coords)
        for i, (ball, coord) in enumerate(zip(f.balls, coords)):
            assert_allclose(
                [ball.coordinate[0], ball.coordinate[1], ball.coordinate[2]],
                coord, err_msg=f"ball {i} coordinate wrong"
            )


# ---------------------------------------------------------------------------
# Fiber radius helpers
# ---------------------------------------------------------------------------

class TestFiberRadius(unittest.TestCase):

    def setUp(self):
        b1 = make_ball([0, 0, 0], radius=1.0)
        b2 = make_ball([1, 0, 0], radius=2.0)
        b3 = make_ball([2, 0, 0], radius=3.0)
        self.f = Fiber(b1)
        self.f.add_ball(b2)
        self.f.add_ball(b3)

    def test_min_radius(self):
        self.assertAlmostEqual(self.f.get_min_radius(), 1.0)

    def test_max_radius(self):
        self.assertAlmostEqual(self.f.get_max_radius(), 3.0)

    def test_mean_radius(self):
        self.assertAlmostEqual(self.f.get_mean_radius(), 2.0)

    def test_uniform_radius(self):
        f = straight_fiber([[0, 0, 0], [1, 0, 0]], radius=2.5)
        self.assertAlmostEqual(f.get_min_radius(), 2.5)
        self.assertAlmostEqual(f.get_max_radius(), 2.5)
        self.assertAlmostEqual(f.get_mean_radius(), 2.5)


# ---------------------------------------------------------------------------
# Fiber.get_length
# ---------------------------------------------------------------------------

class TestFiberLength(unittest.TestCase):

    def test_single_segment_x_axis(self):
        f = straight_fiber([[0, 0, 0], [4, 0, 0]])
        assert_allclose(f.get_length(), 4.0, rtol=1e-6)

    def test_two_segments(self):
        # 0→3 along x, then 3→3+4=7 along y: total = 3 + 4 = 7
        f = straight_fiber([[0, 0, 0], [3, 0, 0], [3, 4, 0]])
        assert_allclose(f.get_length(), 7.0, rtol=1e-6)

    def test_diagonal_segment(self):
        # sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
        f = straight_fiber([[0, 0, 0], [1, 1, 1]])
        assert_allclose(f.get_length(), np.sqrt(3), rtol=1e-6)

    def test_single_ball_length_zero(self):
        f = Fiber(make_ball([0, 0, 0]))
        self.assertAlmostEqual(f.get_length(), 0.0)


# ---------------------------------------------------------------------------
# Fiber.get_direction
# ---------------------------------------------------------------------------

class TestFiberDirection(unittest.TestCase):

    def test_direction_along_x(self):
        f = straight_fiber([[0, 0, 0], [5, 0, 0]])
        assert_allclose(f.get_direction(), [5.0, 0.0, 0.0])

    def test_direction_along_z(self):
        f = straight_fiber([[0, 0, 0], [0, 0, 3]])
        assert_allclose(f.get_direction(), [0.0, 0.0, 3.0])

    def test_direction_diagonal(self):
        f = straight_fiber([[1, 2, 3], [4, 6, 8]])
        assert_allclose(f.get_direction(), [3.0, 4.0, 5.0])

    def test_direction_uses_first_and_last_ball_only(self):
        # Middle ball at a right-angle; direction should still be first→last
        f = straight_fiber([[0, 0, 0], [0, 10, 0], [2, 0, 0]])
        assert_allclose(f.get_direction(), [2.0, 0.0, 0.0])

    def test_single_ball_raises(self):
        # Regression for Bug 3: single-ball fiber has no well-defined direction
        f = Fiber(make_ball([0, 0, 0]))
        with self.assertRaises((ValueError, Exception)):
            f.get_direction()


if __name__ == "__main__":
    unittest.main()