"""
tests/test_fiber_model.py – unit tests for FiberModel.pyx
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.FiberModel import (
    initialize_fiber_system,
    initialize_fiber_system_AJ,
    save_balls_in_fiber_system,
    save_balls_in_fiber_system_endless,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def isotropic_direction(rng):
    """Uniform direction on S^2."""
    d = rng.standard_normal(3)
    d /= np.linalg.norm(d)
    return d[0], d[1], d[2]


def xaxis_direction(rng):
    """Fixed direction along x — deterministic for structural tests."""
    return 1.0, 0.0, 0.0


def make_coords_array(n, start=(0.0, 0.0, 0.0), step=(1.0, 0.0, 0.0)):
    """Straight line of n points as an (n, 3) numpy array."""
    coords = np.zeros((n, 3))
    for i in range(n):
        coords[i] = [start[j] + i * step[j] for j in range(3)]
    return coords


def make_coords_list(n, start=(0.0, 0.0, 0.0), step=(1.0, 0.0, 0.0)):
    """Straight line of n points as a list of arrays (endless format)."""
    return [np.array([start[j] + i * step[j] for j in range(3)]) for i in range(n)]


# ---------------------------------------------------------------------------
# save_balls_in_fiber_system
# ---------------------------------------------------------------------------

class TestSaveBalls(unittest.TestCase):

    def test_adds_one_fiber(self):
        fs = []
        coords = make_coords_array(5)
        save_balls_in_fiber_system(fs, coords, 0, 1.0)
        self.assertEqual(len(fs), 1)

    def test_fiber_has_correct_ball_count(self):
        fs = []
        coords = make_coords_array(5)
        save_balls_in_fiber_system(fs, coords, 0, 1.0)
        self.assertEqual(fs[0].get_number_of_balls(), 5)

    def test_ball_coordinates_match(self):
        fs = []
        coords = make_coords_array(3)
        save_balls_in_fiber_system(fs, coords, 0, 1.0)
        for i, ball in enumerate(fs[0].balls):
            assert_allclose(
                [ball.coordinate[0], ball.coordinate[1], ball.coordinate[2]],
                coords[i], atol=1e-10
            )

    def test_ball_radius_set(self):
        fs = []
        coords = make_coords_array(3)
        save_balls_in_fiber_system(fs, coords, 0, 2.5)
        for ball in fs[0].balls:
            self.assertAlmostEqual(ball.radius, 2.5)

    def test_ball_fiber_label_set(self):
        fs = []
        coords = make_coords_array(3)
        save_balls_in_fiber_system(fs, coords, 0, 1.0)
        for ball in fs[0].balls:
            self.assertEqual(ball.fiber_label, 0)

    def test_ball_ball_label_set(self):
        fs = []
        coords = make_coords_array(4)
        save_balls_in_fiber_system(fs, coords, 0, 1.0)
        for j, ball in enumerate(fs[0].balls):
            self.assertEqual(ball.ball_label, j)

    def test_interior_ball_angle_not_pi_for_bent_fiber(self):
        """A 90-degree bend gives an interior angle != pi."""
        fs = []
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ])
        save_balls_in_fiber_system(fs, coords, 0, 1.0)
        interior_ball = fs[0].balls[1]
        self.assertAlmostEqual(interior_ball.angle, np.pi / 2, places=5)

    def test_last_ball_angle_is_pi(self):
        """Last ball has no outgoing segment → angle defaults to pi."""
        fs = []
        coords = make_coords_array(4)
        save_balls_in_fiber_system(fs, coords, 0, 1.0)
        self.assertAlmostEqual(fs[0].balls[-1].angle, np.pi)

    def test_multiple_fibers(self):
        """Calling twice with i=0,1 appends two distinct fibers."""
        fs = []
        save_balls_in_fiber_system(fs, make_coords_array(3), 0, 1.0)
        save_balls_in_fiber_system(fs, make_coords_array(3, start=(10, 0, 0)), 1, 1.0)
        self.assertEqual(len(fs), 2)


# ---------------------------------------------------------------------------
# save_balls_in_fiber_system_endless
# ---------------------------------------------------------------------------

class TestSaveBallsEndless(unittest.TestCase):

    def test_adds_one_fiber(self):
        fs = []
        coords = make_coords_list(5)
        save_balls_in_fiber_system_endless(fs, coords, 0, 1.0)
        self.assertEqual(len(fs), 1)

    def test_fiber_has_correct_ball_count(self):
        fs = []
        coords = make_coords_list(5)
        save_balls_in_fiber_system_endless(fs, coords, 0, 1.0)
        self.assertEqual(fs[0].get_number_of_balls(), 5)

    def test_ball_coordinates_match(self):
        fs = []
        coords = make_coords_list(3)
        save_balls_in_fiber_system_endless(fs, coords, 0, 1.0)
        for i, ball in enumerate(fs[0].balls):
            assert_allclose(
                [ball.coordinate[0], ball.coordinate[1], ball.coordinate[2]],
                coords[i], atol=1e-10
            )


# ---------------------------------------------------------------------------
# initialize_fiber_system
# ---------------------------------------------------------------------------

class TestInitializeFiberSystem(unittest.TestCase):

    def _run(self, n=5, L=20.0, R=2.0, seed=0):
        return initialize_fiber_system(
            intensity=n,
            L=L,
            R=R,
            direction_distribution=xaxis_direction,
            image_size=np.array([50, 50, 50]),
            seed=seed,
        )

    def test_returns_list_of_fibers(self):
        fs = self._run()
        self.assertIsInstance(fs, list)
        self.assertTrue(all(isinstance(f, Fiber) for f in fs))

    def test_fiber_count(self):
        """Fixed intensity (int) produces exactly that many fibers."""
        fs = self._run(n=7)
        self.assertEqual(len(fs), 7)

    def test_reproducible_with_seed(self):
        fs1 = self._run(seed=42)
        fs2 = self._run(seed=42)
        for f1, f2 in zip(fs1, fs2):
            for b1, b2 in zip(f1.balls, f2.balls):
                assert_allclose(b1.coordinate[0], b2.coordinate[0])

    def test_different_seeds_different_positions(self):
        fs1 = self._run(seed=1)
        fs2 = self._run(seed=2)
        coords1 = [f.balls[0].coordinate[0] for f in fs1]
        coords2 = [f.balls[0].coordinate[0] for f in fs2]
        self.assertFalse(np.allclose(coords1, coords2))

    def test_fibers_have_multiple_balls(self):
        fs = self._run()
        for fiber in fs:
            self.assertGreater(fiber.get_number_of_balls(), 1)

    def test_start_position_within_image(self):
        image_size = np.array([50, 50, 50])
        fs = initialize_fiber_system(
            intensity=20, L=20.0, R=2.0,
            direction_distribution=isotropic_direction,
            image_size=image_size, seed=0,
        )
        for fiber in fs:
            b = fiber.balls[0]
            for dim, size in enumerate(image_size):
                coord = [b.coordinate[0], b.coordinate[1], b.coordinate[2]][dim]
                self.assertGreaterEqual(coord, 0.0)
                self.assertLess(coord, size)

    def test_ball_radius_matches_r(self):
        fs = self._run(R=3.0)
        for fiber in fs:
            for ball in fiber.balls:
                self.assertAlmostEqual(ball.radius, 3.0)

    def test_straight_fiber_direction(self):
        """With xaxis_direction, all fibers should run along x."""
        fs = self._run(n=10, seed=0)
        for fiber in fs:
            d = fiber.get_direction()
            norm = np.linalg.norm(d)
            if norm > 0:
                d_unit = d / norm
                # y and z components should be ~0
                assert_allclose(d_unit[1], 0.0, atol=1e-10)
                assert_allclose(d_unit[2], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# initialize_fiber_system_AJ
# ---------------------------------------------------------------------------

class TestInitializeFiberSystemAJ(unittest.TestCase):

    def _run(self, n=5, L=20.0, R=2.0, kappa1=1.0, kappa2=1.0, seed=0):
        return initialize_fiber_system_AJ(
            intensity=n,
            L=L,
            R=R,
            direction_distribution=xaxis_direction,
            image_size=np.array([50, 50, 50]),
            kappa1=kappa1,
            kappa2=kappa2,
            seed=seed,
        )

    def test_returns_list_of_fibers(self):
        fs = self._run()
        self.assertIsInstance(fs, list)
        self.assertTrue(all(isinstance(f, Fiber) for f in fs))

    def test_fiber_count(self):
        fs = self._run(n=6)
        self.assertEqual(len(fs), 6)

    def test_reproducible_with_seed(self):
        fs1 = self._run(seed=99)
        fs2 = self._run(seed=99)
        for f1, f2 in zip(fs1, fs2):
            for b1, b2 in zip(f1.balls, f2.balls):
                assert_allclose(b1.coordinate[0], b2.coordinate[0])

    def test_fibers_have_multiple_balls(self):
        fs = self._run()
        for fiber in fs:
            self.assertGreater(fiber.get_number_of_balls(), 1)

    def test_mean_direction_aligned_with_mu0(self):
        """
        After the rotation step, each fiber's end-to-end direction should
        be close to mu0 (x-axis here).  Regression for Bug 1 (wrong angle).
        """
        fs = self._run(n=20, kappa1=1.0, kappa2=1.0, seed=0)
        dot_products = []
        for fiber in fs:
            d = fiber.get_direction()
            norm = np.linalg.norm(d)
            if norm > 0:
                d_unit = d / norm
                dot_products.append(d_unit[0])   # x-component = cos(angle with x-axis)
        # Mean dot product with x-axis should be close to 1
        assert_allclose(np.mean(dot_products), 1.0, atol=0.15)

    def test_ball_radius_matches_r(self):
        fs = self._run(R=2.0)
        for fiber in fs:
            for ball in fiber.balls:
                self.assertAlmostEqual(ball.radius, 2.0)


if __name__ == "__main__":
    unittest.main()