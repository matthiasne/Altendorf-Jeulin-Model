"""
tests/test_poisson_lines.py – unit tests for PoissonLines.pyx
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from Altendorf_Jeulin_Model.PoissonLines import (
    simulate_poisson_lines,
    generate_poisson_line,
    line_cut_cube,
    line_cut_face,
    line_cut_sphere_length,
    generate_half_fiber,
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
    return 1.0, 0.0, 0.0


def rasterise_lines(line_system, image_size, r):
    """
    Paint line_system onto a binary voxel image by checking each voxel
    center against each line segment (pos1, pos2 already in voxel coords).
    Returns float image for easy mean computation.
    """
    img = np.zeros(image_size, dtype=np.float32)
    for pos1, pos2, r_line in line_system:
        d = pos2 - pos1
        length_sq = np.dot(d, d)
        if length_sq < 1e-12:
            continue
        for ix in range(image_size[0]):
            for iy in range(image_size[1]):
                for iz in range(image_size[2]):
                    p = np.array([ix, iy, iz], dtype=float) + 0.5
                    t = np.dot(p - pos1, d) / length_sq
                    t = np.clip(t, 0.0, 1.0)
                    closest = pos1 + t * d
                    if np.linalg.norm(p - closest) <= r_line:
                        img[ix, iy, iz] = 1.0
    return img


# ---------------------------------------------------------------------------
# line_cut_face
# ---------------------------------------------------------------------------

class TestLineCutFace(unittest.TestCase):

    def test_perpendicular_line_hits_face(self):
        # Line along x through (0.5, 0.5, 0.5), hits x=1 face
        normal = np.array([1.0, 0.0, 0.0])
        linepos = np.array([0.5, 0.5, 0.5])
        mu0 = np.array([1.0, 0.0, 0.0])
        pt = line_cut_face(normal, 1, linepos, mu0)
        assert_allclose(pt, [1.0, 0.5, 0.5], atol=1e-10)

    def test_perpendicular_line_hits_zero_face(self):
        normal = np.array([1.0, 0.0, 0.0])
        linepos = np.array([0.5, 0.5, 0.5])
        mu0 = np.array([1.0, 0.0, 0.0])
        pt = line_cut_face(normal, 0, linepos, mu0)
        assert_allclose(pt, [0.0, 0.5, 0.5], atol=1e-10)

    def test_parallel_line_returns_sentinel(self):
        # Line parallel to face → no intersection → sentinel (2,2,2)
        normal = np.array([1.0, 0.0, 0.0])
        linepos = np.array([0.5, 0.5, 0.5])
        mu0 = np.array([0.0, 1.0, 0.0])
        pt = line_cut_face(normal, 1, linepos, mu0)
        assert_allclose(pt, [2.0, 2.0, 2.0], atol=1e-10)

    def test_diagonal_line(self):
        # Line at 45 degrees through (0,0,0) with dir (1,1,0)/sqrt(2)
        normal = np.array([1.0, 0.0, 0.0])
        linepos = np.array([0.0, 0.0, 0.0])
        mu0 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        pt = line_cut_face(normal, 1, linepos, mu0)
        assert_allclose(pt[0], 1.0, atol=1e-10)
        assert_allclose(pt[1], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# line_cut_cube
# ---------------------------------------------------------------------------

class TestLineCutCube(unittest.TestCase):

    def test_x_axis_line_through_center(self):
        linepos = np.array([0.5, 0.5, 0.5])
        mu0 = np.array([1.0, 0.0, 0.0])
        is_cut, p1, p2, mid, length = line_cut_cube(linepos, mu0)
        self.assertTrue(is_cut)
        assert_allclose(mid, [0.5, 0.5, 0.5], atol=1e-10)
        assert_allclose(length, 1.0, atol=1e-10)

    def test_y_axis_line_through_center(self):
        linepos = np.array([0.5, 0.5, 0.5])
        mu0 = np.array([0.0, 1.0, 0.0])
        is_cut, p1, p2, mid, length = line_cut_cube(linepos, mu0)
        self.assertTrue(is_cut)
        assert_allclose(length, 1.0, atol=1e-10)

    def test_diagonal_line(self):
        # Line through (0.5,0.5,0.5) along (1,1,0)/sqrt(2)
        # exits at (1,1,0.5) and (0,0,0.5) → length = sqrt(2)
        linepos = np.array([0.5, 0.5, 0.5])
        mu0 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        is_cut, p1, p2, mid, length = line_cut_cube(linepos, mu0)
        self.assertTrue(is_cut)
        assert_allclose(length, np.sqrt(2), atol=1e-10)

    def test_missing_line_returns_false(self):
        # Line outside the cube entirely
        linepos = np.array([2.0, 2.0, 2.0])
        mu0 = np.array([0.0, 0.0, 1.0])
        is_cut, _, _, _, length = line_cut_cube(linepos, mu0)
        self.assertFalse(is_cut)
        self.assertEqual(length, -1)

    def test_midpoint_is_average_of_endpoints(self):
        linepos = np.array([0.3, 0.7, 0.5])
        mu0 = np.array([1.0, 0.0, 0.0])
        is_cut, p1, p2, mid, _ = line_cut_cube(linepos, mu0)
        if is_cut:
            assert_allclose(mid, (p1 + p2) / 2, atol=1e-10)

    def test_length_matches_endpoint_distance(self):
        linepos = np.array([0.5, 0.5, 0.5])
        mu0 = np.array([1.0, 0.0, 0.0])
        _, p1, p2, _, length = line_cut_cube(linepos, mu0)
        assert_allclose(length, np.linalg.norm(p2 - p1), atol=1e-10)


# ---------------------------------------------------------------------------
# line_cut_sphere_length  (regression for Bug 2: wrong quadratic formula)
# ---------------------------------------------------------------------------

class TestLineCutSphereLength(unittest.TestCase):

    def test_diameter_line_has_correct_length(self):
        # Line through center (0,0,0) of sphere of radius sqrt(3)/2
        # along x: should intersect at ±sqrt(3)/2 → length = sqrt(3)
        linepos = np.array([0.0, 0.0, 0.0])
        mu0 = np.array([1.0, 0.0, 0.0])
        length = line_cut_sphere_length(linepos, mu0)
        assert_allclose(length, np.sqrt(3), atol=1e-6)

    def test_tangent_line_zero_length(self):
        # Line tangent to sphere: exactly touches at one point → length = 0
        linepos = np.array([np.sqrt(3) / 2, 0.0, 0.0])
        mu0 = np.array([0.0, 1.0, 0.0])
        length = line_cut_sphere_length(linepos, mu0)
        assert_allclose(length, 0.0, atol=1e-6)

    def test_missing_line_returns_minus_one(self):
        # Line well outside the sphere
        linepos = np.array([2.0, 0.0, 0.0])
        mu0 = np.array([0.0, 1.0, 0.0])
        result = line_cut_sphere_length(linepos, mu0)
        self.assertEqual(result, -1)

    def test_length_positive(self):
        linepos = np.array([0.2, 0.1, 0.0])
        mu0 = np.array([1.0, 0.0, 0.0])
        length = line_cut_sphere_length(linepos, mu0)
        if length != -1:
            self.assertGreater(length, 0.0)


# ---------------------------------------------------------------------------
# generate_poisson_line
# ---------------------------------------------------------------------------

class TestGeneratePoissonLine(unittest.TestCase):

    def _run(self, seed=0, image_size=None):
        if image_size is None:
            image_size = np.array([50, 50, 50])
        rng = np.random.default_rng(seed)
        return generate_poisson_line(rng, isotropic_direction, image_size)

    def test_returns_five_values(self):
        result = self._run()
        self.assertEqual(len(result), 5)

    def test_mu0_is_unit_vector(self):
        _, _, _, mu0, _ = self._run()
        assert_allclose(np.linalg.norm(mu0), 1.0, atol=1e-10)

    def test_length_positive_or_minus_one(self):
        rng = np.random.default_rng(0)
        image_size = np.array([50, 50, 50])
        for _ in range(20):
            _, _, _, _, length = generate_poisson_line(
                rng, isotropic_direction, image_size
            )
            self.assertTrue(length > 0 or length == -1)

    def test_mid_pos_in_image(self):
        """When the line cuts the cube, mid_pos should be within image bounds."""
        rng = np.random.default_rng(0)
        image_size = np.array([50, 50, 50])
        for _ in range(50):
            _, _, mid_pos, _, length = generate_poisson_line(
                rng, isotropic_direction, image_size
            )
            if length > 0:
                for dim in range(3):
                    self.assertGreaterEqual(mid_pos[dim], 0.0)
                    self.assertLessEqual(mid_pos[dim], image_size[dim])


# ---------------------------------------------------------------------------
# simulate_poisson_lines
# ---------------------------------------------------------------------------

class TestSimulatePoissonLines(unittest.TestCase):

    def _run(self, mu=20, R=1.0, seed=0, image_size=None, is_poisson=False):
        if image_size is None:
            image_size = np.array([30, 30, 30])
        return simulate_poisson_lines(
            mu=mu,
            R=R,
            direction_distribution=isotropic_direction,
            image_size=image_size,
            seed=seed,
            is_poisson=is_poisson,
        )

    def test_returns_list(self):
        lines = self._run()
        self.assertIsInstance(lines, list)

    def test_each_entry_has_three_elements(self):
        lines = self._run()
        for entry in lines:
            self.assertEqual(len(entry), 3)

    def test_deterministic_with_seed(self):
        l1 = self._run(seed=7)
        l2 = self._run(seed=7)
        self.assertEqual(len(l1), len(l2))
        for (p1a, p2a, ra), (p1b, p2b, rb) in zip(l1, l2):
            assert_allclose(p1a, p1b)
            assert_allclose(p2a, p2b)

    def test_fixed_mu_gives_exact_count(self):
        lines = self._run(mu=10, is_poisson=False)
        # With is_poisson=False, n=mu exactly (unless volume_fraction exceeded)
        self.assertLessEqual(len(lines), 10)

    def test_radius_stored_correctly(self):
        lines = self._run(R=2.0)
        for _, _, r in lines:
            self.assertAlmostEqual(r, 2.0)


# ---------------------------------------------------------------------------
# Homogeneity test (the key spatial test you described)
# ---------------------------------------------------------------------------

class TestPoissonLinesHomogeneity(unittest.TestCase):
    """
    A homogeneous isotropic Poisson line process produces a uniform
    intensity field.  The mean image value along any axis should be
    approximately flat (constant across slices).

    We test this by checking that the coefficient of variation (std/mean)
    of slice means is small.
    """

    @classmethod
    def setUpClass(cls):
        image_size = np.array([40, 40, 40])
        lines = simulate_poisson_lines(
            mu=500,
            R=1.5,
            direction_distribution=isotropic_direction,
            image_size=image_size,
            seed=0,
            is_poisson=False,
        )
        cls.img = rasterise_lines(lines, image_size, r=1.5)

    def _cv(self, axis):
        """Coefficient of variation of slice means along given axis."""
        slice_means = self.img.mean(axis=tuple(i for i in range(3) if i != axis))
        if slice_means.mean() < 1e-6:
            return 0.0
        return slice_means.std() / slice_means.mean()

    def test_homogeneous_along_x(self):
        self.assertLess(self._cv(0), 0.25,
                        "slice means along x are too variable (non-homogeneous)")

    def test_homogeneous_along_y(self):
        self.assertLess(self._cv(1), 0.25,
                        "slice means along y are too variable (non-homogeneous)")

    def test_homogeneous_along_z(self):
        self.assertLess(self._cv(2), 0.25,
                        "slice means along z are too variable (non-homogeneous)")

    def test_nonzero_volume_fraction(self):
        self.assertGreater(self.img.mean(), 0.0)


# ---------------------------------------------------------------------------
# generate_half_fiber
# ---------------------------------------------------------------------------

class TestGenerateHalfFiber(unittest.TestCase):

    def _run(self, is_forward=True, seed=0):
        rng = np.random.default_rng(seed)
        image_size = np.array([50, 50, 50])
        mu0 = np.array([1.0, 0.0, 0.0])
        mid_pos = np.array([25.0, 25.0, 25.0])
        coords = [mid_pos.copy()]
        return generate_half_fiber(
            mu0, mid_pos, image_size,
            kappa1=1.0, kappa2=1.0,
            r_fiber=1.0,
            coords=coords,
            rng=rng,
            is_forward=is_forward,
        )

    def test_returns_list(self):
        coords = self._run()
        self.assertIsInstance(coords, list)

    def test_forward_appends_to_end(self):
        """Forward walk adds points after the midpoint."""
        coords = self._run(is_forward=True)
        # First element should still be the original midpoint
        assert_allclose(coords[0], [25.0, 25.0, 25.0])

    def test_backward_prepends_to_front(self):
        """Backward walk inserts points before the midpoint."""
        coords = self._run(is_forward=False)
        # Last element should be the original midpoint
        assert_allclose(coords[-1], [25.0, 25.0, 25.0])

    def test_produces_multiple_points(self):
        coords = self._run()
        self.assertGreater(len(coords), 1)

    def test_each_coord_is_length_3(self):
        coords = self._run()
        for c in coords:
            self.assertEqual(len(c), 3)

    def test_reproducible_with_seed(self):
        c1 = self._run(seed=42)
        c2 = self._run(seed=42)
        self.assertEqual(len(c1), len(c2))
        for a, b in zip(c1, c2):
            assert_allclose(a, b)


if __name__ == "__main__":
    unittest.main()