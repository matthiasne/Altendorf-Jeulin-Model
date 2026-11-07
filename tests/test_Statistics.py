"""
tests/test_statistics.py – unit tests for Statistics.pyx
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from Altendorf_Jeulin_Model.Fiber import Fiber
from Altendorf_Jeulin_Model.Fiber import Ball
from Altendorf_Jeulin_Model.utils.Statistics import (
    mean_radius,
    mean_length,
    mean_angle_error,
    estimate_beta,
    h,
    h_dif,
    volume_fraction,
    estimate_kappa1,
    estimate_kappa2,
    calculate_fot,
    acg_log_density,
    kl_to_acg,
    calculate_kl,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_ball(coord, radius=1.0, angle=np.pi):
    return Ball(np.array(coord, dtype=np.float64), radius, angle=angle)


def straight_fiber(start, end, n_balls=3, radius=1.0):
    """n_balls equally-spaced balls along the segment start→end."""
    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)
    fiber = None
    for i in range(n_balls):
        if n_balls > 1:
            t = i / (n_balls - 1)
        else:
            t = 1
        coord = start + t * (end - start)
        ball = make_ball(coord, radius)
        if fiber is None:
            fiber = Fiber(ball)
        else:
            fiber.add_ball(ball)
    return fiber


# ---------------------------------------------------------------------------
# mean_radius
# ---------------------------------------------------------------------------

class TestMeanRadius(unittest.TestCase):

    def test_single_fiber_uniform_radius(self):
        f = straight_fiber([0, 0, 0], [1, 0, 0], n_balls=3, radius=2.0)
        assert_allclose(mean_radius([f]), 2.0)

    def test_multiple_fibers(self):
        f1 = straight_fiber([0, 0, 0], [1, 0, 0], radius=1.0)
        f2 = straight_fiber([0, 0, 0], [0, 1, 0], radius=3.0)
        assert_allclose(mean_radius([f1, f2]), 2.0)


# ---------------------------------------------------------------------------
# mean_length
# ---------------------------------------------------------------------------

class TestMeanLength(unittest.TestCase):

    def test_empty_list_returns_zero(self):
        self.assertEqual(mean_length([]), 0)

    def test_known_length(self):
        # fiber along x-axis, length = 4.0
        f = straight_fiber([0, 0, 0], [4, 0, 0], n_balls=5, radius=0.5)
        assert_allclose(mean_length([f]), 4.0, rtol=1e-6)

    def test_multiple_fibers(self):
        f1 = straight_fiber([0, 0, 0], [3, 0, 0], n_balls=4)
        f2 = straight_fiber([0, 0, 0], [0, 1, 0], n_balls=2)
        assert_allclose(mean_length([f1, f2]), 2.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# mean_angle_error
# ---------------------------------------------------------------------------

class TestMeanAngleError(unittest.TestCase):

    def test_perfectly_straight_fiber_zero_error(self):
        """Consecutive segments are collinear → angle = pi → error vs stored
        angle of pi should be 0."""
        f = straight_fiber([0, 0, 0], [10, 0, 0], n_balls=5)
        # angle stored on each ball defaults to pi (straight)
        assert_allclose(mean_angle_error([f]), 0.0, atol=1e-10)

    def test_empty_list(self):
        self.assertEqual(mean_angle_error([]), 0)

    def test_single_ball_fiber_no_interior(self):
        """Fiber with only one ball has no interior → no angles → 0."""
        f = Fiber(make_ball([0, 0, 0]))
        assert_allclose(mean_angle_error([f]), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# h and h_dif (Newton-Raphson helpers)
# ---------------------------------------------------------------------------

class TestHAndHDif(unittest.TestCase):

    def _thetas(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        return rng.uniform(0, np.pi, size=n)

    def test_h_root_near_one_for_isotropic(self):
        """For isotropic data (uniform thetas), the MLE of beta is ~1."""
        rng = np.random.default_rng(42)
        # isotropic: cos(theta) uniform on [-1,1] → thetas via arccos
        thetas = np.arccos(rng.uniform(-1, 1, size=500))
        result = estimate_beta(list(thetas), beta=1.0)
        # MLE should stay close to 1 for isotropic data
        self.assertAlmostEqual(result, 1.0, delta=0.3)

    def test_h_dif_is_derivative_of_h(self):
        """Finite-difference check: h_dif ≈ (h(β+ε) - h(β-ε)) / (2ε)."""
        thetas = self._thetas()
        for beta in [0.5, 1.0, 2.0, 5.0]:
            eps = 1e-5
            fd = (h(beta + eps, thetas) - h(beta - eps, thetas)) / (2 * eps)
            analytic = h_dif(beta, thetas)
            assert_allclose(analytic, fd, rtol=1e-4,
                            err_msg=f"h_dif wrong at beta={beta}")

    def test_h_at_beta_one_zero_for_isotropic(self):
        """For perfectly isotropic data h(1, thetas) should be near 0."""
        rng = np.random.default_rng(7)
        thetas = np.arccos(rng.uniform(-1, 1, size=2000))
        assert_allclose(h(1.0, thetas), 0.0, atol=20.0)   # stochastic; loose


# ---------------------------------------------------------------------------
# estimate_beta
# ---------------------------------------------------------------------------

class TestEstimateBeta(unittest.TestCase):

    def test_known_root(self):
        """h(beta*,thetas)=0 at the returned value."""
        rng = np.random.default_rng(99)
        thetas = list(np.arccos(rng.uniform(-1, 1, size=30000)))
        beta_est = estimate_beta(thetas, beta=1.0)
        assert_allclose(h(beta_est, thetas), 0.0, atol=0.05)

    def test_returns_float(self):
        thetas = list(np.linspace(0.1, np.pi - 0.1, 50))
        result = estimate_beta(thetas, beta=1.0)
        self.assertIsInstance(result, float)


# ---------------------------------------------------------------------------
# volume_fraction
# ---------------------------------------------------------------------------

class TestVolumeFraction(unittest.TestCase):

    def test_zero_radius_zero_fraction(self):
        f = straight_fiber([5, 5, 5], [10, 5, 5], n_balls=2, radius=0.0)
        vf = volume_fraction([f], shape=(20, 20, 20))
        self.assertAlmostEqual(vf, 0.0, places=5)

    def test_fraction_between_zero_and_one(self):
        f = straight_fiber([5, 5, 5], [15, 5, 5], n_balls=5, radius=1.0)
        vf = volume_fraction([f], shape=(20, 20, 20))
        self.assertGreaterEqual(vf, 0.0)
        self.assertLessEqual(vf, 1.0)

    def test_larger_radius_larger_fraction(self):
        shape = (30, 30, 30)
        f_small = straight_fiber([10, 10, 10], [20, 10, 10], radius=1.0)
        f_large = straight_fiber([10, 10, 10], [20, 10, 10], radius=3.0)
        self.assertLess(
            volume_fraction([f_small], shape=shape),
            volume_fraction([f_large], shape=shape),
        )


# ---------------------------------------------------------------------------
# calculate_fot
# ---------------------------------------------------------------------------

class TestCalculateFOT(unittest.TestCase):

    def test_shape(self):
        f = straight_fiber([0, 0, 0], [1, 0, 0])
        fot = calculate_fot([f])
        self.assertEqual(fot.shape, (3, 3))

    def test_x_aligned_fiber(self):
        """Single fiber along x → FOT = diag(1,0,0)."""
        f = straight_fiber([0, 0, 0], [5, 0, 0])
        fot = calculate_fot([f])
        expected = np.diag([1.0, 0.0, 0.0])
        assert_allclose(fot, expected, atol=1e-10)

    def test_isotropic_fot_trace_one(self):
        """FOT trace must always equal 1 (sum of eigenvalues of outer product)."""
        rng = np.random.default_rng(0)
        fibers = []
        for _ in range(50):
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            end = d * 10
            fibers.append(straight_fiber([0, 0, 0], end))
        fot = calculate_fot(fibers)
        assert_allclose(np.trace(fot), 1.0, atol=1e-10)

    def test_symmetric(self):
        f1 = straight_fiber([0, 0, 0], [1, 1, 0])
        f2 = straight_fiber([0, 0, 0], [0, 1, 1])
        fot = calculate_fot([f1, f2])
        assert_allclose(fot, fot.T, atol=1e-10)


# ---------------------------------------------------------------------------
# acg_log_density
# ---------------------------------------------------------------------------

class TestAcgLogDensity(unittest.TestCase):

    def _unit_sphere_samples(self, n=100, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, 3))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def test_output_shape(self):
        x = self._unit_sphere_samples(50)
        log_p = acg_log_density(x, np.eye(3))
        self.assertEqual(log_p.shape, (50,))

    def test_identity_sigma_uniform(self):
        """For Sigma=I all points on the sphere have the same log density."""
        x = self._unit_sphere_samples(100)
        log_p = acg_log_density(x, np.eye(3))
        # all values identical up to numerical noise
        assert_allclose(log_p, log_p[0], atol=1e-10)

    def test_returns_finite(self):
        x = self._unit_sphere_samples(30)
        Sigma = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        log_p = acg_log_density(x, Sigma)
        self.assertTrue(np.all(np.isfinite(log_p)))


# ---------------------------------------------------------------------------
# kl_to_acg
# ---------------------------------------------------------------------------

class TestKlToAcg(unittest.TestCase):

    def _unit_sphere_samples(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, 3))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def test_non_negative(self):
        """KL divergence is always >= 0."""
        x = self._unit_sphere_samples()
        kl = kl_to_acg(x, np.eye(3))
        self.assertGreaterEqual(kl, -1e-2)

    def test_identity_sigma_minimum(self):
        """KL(uniform || ACG(I)) is minimal; deviating Sigma increases KL."""
        x = self._unit_sphere_samples(500, seed=1)
        kl_identity = kl_to_acg(x, np.eye(3))
        Sigma_biased = np.diag([4.0, 1.0, 1.0])
        kl_biased = kl_to_acg(x, Sigma_biased)
        # uniform data fits isotropic ACG better
        self.assertLess(kl_identity, kl_biased)

    def test_raises_on_bad_shape(self):
        with self.assertRaises(ValueError):
            kl_to_acg(np.ones((10, 2)), np.eye(3))

    def test_raises_on_zero_vector(self):
        x = self._unit_sphere_samples(20)
        x[0] = 0.0
        with self.assertRaises(ValueError):
            kl_to_acg(x, np.eye(3))

    def test_raises_on_bad_k(self):
        x = self._unit_sphere_samples(10)
        with self.assertRaises(ValueError):
            kl_to_acg(x, np.eye(3), k=0)


# ---------------------------------------------------------------------------
# calculate_kl
# ---------------------------------------------------------------------------

class TestCalculateKl(unittest.TestCase):

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        fibers = []
        for _ in range(500):
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            fibers.append(straight_fiber([0, 0, 0], (d * 5).tolist()))
        A = np.eye(3)
        kl = calculate_kl(fibers, A)
        self.assertGreaterEqual(kl, -1e-2)


if __name__ == "__main__":
    unittest.main()