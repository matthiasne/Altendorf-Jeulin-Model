import unittest

import numpy as np
from scipy.linalg import cholesky

from Altendorf_Jeulin_Model.utils.rv_utils import (
    acg_distribution,
    schladitz_distribution,
    vonmises_fisher,
)


N_SAMPLES = 10_000  # sample count for statistical tests


def _rng(seed=0):
    return np.random.default_rng(seed)


def _sample_vmf(kappa, mu, n, seed=0):
    rng = _rng(seed)
    mu_x, mu_y, mu_z = mu
    samples = [vonmises_fisher(kappa, mu_x, mu_y, mu_z, rng) for _ in range(n)]
    return np.array(samples)


def _sample_schladitz(beta, n, seed=0):
    rng = _rng(seed)
    samples = [schladitz_distribution(beta, rng) for _ in range(n)]
    return np.array(samples)


def _sample_acg(matrix, n, seed=0):
    rng = _rng(seed)
    samples = [acg_distribution(matrix, rng) for _ in range(n)]
    return np.array(samples)


# ---------------------------------------------------------------------------
# vonmises_fisher
# ---------------------------------------------------------------------------

class TestVonMisesFisher(unittest.TestCase):

    # --- output shape and type ---

    def test_returns_three_floats(self):
        result = vonmises_fisher(1.0, 1.0, 0.0, 0.0, _rng())
        self.assertEqual(len(result), 3)
        for v in result:
            self.assertIsInstance(v, float)

    def test_output_is_unit_vector(self):
        for mu in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
            mu = np.array(mu, dtype=float)
            mu /= np.linalg.norm(mu)
            result = vonmises_fisher(2.0, *mu, _rng())
            norm = np.sqrt(sum(v**2 for v in result))
            self.assertAlmostEqual(norm, 1.0, places=10,
                                   msg=f"not unit vector for mu={mu}")

    def test_no_nan_or_inf(self):
        samples = _sample_vmf(1.0, (1.0, 0.0, 0.0), 1000)
        self.assertTrue(np.all(np.isfinite(samples)))

    # --- reproducibility ---

    def test_fixed_seed_reproducible(self):
        r1 = vonmises_fisher(1.0, 1.0, 0.0, 0.0, _rng(42))
        r2 = vonmises_fisher(1.0, 1.0, 0.0, 0.0, _rng(42))
        np.testing.assert_array_equal(r1, r2)

    # --- known bug: mu = (0, 0, 1) must not return None (bug 1) ---

    def test_north_pole_mu_does_not_return_none(self):
        result = vonmises_fisher(1.0, 0.0, 0.0, 1.0, _rng())
        self.assertIsNotNone(result,
            "vonmises_fisher returned None for mu=(0,0,1); fix the early return")
        self.assertEqual(len(result), 3)

    def test_north_pole_mu_is_unit_vector(self):
        result = vonmises_fisher(1.0, 0.0, 0.0, 1.0, _rng())
        if result is None:
            self.skipTest("Bug 1 not yet fixed")
        norm = np.sqrt(sum(v**2 for v in result))
        self.assertAlmostEqual(norm, 1.0, places=10)

    # --- known bug: kappa = 0 must not crash (bug 2) ---

    def test_kappa_zero_does_not_crash(self):
        try:
            result = vonmises_fisher(0.0, 1.0, 0.0, 0.0, _rng())
        except ZeroDivisionError:
            self.fail("vonmises_fisher crashed with ZeroDivisionError for kappa=0")

    def test_kappa_zero_returns_unit_vector(self):
        try:
            result = vonmises_fisher(0.0, 1.0, 0.0, 0.0, _rng())
        except ZeroDivisionError:
            self.skipTest("Bug 2 not yet fixed")
        if result is None:
            self.fail("vonmises_fisher returned None for kappa=0")
        norm = np.sqrt(sum(v**2 for v in result))
        self.assertAlmostEqual(norm, 1.0, places=10)

    # --- statistical properties ---

    def test_high_kappa_concentrates_near_mu(self):
        # with kappa=50 samples should be very close to mu
        mu = np.array([1.0, 0.0, 0.0])
        samples = _sample_vmf(500.0, mu, N_SAMPLES)
        mean_dot = np.mean(samples @ mu)
        self.assertGreater(mean_dot, 0.99,
            "High-kappa vMF should concentrate near mu")

    def test_low_kappa_is_roughly_isotropic(self):
        # kappa=0.01 ~ uniform; mean direction should be near zero
        mu = np.array([1.0, 0.0, 0.0])
        samples = _sample_vmf(0.01, mu, N_SAMPLES, seed=1)
        mean_dir = np.abs(np.mean(samples, axis=0))
        self.assertLess(mean_dir[0], 0.1,
            "Low-kappa vMF should be roughly isotropic")

    def test_samples_lie_on_unit_sphere(self):
        samples = _sample_vmf(2.0, (0.0, 1.0, 0.0), N_SAMPLES)
        norms = np.linalg.norm(samples, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_different_mu_directions(self):
        # all three axis-aligned mu directions should give unit-vector outputs
        for mu in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            result = vonmises_fisher(2.0, *mu, _rng())
            norm = np.sqrt(sum(v**2 for v in result))
            self.assertAlmostEqual(norm, 1.0, places=10, msg=f"mu={mu}")

    def test_mean_direction_matches_mu(self):
        # mean of many samples should be in the same half-sphere as mu
        mu = np.array([0.0, 0.0, 1.0])
        samples = _sample_vmf(3.0, mu, N_SAMPLES, seed=2)
        self.assertGreater(np.mean(samples[:, 2]), 0.0)


# ---------------------------------------------------------------------------
# schladitz_distribution
# ---------------------------------------------------------------------------

class TestSchladitzDistribution(unittest.TestCase):

    # --- output shape and type ---

    def test_returns_three_floats(self):
        result = schladitz_distribution(1.0, _rng())
        self.assertEqual(len(result), 3)

    def test_output_is_unit_vector(self):
        x, y, z = schladitz_distribution(1.0, _rng())
        self.assertAlmostEqual(x**2 + y**2 + z**2, 1.0, places=10)

    def test_no_nan_or_inf(self):
        samples = _sample_schladitz(2.0, 1000)
        self.assertTrue(np.all(np.isfinite(samples)))

    # --- reproducibility ---

    def test_fixed_seed_reproducible(self):
        r1 = schladitz_distribution(1.0, _rng(7))
        r2 = schladitz_distribution(1.0, _rng(7))
        self.assertEqual(r1, r2)

    # --- error cases ---

    def test_beta_zero_raises(self):
        # once bug 5 is fixed this should raise ValueError
        with self.assertRaises((ValueError, ZeroDivisionError)):
            schladitz_distribution(0.0, _rng())

    # --- statistical properties ---

    def test_beta_one_is_isotropic(self):
        # beta=1: uniform distribution on sphere
        # each axis should have mean ~0 and variance ~1/3
        samples = _sample_schladitz(1.0, N_SAMPLES)
        for axis in range(3):
            self.assertAlmostEqual(np.mean(samples[:, axis]), 0.0, delta=0.05,
                                   msg=f"axis {axis} mean should be ~0 for beta=1")
            self.assertAlmostEqual(np.mean(samples[:, axis]**2), 1/3, delta=0.05,
                                   msg=f"axis {axis} variance should be ~1/3 for beta=1")

    def test_large_beta_concentrates_in_xy_plane(self):
        # large beta -> fibers prefer the x-y plane (small |z|)
        samples = _sample_schladitz(10.0, N_SAMPLES)
        mean_abs_z = np.mean(np.abs(samples[:, 2]))
        self.assertLess(mean_abs_z, 0.3,
            "Large beta should concentrate directions in the x-y plane")

    def test_small_beta_concentrates_near_z_axis(self):
        # beta < 1 -> fibers prefer the z-axis (large |z|)
        samples = _sample_schladitz(0.1, N_SAMPLES)
        mean_abs_z = np.mean(np.abs(samples[:, 2]))
        self.assertGreater(mean_abs_z, 0.7,
            "Small beta should concentrate directions near the z-axis")

    def test_samples_lie_on_unit_sphere(self):
        samples = _sample_schladitz(2.0, N_SAMPLES)
        norms = np.linalg.norm(samples, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_distribution_is_symmetric_in_phi(self):
        # the distribution is rotationally symmetric around the z-axis
        # so mean(x) and mean(y) should both be ~0
        samples = _sample_schladitz(3.0, N_SAMPLES)
        self.assertAlmostEqual(np.mean(samples[:, 0]), 0.0, delta=0.05)
        self.assertAlmostEqual(np.mean(samples[:, 1]), 0.0, delta=0.05)


# ---------------------------------------------------------------------------
# acg_distribution
# ---------------------------------------------------------------------------

class TestAcgDistribution(unittest.TestCase):

    def _identity(self):
        return np.eye(3)

    def _anisotropic(self):
        return np.array([
            [1.697, 0.023, -0.028],
            [0.023, 0.873, -0.031],
            [-0.028, -0.031, 0.324],
        ])

    # --- output shape and type ---

    def test_returns_array(self):
        result = acg_distribution(self._identity(), _rng())
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))

    def test_output_is_unit_vector(self):
        result = acg_distribution(self._identity(), _rng())
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=10)

    def test_no_nan_or_inf(self):
        samples = _sample_acg(self._identity(), 1000)
        self.assertTrue(np.all(np.isfinite(samples)))

    # --- reproducibility ---

    def test_fixed_seed_reproducible(self):
        r1 = acg_distribution(self._identity(), _rng(99))
        r2 = acg_distribution(self._identity(), _rng(99))
        np.testing.assert_array_equal(r1, r2)

    # --- error cases ---

    def test_non_positive_definite_matrix_raises(self):
        bad_matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [0.0, 0.0, 1.0]])
        with self.assertRaises(Exception):
            acg_distribution(bad_matrix, _rng())

    # --- statistical properties ---

    def test_identity_matrix_is_isotropic(self):
        # ACG with identity matrix is uniform on the sphere
        samples = _sample_acg(self._identity(), N_SAMPLES)
        for axis in range(3):
            self.assertAlmostEqual(np.mean(samples[:, axis]), 0.0, delta=0.05,
                                   msg=f"axis {axis} mean should be ~0 for identity matrix")
            self.assertAlmostEqual(np.mean(samples[:, axis]**2), 1/3, delta=0.05,
                                   msg=f"axis {axis} variance should be ~1/3 for identity matrix")

    def test_samples_lie_on_unit_sphere(self):
        samples = _sample_acg(self._anisotropic(), N_SAMPLES)
        norms = np.linalg.norm(samples, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_anisotropic_matrix_produces_anisotropic_distribution(self):
        # the anisotropic matrix has largest eigenvalue along x, smallest along z
        # so variance of x-component should be largest and z smallest
        samples = _sample_acg(self._anisotropic(), N_SAMPLES)
        var_x = np.var(samples[:, 0])
        var_z = np.var(samples[:, 2])
        self.assertGreater(var_x, var_z,
            "x-axis variance should exceed z-axis for this anisotropic matrix")

    def test_diagonal_matrix_concentrates_along_largest_eigenvalue(self):
        # matrix with large value along z concentrates samples near z-axis
        matrix = np.diag([0.1, 0.1, 10.0])
        samples = _sample_acg(matrix, N_SAMPLES)
        mean_abs_z = np.mean(np.abs(samples[:, 2]))
        self.assertGreater(mean_abs_z, 0.7,
            "ACG with large z-eigenvalue should concentrate near z-axis")


if __name__ == "__main__":
    unittest.main()