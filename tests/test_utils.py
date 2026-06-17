import unittest

import numpy as np

from Altendorf_Jeulin_Model.utils import (
    angle_between,
    cartesian_to_spherical,
    normalized,
    periodic_distance,
    spherical_to_cartesian,
    spherical_to_matrix,
)


class test_utils(unittest.TestCase):
    def test_periodic_distance(self):
        coord1 = np.array([30.0, 30.0, 30.0])
        coord2 = np.array([32.0, 30.0, 30.0])
        coord3 = np.array([1., 1., 1.])
        coord4 = np.array([63., 1., 1.])
        image_size = (64, 64, 64)
        dist, _ = periodic_distance(coord1, coord2, image_size)
        self.assertEqual(dist, 2,
                         "Error in periodic distance")
        dist, _ = periodic_distance(coord3, coord4, image_size)
        self.assertEqual(dist, 2,
                         "Error in periodic distance")

    def test_angle_between(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        v4 = np.array([0.0, 0.0, 0.0])
        v5 = np.array([1.0, 0.0, 1e-12])
        self.assertEqual(angle_between(v1, v2), np.pi / 2.0)
        self.assertEqual(angle_between(v2, v3), np.pi / 2.0)
        self.assertEqual(angle_between(v1, 200 * v1), 0)
        self.assertEqual(angle_between(v1, -v1), np.pi)
        self.assertEqual(angle_between(v1, v5), 0)
        self.assertEqual(angle_between(v2, 2 * v1), np.pi / 2.0)
        with self.assertRaises(ValueError):
            angle_between(v1, v4)

    def test_normalized(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        v4 = np.array([0.0, 0.0, 0.0])
        l1, vl = normalized(v1)
        np.testing.assert_array_equal(vl, v1)
        self.assertEqual(l1, 1)
        l2, vl = normalized(v2)
        np.testing.assert_array_equal(vl, v2)
        self.assertEqual(l2, 1)
        l3, vl = normalized(v3)
        np.testing.assert_array_equal(vl, v3)
        self.assertEqual(l3, 1)
        l4, vl = normalized(v4)
        np.testing.assert_array_equal(vl, v4)
        self.assertEqual(l4, 0)
        l5, vl = normalized(2*v1)
        np.testing.assert_array_equal(vl, v1)
        self.assertEqual(l5, 2)

    def test_cartesian_to_spherical(self):
        r, theta, phi = cartesian_to_spherical(1,0,0)
        self.assertAlmostEqual(r, 1)
        self.assertAlmostEqual(theta, np.pi / 2.0)
        self.assertAlmostEqual(phi, 0)
        r, theta, phi = cartesian_to_spherical(0, 1, 0)
        self.assertAlmostEqual(r, 1)
        self.assertAlmostEqual(theta, np.pi / 2.0)
        self.assertAlmostEqual(phi, np.pi / 2.0)
        r, theta, phi = cartesian_to_spherical(0, 0, 2)
        self.assertAlmostEqual(r, 2)
        self.assertAlmostEqual(theta, 0)
        self.assertAlmostEqual(phi, 0)
        r, theta, phi = cartesian_to_spherical(0, 0, 0)
        self.assertAlmostEqual(r, 0)
        self.assertAlmostEqual(theta, 0)
        self.assertAlmostEqual(phi, 0)
        
    def test_spherical_to_cartesian(self):
        x, y, z = spherical_to_cartesian(1, np.pi/2.0, 0)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(z, 0)
        x, y, z = spherical_to_cartesian(1, np.pi/2.0, np.pi / 2.0)
        self.assertAlmostEqual(x, 0)
        self.assertAlmostEqual(y, 1)
        self.assertAlmostEqual(z, 0)
        x, y, z = spherical_to_cartesian(2, 0, 0)
        self.assertAlmostEqual(x, 0)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(z, 2)
        x, y, z = spherical_to_cartesian(0, 0, np.pi / 2.0)
        self.assertAlmostEqual(x, 0)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(z, 0)

        x = 1
        y = 2
        z = 3
        r, theta, phi = cartesian_to_spherical(x, y, z)
        xt, yt, zt = spherical_to_cartesian(r, theta, phi)
        self.assertAlmostEqual(xt, x)
        self.assertAlmostEqual(yt, y)
        self.assertAlmostEqual(zt, z)


    def test_spherical_to_matrix(self):
        mat = spherical_to_matrix(0,0)
        mat_test = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
        np.testing.assert_array_equal(mat, mat_test)
    