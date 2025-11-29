import unittest
import numpy as np
from utils import periodic_distance, angle_between


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
        self.assertEqual(angle_between(v1, v2), np.pi / 2.0)
        self.assertEqual(angle_between(v2, v3), np.pi / 2.0)
        with self.assertRaises(ValueError):
            angle_between(v1, v4)
