import unittest
import numpy as np
from utils import periodic_distance


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
