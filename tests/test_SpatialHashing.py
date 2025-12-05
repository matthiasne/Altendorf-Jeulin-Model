import unittest
import SpatialHashing as sh
import numpy as np
from Fiber import Ball
import FiberModel as fm


class TestSpatialHashing(unittest.TestCase):
    def test_in_bounds(self):
        grid = sh.SpatialHashing((64, 64, 64), 32)
        self.assertEqual(grid._in_bounds([0, 1, 1]), True,
                         "_in_bounds considers correct indices as false")
        self.assertEqual(grid._in_bounds([0, 1, 2]), False,
                         "_in_bounds does not interpret index out of bound")
        self.assertEqual(grid._in_bounds([-1, 1, 0]), False,
                         "_in_bounds does not interpret index out of bound")

    def test_get_cell_index_of_coord(self):
        grid = sh.SpatialHashing((64, 64, 64), 32)
        self.assertEqual(grid.get_cell_index_of_coord(np.array([40, 40, 40])), (1, 1, 1),
                         "Calculation of cell index is wrong")
        self.assertEqual(grid.get_cell_index_of_coord(np.array([0, 30, 30])), (0, 0, 0),
                         "Calculation of cell index is wrong")
        with self.assertRaises(ValueError):
            grid.get_cell_index_of_coord(np.array([-1, 30, 30]))
        with self.assertRaises(ValueError):
            grid.get_cell_index_of_coord(np.array([1, 65, 30]))

    def test_get_neighbor_cell_indices(self):
        grid = sh.SpatialHashing((64, 64, 64), 32)
        self.assertEqual(sorted(grid.get_neighbor_cell_indices((0, 0, 0))),
                         sorted([(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]),
                         "Calculation of the neighboring cells is erroneous")
        with self.assertRaises(ValueError):
            grid.get_neighbor_cell_indices([0, 1, 2])
        with self.assertRaises(ValueError):
            grid.get_neighbor_cell_indices([0, 1, -1])

    def test_add_ball(self):
        grid = sh.SpatialHashing((64, 64, 64), 32)
        grid.add_ball(Ball(np.array([30, 30, 30]), 2))
        self.assertNotEqual(len(grid.cells[0]), 0,
                            "Ball was inserted wrongly in the SpatialHashing")
        self.assertEqual(len(grid.cells[3]), 0,
                         "Ball was inserted wrongly in the SpatialHashing")

    def test_add_fiber_system(self):
        N = 2
        radius = 2
        fs = fm.initialize_fiber_system(N, 20, radius, 1, 10, 100)
        grid = sh.SpatialHashing((64, 64, 64), 32)
        grid.add_fiber_system(fs)
        self.assertEqual(len(grid.cells[0]), 9,
                         "FiberSystem was inserted wrongly in the SpatialHashing")
        self.assertEqual(len(grid.cells[1]), 0,
                         "FiberSystem was inserted wrongly in the SpatialHashing")
        self.assertEqual(len(grid.cells[2]), 5,
                         "FiberSystem was inserted wrongly in the SpatialHashing")
        self.assertEqual(len(grid.cells[3]), 6,
                         "FiberSystem was inserted wrongly in the SpatialHashing")
        self.assertEqual(len(grid.cells[4]), 0,
                         "FiberSystem was inserted wrongly in the SpatialHashing")
        self.assertEqual(len(grid.cells[5]), 0,
                         "FiberSystem was inserted wrongly in the SpatialHashing")
        self.assertEqual(len(grid.cells[6]), 0,
                         "FiberSystem was inserted wrongly in the SpatialHashing")
        self.assertEqual(len(grid.cells[7]), 0,
                         "FiberSystem was inserted wrongly in the SpatialHashing")


if __name__ == '__main__':
    unittest.main()
