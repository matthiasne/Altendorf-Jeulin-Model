import unittest
import numpy as np
import Altendorf_Jeulin_Model.SpatialHashing as sh
import Altendorf_Jeulin_Model.FiberModel as fm
from Altendorf_Jeulin_Model.CalculateForces import calculate_forces
from Altendorf_Jeulin_Model.rv_utils import schladitz_distribution


def make_fiber_system(seed=42):
    image_size = np.array([128, 128, 128])
    rng = np.random.default_rng(seed)
    direction_dist = lambda rng: schladitz_distribution(0.1, rng)
    return fm.initialize_fiber_system(10, 400, 8.5, direction_dist, image_size, seed=seed), image_size


class TestOptimSumCellSizeInvariance(unittest.TestCase):

    def test_optim_sum_invariant_to_cell_size_periodic(self):
        """optim_sum must not change when only cell size changes (periodic)"""
        fs, image_size = make_fiber_system()
        results = {}
        for cell_size in [20, 40]:
            # fresh fiber system each time — same seed, same positions
            fs, image_size = make_fiber_system()
            grid = sh.SpatialHashing(image_size, cell_size)
            grid.add_fiber_system(fs, is_periodic=True)
            _, _, _, _, _, _, optim_sum = calculate_forces(
                grid, fiber_system=fs, is_periodic=True, hardcore_ratio=1.0
            )
            results[cell_size] = optim_sum

        values = list(results.values())
        for cell_size, val in results.items():
            self.assertAlmostEqual(val, values[0], places=6,
                msg=f"optim_sum={val} for cell_size={cell_size} differs from {values[0]} for cell_size={list(results.keys())[0]}")

    def test_optim_sum_invariant_to_cell_size_nonperiodic(self):
        """optim_sum must not change when only cell size changes (non-periodic)"""
        results = {}
        for cell_size in [5*8.5, 6*8.5, 10*8.5]:
            fs, image_size = make_fiber_system()
            grid = sh.SpatialHashing(image_size, cell_size)
            grid.add_fiber_system(fs, is_periodic=False)
            _, _, _, _, _, _, optim_sum = calculate_forces(
                grid, fiber_system=fs, is_periodic=False, hardcore_ratio=1.0
            )
            results[cell_size] = optim_sum

        values = list(results.values())
        for cell_size, val in results.items():
            self.assertAlmostEqual(val, values[0], places=6,
                msg=f"optim_sum={val} for cell_size={cell_size} differs from {values[0]}")

    def test_all_balls_inserted(self):
        """Every ball must appear in exactly one cell regardless of cell size"""
        fs, image_size = make_fiber_system()
        total_balls = sum(len(f.balls) for f in fs)
        for cell_size in [5*8.5, 6*8.5, 10*8.5]:
            fs, image_size = make_fiber_system()
            grid = sh.SpatialHashing(image_size, cell_size)
            grid.add_fiber_system(fs, is_periodic=True)
            balls_in_grid = sum(len(cell) for cell in grid.cells)
            self.assertEqual(balls_in_grid, total_balls,
                msg=f"cell_size={cell_size}: {balls_in_grid} balls in grid, expected {total_balls}")

    def test_younger_neighbors_count(self):
        """Each interior cell must have exactly 13 younger neighbors"""
        image_size = np.array([128, 128, 128])
        grid = sh.SpatialHashing(image_size, 20)  # 6x6x6 cells
        # pick a cell well away from boundaries
        neighbors = grid.get_younger_neighbor_cell_indices(3, 3, 3, is_periodic=False)
        self.assertEqual(len(neighbors), 13,
            msg=f"Expected 13 younger neighbors, got {len(neighbors)}")

    def test_younger_neighbors_no_self(self):
        """Cell must never appear in its own younger neighbor list"""
        image_size = np.array([128, 128, 128])
        grid = sh.SpatialHashing(image_size, 20)
        for ix in range(grid.division[0]):
            for iy in range(grid.division[1]):
                for iz in range(grid.division[2]):
                    flat = ix + iy*grid.division[0] + iz*grid.division[0]*grid.division[1]
                    neighbors = grid.get_younger_neighbor_cell_indices(ix, iy, iz, is_periodic=False)
                    self.assertNotIn(flat, neighbors,
                        msg=f"Cell ({ix},{iy},{iz}) is its own younger neighbor")

    def test_each_pair_covered_exactly_once(self):
        """Every adjacent cell pair (A,B) must appear exactly once across all neighbor lists"""
        image_size = np.array([64, 64, 64])
        grid = sh.SpatialHashing(image_size, 20)
        pair_counts = {}
        for flat_idx in range(grid.n_cells):
            iz = flat_idx // (grid.division[0] * grid.division[1])
            iy = (flat_idx % (grid.division[0] * grid.division[1])) // grid.division[0]
            ix = flat_idx % grid.division[0]
            neighbors = grid.get_younger_neighbor_cell_indices(ix, iy, iz, is_periodic=False)
            for nb in neighbors:
                pair = (min(flat_idx, nb), max(flat_idx, nb))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        for pair, count in pair_counts.items():
            self.assertEqual(count, 1,
                msg=f"Cell pair {pair} covered {count} times, expected 1")


if __name__ == "__main__":
    unittest.main()