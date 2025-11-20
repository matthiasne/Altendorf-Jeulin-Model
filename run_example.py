import numpy as np

import FiberModel as fm
import io_utils as io
import SpatialHashing as sh
from CalculateForces import calculate_forces, apply_forces


def main():
    # create a fiber system
    print("This is the Altendorf-Jeulin model")
    N = 2
    radius = 2
    fs = fm.initialize_fiber_system(N, 20, radius, 1, 10, 100)

    # output fiber system
    print("We generated the following fibers")
    io.print_fiber_positions(fs, 5)
    io.plot_fibers_in_2D(fs)
    io.save_fibers_as_tif(fs)

    # set up spatial hashing
    grid = sh.SpatialHashing((64, 64, 64), 32)
    index = grid.get_cell_index_of_coord(np.array([30, 30, 30]))
    neighbor_cells = grid.get_neighbor_cell_indices(index)
    print("index: ", index)
    print("neighbor cells:", neighbor_cells)
    grid.add_fiber_system(fs)
    print("Cells filled with the fiber system:")
    io.print_grid(grid)
    print("We calculate and apply forces:")
    calculate_forces(grid)
    apply_forces(fs)
    io.print_fiber_positions(fs, 5)
    io.plot_fibers_in_2D(fs, path="spheres+.png")
    io.save_fibers_as_tif(fs, path="spheres+.tif")

if __name__ == "__main__":
    main()
