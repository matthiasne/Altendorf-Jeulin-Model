import time
import numpy as np

import Altendorf_Jeulin_Model.FiberModel as fm
import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.ForceBiased import run_force_biased
from Altendorf_Jeulin_Model.io_utils import print_fiber_positions
from Altendorf_Jeulin_Model.utils import cut_border


def main():
    example_AJ_finite()
    example_AJ_endless()

def example_AJ_finite():
    print("This is the Altendorf-Jeulin model")
    image_size = (100, 100, 100)
    N = 100
    L = 50
    R = 5
    beta = 1

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system(N, L, R, beta, image_size, 10, 100)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased(fs, image_size, beta, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    io.save_fibers_as_tif(
        fs, shape=image_size, path="examples/outputs/AJ_model.tif", is_periodic=True
    )

def example_AJ_endless():
    print("This is the Altendorf-Jeulin model for endless fibers")
    image_size = (1800, 1800, 1800)
    boundary_size = 100
    VV = 0.06
    R = 17
    L = np.sqrt(3) / 2 * VV * (image_size[0] + 200) ** 2 / R**2
    mu = 3 / 4 * np.pi * L * (image_size[0] + 200) / image_size[0]
    N = int(mu)  # TODO
    A = np.array(
        [[1.697, 0.023, -0.028], [0.023, 0.873, -0.031], [-0.028, -0.031, 0.324]]
    )

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system_endless(
        N, R, A, image_size, boundary_size, 100, 100, has_beta=False, seed=43
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased(fs, image_size, is_periodic=False, has_beta=True, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")
    print_fiber_positions(fs, max_balls=20)

    io.save_fibers_as_tif(
        fs,
        scale=4,
        shape=(450, 450, 450),
        boundary=(boundary_size, boundary_size, boundary_size),
        path="examples/outputs/AJ_model_endless.tif",
        is_periodic=False,
    )
    fs_cut = cut_border(fs, image_size, boundary_size)
    io.save_fibers_as_small_graph("examples/outputs/nonwoven", fs_cut)
    


if __name__ == "__main__":
    main()
