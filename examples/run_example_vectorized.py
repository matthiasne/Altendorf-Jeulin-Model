import time

import Altendorf_Jeulin_Model.FiberModel as fm
import numpy as np
from Altendorf_Jeulin_Model.VectorizedForceBiased import run_force_biased_vectorized
from Altendorf_Jeulin_Model.utils import cut_border

import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.io_utils import (
    print_fiber_positions_to_file,
)


def main():
    example_AJ_finite()
    example_AJ_endless()


def example_AJ_finite():
    print("This is the Altendorf-Jeulin model")
    image_size = np.array([100, 100, 100])
    intensity = 50
    L = 100
    R = 5
    beta = 1.0

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system(
        intensity, L, R, beta, image_size, 10, 100, is_poisson=False
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased_vectorized(fs, image_size, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    io.save_fibers_as_tif(
        fs, domain=image_size, path="examples/outputs/AJ_model_vectorized.tif", is_periodic=True
    )
    print_fiber_positions_to_file(fs, "examples/outputs/fibers_vectorized.txt")


def example_AJ_endless():
    print("This is the Altendorf-Jeulin model for endless fibers")
    image_size = (400, 400, 400)
    boundary_size = 50
    VV = 0.12
    R = np.random.normal(loc=8.5, scale=1.0)
    L = np.sqrt(3) / 2 * VV * (image_size[0] + 2 * boundary_size) ** 2 / R**2
    mu = 3 / 4 * np.pi * L * (image_size[0] + 2 * boundary_size) / image_size[0]
    A = np.array(
        [[1.697, 0.023, -0.028], [0.023, 0.873, -0.031], [-0.028, -0.031, 0.324]]
    )

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system_endless(
        mu,
        R,
        A,
        image_size,
        boundary_size,
        10,
        100,
        volume_fraction_should=VV,
        has_beta=False
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased_vectorized(fs, image_size, is_periodic=False, verbose=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    io.save_fibers_as_tif(
        fs,
        scale=4,
        domain=image_size,
        boundary=(boundary_size, boundary_size, boundary_size),
        path="examples/outputs/AJ_model_endless_vectorized.tif",
        is_periodic=False,
    )
    fs_cut = cut_border(fs, image_size, boundary_size)
    io.save_fibers_as_small_graph("examples/outputs/nonwoven_vectorized", fs_cut)


if __name__ == "__main__":
    main()
