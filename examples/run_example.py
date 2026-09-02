import time

import Altendorf_Jeulin_Model.FiberModel as fm
import numpy as np
from Altendorf_Jeulin_Model.utils import cut_border

import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.ForceBiased import run_force_biased
from Altendorf_Jeulin_Model.io_utils import (
    print_fiber_positions_to_file,
)
from Altendorf_Jeulin_Model.ContactModel import (
    find_contact_areas,
    find_contact_candidates
)


def main():
    example_AJ_finite()
    example_AJ_endless()


def example_AJ_finite():
    print("This is the Altendorf-Jeulin model")
    image_size = np.array([150, 150, 150])
    intensity = 90
    L = 80
    R = 4
    beta = 3.0

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system(
        intensity, L, R, beta, image_size, 100, 100, is_poisson=False, seed=42
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    start_time = time.time()
    run_force_biased(fs, image_size, verbose=True, softcore_ratio = 0.0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    n_cc, n_clots, n_contacts, contact_surface = find_contact_areas(fs, image_size, is_periodic=True, contact_distance= 0.1*R)
    print("connected components: ", n_cc, "clots: ", n_clots, " contacts: ", n_contacts,
          " contact surface: ", contact_surface)
    shortlist = find_contact_candidates(fs, image_size, is_periodic = True, interaction_distance=0.2*R)
    print("shortlist has ", len(shortlist), " elements")
    run_force_biased(fs, image_size, verbose=True, shortlist=shortlist, softcore_ratio=0, contact_distance=0.1*R,
                     method="Contact++")
    n_cc, n_clots, n_contacts, contact_surface = find_contact_areas(fs, image_size, is_periodic=True, contact_distance=0.1*R)
    print("connected components: ", n_cc, "clots: ", n_clots, " contacts: ", n_contacts,
          " contact surface: ", contact_surface)

    io.save_fibers_as_tif(
        fs, domain=image_size, path="examples/outputs/AJ_model.tif", is_periodic=True
    )
    print_fiber_positions_to_file(fs, "examples/outputs/fibers.txt")


def example_AJ_endless():
    print("This is the Altendorf-Jeulin model for endless fibers")
    image_size = np.array([200, 200, 200])
    boundary_size = 50
    VV = 0.12
    R = 8.5 #np.random.normal(loc=8.5, scale=1.0)
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
        has_beta=False,
        seed = 42
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    boundary_size = 100
    start_time = time.time()
    run_force_biased(fs, image_size, is_periodic=False, verbose=True, softcore_ratio=0.1, boundary_size=boundary_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    n_cc, n_clots, n_contacts, contact_surface = find_contact_areas(fs, image_size, is_periodic = False,
                                                                    contact_distance=0.5, boundary_size=boundary_size)
    print("connected components: ", n_cc, "clots: ", n_clots, " contacts: ", n_contacts,
          " contact surface: ", contact_surface)
    shortlist = find_contact_candidates(fs, image_size, is_periodic=False, interaction_distance=8.5,
                                        boundary_size=boundary_size)
    print("shortlist has ", len(shortlist), " elements")
    run_force_biased(
        fs,
        image_size,
        is_periodic=False,
        verbose=True,
        shortlist=shortlist,
        softcore_ratio=0.1,
        contact_distance=0.5,
        boundary_size=boundary_size,
        method="Contact++",
    )
    n_cc, n_clots, n_contacts, contact_surface = find_contact_areas(fs, image_size, is_periodic=False,
                                                                    contact_distance=0.5, boundary_size=boundary_size)
    print("connected components: ", n_cc, "clots: ", n_clots, " contacts: ", n_contacts,
          " contact surface: ", contact_surface)

    io.save_fibers_as_tif(
        fs,
        scale=4,
        domain=image_size,
        boundary=(boundary_size, boundary_size, boundary_size),
        path="examples/outputs/AJ_model_endless.tif",
        is_periodic=False,
    )
    fs_cut = cut_border(fs, image_size, boundary_size)
    io.save_fibers_as_small_graph("examples/outputs/nonwoven", fs_cut)


if __name__ == "__main__":
    main()
