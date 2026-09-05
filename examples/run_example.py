import time
import scipy

import Altendorf_Jeulin_Model.FiberModel as fm
import numpy as np
from Altendorf_Jeulin_Model.utils import cut_border
from Altendorf_Jeulin_Model.rv_utils import schladitz_distribution

import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.ForceBiased import run_force_biased
from Altendorf_Jeulin_Model.io_utils import (
    print_fiber_positions_to_file,
)
from Altendorf_Jeulin_Model.ContactModel import (
    find_contact_areas,
    find_contact_candidates
)
from Altendorf_Jeulin_Model.Statistics import volume_fraction


def main():
    #main_example()
    example_AJ_finite()

def main_example():
    print("This is the most basic example to generate finite fibers allowing intersection.\n"
          "We use a Schladitz distribution with parameter beta and save the result in an image.")
    image_size = np.array([150, 150, 150])
    intensity = 90
    L = 100
    R = 5
    beta = 3.0
    fs = fm.initialize_fiber_system(
        intensity, L, R, beta, image_size, 10, 100, seed = 42
    )
    io.save_fibers_as_tif(
        fs, domain=image_size, path="examples/outputs/AJ_model_intersect.tif", is_periodic=True
    )

    print("Next up, we pack the fibers to remove intersections. You can choose between the original force-biased\n"
          "packing by Altendorf-Jeulin, called 'AJ', and the tuned version, called 'AJ++'. We recommend using the\n"
          "tuned version, please refer to our paper for the explanation.\n"
          "\n"
          "If you have core-sheath fibers, you can allow for intersections corresponding to the percentage of the\n"
          "sheath by setting the parameter hardcore_ratio. It's default is 1.0 (no sheath). Another reason for such\n"
          "a choice may be a low resolution, which is why your image could not resolve this overlap clearly anyway.\n"
          "On the other hand, you may ant to ensure a minimum distance between fibers, e.g., for avoiding stress\n"
          "peaks. In this case, please choose the hardcore_ratio larger 1 according to the ratio of voxel size\n"
          "and radius. The following output provides statistics on the packing procedure and result.")
    run_force_biased(fs, image_size)
    io.save_fibers_as_tif(
        fs,
        domain=image_size,
        path="examples/outputs/AJ_model_nointersect.tif",
        is_periodic=True,
    )

    n_cc, n_clots, n_contacts, contact_surface = find_contact_areas(
        fs, image_size, is_periodic=True, contact_distance=1
    )
    print(
        "As mentioned above, images may not be able to resolve a distance lower than the voxel size. Therefore, we\n"
        "consider a distance between fibers lower than the voxel size (1um in this case) to be a contact area.\n"
        "This system has ",
        n_contacts,
        " inter-fiber contacts. Their surface is approximated by the length of fibers that touch,\nwhich amounts to ",
        contact_surface, "um.\n "
        "We can increase the amount of contact using the contact packing. For this, we draw in fibers when they are\n"
        "closer than the fiber radius."
    )
    shortlist = find_contact_candidates(
        fs, image_size, is_periodic=True, interaction_distance= R
    )
    run_force_biased(
        fs,
        image_size,
        verbose=True,
        shortlist=shortlist,
        hardcore_ratio=1.0,
        contact_distance=1,
        method="Contact++",
    )
    n_cc, n_clots, n_contacts, contact_surface = find_contact_areas(
        fs, image_size, is_periodic=True, contact_distance=1
    )
    print(
        "This system has ",
        n_contacts,
        " inter-fiber contacts and a contact 'length' of ",
        contact_surface, "um."
    )
    io.save_fibers_as_tif(
        fs,
        domain=image_size,
        path="examples/outputs/AJ_model_increasecontact.tif",
        is_periodic=True,
    )

    print("The code allows for plenty of customization, e.g. nonperiodic boundaries, different stop criteria,\n"
          "or different fiber models. We also offer an endless fiber model and a Poisson line process. Please\n"
          "don't hesitate to reach out to us if you miss any functionality.")

def direction_distribution(rng):
    beta = 3.0
    return schladitz_distribution(beta, rng)

def example_AJ_finite():
    print("This is the Altendorf-Jeulin model")
    image_size = np.array([384, 384, 384])
    N = 6000
    L = 200
    R = 2

    # create a fiber system
    start_time = time.time()
    fs = fm.initialize_fiber_system(
        N, L, R, direction_distribution, image_size, 100, 100, seed=42
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fiber initialization - Elapsed time: {elapsed_time:.6f} seconds")
    start_time = time.time()
    vf = volume_fraction(fs, image_size, True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"volume fraction from image {vf:.3f} - elapsed time: {elapsed_time:.6f} seconds")

    # pack the fibers
    """start_time = time.time()
    run_force_biased(fs, image_size, verbose=True, hardcore_ratio = 1.0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Packing - Elapsed time: {elapsed_time:.6f} seconds")

    n_cc, n_clots, n_contacts, contact_surface = find_contact_areas(fs, image_size, is_periodic=True, contact_distance= 0.1*R)
    print("connected components: ", n_cc, "clots: ", n_clots, " contacts: ", n_contacts,
          " contact surface: ", contact_surface)
    shortlist = find_contact_candidates(fs, image_size, is_periodic = True, interaction_distance=0.2*R)
    print("shortlist has ", len(shortlist), " elements")
    run_force_biased(fs, image_size, verbose=True, shortlist=shortlist, hardcore_ratio=1.0, contact_distance=0.1*R,
                     method="Contact++")
    n_cc, n_clots, n_contacts, contact_surface = find_contact_areas(fs, image_size, is_periodic=True, contact_distance=0.1*R)
    print("connected components: ", n_cc, "clots: ", n_clots, " contacts: ", n_contacts,
          " contact surface: ", contact_surface)
    """
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
    run_force_biased(fs, image_size, is_periodic=False, verbose=True, hardcore_ratio=0.9, boundary_size=boundary_size)
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
        hardcore_ratio=0.9,
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
