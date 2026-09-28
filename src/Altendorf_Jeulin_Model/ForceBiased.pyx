# cython: language_level=3, infer_type=True
import cython
import numpy as np
from Altendorf_Jeulin_Model.CalculateForces import (
    apply_forces,
    calculate_forces,
    calculate_forces_endstep
)

import Altendorf_Jeulin_Model.Fiber as Fiber
import Altendorf_Jeulin_Model.SpatialHashing as sh

from Altendorf_Jeulin_Model.ContactModel import test_in_contact
from Altendorf_Jeulin_Model.io_utils import print_stats, print_stats_row


def run_force_biased(
    fs: list[Fiber],
    image_size,
    use_end_step_radius: bool = False,
    use_end_step_repulsion: bool = False,
    output_file: str = "results.csv",
    verbose: bool = True,
    is_periodic: bool = True,
    shortlist = [],
    hardcore_ratio: float = 1.0,
    contact_distance: float = 1.0,
    method: str = "AJ++",
    boundary_size: int = 0,
    output_step_size: int = 100,
    max_steps: int = 1000,
    stop_criterion: Callable[[float], bool] = default_stop_criterion,
):
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    if not is_periodic:
        image_size = image_size + 2 * boundary_size_vec
    if method == "AJ":
        run_force_biased_AJ(fs, image_size, use_end_step_radius, use_end_step_repulsion, output_file, verbose,
                            is_periodic, hardcore_ratio, output_step_size, max_steps)
    elif method == "AJ++":
        run_force_biased_AJpp(fs, image_size, output_file, verbose, is_periodic, hardcore_ratio = hardcore_ratio,
                              output_step_size=output_step_size, max_steps=max_steps)
    elif method == "AJX":
        run_force_biased_AJX(fs, image_size, output_file, verbose, is_periodic, hardcore_ratio = hardcore_ratio,
                              output_step_size=output_step_size, max_steps=max_steps)
    elif method == "Contact++":
        run_force_biased_Contactpp(fs, image_size, output_file, verbose, is_periodic, shortlist, hardcore_ratio,
                              contact_distance, output_step_size, max_steps)
    else:
        run_force_biased_general(fs, image_size, stop_criterion, output_file, verbose, is_periodic, hardcore_ratio = hardcore_ratio,
                              output_step_size=output_step_size, max_steps=max_steps)

def run_force_biased_AJ(
    fs: list[Fiber],
    image_size,
    use_end_step_radius: bool = False,
    use_end_step_repulsion: bool = False,
    output_file: str = "results.csv",
    verbose: bool = True,
    is_periodic: bool = True,
    hardcore_ratio: float = 1.0,
    output_step_size: int = 100,
    max_steps: int = 1000,
):
    """
    Run the force-biased packing by Altendorf & Jeulin, using the original end criteria

    :param fs: list[Fiber]
        the fiber system to be packed
    :param image_size: tuple[int, int, int]     the image size/domain to be modeled on
    :param use_end_step_radius: bool            if necessary, reduces the radius to remove intersections at the end
    :param use_end_step_repulsion: bool         if necessary, applies repulsion force to remove intersections at the end
    :param output_file: str                     file path to store packing step statistics
    :param verbose: bool                        true: output information on packing statistics
    :param is_periodic: bool                    uses periodic boundary conditions
    :param hardcore_ratio: float                hardcore ratio percentage of fiber core that must not be intersected
    """
    rows = []

    max_radius = np.max([fiber.get_max_radius() for fiber in fs])
    min_radius = np.min([fiber.get_min_radius() for fiber in fs])

    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, contact_distances, optim_sum\
        = calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio, repulsion_factor=1
    )
    if verbose:
        rows.append(print_stats_row(fs, 0, total_force_strength, overlap, neighbor_dist, contact_distances))
    print("We run the force-biased algorithm:")
    end_force_biased = 0.002 * max(image_size) * len(fs)

    for i in range(1, max_steps):
        if total_force_strength < end_force_biased and overlap < 0.1 * min_radius:
            break

        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, contact_distances = calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
            repulsion_factor=1
        )
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, total_force_strength, overlap, neighbor_dist, contact_distances))
    if use_end_step_radius and is_periodic:
        end_step_radius(fs, overlap, 0.1 * min_radius)
    if use_end_step_repulsion:
        end_step_repulsion(fs, max_radius, overlap, image_size)

    if verbose:
        rows.append(print_stats_row(fs, i, total_force_strength, overlap, neighbor_dist, contact_distances))
        print_stats(output_file, rows)


def run_force_biased_AJpp(
    fs: list[Fiber],
    image_size,
    output_file: str = "results.csv",
    verbose: bool = True,
    is_periodic: bool = True,
    hardcore_ratio: float = 1.0,
    output_step_size: int = 100,
    max_steps: int = 1000,
):
    """
    Run the force-biased packing by Altendorf & Jeulin, using the original end criteria

    :param fs: list[Fiber]
        the fiber system to be packed
    :param image_size: tuple[int, int, int]     the image size/domain to be modeled on
    :param use_end_step_radius: bool            if necessary, reduces the radius to remove intersections at the end
    :param use_end_step_repulsion: bool         if necessary, applies repulsion force to remove intersections at the end
    :param output_file: str                     file path to store packing step statistics
    :param verbose: bool                        true: output information on packing statistics
    :param is_periodic: bool                    uses periodic boundary conditions
    :param has_beta: bool                       uses the Schladitz distribution with parameter beta for the direction
                                                distribution, otherwise the ACG distribution with parameter matrix is used
    :param beta: float                          parameter of direction distribution
    """
    rows = []

    max_radius = max(fiber.get_max_radius() for fiber in fs)

    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum =\
        calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
    )
    if verbose:
        rows.append(print_stats_row(fs, 0, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum))
    eps = np.finfo(float).eps
    end_force_biased = 0.1*max_radius

    for i in range(1, max_steps):
        if max_force_strength < end_force_biased and overlap < eps:
            break
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
            = calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio
        )
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                        shortlist_distance_sum, optim_sum))

    if verbose:
        rows.append(print_stats_row(fs, i, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum))
        print_stats(output_file, rows)


def run_force_biased_AJX(
    fs: list[Fiber],
    image_size,
    output_file: str = "results.csv",
    verbose: bool = True,
    is_periodic: bool = True,
    hardcore_ratio: float = 1.0,
    output_step_size: int = 100,
    max_steps: int = 1000,
):
    """
    Run the force-biased packing by Altendorf & Jeulin, using the original end criteria

    :param fs: list[Fiber]
        the fiber system to be packed
    :param image_size: tuple[int, int, int]     the image size/domain to be modeled on
    :param use_end_step_radius: bool            if necessary, reduces the radius to remove intersections at the end
    :param use_end_step_repulsion: bool         if necessary, applies repulsion force to remove intersections at the end
    :param output_file: str                     file path to store packing step statistics
    :param verbose: bool                        true: output information on packing statistics
    :param is_periodic: bool                    uses periodic boundary conditions
    :param has_beta: bool                       uses the Schladitz distribution with parameter beta for the direction
                                                distribution, otherwise the ACG distribution with parameter matrix is used
    :param beta: float                          parameter of direction distribution
    """
    rows = []

    max_radius = max(fiber.get_max_radius() for fiber in fs)

    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum =\
        calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
    )
    if verbose:
        rows.append(print_stats_row(fs, 0, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum))
    eps = np.finfo(float).eps
    end_force_biased = 1e-6

    for i in range(1, max_steps):
        if max_force_strength < end_force_biased and overlap < eps:
            break
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
            = calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio
        )
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                        shortlist_distance_sum, optim_sum))

    if verbose:
        rows.append(print_stats_row(fs, i, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum))
        print_stats(output_file, rows)

def run_force_biased_general(
    fs: list[Fiber],
    image_size,
    stop_criterion: Callable[[float], bool],
    output_file: str = "results.csv",
    verbose: bool = True,
    is_periodic: bool = True,
    hardcore_ratio: float = 1.0,
    output_step_size: int = 100,
    max_steps: int = 1000,
):
    """
    Run the force-biased packing by Altendorf & Jeulin, using the original end criteria

    :param fs: list[Fiber]
        the fiber system to be packed
    :param image_size: tuple[int, int, int]     the image size/domain to be modeled on
    :param use_end_step_radius: bool            if necessary, reduces the radius to remove intersections at the end
    :param use_end_step_repulsion: bool         if necessary, applies repulsion force to remove intersections at the end
    :param output_file: str                     file path to store packing step statistics
    :param verbose: bool                        true: output information on packing statistics
    :param is_periodic: bool                    uses periodic boundary conditions
    :param has_beta: bool                       uses the Schladitz distribution with parameter beta for the direction
                                                distribution, otherwise the ACG distribution with parameter matrix is used
    :param beta: float                          parameter of direction distribution
    """
    rows = []

    max_radius = max(fiber.get_max_radius() for fiber in fs)

    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum =\
        calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
    )
    if verbose:
        rows.append(print_stats_row(fs, 0, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum))
    eps = np.finfo(float).eps
    end_force_biased = 0.1*max_radius

    for i in range(1, max_steps):
        if not stop_criterion(overlap, eps, total_force_strength, max_force_strength, shortlist_distance_sum,
                              end_force_biased):
            break
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
            = calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio
        )
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                        shortlist_distance_sum, optim_sum))

    if verbose:
        rows.append(print_stats_row(fs, i, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum))
        print_stats(output_file, rows)





def run_force_biased_Contactpp(
    fs: list[Fiber],
    image_size,
    output_file: str = "results.csv",
    verbose: bool = True,
    is_periodic: bool = True,
    shortlist = [],
    hardcore_ratio: float = 1.0,
    contact_distance: float = 1.0,
    output_step_size: int = 100,
    max_steps: int = 1000,
):
    """
    Run the force-biased packing by Altendorf & Jeulin, using the original end criteria

    :param fs: list[Fiber]
        the fiber system to be packed
    :param image_size: tuple[int, int, int]     the image size/domain to be modeled on
    :param use_end_step_radius: bool            if necessary, reduces the radius to remove intersections at the end
    :param use_end_step_repulsion: bool         if necessary, applies repulsion force to remove intersections at the end
    :param output_file: str                     file path to store packing step statistics
    :param verbose: bool                        true: output information on packing statistics
    :param is_periodic: bool                    uses periodic boundary conditions
    :param has_beta: bool                       uses the Schladitz distribution with parameter beta for the direction
                                                distribution, otherwise the ACG distribution with parameter matrix is used
    :param beta: float                          parameter of direction distribution
    """
    rows = []

    max_radius = max(fiber.get_max_radius() for fiber in fs)

    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
        = (calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, shortlist=shortlist, hardcore_ratio = hardcore_ratio,
        contact_distance=contact_distance, repulsion_factor=1.05
    ))
    if verbose:
        rows.append(print_stats_row(fs, 0, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum))
    print("We run the force-biased algorithm:")
    eps = np.finfo(float).eps
    end_force_biased = 0.1*max_radius

    for i in range(1, max_steps):
        if (max_force_strength < end_force_biased and overlap < eps):
            break
        if i == int(max_steps/2):
            print("remove tense links")
            tense_links = list()
            for contact_edge in shortlist:
                ball = fs[contact_edge[0][0]].balls[contact_edge[0][1]]
                ball2 = fs[contact_edge[1][0]].balls[contact_edge[1][1]]
                is_in_contact, _ = test_in_contact(ball,
                    ball2,
                    image_size=grid.image_size,
                    is_periodic=is_periodic,
                    contact_distance=contact_distance,
                )
                if not is_in_contact:
                    tense_links.append(contact_edge)
            shortlist = [x for x in shortlist if x not in tense_links]
            print(len(shortlist))

        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum,\
            optim_sum = (calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
            shortlist=shortlist, contact_distance=contact_distance, repulsion_factor=1.05
        ))
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                        shortlist_distance_sum, optim_sum))

    if verbose:
        rows.append(print_stats_row(fs, i, total_force_strength, max_force_strength, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum))
        print_stats(output_file, rows)


def end_step_radius(fs: list[Fiber], overlap: float, max_overlap: float):
    """
    The end step where radii are reduced

    :param fs: list[list[Ball]]
        the fiber system to be packed
    :param overlap: float
        The currently maximal overlap in the fiber system
    :param max_overlap: float
        The maximal overlap that is permitted for the fiber system
    """
    if overlap > max_overlap:  # why not only do this for radii that are too large?
        for fiber in fs:
            for ball in fiber.balls:
                new_radius = ball.radius - ball.overlap
                if new_radius <= 0:
                    raise ValueError("Radius cannot be reduced sufficiently.")
                ball.radius = new_radius
                ball.overlap = 0


def end_step_repulsion(
    fs: list[Fiber], max_radius: float, overlap: float, image_size: tuple[int, int, int]
):
    """
    The end step where only the repulsion force is applied

    :param fs: list[list[Ball]]
        the fiber system to be packed
    :param max_radius: float
        The maximal radius in the fiber system
    :param overlap: float
        The currently maximal overlap in the fiber system
    :param image_size: tuple[int, int, int]
    """
    for fiber in fs:
        for ball in fiber.balls:
            ball.force = np.array([0, 0, 0])
            ball.overlap = 0
    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs)
    while overlap > 0:
        force_strength, overlap = calculate_forces_endstep(grid, fiber_system=fs)
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs)

cdef inline bint default_stop_criterion(double overlap, double eps,
                                double total_force_strength,
                                double max_force_strength, double contact_distances,
                                double end_force_biased):
    if max_force_strength < end_force_biased and overlap < eps:
        return True
    return False