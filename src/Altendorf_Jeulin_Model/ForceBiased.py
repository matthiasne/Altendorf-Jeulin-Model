# cython: language_level=3, infer_type=True
import cython
import numpy as np
from Altendorf_Jeulin_Model.CalculateForces import (
    apply_forces,
    calculate_forces,
    calculate_forces_endstep,
)

import Altendorf_Jeulin_Model.Fiber as Fiber
import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.io_utils import print_stats, print_stats_row, write_gad

MAX_STEPS = 1500
MAX_OVERLAP = 0.1
BOUNDARY_SIZE = 100


def run_force_biased(
    fs: list[Fiber],
    image_size,
    use_end_step_radius: bool = False,
    use_end_step_repulsion: bool = False,
    output_path: str = "examples/outputs/",
    verbose: bool = False,
    step_size_verbose: int = 100,
    is_periodic: bool = True,
):
    """
    Run the force-biased packing by Altendorf & Jeulin, using the original end criteria

    :param fs: list[Fiber]
        the fiber system to be packed
    :param image_size: tuple[int, int, int]     the image size/domain to be modeled on
    :param use_end_step_radius: bool            if necessary, reduces the radius to remove intersections at the end
    :param use_end_step_repulsion: bool         if necessary, applies repulsion force to remove intersections at the end
    :param output_path: str                     file path to store packing step statistics and intermediate fiber
                                                configuration
    :param verbose: bool                        true: output information on packing statistics and intermediate fiber
                                                configuration
    :param step_size_verbose: int               number of steps after which statistics and intermediate fiber
                                                configuration are saved
    :param is_periodic: bool                    uses periodic boundary conditions
    """
    rows = []

    max_radius = max(fiber.get_max_radius() for fiber in fs)
    min_radius = min(fiber.get_max_radius() for fiber in fs)

    boundary_size_vec = np.array([BOUNDARY_SIZE, BOUNDARY_SIZE, BOUNDARY_SIZE])
    if not is_periodic:
        image_size = image_size + 2 * boundary_size_vec
    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    force_strength, overlap, neighbor_dist, angle_diff = calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic
    )
    if verbose:
        rows.append(print_stats_row(fs, 0, force_strength, overlap, neighbor_dist))
    print("We run the force-biased algorithm:")
    end_force_biased = 0.002 * max(image_size) * len(fs)
    eps = np.finfo(float).eps
    for i in range(1, MAX_STEPS):
        if force_strength < end_force_biased and overlap < eps:
            break
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        force_strength, overlap, neighbor_dist, angle_diff = calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic
        )
        if verbose and i % step_size_verbose == 0:
            rows.append(print_stats_row(fs, i, force_strength, overlap, neighbor_dist))
            print_stats(output_path + "results.csv", rows)
            write_gad(
                fs,
                output_path + "model.gad",
                image_size,
                is_periodic=is_periodic,
            )
    if use_end_step_radius and is_periodic:
        end_step_radius(fs, overlap, MAX_OVERLAP * min_radius)
    if use_end_step_repulsion:
        end_step_repulsion(fs, max_radius, overlap, image_size)

    if verbose:
        rows.append(print_stats_row(fs, i, force_strength, overlap, neighbor_dist))
        print_stats(output_path + "results.csv", rows)
        write_gad(
            fs,
            output_path + "model.gad",
            image_size,
            is_periodic=is_periodic,
        )


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
