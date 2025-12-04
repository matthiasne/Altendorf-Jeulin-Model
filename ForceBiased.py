import SpatialHashing as sh
import numpy as np
from Fiber import Ball
from CalculateForces import calculate_forces, apply_forces, calculate_forces_endstep

MAX_STEPS = 10000
MAX_OVERLAP = 0.1


def run_force_biased(fs: list[list[Ball]], image_size: tuple[int, int, int],
                     use_end_step_radius: bool = False, use_end_step_repulsion: bool = False):
    """
    Run the force-biased packing by Altendorf & Jeulin, using the original end criteria

    Attributes
    ---------------------
    :param fs: list[list[Ball]]
        the fiber system to be packed
    :param image_size: tuple[int, int, int]
    """
    # calculate maximal radius TODO: use implementation from Fiber
    max_radius = max(ball.radius for fiber in fs for ball in fiber)
    min_radius = min(ball.radius for fiber in fs for ball in fiber)

    # TODO: displacement similar to RSA
    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs)
    force_strength, overlap = calculate_forces(grid, fiber_system=fs)

    end_force_biased = 0.002 * max(image_size) * len(fs)
    for i in range(MAX_STEPS):
        if force_strength < end_force_biased and overlap < 0.1 * min_radius:
            break
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs)
        force_strength, overlap = calculate_forces(grid, fiber_system=fs)

    if use_end_step_radius:
        end_step_radius(fs, overlap, MAX_OVERLAP*min_radius)
    if use_end_step_repulsion:
        end_step_repulsion(fs, max_radius, overlap, image_size)


def end_step_radius(fs:list[list[Ball]], overlap: float, max_overlap: float):
    """
    The end step where radii are reduced

    Attributes
    ---------------------
    :param fs: list[list[Ball]]
        the fiber system to be packed
    :param overlap: float
        The currently maximal overlap in the fiber system
    :param max_overlap: float
        The maximal overlap that is permitted for the fiber system
    """
    if overlap > max_overlap: # why not only do this for radii that are too large?
        for fiber in fs:
            for ball in fiber:
                new_radius = ball.radius - ball.overlap
                if new_radius <= 0:
                    raise ValueError("Radius cannot be reduced sufficiently.")
                ball.radius = new_radius
                ball.overlap = 0

def end_step_repulsion(fs:list[list[Ball]], max_radius: float,
                       overlap: float, image_size: tuple[int, int, int]):
    """
    The end step where only the repulsion force is applied

    Attributes
    ---------------------
    :param fs: list[list[Ball]]
        the fiber system to be packed
    :param max_radius: float
        The maximal radius in the fiber system
    :param overlap: float
        The currently maximal overlap in the fiber system
    :param image_size: tuple[int, int, int]
    """
    for fiber in fs:
        for ball in fiber:
            ball.force = np.array([0, 0, 0])
            ball.overlap = 0
    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs)
    while overlap > 0:
        force_strength, overlap = calculate_forces_endstep(grid, fiber_system=fs)
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs)
