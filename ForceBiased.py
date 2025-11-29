import SpatialHashing as sh
from Fiber import Ball
from CalculateForces import calculate_forces, apply_forces

MAX_STEPS = 10000

def run_force_biased(fs:list[list[Ball]], image_size: tuple[int, int, int]):
    # calculate maximal radius TODO: use implementation from Fiber
    max_radius = max(ball.radius for fiber in fs for ball in fiber)
    min_radius = min(ball.radius for fiber in fs for ball in fiber)

    # TODO: displacement similar to RSA
    grid = sh.SpatialHashing(image_size, 2.5*max_radius)
    grid.add_fiber_system(fs)
    force_strength, overlap = calculate_forces(grid, fiber_system=fs)

    end_force_biased = 0.002*max(image_size)*len(fs)
    for i in range(MAX_STEPS):
        if force_strength < end_force_biased and overlap < 0.1*min_radius:
            break
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
        grid.add_fiber_system(fs)
        force_strength, overlap = calculate_forces(grid, fiber_system=fs)



    # TODO: end step(s)
