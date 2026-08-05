# cython: language_level=3, infer_type=True, exception_check=False, cdivision=True
import numpy as np
import cython
cimport numpy as np
np.import_array()

import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.Fiber import Ball, Fiber

MIN_REPULSION_DISTANCE = 5
X_S:cython.double = 0.05
X_E:cython.double = 0.1
ALPHA_S:cython.double = 0.1 * np.pi / 180
ALPHA_E:cython.double = 0.2 * np.pi / 180
# factors to balance forces, see Altendorf & Jeulin
TAU:cython.double = 0.25
RHO:cython.double = 0.25

def calculate_forces_vectorized(grid: sh, fiber_system: list[Fiber], is_periodic: bool = True):
    """
    Calculates forces in the fiber system and adds them to corresponding ball

    :param grid: SpatialHashing
        The spatial hashing grid for the model
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :return: np.ndarray
        total force of the fiber system
    """
    cdef int f_idx, b_idx

    coord_array, radius_array, ball_label_array, fiber_label_array = sh_to_numpy(grid)
    total_grid_forces = np.zeros_like(coord_array)
    total_grid_overlaps = np.zeros_like(radius_array, dtype=float)

    # calculate forces for 13 unique directions + same cell
    directions = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, -1), (1, 0, 0), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, 1, 1), (1, 1, -1), (1, -1, 0), (1, -1, 1), (1, -1, -1))
    for di, dj, dk in directions:
        fdirect, freact, odirect, oreact = calculate_repulsion_forces_numpy(
            coord_array, radius_array, ball_label_array, fiber_label_array, grid.image_size, 
            is_periodic=is_periodic, shift=(di, dj, dk)
        )
        total_grid_forces += fdirect
        total_grid_overlaps = np.maximum(total_grid_overlaps, odirect)
        # apply Newton's third law
        if (di, dj, dk) != (0, 0, 0):
            total_grid_forces += freact
            total_grid_overlaps = np.maximum(total_grid_overlaps, oreact)

    Nx, Ny, Nz = grid.division    
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                cell_1d_index = x + (y * Nx) + (z * Nx * Ny)
                cell = grid.cells[cell_1d_index]
                for m, ball in enumerate(cell):
                    ball.force = ball.force + total_grid_forces[x, y, z, m]
                    ball.overlap = max(ball.overlap, total_grid_overlaps[x, y, z, m])

    nb_fibers = len(fiber_system)
    if nb_fibers > 0:
        coord_array, label_array, radius_array, neighbor_distances_array, angle_array = fibers_to_numpy(fiber_system)

        spring_forces, neighbor_distances_array = calculate_spring_force_numpy(
                coord_array, label_array, radius_array, neighbor_distances_array
            )
        angle_forces, alpha_diff_array = calculate_angle_force_numpy(
                coord_array, label_array, angle_array
            )

        for f_idx, fiber in enumerate(fiber_system):
            for b_idx, ball in enumerate(fiber.balls):
                    ball.force = ball.force + spring_forces[f_idx, b_idx]
                    ball.neighbor_dist = neighbor_distances_array[f_idx, b_idx]

                    if b_idx - 1 >= 0 and b_idx + 1 < len(fiber.balls):
                        ball.force = ball.force + angle_forces[f_idx, b_idx - 1]
                        ball.angle_diff = alpha_diff_array[f_idx, b_idx - 1]

    total_force = np.array([0.0, 0.0, 0.0])
    total_overlap = 0
    total_neighbor_dist = 0
    total_angle_diff = 0
    for fiber in fiber_system:
        for ball in fiber.balls:
            total_force = total_force + ball.force
            total_overlap = max(total_overlap, ball.overlap)
            total_neighbor_dist = max(total_neighbor_dist, ball.neighbor_dist)
            total_angle_diff = max(total_angle_diff, abs(ball.angle_diff))
    return (
        np.linalg.norm(total_force),
        total_overlap,
        total_neighbor_dist,
        total_angle_diff,
    )

def calculate_repulsion_forces_numpy(
    np.ndarray coord_array, 
    np.ndarray radius_array,
    np.ndarray ball_label_array,
    np.ndarray fiber_label_array, 
    np.ndarray image_size, 
    is_periodic: bool = True, 
    repulsion_factor: float = 1.1,
    tuple shift = (0, 0, 0)
):
    """
    Calculates the repulsion forces between all balls in the system using numpy broadcasting.
    :param coord_array: np.ndarray
        The coordinates of the balls in the system, shape (x, y, z, max_balls, 3)
    :param radius_array: np.ndarray
        The radii of the balls in the system, shape (x, y, z, max_balls)
    :param ball_label_array: np.ndarray
        The labels of the balls in the system, shape (x, y, z, max_balls)
    :param fiber_label_array: np.ndarray
        The labels of the fibers in the system, shape (x, y, z, max_balls)
    :param image_size: np.ndarray
        The size of the image for periodic boundary conditions, shape (3,)
    :param is_periodic: bool
        Whether to apply periodic boundary conditions
    :param repulsion_factor: float
        The factor to scale the repulsion force
    :param shift: tuple
        The shift to apply to the neighbor cells, default is (0, 0, 0)
    :return: tuple
        fdirect: np.ndarray
            The direct forces on each ball, shape (x, y, z, max_balls, 3)
        freact: np.ndarray
            The reactive forces on each ball, shape (x, y, z, max_balls, 3)
        odirect: np.ndarray
            The direct overlaps for each ball, shape (x, y, z, max_balls)
        oreact: np.ndarray
            The reactive overlaps for each ball, shape (x, y, z, max_balls)
    """
    # declare all variables for a better performance
    cdef tuple[int, int, int] neg_shift
    cdef np.int max_balls
    cdef np.ndarray valid_balls, valid_cells, has_balls_j, active_cells
    cdef np.ndarray coord_active, radius_active, fiber_active, ball_active, valid_balls_active
    cdef np.int Nx, Ny, Nz
    cdef np.ndarray x_indices, y_indices, z_indices
    cdef np.ndarray x_neigh, y_neigh, z_neigh
    cdef np.ndarray coord_j_active, radius_j_active, fiber_shifted, ball_shifted, valid_balls_j_active
    cdef np.ndarray valid_pairs_active, different_fiber, far_enough, valid_interactions
    cdef np.ndarray r_sum, pos_diff, distances_sq, overlaps_sq, is_overlapping
    cdef np.ndarray distances, overlaps_true, overlaps, safe_distances
    cdef np.ndarray compute_mask, border_mask
    cdef int axis, direction
    cdef np.ndarray force_mags, force_vecs, forces
    cdef np.ndarray net_forces_active, max_overlaps_active
    cdef np.ndarray reaction_forces_active, reaction_overlaps_active
    cdef np.ndarray direct_forces, direct_overlaps
    cdef np.ndarray reaction_forces, reaction_overlaps

    neg_shift = (-shift[0], -shift[1], -shift[2])

    max_balls = coord_array.shape[3]
    valid_balls = ball_label_array[..., :] != -1 # -1 are empty cells

    # find cells containing at least one ball
    valid_cells = np.any(valid_balls, axis=-1)
    # find cells such that (pos + shift) contains at least one ball
    has_balls_j = np.roll(valid_cells, shift=neg_shift, axis=(0, 1, 2))
    # cells that contain at least one neighboring ball in the shifted cell
    active_cells = valid_cells & has_balls_j

    # extract active cells only. This should reduce the number of computations significantly
    coord_active = coord_array[active_cells]
    radius_active = radius_array[active_cells]
    fiber_active = fiber_label_array[active_cells]
    ball_active = ball_label_array[active_cells]
    valid_balls_active = valid_balls[active_cells]

    # find corresponding neighbor indices for each active cell
    # we need the exact indices to preserve the order of interacting balls
    Nx, Ny, Nz = coord_array.shape[0], coord_array.shape[1], coord_array.shape[2]
    x_indices, y_indices, z_indices = np.where(active_cells)
    x_neigh = (x_indices + shift[0]) % Nx
    y_neigh = (y_indices + shift[1]) % Ny
    z_neigh = (z_indices + shift[2]) % Nz

    # extract the same properties for the neighboring cells
    coord_j_active = coord_array[x_neigh, y_neigh, z_neigh]
    radius_j_active = radius_array[x_neigh, y_neigh, z_neigh]
    fiber_shifted = fiber_label_array[x_neigh, y_neigh, z_neigh]
    ball_shifted = ball_label_array[x_neigh, y_neigh, z_neigh]
    valid_balls_j_active = valid_balls[x_neigh, y_neigh, z_neigh]

    # (N_active, max_balls, max_balls) indicating if (cell, ball_i, ball_j) is a pair of interacting balls
    valid_pairs_active = valid_balls_active[:, :, np.newaxis] & valid_balls_j_active[:,np.newaxis, :]
    
    # ignore balls from the same fiber if their ball index distance is less than the limit
    different_fiber = fiber_active[:, :, np.newaxis] != fiber_shifted[:, np.newaxis, :]
    far_enough = np.abs(ball_active[:, :, np.newaxis] - ball_shifted[:, np.newaxis, :]) >= MIN_REPULSION_DISTANCE
    valid_interactions = different_fiber | far_enough

    # vector calculations
    r_sum = radius_active[..., :, np.newaxis] + radius_j_active[..., np.newaxis, :] # (x, y, z, max_balls, max_balls)
    pos_diff = coord_active[..., :, np.newaxis, :] - coord_j_active[..., np.newaxis, :, :] # (x, y, z, max_balls, max_balls, 3)
    if is_periodic:
        pos_diff = pos_diff - image_size * np.round(pos_diff / image_size)

    # squared distances are much faster to compute
    distances_sq = np.sum(pos_diff * pos_diff, axis=-1)

    # overlap logic
    overlaps_sq = (repulsion_factor * r_sum) * (repulsion_factor * r_sum) - distances_sq
    is_overlapping = overlaps_sq > 0

    # now we compute the actual distance for the overlapping pairs only
    distances = np.zeros_like(distances_sq)
    distances[is_overlapping] = np.sqrt(distances_sq[is_overlapping])

    # we only need this for overlapping pairs
    overlaps_true = np.zeros_like(distances_sq)
    overlaps_true[is_overlapping] = r_sum[is_overlapping] - distances[is_overlapping]
    overlaps = repulsion_factor * r_sum - distances
    # avoid division by zero for same / very close balls
    safe_distances = np.where(distances < 1e-8, 1.0, distances) # replace zeros

    # ignore self-interactions in case of the same cell
    compute_mask = valid_pairs_active & is_overlapping & valid_interactions
    # ignore self-interactions
    if shift == (0, 0, 0):
        compute_mask = compute_mask & ~np.eye(max_balls, dtype=bool)

    # if it is not periodic, we need to ignore border cells
    if not is_periodic:
        border_mask = np.ones((Nx, Ny, Nz), dtype=bool)
        for axis, direction in enumerate(shift):
            index = [slice(None)] * 3 # [:, :, :]
            if direction == -1:
                index[axis] = 0 # [-1, :, :]
                border_mask[tuple(index)] = False
            if direction == 1:
                index[axis] = -1 # [0, :, :]
                border_mask[tuple(index)] = False
        compute_mask = compute_mask & border_mask[active_cells][:, np.newaxis, np.newaxis]

    # forces
    force_mags = TAU * overlaps / 2.0
    force_vecs = (force_mags / safe_distances)[..., np.newaxis] * pos_diff
    forces = force_vecs * compute_mask[..., np.newaxis]
    
    # collapse direct forces and overlaps
    net_forces_active = np.sum(forces, axis=-2) 
    max_overlaps_active = np.max(overlaps_true * compute_mask, axis=-1)

    # collapse reactive forces and overlaps for the neighbor cells
    reaction_forces_active = np.sum(-forces, axis=-3)
    reaction_overlaps_active = np.max(overlaps_true * compute_mask, axis=-2)

    # recunstruct full arrays
    direct_forces = np.zeros_like(coord_array)
    direct_overlaps = np.zeros_like(radius_array)
    reaction_forces = np.zeros_like(coord_array)
    reaction_overlaps = np.zeros_like(radius_array)

    direct_forces[active_cells] = net_forces_active
    direct_overlaps[active_cells] = max_overlaps_active
    reaction_forces[x_neigh, y_neigh, z_neigh] = reaction_forces_active
    reaction_overlaps[x_neigh, y_neigh, z_neigh] = reaction_overlaps_active

    return direct_forces, reaction_forces, direct_overlaps, reaction_overlaps

def sh_to_numpy(
    grid: sh
    ):
    """
    Converts the SpatialHashing to a tuple (coord_array, radius_array, ball_label_array, fiber_label_array) of numpy arrays

    :return: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The numpy array representations of the SpatialHashing
    """
    max_balls_per_cell = max(len(cell) for cell in grid.cells)
    coord_array = np.zeros((grid.division[0], grid.division[1], grid.division[2], max_balls_per_cell, 3))
    radius_array = np.zeros((grid.division[0], grid.division[1], grid.division[2], max_balls_per_cell), dtype=float)
    ball_label_array = np.full((grid.division[0], grid.division[1], grid.division[2], max_balls_per_cell), -1, dtype=int) # -1 for empty cells
    fiber_label_array = np.full((grid.division[0], grid.division[1], grid.division[2], max_balls_per_cell), -1, dtype=int) # -1 for empty cells

    for idx, cell in enumerate(grid.cells):
        i = idx % grid.division[0]
        j = (idx // grid.division[0]) % grid.division[1]
        k = idx // (grid.division[0] * grid.division[1])
        for b_idx, ball in enumerate(cell):
            coord_array[i, j, k, b_idx] = ball.coordinate
            radius_array[i, j, k, b_idx] = ball.radius
            ball_label_array[i, j, k, b_idx] = ball.ball_label
            fiber_label_array[i, j, k, b_idx] = ball.fiber_label

    return coord_array, radius_array, ball_label_array, fiber_label_array

def fibers_to_numpy(fiber_system: list[Fiber]):
    """
    Converts the fiber system to numpy arrays for coordinates, labels, and radii and neighbor distances.
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :return: tuple
        coord_array: np.ndarray
            The coordinates of the balls in the system, shape (nb_fibers, max_balls, 3)
        label_array: np.ndarray
            The labels of the balls in the system, shape (nb_fibers, max_balls)
        radius_array: np.ndarray
            The radii of the balls in the system, shape (nb_fibers, max_balls)
        neighbor_distances_array: np.ndarray
            The distances to the neighboring balls, shape (nb_fibers, max_balls)
        angle_array: np.ndarray
            The angles of the balls in the system, shape (nb_fibers, max_balls)
    """
    nb_fibers = len(fiber_system)
    max_balls = max(len(fiber.balls) for fiber in fiber_system)
    coord_array = np.zeros((nb_fibers, max_balls, 3))
    label_array = np.full((nb_fibers, max_balls), -1, dtype=int)
    radius_array = np.ones((nb_fibers, max_balls)) # intialize with 1 to avoid divison by zero problems
    neighbor_distances_array = np.full((nb_fibers, max_balls), -1.0)
    angle_array = np.ones((nb_fibers, max_balls)) # initialize with 1 to avoid divison by zero problems
    for i, fiber in enumerate(fiber_system):
        for j, ball in enumerate(fiber.balls):
            coord_array[i, j] = ball.coordinate
            label_array[i, j] = ball.fiber_label
            radius_array[i, j] = ball.radius
            neighbor_distances_array[i, j] = ball.neighbor_dist
            angle_array[i, j] = ball.angle
    return coord_array, label_array, radius_array, neighbor_distances_array, angle_array

def calculate_spring_force_numpy(
    np.ndarray coord_array,
    np.ndarray label_array,
    np.ndarray radius_array,
    np.ndarray neighbor_distances_array,
):
    """
    Calculates the spring forces between neighboring balls in the fiber system using numpy broadcasting.
    :param coord_array: np.ndarray
        The coordinates of the balls in the system, shape (nb_fibers, max_balls, 3)
    :param radius_array: np.ndarray
        The radii of the balls in the system, shape (nb_fibers, max_balls)
    :param label_array: np.ndarray
        The labels of the balls in the system, shape (nb_fibers, max_balls)
    :param neighbor_distances_array: np.ndarray
        The distances to the neighboring balls, shape (nb_fibers, max_balls)
    :return: np.ndarray
        The net spring forces on each ball, shape (nb_fibers, max_balls, 3)
    """
    valid_cells = label_array[..., :] != -1 # -1 are empty cells
    valid_links = valid_cells[:, :-1] & valid_cells[:, 1:]

    forward_diff = coord_array[:, 1:, :] - coord_array[:, :-1, :]
    distances = np.linalg.norm(forward_diff, axis=-1)

    # avoid division by zero in empty cells
    safe_dist = np.where(distances < 1e-8, 1.0, distances)

    # original code always uses the first ball's radius
    correct_dist = radius_array[:, :-1] / 2.0
    displaced = distances - correct_dist
    ratio = np.abs(displaced) / correct_dist

    # vectorized smoothing calculation
    rescale = np.clip((ratio - X_S) / (X_E - X_S), 0, 1)
    smoothing = 0.5 * (1 - np.cos(rescale * np.pi)) * RHO * displaced / safe_dist
    forward_forces = forward_diff * smoothing[..., np.newaxis]

    # accumulate forces in the correct direction
    total_forces = np.zeros_like(coord_array)
    total_forces[:, :-1, :] += forward_forces * valid_links[..., np.newaxis]
    total_forces[:, 1:, :] -= forward_forces * valid_links[..., np.newaxis]

    # only update distances for the existing links
    valid_dist = np.where(valid_links, distances, 0.0)
    neighbor_distances_array[:, :-1] = np.maximum(neighbor_distances_array[:, :-1], valid_dist)
    neighbor_distances_array[:, 1:] = np.maximum(neighbor_distances_array[:, 1:], valid_dist)

    return total_forces, neighbor_distances_array

def calculate_angle_force_numpy(
    np.ndarray coord_array,
    np.ndarray label_array,
    np.ndarray angle_array,
):
    """
    Calculates the angle forces between neighboring balls in the fiber system using numpy broadcasting.
    :param coord_array: np.ndarray
        The coordinates of the balls in the system, shape (nb_fibers, max_balls, 3)
    :param label_array: np.ndarray
        The labels of the balls in the system, shape (nb_fibers, max_balls)
    :param angle_array: np.ndarray
        The angles of the balls in the system, shape (nb_fibers, max_balls)
    :return: np.ndarray
        The net angle forces on each ball, shape (nb_fibers, max_balls, 3)
    """

    valid_cells = label_array[..., :] != -1 # -1 are empty cells
    valid_links = valid_cells[:, :-2] & valid_cells[:, 2:]

    diff_next = coord_array[:, 2:, :] - coord_array[:, 1:-1, :]
    diff_prev = coord_array[:, 1:-1, :] - coord_array[:, :-2, :]
    diff_a = coord_array[:, 2:, :] - coord_array[:, :-2, :]

    norm_next = np.linalg.norm(diff_next, axis=-1)
    safe_norm_next = np.where(norm_next < 1e-8, 1.0, norm_next)
    norm_prev = np.linalg.norm(diff_prev, axis=-1)
    safe_norm_prev = np.where(norm_prev < 1e-8, 1.0, norm_prev)
    norm_a = np.linalg.norm(diff_a, axis=-1)
    safe_norm_a = np.where(norm_a < 1e-8, 1.0, norm_a)

    dir_next = diff_next / safe_norm_next[..., np.newaxis]
    dir_prev = diff_prev / safe_norm_prev[..., np.newaxis]
    a = diff_a / safe_norm_a[..., np.newaxis]    

    d = np.einsum('ijk,ijk->ij', diff_prev, a)
    m = coord_array[:, :-2, :] + d[..., np.newaxis] * a
    alpha = np.pi - np.arccos(np.einsum('ijk,ijk->ij', dir_prev, dir_next))

    h1 = np.abs(d)
    h2 = np.linalg.norm(m - coord_array[:, 2:, :], axis=-1)
    z = np.linalg.norm(m - coord_array[:, 1:-1, :], axis=-1)
    # this can be zero for collinear points
    safe_z = np.where(z < 1e-8, 1.0, z)

    tan_alpha0 = np.tan(angle_array[:, 1:-1])
    z0 = (h1 + h2  + (np.sqrt(np.square((h1 + h2)) + 4 * h1 * h2 * np.square(tan_alpha0)) * np.sign(tan_alpha0))) / (2 * tan_alpha0)

    alpha_diff = angle_array[:, 1:-1] - alpha
    rescale = np.clip((alpha_diff - ALPHA_S) / (ALPHA_E - ALPHA_S), 0, 1)
    smoothing = 0.5 * (1 - np.cos(rescale * np.pi)) / safe_z * RHO * (z - z0) / 2.0

    total_forces = (m - coord_array[:, 1:-1, :]) * smoothing[..., np.newaxis]
    total_forces = total_forces * valid_links[..., np.newaxis]
    
    return total_forces, alpha_diff
