# cython: language_level=3, infer_type=True, exception_check=False, cdivision=True
import numpy as np
import cython
from libc.stdint cimport int64_t
from itertools import product
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

def calculate_forces(grid: sh, fiber_system: list[Fiber], is_periodic: bool = True):
    """
    Calculates forces in the fiber system and adds them to corresponding ball

    :param grid: SpatialHashing
        The spatial hashing grid for the model
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :return: np.ndarray
        total force of the fiber system
    """
    # for cell in grid.cells:
    #     if len(cell) > 0:
    #         neighbor_cells = grid.get_younger_neighbor_cell_indices(
    #             grid.get_cell_index_of_coord(cell[0].coordinate)
    #         )
    #         for i, ball in enumerate(cell):
    #             calculate_repulsion_forces(
    #                 i, ball, cell, grid, neighbor_cells, is_periodic=is_periodic
    #             )

    # Numpy version of the above code

    coord_array, label_array, radius_array = grid.to_numpy_array()
    total_grid_forces = np.zeros_like(coord_array)
    total_grid_overlaps = np.zeros_like(radius_array, dtype=float)

    # calculate forces for 13 unique directions + same cell
    directions = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, -1), (1, 0, 0), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, 1, 1), (1, 1, -1), (1, -1, 0), (1, -1, 1), (1, -1, -1))
    for di, dj, dk in directions:
        np_shift = (-di, -dj, -dk)
        f, o = calculate_repulsion_forces_numpy(
            coord_array, label_array, radius_array, grid.image_size, 
            is_periodic=is_periodic, shift=np_shift
        )
        total_grid_forces += f
        # apply Newton's third law
        if (di, dj, dk) != (0, 0, 0):
            total_grid_forces -= np.roll(f, shift=(di, dj, dk), axis=(0, 1, 2))
            total_grid_overlaps = np.maximum(total_grid_overlaps, np.roll(o, shift=(di, dj, dk), axis=(0, 1, 2)))
        total_grid_overlaps = np.maximum(total_grid_overlaps, o)

    # TODO : This can be optimized by storing balls / forces as arrays directly
    Nx, Ny, Nz = grid.division    
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                cell_1d_index = x + (y * Nx) + (z * Nx * Ny)
                cell = grid.cells[cell_1d_index]
                for m, ball in enumerate(cell):
                    ball.force = ball.force + total_grid_forces[x, y, z, m]
                    ball.overlap = max(ball.overlap, total_grid_overlaps[x, y, z, m])

    # for fiber in fiber_system:
    #     for i, ball in enumerate(fiber.balls):
    #         if i + 1 < len(fiber.balls):
    #             calculate_spring_force(ball, fiber.balls[i + 1], is_next=True)
    #         if i - 1 >= 0:
    #             calculate_spring_force(ball, fiber.balls[i - 1], is_next=False)
    #         if i - 1 >= 0 and i + 1 < len(fiber.balls):
    #             calculate_angle_force(ball, fiber.balls[i - 1], fiber.balls[i + 1])

    # Numpy version of the above code

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


def calculate_forces_endstep(
    grid: sh, fiber_system: list[Fiber], is_periodic: bool = True
):
    """
    Calculates forces in the fiber system and adds them to corresponding ball

    :param grid: SpatialHashing
        The spatial hashing grid for the model
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :return: np.ndarray
        total force of the fiber system
    """

    for cell in grid.cells:
        for i, ball in enumerate(cell):
            calculate_repulsion_forces(i, ball, cell, grid, is_periodic=is_periodic)

    total_force = np.array([0.0, 0.0, 0.0])
    total_overlap = 0
    for fiber in fiber_system:
        for ball in fiber.balls:
            total_force = total_force + ball.force
            total_overlap = max(total_overlap, ball.overlap)
    return np.linalg.norm(total_force), total_overlap


def calculate_repulsion_forces(
    i: cython.int,
    ball: Ball,
    cell: list[Ball],
    grid: sh,
    neighbor_cells,
    is_periodic: bool = True,
):
    """
    Calculates the repulsion force for the whole fiber system
    and adds it to corresponding ball

    :param i: int
        cell index of the current cell
    :param ball: Ball
        The ball whose neighbors are currently considered
    :param cell: list[Ball]
        The cell that the ball is saved in
    :param grid: SpatialHashing
        The spatial hashing grid of the model
    """
    fiber_label: cython.int = ball.fiber_label
    label: cython.int = ball.ball_label
    coord = ball.coordinate
    # compare within cell
    for ball2 in cell[i + 1 :]:
        calculate_repulsion_force(
            ball, ball2, fiber_label, label, is_periodic, coord, grid.image_size
        )
    # compare with neighbor cells
    for cell_index in neighbor_cells:
        cell = grid.cells[cell_index]
        for ball2 in cell:
            calculate_repulsion_force(
                ball, ball2, fiber_label, label, is_periodic, coord, grid.image_size
            )


cdef calculate_repulsion_force(
    ball, ball2, fiber_label: int, label: int, is_periodic: bool, double[:] coord,int64_t[:] image_size,
    repulsion_factor: float = 1.1
):
    """
    calculates the repulsion force between two balls

    :param ball: Ball
        The ball whose neighbors are currently considered
    :param ball2: Ball
        The neighboring ball that is currently considered
    :param fiber_label: int
        The fiber label of ball
    :param label: int
        The ball label of ball
    :param is_periodic: bool
        Whether the repulsion force is to be calculated on the torus, i.e., periodically
    :param coord: double
        The coordinate of ball
    :param image_size: int64_t
        The image size (relevant for periodic case)
    :param repulsion_factor: float, default 1.1
        This factor is 1 in the Altendorf-Jeulin model.
        However, this leads to incredibly low convergence (explainable with limit of explicit Euler?),
        which is also why they stop packing when the overlap is 0.1*radius and then need an end_step
        A factor of 1.1 turned out as trade-off between runtime and highest volume fraction
        TODO: add enforced distance as in contact model or fSAM, which may be relevant when voxelizing fiber system
    """
    if (
        fiber_label != ball2.fiber_label
        or abs(label - ball2.ball_label) >= MIN_REPULSION_DISTANCE
    ):
        if is_periodic:
            # calculate periodic distance of the balls' coordinates
            coord2mod = np.mod(ball2.coordinate, image_size)
            disp: cython.double
            for i in range(3):
                disp = coord2mod[i] - coord[i]
                if abs(disp) > image_size[i] / 2.0:
                    if disp > 0:
                        coord2mod[i] -= image_size[i]
                    else:
                        coord2mod[i] += image_size[i]
                coord2mod[i] -= coord[i]
            dist: cython.double = np.linalg.norm(coord2mod)

            # calculate the force if balls are indeed overlapping
            overlap: cython.double = ball.radius + ball2.radius
            overlap_true: cython.double = overlap - dist
            overlap = repulsion_factor*overlap - dist
            if overlap > 0:
                coord2mod = coord2mod / dist
                force = TAU * overlap / 2.0 * coord2mod
                ball.force = ball.force - force
                ball.overlap = max(ball.overlap, overlap_true)
                ball2.force = ball2.force + force
                ball2.overlap = max(ball2.overlap, overlap_true)

        else:
            coord2 = ball2.coordinate
            dist: cython.double = np.linalg.norm(coord2 - coord)
            overlap: cython.double = ball.radius + ball2.radius
            overlap_true: cython.double = overlap - dist
            overlap = repulsion_factor*overlap - dist
            if overlap > 0:
                dir = (coord2 - coord)/dist
                ball.force = ball.force - TAU * overlap / 2.0 * dir
                ball.overlap = max(ball.overlap, overlap_true)
                ball2.force = ball2.force + TAU * overlap / 2.0 * dir
                ball2.overlap = max(ball2.overlap, overlap_true)


cdef double smoothing_factor(x: cython.double, x_s: cython.double, x_e: cython.double):
    """
    Calculate the smoothing factor
    (arguments named after Altendorf&Jeulin 2011)

    :param x: float
        The ratio that is the argument of smoothing factor
    :param x_s: float
        if x < x_y, the factor is 0
    :param x_e: float
        if x > x_e, the factor is 1
    :return: float
        the smoothing factor
    """
    if x < x_s:
        return 0
    elif x > x_e:
        return 1
    else:
        ratio: cython.double = (x - x_s) / (x_e - x_s)
        factor: cython.double = 0.5 * (1 - np.cos(ratio * np.pi))
        return factor


cdef calculate_spring_force(ball1: Ball, ball2: Ball, is_next: bool):
    """
    Calculates the spring force between 2 balls and adds it to corresponding balls

    :param ball1: Ball
        the ball that the force is added to
    :param ball2: Ball
        neighbor to ball1
    :param is_next: bool
        indicates whether ball2 comes before or after ball1 in the fiber
    """
    # displacement
    force_dir = ball2.coordinate - ball1.coordinate
    dist_is: cython.double = np.linalg.norm(force_dir)

    # distance to the next ball is currently always radius/2.0
    # - may need to adapt for different random walks
    dist_should = ball1.radius / 2.0 if is_next else ball2.radius / 2.0
    dist_displaced = dist_is - dist_should
    ratio_displaced = abs(dist_displaced) / dist_should
    # smoothing_factor
    s_f:cython.double = smoothing_factor(ratio_displaced, X_S, X_E) * RHO * dist_displaced / dist_is
    # add to recover force
    force_dir *= s_f
    ball1.force = ball1.force + force_dir
    ball1.neighbor_dist = max(ball1.neighbor_dist, dist_is)


cdef calculate_angle_force(ball: Ball, ball_prev: Ball, ball_next: Ball):
    """
    Calculates angle force between 3 neighboring balls and adds it to the center ball
    Note: this code does not directly follow the paper by Altendorf&Jeulin
    because this caused strange errors. Instead, it uses an equivalent calculation
    that was proposed yet undocumented in the original code (MAVIlib)

    :param ball: Balls / 2.0
        The center ball - this is where the force will be applied
    :param ball_prev: Ball
        The previous ball
    :param ball_next: Ball
        The next ball
    """
    coord = ball.coordinate
    coord_prev = ball_prev.coordinate
    coord_next = ball_next.coordinate

    # calculate and normalize vectors
    alpha0:cython.double = ball.angle
    norm_prev:cython.double = np.linalg.norm(coord - coord_prev) 
    norm_next:cython.double = np.linalg.norm(coord_next - coord) 
    norm_a:cython.double = np.linalg.norm(coord_next - coord_prev)
    dir_next = (coord_next - coord) / norm_next
    dir_prev = coord - coord_prev
    a = (coord_next - coord_prev) / norm_a

    # calculate m, the point where the line hits the plane
    d = np.dot(dir_prev, a)
    dir_prev /= norm_prev
    m = coord_prev + d*a
    alpha = np.pi - np.arccos(np.dot(dir_prev, dir_next))

    # z, z0: calculate distances of ball.coordinate to m
    h1:cython.double = abs(d)
    h2:cython.double = np.linalg.norm(m - coord_next)
    z:cython.double = np.linalg.norm(m - coord)

    tan_alpha0 = np.tan(alpha0)
    if tan_alpha0 < 0:
        z0 = (
            h1 + h2 - np.sqrt(np.square((h1 + h2)) + 4 * h1 * h2 * np.square(tan_alpha0))
        ) / (2 * tan_alpha0)
    else:
        z0 = (
            h1 + h2 + np.sqrt(np.square(h1 + h2) + 4 * h1 * h2 * np.square(tan_alpha0))
        ) / (2 * tan_alpha0)

    # calculate force
    f:cython.double = smoothing_factor(alpha0 - alpha, ALPHA_S, ALPHA_E) / z * RHO * (z - z0) / 2.0
    ball.force += (m-coord)*f
    ball.angle_diff = alpha0 - alpha


def apply_forces(fiber_system: list[Fiber]):
    """
    Applies forces to the fiber system
    - it adds their forces to their coordinates, thus moves the balls
    - it sets all forces to 0

    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    """
    for fiber in fiber_system:
        for ball in fiber.balls:
            old_coord = ball.coordinate
            new_coord = old_coord + ball.force
            ball.coordinate = new_coord
            ball.force = np.array([0, 0, 0])
            ball.overlap = 0
            ball.neighbor_dist = ball.radius / 2.0
            ball.angle_diff = 0

def calculate_repulsion_forces_numpy(
    np.ndarray coord_array, 
    np.ndarray label_array, 
    np.ndarray radius_array, 
    np.ndarray image_size, 
    is_periodic: bool = True, 
    repulsion_factor: float = 1.1,
    tuple shift = (0, 0, 0)
):
    """
    Calculates the repulsion forces between all balls in the system using numpy broadcasting.
    :param coord_array: np.ndarray
        The coordinates of the balls in the system, shape (x, y, z, max_cells, 3)
    :param label_array: np.ndarray
        The labels of the balls in the system, shape (x, y, z, max_cells)
    :param radius_array: np.ndarray
        The radii of the balls in the system, shape (x, y, z, max_cells)
    :param image_size: np.ndarray
        The size of the image for periodic boundary conditions, shape (3,)
    :param is_periodic: bool
        Whether to apply periodic boundary conditions
    :param repulsion_factor: float
        The factor to scale the repulsion force
    :param shift: tuple
        The shift to apply to the neighbor cells, default is (0, 0, 0)
    :return: tuple
        net_forces: np.ndarray
            The net forces on each ball, shape (x, y, z, max_cells, 3)
        max_overlaps: np.ndarray
            The maximum overlaps for each ball, shape (x, y, z, max_cells)
    """
    max_cells = coord_array.shape[3]
    valid_cells = label_array[..., :] != -1 # -1 are empty cells

    # roll the array for neighboring cells
    pos_j = np.roll(coord_array, shift=shift, axis=(0,1,2))
    r_j = np.roll(radius_array, shift=shift, axis=(0,1,2))
    valid_j = np.roll(valid_cells, shift=shift, axis=(0,1,2))

    # vector calculations
    r_sum = radius_array[..., :, np.newaxis] + r_j[..., np.newaxis, :] # (x, y, z, max_cells, max_cells)
    valid_pairs = valid_cells[..., :, np.newaxis] & valid_j[..., np.newaxis, :] # (x, y, z, max_cells, max_cells)
    pos_diff = coord_array[..., :, np.newaxis, :] - pos_j[..., np.newaxis, :, :] # (x, y, z, max_cells, max_cells, 3)
    if is_periodic:
        pos_diff = pos_diff - image_size * np.round(pos_diff / image_size)

    # because we compute the whole matrix, we need to avoid accidental division by zero
    distances = np.linalg.norm(pos_diff, axis=-1)
    safe_distances = np.where(distances < 1e-8, 1.0, distances) # replace zeros

    # overlap logic
    overlaps = repulsion_factor * r_sum - distances
    overlaps_true = r_sum - distances
    is_overlapping = overlaps > 0

    # ignore self-interactions in case of the same cell
    if shift == (0, 0, 0):
        compute_mask = valid_pairs & is_overlapping & ~np.eye(max_cells, dtype=bool)
    else:
        compute_mask = valid_pairs & is_overlapping

    # if it is not periodic, we need to ignore border cells
    if not is_periodic:
        border_mask = np.ones_like(compute_mask, dtype=bool)
        for axis, direction in enumerate(shift):
            index = [slice(None)] * 3 # [:, :, :]
            if direction == -1:
                index[axis] = -1 # [-1, :, :]
            if direction == 1:
                index[axis] = 0 # [0, :, :]
            compute_mask[tuple(index)] = False
        compute_mask = compute_mask & border_mask

    # forces
    force_mags = TAU * overlaps / 2.0
    force_vecs = (force_mags / safe_distances)[..., np.newaxis] * pos_diff
    forces = force_vecs * compute_mask[..., np.newaxis]
    
    # collapse forces and overlaps
    net_forces = np.sum(forces, axis=-2) 
    masked_overlap_true = np.where(compute_mask, overlaps_true, 0.0)
    max_overlaps = np.max(masked_overlap_true, axis=-1)

    return net_forces, max_overlaps

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
    radius_array = np.zeros((nb_fibers, max_balls))
    neighbor_distances_array = np.full((nb_fibers, max_balls), -1.0)
    angle_array = np.zeros((nb_fibers, max_balls))
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

    dir_next = diff_next / safe_norm_next[..., np.newaxis]
    dir_prev = diff_prev / safe_norm_prev[..., np.newaxis]
    a = diff_a / norm_a[..., np.newaxis]    

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
