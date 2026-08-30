# cython: language_level=3, infer_type=True, exception_check=False, cdivision=True
import numpy as np
import cython
from libc.stdint cimport int64_t
from libc.math cimport cos, sqrt, tan, acos
cimport numpy as np
np.import_array()


import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.Fiber cimport Ball
from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.utils cimport cdirection, cdistance3

MIN_REPULSION_DISTANCE = 5
X_S:cython.double = 0.05
X_E:cython.double = 0.1
ALPHA_S:cython.double = 0.1 * np.pi / 180
ALPHA_E:cython.double = 0.2 * np.pi / 180
# factors to balance forces, see Altendorf & Jeulin
TAU:cython.double = 0.25
RHO:cython.double = 0.25
REPULSION_FACTOR:cython.double = 1.0
cdef double PI = 3.141592653589793


def calculate_forces(grid: sh, fiber_system: list[Fiber], is_periodic: bool = True,
                     shortlist = [], softcore_ratio: float = 0.0, contact_distance = 1):
    """
    Calculates forces in the fiber system and adds them to corresponding ball

    :param grid: SpatialHashing
        The spatial hashing grid for the model
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :return: np.ndarray
        total force of the fiber system
    """
    cdef list balls
    cdef Ball ball, ball_prev, ball_next
    cdef int n
    for cell in grid.cells:
        if len(cell) > 0:
            neighbor_cells = grid.get_younger_neighbor_cell_indices(
                grid.get_cell_index_of_coord(cell[0].coordinate), is_periodic=is_periodic
            )
            for i, ball in enumerate(cell):
                calculate_repulsion_forces(
                    i, ball, cell, grid, neighbor_cells, is_periodic=is_periodic,
                    softcore_ratio=softcore_ratio
                )
    for fiber in fiber_system:
        balls = fiber.balls
        n = len(balls)
        for i in range(n):
            ball = balls[i]
            if i + 1 < n:
                next_ball = balls[i+1]
                calculate_spring_force(ball, next_ball, is_next=True)
            if i > 0:
                prev_ball = balls[i-1]
                calculate_spring_force(ball, prev_ball, is_next=False)
            if i > 0 and i + 1 < n:
                calculate_angle_force(ball, balls[i - 1], balls[i + 1])

    contact_distances = 0
    for contact_edge in shortlist:
        ball = fiber_system[contact_edge[0][0]].balls[contact_edge[0][1]]
        ball2 = fiber_system[contact_edge[1][0]].balls[contact_edge[1][1]]
        distance = calculate_contact_force(ball, ball2, image_size = grid.image_size, is_periodic=is_periodic,
                                 contact_distance=contact_distance)
        contact_distances += distance

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
        contact_distances
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
    softcore_ratio: float = 0.0,
    repulsion_factor: float = 1.1
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
    # compare within cell
    for ball2 in cell[i + 1 :]:
        calculate_repulsion_force(
            ball, ball2, fiber_label, label, is_periodic, grid.image_size, softcore_ratio, repulsion_factor
        )
    # compare with neighbor cells
    for cell_index in neighbor_cells:
        cell = grid.cells[cell_index]
        for ball2 in cell:
            calculate_repulsion_force(
                ball, ball2, fiber_label, label, is_periodic, grid.image_size, softcore_ratio, repulsion_factor
            )


def calculate_repulsion_force(
    ball, ball2, fiber_label: int, label: int, is_periodic: bool, int64_t[:] image_size,
    softcore_ratio: float = 0.0,
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
            coord = ball.coordinate
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
            overlap: cython.float = ball.radius + ball2.radius
            overlap_true: cython.float = (1 - softcore_ratio)*overlap - dist
            overlap = repulsion_factor*(1 - softcore_ratio)*overlap - dist
            if overlap > 0:
                coord2mod = coord2mod / dist
                force = TAU * overlap / 2.0 * coord2mod
                ball.force = ball.force - force
                ball.overlap = max(ball.overlap, overlap_true)
                ball2.force = ball2.force + force
                ball2.overlap = max(ball2.overlap, overlap_true)

        else:
            coord = ball.coordinate
            coord2 = ball2.coordinate
            dist: cython.double = np.linalg.norm(coord2 - coord)
            overlap: cython.float = ball.radius + ball2.radius
            overlap_true: cython.float = (1 - softcore_ratio)*overlap - dist
            overlap = repulsion_factor*(1 - softcore_ratio)*overlap - dist
            if overlap > 0:
                dir = (coord2 - coord)/dist
                ball.force = ball.force - TAU * overlap / 2.0 * dir
                ball.overlap = max(ball.overlap, overlap_true)
                ball2.force = ball2.force + TAU * overlap / 2.0 * dir
                ball2.overlap = max(ball2.overlap, overlap_true)


cdef inline double smoothing_factor(double x, double x_s, double x_e) noexcept:
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
    cdef double ratio
    if x < x_s:
        return 0
    elif x > x_e:
        return 1
    ratio = (x - x_s) / (x_e - x_s)
    return 0.5 * (1 - cos(ratio * PI))

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
    cdef double dx, dy, dz, s_f
    cdef double dist_is, dist_should, dist_displaced, ratio_displaced
    dist_is = cdirection(ball1, ball2, &dx, &dy, &dz)

    # distance to the next ball is currently always radius/2.0
    # - may need to adapt for different random walks
    dist_should = ball1.radius / 2.0 if is_next else ball2.radius / 2.0
    dist_displaced = dist_is - dist_should
    ratio_displaced = abs(dist_displaced) / dist_should
    # smoothing_factor
    s_f = smoothing_factor(ratio_displaced, X_S, X_E) * RHO * dist_displaced
    # add to recover force
    ball1.force[0] += dx*s_f
    ball1.force[1] += dy*s_f
    ball1.force[2] += dz*s_f
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
    cdef double ax, ay, az, dir_prev_x, dir_prev_y, dir_prev_z, dir_next_x, dir_next_y, dir_next_z, mx, my, mz
    cdef double length_prev, alpha0, alpha, d, h1, h2, z, tan_alpha0, f
    cdirection(ball_prev, ball_next, &ax, &ay, &az)
    length_prev = cdirection(ball_prev, ball, &dir_prev_x, &dir_prev_y, &dir_prev_z)
    cdirection(ball, ball_next, &dir_next_x, &dir_next_y, &dir_next_z)
    coord = ball.coordinate
    coord_prev = ball_prev.coordinate
    coord_next = ball_next.coordinate

    # calculate and normalize vectors
    alpha0 = ball.angle

    # calculate m, the point where the line hits the plane
    d =  length_prev*(ax*dir_prev_x + ay*dir_prev_y + az*dir_prev_z)
    mx = coord_prev[0] + d*ax
    my = coord_prev[1] + d*ay
    mz = coord_prev[2] + d*az
    alpha = PI - acos(dir_prev_x*dir_next_x + dir_prev_y*dir_next_y + dir_prev_z*dir_next_z) #TODO np

    # z, z0: calculate distances of ball.coordinate to m
    h1 = abs(d)
    h2  = cdistance3(mx, my, mz, coord_next[0], coord_next[1], coord_next[2])
    z = cdistance3(mx, my, mz, coord[0], coord[1], coord[2])

    tan_alpha0 = tan(alpha0)
    if tan_alpha0 < 0:
        z0 = (
            h1
            + h2
            - sqrt((h1 + h2)*(h1 + h2) + 4 * h1 * h2 * tan_alpha0*tan_alpha0)
        ) / (2 * tan_alpha0)
    else:
        z0 = (
            h1 + h2 + sqrt((h1 + h2)*(h1 + h2) + 4 * h1 * h2 * tan_alpha0*tan_alpha0)
        ) / (2 * tan_alpha0)

    # calculate force
    f = smoothing_factor(alpha0 - alpha, ALPHA_S, ALPHA_E) / z * RHO * (z - z0) / 2.0
    ball.force[0] += (mx-coord[0])*f
    ball.force[1] += (my-coord[1])*f
    ball.force[2] += (mz-coord[2])*f
    ball.angle_diff = alpha0 - alpha


def calculate_contact_force(ball, ball2, is_periodic: bool, int64_t[:] image_size, contact_distance: float,
    repulsion_factor:float = 1.1):
    """
    calculates the contact force between two balls

    :param ball: Ball
        The ball whose neighbors are currently considered
    :param ball2: Ball
        The neighboring ball that is currently considered
    :param is_periodic: bool
        Whether the repulsion force is to be calculated on the torus, i.e., periodically
    :param image_size: int64_t
        The image size (relevant for periodic case)
    :param contact_distance: float
        The maximal distance that balls can have to be considered in contact
    :param repulsion_factor: float, default 1.1
        This factor is 1 in the Altendorf-Jeulin model.
        However, this leads to incredibly low convergence (explainable with limit of explicit Euler?),
        which is also why they stop packing when the overlap is 0.1*radius and then need an end_step
        A factor of 1.1 turned out as trade-off between runtime and highest volume fraction
    """
    coord = ball.coordinate
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
        dist_perfect: cython.double = ball.radius + ball2.radius
        displace: cython.double = dist - dist_perfect
        if displace > 0:
            coord2mod = coord2mod / dist
            force = displace / 2.0 * coord2mod*repulsion_factor
            ball.force = ball.force + force
            ball2.force = ball2.force - force
            return max(0, displace - contact_distance)
    else:
        coord2 = ball2.coordinate
        dist: cython.double = np.linalg.norm(coord2 - coord)
        dist_perfect: cython.double = ball.radius + ball2.radius
        displace: cython.double = dist - dist_perfect
        if displace > 0:
            dir = (coord2 - coord)/dist
            force = TAU * displace / 2.0 * dir*smoothing_factor(displace, 0, contact_distance)*repulsion_factor
            ball.force = ball.force + force
            ball2.force = ball2.force - force
            return max(0, displace - contact_distance)
    return 0


def apply_forces(fiber_system: list[Fiber]):
    """
    Applies forces to the fiber system
    - it adds their forces to their coordinates, thus moves the balls
    - it sets all forces to 0

    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    """
    cdef Ball ball
    for fiber in fiber_system:
        for ball in fiber.balls:
            ball.coordinate[0] += ball.force[0]
            ball.coordinate[1] += ball.force[1]
            ball.coordinate[2] += ball.force[2]
            ball.force[0] = 0
            ball.force[1] = 0
            ball.force[2] = 0
            ball.overlap = 0
            ball.neighbor_dist = ball.radius / 2.0
            ball.angle_diff = 0
