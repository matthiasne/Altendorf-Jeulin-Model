# cython: language_level=3, infer_type=True, exception_check=False, cdivision=True
import numpy as np
import cython
from Altendorf_Jeulin_Model import SpatialHashing
from libc.math cimport cos, sqrt, tan, acos
cimport numpy as np
np.import_array()

from Altendorf_Jeulin_Model.SpatialHashing cimport SpatialHashing as sh
from Altendorf_Jeulin_Model.Fiber cimport Ball
from Altendorf_Jeulin_Model.Fiber import Fiber
from Altendorf_Jeulin_Model.utils cimport cdirection, cdistance3, clip

MIN_REPULSION_DISTANCE = 5
X_S:cython.double = 0.25
X_E:cython.double = 0.5
ALPHA_S:cython.double = 2 * np.pi / 180
ALPHA_E:cython.double = 3 * np.pi / 180
# factors to balance forces, see Altendorf & Jeulin
TAU:cython.double = 0.25
RHO:cython.double = 0.25
cdef double PI = 3.141592653589793


def calculate_forces(grid: sh, fiber_system: list[Fiber], is_periodic: bool = True,
                     shortlist = [], hardcore_ratio: float = 1.0, contact_distance = 1,
                     repulsion_factor = 1.1):
    """
    Calculates forces in the fiber system and adds them to corresponding ball

    :param grid: SpatialHashing
        The spatial hashing grid for the model
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :return: np.ndarray
        total force of the fiber system
    """
    cdef list balls, cell
    cdef set neighbor_cells
    cdef Ball ball, ball_prev, ball_next, ball2
    cdef int n, ix, iy, iz
    cdef double distance, shortlist_distance_sum
    cdef total_force_x, total_force_y, total_force_z, total_force_norm, force_norm, max_force_norm
    cdef double total_overlap, total_neighbor_dist, total_angle_diff, optim_sum
    cdef int[3] image_size = grid.image_size
    for cell in grid.cells:
        n = len(cell)
        if n > 0:
            ix, iy, iz = grid.get_cell_index_of_coord(cell[0].coordinate[0], cell[0].coordinate[1], cell[0].coordinate[2])
            neighbor_cells = grid.get_younger_neighbor_cell_indices(
                ix, iy, iz, is_periodic=is_periodic
            )
            for i in range(n):
                ball = cell[i]
                calculate_repulsion_forces(
                    i, ball, cell, grid, neighbor_cells, is_periodic=is_periodic,
                    hardcore_ratio=hardcore_ratio, repulsion_factor=repulsion_factor
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

    shortlist_distance_sum = 0
    for contact_edge in shortlist:
        ball = fiber_system[contact_edge[0][0]].balls[contact_edge[0][1]]
        ball2 = fiber_system[contact_edge[1][0]].balls[contact_edge[1][1]]
        if is_periodic:
            distance = calculate_contact_force_periodic(ball, ball2, image_size, contact_distance)
        else:
            distance = calculate_contact_force_nonperiodic(ball, ball2, contact_distance)
        shortlist_distance_sum += distance
    total_force_x = 0
    total_force_y = 0
    total_force_z = 0
    total_overlap = 0
    total_neighbor_dist = 0
    total_angle_diff = 0
    max_force_norm = 0
    optim_sum = 0
    for fiber in fiber_system:
        for ball in fiber.balls:
            force_norm = sqrt(ball.force[0]*ball.force[0] + ball.force[1]*ball.force[1] + ball.force[2]*ball.force[2])
            if force_norm > max_force_norm:
                max_force_norm = force_norm
            total_force_x = total_force_x + ball.force[0]
            total_force_y = total_force_y + ball.force[1]
            total_force_z = total_force_z + ball.force[2]
            total_overlap = max(total_overlap, ball.overlap)
            total_neighbor_dist = max(total_neighbor_dist, ball.neighbor_dist)
            total_angle_diff = max(total_angle_diff, abs(ball.angle_diff))
            optim_sum += ball.optim_sum
    total_force_norm = sqrt(total_force_x*total_force_x + total_force_y*total_force_y + total_force_z*total_force_z)
    return (
        total_force_norm,
        max_force_norm,
        total_overlap,
        total_neighbor_dist,
        total_angle_diff,
        shortlist_distance_sum,
        optim_sum
    )


def calculate_forces_endstep(
    grid: sh, fiber_system: list[Fiber], is_periodic: bool = True, hardcore_ratio: float = 1.0
):
    """
    Calculates forces in the fiber system and adds them to corresponding ball#

    :param grid: SpatialHashing
        The spatial hashing grid for the model
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :return: np.ndarray
        total force of the fiber system
    """

    cdef list balls, cell
    cdef set neighbor_cells
    cdef Ball ball, ball_prev, ball_next, ball2
    cdef int n, ix, iy, iz
    cdef total_force_x, total_force_y, total_force_z, total_force_norm
    cdef double total_overlap, total_neighbor_dist, total_angle_diff
    cdef int[3] image_size = grid.image_size
    for cell in grid.cells:
        n = len(cell)
        if n > 0:
            ix, iy, iz = grid.get_cell_index_of_coord(cell[0].coordinate[0], cell[0].coordinate[1], cell[0].coordinate[2])
            neighbor_cells = grid.get_younger_neighbor_cell_indices(
                ix, iy, iz, is_periodic=is_periodic
            )
            for i in range(n):
                ball = cell[i]
                calculate_repulsion_forces(
                    i, ball, cell, grid, neighbor_cells, is_periodic=is_periodic,
                    hardcore_ratio=hardcore_ratio
                )

    total_force_x = 0
    total_force_y = 0
    total_force_z = 0
    total_overlap = 0
    total_neighbor_dist = 0
    total_angle_diff = 0
    for fiber in fiber_system:
        for ball in fiber.balls:
            total_force_x = total_force_x + ball.force[0]
            total_force_y = total_force_y + ball.force[1]
            total_force_z = total_force_z + ball.force[2]
            total_overlap = max(total_overlap, ball.overlap)
            total_neighbor_dist = max(total_neighbor_dist, ball.neighbor_dist)
            total_angle_diff = max(total_angle_diff, abs(ball.angle_diff))
    total_force_norm = sqrt(total_force_x*total_force_x + total_force_y*total_force_y + total_force_z*total_force_z)
    return (
        total_force_norm,
        total_overlap,
        total_neighbor_dist,
        total_angle_diff
    )


cdef inline void calculate_repulsion_forces(
    int i,
    Ball ball,
    list cell,
    sh grid,
    set neighbor_cells,
    bint is_periodic = True,
    double hardcore_ratio = 1.0,
    double repulsion_factor = 1.1
) noexcept:
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
    cdef Ball ball2
    cdef list neighbor_cell
    cdef int j, cell_index
    cdef int n = len(cell)
    cdef int[3] image_size = grid.image_size
    # compare within cell
    if is_periodic:
        for j in range(i+1, n):
            ball2 = cell[j]
            calculate_repulsion_force_periodic(
                ball, ball2, image_size, hardcore_ratio, repulsion_factor
            )
        # compare with neighbor cells
        for cell_index in neighbor_cells:
            neighbor_cell = grid.cells[cell_index]
            for ball2 in neighbor_cell:
                calculate_repulsion_force_periodic(
                    ball, ball2, image_size, hardcore_ratio, repulsion_factor
                )
    else:
        for j in range(i+1, n):
            ball2 = cell[j]
            calculate_repulsion_force_nonperiodic(
                ball, ball2, hardcore_ratio, repulsion_factor
            )
        # compare with neighbor cells
        for cell_index in neighbor_cells:
            neighbor_cell = grid.cells[cell_index]
            for ball2 in neighbor_cell:
                calculate_repulsion_force_nonperiodic(
                    ball, ball2, hardcore_ratio, repulsion_factor
                )



cdef void calculate_repulsion_force_periodic(
    Ball ball, Ball ball2, int[3] image_size,
    double hardcore_ratio,
    double repulsion_factor
) noexcept:
    """
    calculates the repulsion force between two balls in the periodic case

    :param ball: Ball
        The ball whose neighbors are currently considered
    :param ball2: Ball
        The neighboring ball that is currently considered
    :param image_size: int[3]
    :param repulsion_factor: float, default 1.1
        This factor is 1 in the Altendorf-Jeulin model.
        However, this leads to incredibly low convergence (explainable with limit of explicit Euler?),
        which is also why they stop packing when the overlap is 0.1*radius and then need an end_step
        A factor of 1.1 turned out as trade-off between runtime and highest volume fraction
    """
    cdef double dist, displaced, overlap, overlap_true, force_strength
    cdef double dx, dy, dz, coordx, coordy, coordz, coord2x, coord2y, coord2z
    cdef double optim_summand
    if (
        ball.fiber_label != ball2.fiber_label
        or abs(ball.ball_label - ball2.ball_label) >= MIN_REPULSION_DISTANCE
    ):
        coordx = ball.coordinate[0]
        coordy = ball.coordinate[1]
        coordz = ball.coordinate[2]
        coord2x = ball2.coordinate[0]
        coord2y = ball2.coordinate[1]
        coord2z = ball2.coordinate[2]

        # calculate periodic distance/direction
        displaced = coord2x - coordx
        dx = displaced - image_size[0]*round(displaced/image_size[0])
        displaced = coord2y - coordy
        dy = displaced - image_size[1]*round(displaced/image_size[1])
        displaced = coord2z - coordz
        dz = displaced - image_size[2]*round(displaced/image_size[2])
        dist = sqrt(dx*dx + dy*dy + dz*dz)
        overlap = ball.radius + ball2.radius
        overlap_true = hardcore_ratio*overlap - dist
        overlap = repulsion_factor*hardcore_ratio*overlap - dist
        if overlap > 0:
            force_strength = TAU*overlap / 2.0
            optim_summand = overlap*overlap/4.0
            if dist > 0.0:
                force_strength /= dist
            ball.force[0] -= force_strength * dx
            ball.force[1] -= force_strength * dy
            ball.force[2] -= force_strength * dz
            ball.overlap = max(ball.overlap, overlap_true)
            ball.optim_sum += optim_summand
            ball2.force[0] += force_strength * dx
            ball2.force[1] += force_strength * dy
            ball2.force[2] += force_strength * dz
            ball2.overlap = max(ball2.overlap, overlap_true)
            ball2.optim_sum += optim_summand


cdef void calculate_repulsion_force_nonperiodic(
    Ball ball, Ball ball2,
    double hardcore_ratio,
    double repulsion_factor
) noexcept:
    """
    calculates the repulsion force between two balls in the nonperiodic case

    :param ball: Ball
        The ball whose neighbors are currently considered
    :param ball2: Ball
        The neighboring ball that is currently considered
    :param repulsion_factor: float, default 1.1
        This factor is 1 in the Altendorf-Jeulin model.
        However, this leads to incredibly low convergence (explainable with limit of explicit Euler?),
        which is also why they stop packing when the overlap is 0.1*radius and then need an end_step
        A factor of 1.1 turned out as trade-off between runtime and highest volume fraction
        TODO: add enforced distance as in contact model or fSAM, which may be relevant when voxelizing fiber system
    """
    cdef double dist, overlap, overlap_true, force_strength
    cdef double dx, dy, dz
    if (
        ball.fiber_label != ball2.fiber_label
        or abs(ball.ball_label - ball2.ball_label) >= MIN_REPULSION_DISTANCE
    ):
        dist = cdirection(ball, ball2, &dx, &dy, &dz)
        overlap = ball.radius + ball2.radius
        overlap_true = hardcore_ratio*overlap - dist
        overlap = repulsion_factor*hardcore_ratio*overlap - dist
        if overlap > 0:
            force_strength = TAU * overlap / 2.0
            optim_summand = overlap*overlap/4.0
            ball.force[0] -= force_strength * dx
            ball.force[1] -= force_strength * dy
            ball.force[2] -= force_strength * dz
            ball.overlap = max(ball.overlap, overlap_true)
            ball.optim_sum += optim_summand
            ball2.force[0] += force_strength * dx
            ball2.force[1] += force_strength * dy
            ball2.force[2] += force_strength * dz
            ball2.overlap = max(ball2.overlap, overlap_true)
            ball.optim_sum += optim_summand


cdef void calculate_spring_force(Ball ball1, Ball ball2, bint is_next) noexcept:
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
    ball1.neighbor_dist = dist_is#max(ball1.neighbor_dist, dist_is)
    ball1.optim_sum += s_f*s_f/(RHO*RHO)


cdef void calculate_angle_force(Ball ball, Ball ball_prev, Ball ball_next) noexcept:
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
    alpha = PI - acos(clip(dir_prev_x*dir_next_x + dir_prev_y*dir_next_y + dir_prev_z*dir_next_z))
    if abs(alpha - alpha0) < 1e-6:
        return
    # z, z0: calculate distances of ball.coordinate to m
    h1 = abs(d)
    h2  = cdistance3(mx, my, mz, coord_next[0], coord_next[1], coord_next[2])
    z = cdistance3(mx, my, mz, coord[0], coord[1], coord[2])
    if np.isclose(z,0):
        return

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
    ball.optim_sum += f*f/(RHO*RHO)


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


cdef double calculate_contact_force_periodic(Ball ball, Ball ball2, int[3] image_size,
    double contact_distance, double repulsion_factor = 1.1) noexcept:
    """
    calculates the contact force between two balls

    :param ball: Ball
        The ball whose neighbors are currently considered
    :param ball2: Ball
        The neighboring ball that is currently considered
    :param is_periodic: bool
        Whether the repulsion force is to be calculated on the torus, i.e., periodically
    :param image_size: int
        The image size (relevant for periodic case)
    :param contact_distance: float
        The maximal distance that balls can have to be considered in contact
    :param repulsion_factor: float, default 1.1
        This factor is 1 in the Altendorf-Jeulin model.
        However, this leads to incredibly low convergence (explainable with limit of explicit Euler?),
        which is also why they stop packing when the overlap is 0.1*radius and then need an end_step
        A factor of 1.1 turned out as trade-off between runtime and highest volume fraction
    """
    cdef double dist, displaced, dist_perfect, displace, true_displace, force_strength
    cdef double dx, dy, dz, coordx, coordy, coordz, coord2x, coord2y, coord2z
    coordx = ball.coordinate[0]
    coordy = ball.coordinate[1]
    coordz = ball.coordinate[2]
    coord2x = ball2.coordinate[0]
    coord2y = ball2.coordinate[1]
    coord2z = ball2.coordinate[2]

    # calculate periodic distance/direction
    displaced = coord2x - coordx
    dx = displaced - image_size[0]*round(displaced/image_size[0])
    displaced = coord2y - coordy
    dy = displaced - image_size[1]*round(displaced/image_size[1])
    displaced = coord2z - coordz
    dz = displaced - image_size[2]*round(displaced/image_size[2])
    dist = sqrt(dx*dx + dy*dy + dz*dz)
    dist_perfect = ball.radius + ball2.radius
    true_displace = dist - dist_perfect - contact_distance
    displace = dist - dist_perfect

    if displace > 0:
        force_strength = displace / 2.0*RHO*smoothing_factor(displace/dist, contact_distance/2., contact_distance)
        if dist > 0.0:
            force_strength /= dist
        ball.force[0] += force_strength * dx
        ball.force[1] += force_strength * dy
        ball.force[2] += force_strength * dz
        ball.optim_sum += force_strength
        ball2.force[0] -= force_strength * dx
        ball2.force[1] -= force_strength * dy
        ball2.force[2] -= force_strength * dz
        ball2.optim_sum += force_strength

        return max(0, true_displace)
    return 0


cdef double calculate_contact_force_nonperiodic(Ball ball, Ball ball2,
    double contact_distance, double repulsion_factor = 1.1) noexcept:
    """
    calculates the contact force between two balls in the nonperiodic case
    
    :param ball: Ball
        The ball whose neighbors are currently considered
    :param ball2: Ball
        The neighboring ball that is currently considered
    :param contact_distance: float
        The maximal distance that balls can have to be considered in contact
    :param repulsion_factor: float, default 1.1
        This factor is 1 in the Altendorf-Jeulin model.
        However, this leads to incredibly low convergence (explainable with limit of explicit Euler?),
        which is also why they stop packing when the overlap is 0.1*radius and then need an end_step
        A factor of 1.1 turned out as trade-off between runtime and highest volume fraction
    """
    cdef double dist, displaced, dist_perfect, displace, force_strength
    cdef double dx, dy, dz

    dist = cdirection(ball2, ball, &dx, &dy, &dz)
    dist_perfect = ball.radius + ball2.radius
    displace = dist - dist_perfect

    if displace > 0:
        force_strength = displace / 2.0*repulsion_factor
        if dist > 0.0:
            force_strength /= dist
        ball.force[0] += force_strength * dx
        ball.force[1] += force_strength * dy
        ball.force[2] += force_strength * dz
        ball2.force[0] -= force_strength * dx
        ball2.force[1] -= force_strength * dy
        ball2.force[2] -= force_strength * dz

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
            ball.optim_sum = 0
