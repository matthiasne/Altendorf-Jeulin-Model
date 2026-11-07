# cython: language_level=3, infer_type=True, exception_check=False, cdivision=True
import numpy as np
import cython
from Altendorf_Jeulin_Model.utils import SpatialHashing
from libc.math cimport cos, sqrt, tan, acos
cimport numpy as np
np.import_array()

from Altendorf_Jeulin_Model.utils.SpatialHashing cimport SpatialHashing as sh
from Altendorf_Jeulin_Model.Fiber cimport Ball
from Altendorf_Jeulin_Model.Fiber import Fiber
from Altendorf_Jeulin_Model.utils.utils cimport cdirection, cdistance3, clip

MIN_REPULSION_DISTANCE = 5
cdef double PI = 3.141592653589793


def calculate_forces(grid: sh, fiber_system: list[Fiber], is_periodic: bool = True,
                     shortlist = [], hardcore_ratio: float = 1.0, contact_distance = 1,
                     repulsion_factor = 1.1, tau: float = 0.25, rho: float = 0.25,
                     alpha_s: float = 1*PI/180, alpha_e: float = 2*PI/180,
                     x_s: float = 0.25, x_e: float = 0.5):
    """
    Compute all forces in the fiber system for one packing step.

    Applies three/four force types in order:
    1. Repulsion forces between all nearby ball pairs (via spatial hashing).
    2. Spring forces between consecutive balls within each fiber.
    3. Angle forces at each interior ball to enforce the target bending angle.
    4. Contact forces along edges in the shortlist (only in the Contact++ model).

    Forces are *added* to ball.force; the caller must ensure ball.force is
    zeroed before calling (apply_forces resets them after each step).

    :param grid: SpatialHashing
        spatial hashing grid containing all balls
    :param fiber_system: list[Fiber]
        the full fiber system
    :param is_periodic: bool, optional (default: True)
        whether to use periodic boundary conditions
    :param shortlist: list, optional (default: [], i.e. no contact forces)
        list of contact edges ((fiber_i, ball_i), (fiber_j, ball_j));
    :param hardcore_ratio: float, optional (default: 1.0)
        fraction of the sum of radii that counts as the hard-core exclusion
        distance
    :param contact_distance: float, optional (default: 1)
        maximum surface-to-surface distance for contact force activation
    :param repulsion_factor: float, optional (default 1.1)
        multiplier on the exclusion distance for the soft repulsion zone;
        values > 1 widen the repulsion shell
    :param tau: float, optional (default: 0.25)
        repulsion force magnitude scaling factor
    :param rho: float, optional (default: 0.25)
        spring and angle force magnitude scaling factor
    :param alpha_s: float, optional (default: 1°)
        angle smoothing onset (radians)
    :param alpha_e: float, optional (default: 2°)
        angle smoothing saturation (radians)
    :param x_s: float, optional (default: 0.25)
        spring smoothing onset ratio
    :param x_e: float, optional (default: 0.5)
        spring smoothing saturation ratio
    :return: tuple
        (total_force_norm, max_force_norm, total_overlap,
         total_neighbor_dist, total_angle_diff,
         shortlist_distance_sum, optim_sum)
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
                    hardcore_ratio=hardcore_ratio, repulsion_factor=repulsion_factor, tau=tau
                )
    for fiber in fiber_system:
        balls = fiber.balls
        n = len(balls)
        for i in range(n):
            ball = balls[i]
            if i + 1 < n:
                next_ball = balls[i+1]
                calculate_spring_force(ball, next_ball, is_next=True, rho=rho, x_s=x_s, x_e=x_e)
            if i > 0:
                prev_ball = balls[i-1]
                calculate_spring_force(ball, prev_ball, is_next=False, rho=rho, x_s=x_s, x_e=x_e)
            if i > 0 and i + 1 < n:
                calculate_angle_force(ball, balls[i - 1], balls[i + 1], rho=rho, alpha_s=alpha_s, alpha_e=alpha_e)

    shortlist_distance_sum = 0
    for contact_edge in shortlist:
        ball = fiber_system[contact_edge[0][0]].balls[contact_edge[0][1]]
        ball2 = fiber_system[contact_edge[1][0]].balls[contact_edge[1][1]]
        if is_periodic:
            distance = calculate_contact_force_periodic(ball, ball2, image_size, contact_distance, rho)
        else:
            distance = calculate_contact_force_nonperiodic(ball, ball2, contact_distance, rho)
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
    Compute only repulsion forces for the end-step of the packing algorithm.

    Used after the main packing loop converges (overlap < threshold) to
    remove residual overlaps without applying spring or angle forces.
    Forces are added to ball.force; the caller must ensure ball.force is
    zeroed before calling.

    :param grid: SpatialHashing
        spatial hashing grid containing all balls
    :param fiber_system: list[Fiber]
        the full fiber system
    :param is_periodic: bool, optional (default: True)
        whether to use periodic boundary conditions
    :param hardcore_ratio: float, optional (default: 1.0)
        fraction of the sum of radii that counts as the hard-core exclusion
        distance
    :return: tuple
        (total_force_norm, total_overlap, total_neighbor_dist, total_angle_diff)
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
    double repulsion_factor = 1.1,
    double tau = 0.25
) noexcept:
    """
    Dispatch repulsion force calculations for one ball against its cell
    neighbours and all balls in younger neighbouring cells.

    Within-cell pairs are compared with index j > i to avoid double-counting.
    Neighbouring-cell pairs are compared with all balls (the younger-cell
    convention ensures each inter-cell pair is visited exactly once).

    :param i: int
        index of ball within its cell; pairs with j > i within the same cell
    :param ball: Ball
        the ball whose interactions are computed
    :param cell: list[Ball]
        the cell containing ball
    :param grid: SpatialHashing
        the spatial hashing grid
    :param neighbor_cells: set[int]
        flat indices of the younger neighbouring cells returned by
        grid.get_younger_neighbor_cell_indices
    :param is_periodic: bool
        whether to use periodic boundary conditions
    :param hardcore_ratio: float
        fraction of radii sum used as hard-core exclusion distance
    :param repulsion_factor: float
        multiplier on the exclusion distance for the soft repulsion shell
    :param tau: float
        repulsion force magnitude scaling factor
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
                ball, ball2, image_size, hardcore_ratio, repulsion_factor, tau
            )
        # compare with neighbor cells
        for cell_index in neighbor_cells:
            neighbor_cell = grid.cells[cell_index]
            for ball2 in neighbor_cell:
                calculate_repulsion_force_periodic(
                    ball, ball2, image_size, hardcore_ratio, repulsion_factor, tau
                )
    else:
        for j in range(i+1, n):
            ball2 = cell[j]
            calculate_repulsion_force_nonperiodic(
                ball, ball2, hardcore_ratio, repulsion_factor, tau
            )
        # compare with neighbor cells
        for cell_index in neighbor_cells:
            neighbor_cell = grid.cells[cell_index]
            for ball2 in neighbor_cell:
                calculate_repulsion_force_nonperiodic(
                    ball, ball2, hardcore_ratio, repulsion_factor, tau
                )



cdef void calculate_repulsion_force_periodic(
    Ball ball, Ball ball2, int[3] image_size,
    double hardcore_ratio,
    double repulsion_factor,
    double tau = 0.25
) noexcept:
    """
    Compute and apply the repulsion force between two balls under
    periodic boundary conditions.

    The force is active when the periodic distance is less than
    repulsion_factor * hardcore_ratio * (r1 + r2).  Its magnitude is
    proportional to the overlap depth and scaled by tau.  Equal and
    opposite forces are applied to both balls.  Balls on the same fiber
    within MIN_REPULSION_DISTANCE indices of each other are skipped.

    :param ball: Ball
        first ball; force subtracted (pushed away from ball2)
    :param ball2: Ball
        second ball; force added (pushed away from ball)
    :param image_size: int[3]
        domain dimensions for periodic wrapping
    :param hardcore_ratio: float
        fraction of radii sum used as hard-core exclusion distance
    :param repulsion_factor: float
        multiplier on the exclusion distance for the soft repulsion shell
    :param tau: float (default: 0.25)
        force magnitude scaling factor
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
            force_strength = tau*overlap / 2.0
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
    double repulsion_factor,
    double tau = 0.25
) noexcept:
    """
    Compute and apply the repulsion force between two balls without
    periodic boundary conditions.

    Identical to calculate_repulsion_force_periodic except distances are
    Euclidean rather than periodic.  Balls on the same fiber within
    MIN_REPULSION_DISTANCE indices of each other are skipped.

    :param ball: Ball
        first ball; force subtracted (pushed away from ball2)
    :param ball2: Ball
        second ball; force added (pushed away from ball)
    :param hardcore_ratio: float
        fraction of radii sum used as hard-core exclusion distance
    :param repulsion_factor: float
        multiplier on the exclusion distance for the soft repulsion shell
    :param tau: float (default: 0.25)
        force magnitude scaling factor
    """
    cdef double dist, overlap, overlap_true, force_strength, optim_summand
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
            force_strength = tau * overlap / 2.0
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
            ball2.optim_sum += optim_summand


cdef void calculate_spring_force(Ball ball1, Ball ball2, bint is_next, double rho = 0.25,
        double x_s = 0.25, double x_e = 0.5) noexcept:
    """
    Compute and apply the spring force between two consecutive balls.

    The target distance between neighbours is ball.radius / 2 (for the
    forward neighbour) or ball2.radius / 2 (for the backward neighbour).
    The force magnitude is proportional to the displacement from the target
    distance, smoothed by smoothing_factor to suppress small fluctuations.

    :param ball1: Ball
        the ball to which the force is applied
    :param ball2: Ball
        the neighbouring ball (previous or next in the fiber)
    :param is_next: bool
        True if ball2 follows ball1 in the fiber (use ball1.radius);
        False if ball2 precedes ball1 (use ball2.radius)
    :param rho: float (default: 0.25)
        force magnitude scaling factor; default 0.25
    :param x_s: float (default: 0.25)
        smoothing onset ratio (fraction of target distance)
    :param x_e: float ((default: 0.5)
        smoothing saturation ratio (fraction of target distance)
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
    s_f = smoothing_factor(ratio_displaced, x_s, x_e) * rho * dist_displaced
    # add to recover force
    ball1.force[0] += dx*s_f
    ball1.force[1] += dy*s_f
    ball1.force[2] += dz*s_f
    ball1.neighbor_dist = dist_is#max(ball1.neighbor_dist, dist_is)
    ball1.optim_sum += s_f*s_f/(rho*rho)


cdef void calculate_angle_force(Ball ball, Ball ball_prev, Ball ball_next, double rho = 0.25,
        double alpha_s = 1*PI/180, double alpha_e = 2*PI/180) noexcept:
    """
    Compute and apply the bending-angle restoring force at an interior ball.

    The force drives the bending angle at ball toward the target angle
    ball.angle, using the geometric construction from the original MAVIlib
    implementation (equivalent to Altendorf & Jeulin 2011 but numerically
    more stable).  The force magnitude is smoothed by smoothing_factor.
    Returns immediately (no force) if the angle error is below 1e-6 rad
    or if ball lies exactly on the axis between ball_prev and ball_next.

    :param ball: Ball
        the interior ball where the angle force is applied
    :param ball_prev: Ball
        the preceding ball in the fiber
    :param ball_next: Ball
        the following ball in the fiber
    :param rho: float (default: 0.25)
        force magnitude scaling factor
    :param alpha_s: float (default: 1°)
        smoothing onset angle in radians
    :param alpha_e: float (default: 2°)
        smoothing saturation angle in radians
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
    f = smoothing_factor(alpha0 - alpha, alpha_s, alpha_e) / z * rho * (z - z0) / 2.0
    ball.force[0] += (mx-coord[0])*f
    ball.force[1] += (my-coord[1])*f
    ball.force[2] += (mz-coord[2])*f
    ball.angle_diff = alpha0 - alpha
    ball.optim_sum += f*f/(rho*rho)


cdef inline double smoothing_factor(double x, double x_s, double x_e) noexcept:
    """
    Cut-off smoothing function mapping a ratio to [0, 1].

    Returns 0 for x < x_s, 1 for x > x_e, and a smooth cosine ramp
    0.5*(1 - cos(pi*(x-x_s)/(x_e-x_s))) in between.  Named after the
    notation in Altendorf & Jeulin (2011).

    :param x: float
        input value (typically a normalised displacement or angle)
    :param x_s: float
        lower threshold; below this the factor is 0
    :param x_e: float
        upper threshold; above this the factor is 1
    :return: float
        smoothing factor in [0, 1]
    """
    cdef double ratio
    if x < x_s:
        return 0
    elif x > x_e:
        return 1
    ratio = (x - x_s) / (x_e - x_s)
    return 0.5 * (1 - cos(ratio * PI))


cdef double calculate_contact_force_periodic(Ball ball, Ball ball2, int[3] image_size,
    double contact_distance, double rho = 0.25) noexcept:
    """
    Compute and apply the contact attraction force between two balls under
    periodic boundary conditions.

    The force is active when the surface-to-surface distance
    dist - (r1 + r2) is positive (balls not overlapping) and pulls the
    balls toward their touching distance.  The magnitude is smoothed by
    smoothing_factor over the range [contact_distance/2, contact_distance].

    :param ball: Ball
        first ball; force added (pulled toward ball2)
    :param ball2: Ball
        second ball; force subtracted (pulled toward ball)
    :param image_size: int[3]
        domain dimensions for periodic wrapping
    :param contact_distance: float
        maximum surface-to-surface distance for force activation
    :param rho: float (default: 0.25)
        force magnitude scaling factor
    :return: float
        residual gap max(0, dist - (r1+r2) - contact_distance), i.e. how
        far outside the contact zone the balls still are; 0 if in contact
    """
    cdef double dist, displaced, dist_perfect, displace, true_displace, force_strength
    cdef double dx, dy, dz, coordx, coordy, coordz, coord2x, coord2y, coord2z
    cdef double optim_summand
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
        force_strength = displace / 2.0*smoothing_factor(displace/dist, contact_distance/2., contact_distance)
        if dist > 0.0:
            force_strength /= dist
        optim_summand = force_strength*force_strength
        force_strength *= rho
        ball.force[0] += force_strength * dx
        ball.force[1] += force_strength * dy
        ball.force[2] += force_strength * dz
        ball.optim_sum += optim_summand
        ball2.force[0] -= force_strength * dx
        ball2.force[1] -= force_strength * dy
        ball2.force[2] -= force_strength * dz
        ball2.optim_sum += optim_summand

        return max(0, true_displace)
    return 0


cdef double calculate_contact_force_nonperiodic(Ball ball, Ball ball2,
    double contact_distance, double rho = 0.25) noexcept:
    """
    Compute and apply the contact attraction force between two balls without
    periodic boundary conditions.

    Identical to calculate_contact_force_periodic except distances are
    Euclidean rather than periodic.

    :param ball: Ball
        first ball; force added (pulled toward ball2)
    :param ball2: Ball
        second ball; force subtracted (pulled toward ball)
    :param contact_distance: float
        maximum surface-to-surface distance for force activation
    :param rho: float (default: 0.25)
        force magnitude scaling factor
    :return: float
        residual gap max(0, dist - (r1+r2) - contact_distance); 0 if in contact
    """
    cdef double dist, displaced, dist_perfect, displace, force_strength
    cdef double dx, dy, dz

    dist = cdirection(ball2, ball, &dx, &dy, &dz)
    dist_perfect = ball.radius + ball2.radius
    displace = dist - dist_perfect

    if displace > 0:
        force_strength = displace / 2.0*smoothing_factor(displace/dist, contact_distance/2., contact_distance)
        if dist > 0.0:
            force_strength /= dist
        optim_summand = force_strength*force_strength
        force_strength *= rho
        ball.force[0] += force_strength * dx
        ball.force[1] += force_strength * dy
        ball.force[2] += force_strength * dz
        ball.optim_sum += optim_summand
        ball2.force[0] -= force_strength * dx
        ball2.force[1] -= force_strength * dy
        ball2.force[2] -= force_strength * dz
        ball2.optim_sum += optim_summand

        return max(0, displace - contact_distance)
    return 0





def apply_forces(fiber_system: list[Fiber]):
    """
    Apply accumulated forces to ball positions and reset per-step state.

    For each ball: adds ball.force to ball.coordinate, then resets
    ball.force to (0,0,0), ball.overlap to 0, ball.neighbor_dist to
    radius/2, ball.angle_diff to 0, and ball.optim_sum to 0.

    Must be called once per packing step, after calculate_forces.

    :param fiber_system: list[Fiber]
        the full fiber system; modified in place
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
