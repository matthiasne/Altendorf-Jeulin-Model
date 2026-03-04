import numpy as np
from skspatial.objects import Line, Plane

import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.Fiber import Fiber, Ball

from line_profiler import profile

MIN_REPULSION_DISTANCE = 5
X_S = 0.05
X_E = 0.1
ALPHA_S = 0.1 * np.pi / 180
ALPHA_E = 0.2 * np.pi / 180
TAU = 0.25

@profile
def calculate_forces(grid: sh, fiber_system: list[Fiber], max_step_size, rho: float = 0.2):
    """
    Calculates forces in the fiber system and adds them to corresponding ball

    Attributes
    ---------------------
    :param grid: SpatialHashing
        The spatial hashing grid for the model
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :param rho: float
        factor to balance forces, [0, 1]
        0 means no recover forces
        default 0.2 from Altendorf&Jeulin 2011
    :return: np.ndarray
        total force of the fiber system
    """
    for cell in grid.cells:
        if len(cell) > 0:
            neighbor_cells = grid.get_younger_neighbor_cell_indices(
                grid.get_cell_index_of_coord(cell[0].coordinate))
            for i, ball in enumerate(cell):
                calculate_repulsion_force(i, ball, cell, grid, neighbor_cells)
    for fiber in fiber_system:
        for i, ball in enumerate(fiber.balls):
            if (i + 1 < len(fiber.balls)):
                calculate_spring_force(ball, fiber.balls[i + 1], is_next=True, rho=rho)
            if (i - 1 >= 0):
                calculate_spring_force(ball, fiber.balls[i - 1], is_next=False, rho=rho)
            if (i - 1 >= 0 and i + 1 < len(fiber.balls)):
                calculate_angle_force(ball, fiber.balls[i - 1], fiber.balls[i + 1], rho)

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
    return np.linalg.norm(total_force), total_overlap, total_neighbor_dist, total_angle_diff


def calculate_forces_endstep(grid: sh, fiber_system: list[Fiber]):
    """
    Calculates forces in the fiber system and adds them to corresponding ball

    Attributes
    ---------------------
    :param grid: SpatialHashing
        The spatial hashing grid for the model
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    :param rho: float
        factor to balance forces, [0, 1]
        0 means no recover forces
        default 0.2 from Altendorf&Jeulin 2011
    :return: np.ndarray
        total force of the fiber system
    """

    for cell in grid.cells:
        for i, ball in enumerate(cell):
            calculate_repulsion_force(i, ball, cell, grid)

    total_force = np.array([0.0, 0.0, 0.0])
    total_overlap = 0
    for fiber in fiber_system:
        for ball in fiber.balls:
            total_force = total_force + ball.force
            total_overlap = max(total_overlap, ball.overlap)
    return np.linalg.norm(total_force), total_overlap

@profile
def calculate_repulsion_force(i: int, ball: Ball, cell: list[Ball], grid: sh, neighbor_cells):
    """
    Calculates the repulsion force for the whole fiber system
    and adds it to corresponding ball

    Attributes
    ---------------------
    :param i: int
        cell index of the current cell
    :param ball: Ball
        The ball whose neighbors are currently considered
    :param cell: list[Ball]
        The cell that the ball is saved in
    :param grid: SpatialHashing
        The spatial hashing grid of the mdel
    """
    # compare within cell
    image_size = np.asarray(grid.image_size, dtype=float)
    coord2mod = np.empty_like(image_size)
    coord = ball.coordinate%image_size
    label = ball.ball_label
    fiber_label = ball.fiber_label
    for ball2 in cell[i + 1:]:
        # calculate repulsion forces
        if (fiber_label != ball2.fiber_label
                or abs(label - ball2.ball_label) >= MIN_REPULSION_DISTANCE):
            np.mod(ball2.coordinate, image_size, out=coord2mod)

            for i in range(3):
                disp = coord2mod[i] - coord[i]
                if abs(disp) > image_size[i] / 2.:
                    if disp > 0:
                        coord2mod[i] -= image_size[i]
                    else:
                        coord2mod[i] += image_size[i]
                coord2mod[i] -= coord[i]
            dist = np.sqrt(np.square(coord2mod[0]) + np.square(coord2mod[1]) + np.square(coord2mod[2]))
            overlap = ball.radius + ball2.radius - dist
            if overlap > 0:
                coord2mod = coord2mod / dist
                ball.force = ball.force - TAU * overlap / 2.0 * coord2mod
                ball.overlap = max(ball.overlap, overlap)
                ball2.force = ball2.force + TAU * overlap / 2.0 * coord2mod
                ball2.overlap = max(ball2.overlap, overlap)

    # compare with neighbor cells
    #cell_index_ball = grid.get_cell_index_of_coord(coord)
    #neighbor_cells = grid.get_younger_neighbor_cell_indices(cell_index_ball)
    for cell_index in neighbor_cells:
        cell = grid.cells[cell_index]
        for ball2 in cell:
            if (fiber_label != ball2.fiber_label or
                    abs(label - ball2.ball_label) >= MIN_REPULSION_DISTANCE):
                np.mod(ball2.coordinate, image_size, out=coord2mod)

                for i in range(3):
                    disp = coord2mod[i] - coord[i]
                    if abs(disp) > image_size[i] / 2.:
                        if disp > 0:
                            coord2mod[i] -= image_size[i]
                        else:
                            coord2mod[i] += image_size[i]
                    coord2mod[i] -= coord[i]
                dist = np.sqrt(np.square(coord2mod[0]) + np.square(coord2mod[1]) + np.square(coord2mod[2]))
                overlap = ball.radius + ball2.radius - dist
                if overlap > 0:
                    coord2mod = coord2mod / dist
                    ball.force = ball.force - TAU * overlap / 2.0 * coord2mod
                    ball.overlap = max(ball.overlap, overlap)
                    ball2.force = ball2.force + TAU * overlap / 2.0 * coord2mod
                    ball2.overlap = max(ball2.overlap, overlap)

@profile
def add_repulsion_force(ball: Ball, neighbor: Ball, coord1mod, image_size):
    """
    Adds repulsion force to the balls it applies to

    Attributes
    ---------------------
    :param ball: Ball
        One of the balls
    :param neighbor: Ball
        The other ball
    :param dist: float
        The periodic distance between the balls
    :param dir: np.ndarray
        The direction vector of the periodic distance
    :param max_step_size: float
        discretization parameter from MAVIlib (??)
    """
    coord2mod = neighbor.coordinate % image_size

    for i in range(3):
        disp = coord2mod[i] - coord1mod[i]
        if abs(disp) > image_size[i] / 2.:
            if disp > 0:
                coord2mod[i] -= image_size[i]
            else:
                coord2mod[i] += image_size[i]
        coord2mod[i] -= coord1mod[i]
    dist = np.sqrt(np.square(coord2mod[0]) + np.square(coord2mod[1]) + np.square(coord2mod[2]))
    overlap = ball.radius + neighbor.radius - dist
    if overlap > 0:
        coord2mod = coord2mod/dist
        ball.force = ball.force - TAU*overlap / 2.0 * coord2mod
        ball.overlap = max(ball.overlap, overlap)
        neighbor.force = neighbor.force + TAU*overlap / 2.0 * coord2mod
        neighbor.overlap = max(neighbor.overlap, overlap)

@profile
def add_recover_force(ball: Ball, force: np.ndarray, dist: float):
    """
    Adds a recovery force to a ball

    Attributes
    ---------------------
    :param ball: Ball
        The ball that the force is added to
    :param force: np.ndarray
        The force that is added to the ball
    """
    ball.force = ball.force + force
    ball.neighbor_dist = max(ball.neighbor_dist, dist)

@profile
def add_spring_force(ball: Ball, force: np.ndarray, dist: float):
    """
    Adds a recovery force to a ball

    Attributes
    ---------------------
    :param ball: Ball
        The ball that the force is added to
    :param force: np.ndarray
        The force that is added to the ball
    """
    ball.force = ball.force + force
    ball.neighbor_dist = max(ball.neighbor_dist, dist)

@profile
def add_angle_force(ball: Ball, force: np.ndarray, angle_diff: float):
    """
    Adds a recovery force to a ball

    Attributes
    ---------------------
    :param ball: Ball
        The ball that the force is added to
    :param force: np.ndarray
        The force that is added to the ball
    """
    ball.force = ball.force + force
    ball.angle_diff = angle_diff

@profile
def smoothing_factor(x: float, x_s: float, x_e: float):
    """
    Calculate smoothing factor
    (arguments named after Altendorf&Jeulin 2011)

    Attributes
    ---------------------
    :param x: float
        ratio that is argument of smoothing factor
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
        ratio = (abs(x) - x_s) / (x_e - x_s)
        factor = 0.5 * (1 - np.cos(ratio * np.pi))
        return factor

@profile
def calculate_spring_force(ball1: Ball, ball2: Ball, is_next: bool, rho: float = 0.2):
    """
    Calculates the spring force between 2 balls and adds it to corresponding balls

    Attributes
    ---------------------
    :param ball1: Ball
        the ball that the force is added to
    :param ball2: Ball
        neighbor to ball1
    :param rho: float
        factor to balance forces, [0, 1]
        default 0.2 from Altendorf&Jeulin 2011
    :param is_next: bool
        indicates whether ball2 comes before or after ball1 in the fiber
    """
    # displacement
    force_dir = ball2.coordinate - ball1.coordinate
    dist_is = np.sqrt(np.square(force_dir[0]) + np.square(force_dir[1]) + np.square(force_dir[2]))

    # distance to next ball is currently always radius
    # - may need to adapt for different random walks
    dist_should = ball1.radius / 2.0 if is_next else ball2.radius / 2.0  # TODO
    dist_displaced = dist_is - dist_should
    ratio_displaced = abs(dist_displaced) / dist_should
    # smoothing_factor
    s_f = smoothing_factor(ratio_displaced, X_S, X_E) * rho * dist_displaced / dist_is
    # add to recoverforce
    for i in range(0,3):
        force_dir[i] = s_f * force_dir[i]
    ball1.force = ball1.force + force_dir
    ball1.neighbor_dist = max(ball1.neighbor_dist, dist_is)

@profile
def calculate_angle_force(ball: Ball, ball_prev: Ball, ball_next: Ball, rho=0.2):
    """
    Calculates angle force between 3 neighboring balls and adds it to the center ball
    Note: this code does not directly follow the paper by Altendorf&Jeulin
    because this caused strange errors. Instead, it uses an equivalent calculation
    that was proposed yet undocumented in the original code (MAVIlib)

    Attributes
    ---------------------
    :param ball: Ball
        center ball - this is where the force will be applied
    :param ball_prev: Ball
        previous ball
    :param ball_next: Ball
        next ball
    :param rho: float
        factor to balance forces, [0, 1]
        default 0.2 from Altendorf&Jeulin 2011
    """
    coord = ball.coordinate
    coord_prev = ball_prev.coordinate
    coord_next = ball_next.coordinate
    dir_next = np.empty_like(coord)
    dir_prev = np.empty_like(coord)
    a = np.empty_like(coord)
    m = np.empty_like(coord)

    alpha0 = ball.angle
    norm_prev = np.sqrt(np.square(coord[0] - coord_prev[0]) + np.square(coord[1] - coord_prev[1]) + np.square(coord[2] - coord_prev[2]))
    norm_next = np.sqrt(np.square(coord_next[0] - coord[0]) + np.square(coord_next[1] - coord[1]) + np.square(coord_next[2] - coord[2]))
    norm_a = np.sqrt(
        np.square(coord_next[0] - coord_prev[0]) + np.square(coord_next[1] - coord_prev[1]) + np.square(coord_next[2] - coord_prev[2]))
    for i in range(0,3):
        dir_next[i] = (coord_next[i] - coord[i]) / norm_next
        dir_prev[i] = (coord[i] - coord_prev[i])# / norm_prev
        a[i] = (coord_next[i] - coord_prev[i])/norm_a

    # m: line cutting plane
    #line = Line(coord_prev, direction=a)
    #plane = Plane(coord, normal=a)
    #m = plane.intersect_line(line)
    d = dir_prev[0]*a[0] + dir_prev[1]*a[1] + dir_prev[2]*a[2]
    for i in range(0,3):
        dir_prev[i] = dir_prev[i] / norm_prev
        m[i] = coord_prev[i] + d*a[i]
    alpha = np.pi - np.arccos(np.dot(dir_prev, dir_next))

    # z, z0: calculate distances of ball.coordinate to m
    h1 = d
    h2 = np.sqrt(np.square(m[0] - coord_next[0]) + np.square(m[1] - coord_next[1]) + np.square(m[2] - coord_next[2]))
    z = np.sqrt(np.square(m[0] - coord[0]) + np.square(m[1] - coord[1]) + np.square(m[2] - coord[2]))

    tan_alpha0 = np.tan(alpha0)
    if tan_alpha0 < 0:
        z0 = (h1 + h2 - np.sqrt(np.square((h1 + h2)) + 4 * h1 * h2 * np.square(np.tan(alpha0)))) / (2 * np.tan(alpha0))
    else:
        z0 = (h1 + h2 + np.sqrt(np.square(h1 + h2) + 4 * h1 * h2 * np.square(np.tan(alpha0)))) / (2 * np.tan(alpha0))

    # calculate force
    f = smoothing_factor(alpha0 - alpha, ALPHA_S, ALPHA_E)/ z*rho * (z - z0)/2.
    for i in range(0,3):
        ball.force[i] += (m[i] - coord[i]) * f
    ball.angle_diff = alpha0 - alpha

@profile
def apply_forces(fiber_system: list[Fiber]):
    """
    Applies forces to the fiber system
    - it adds their forces to their coordinates, thus moves the balls
    - it sets all forces to 0

    Attributes
    ---------------------
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
            ball.neighbor_dist = ball.radius / 2.
            ball.angle_diff = 0
