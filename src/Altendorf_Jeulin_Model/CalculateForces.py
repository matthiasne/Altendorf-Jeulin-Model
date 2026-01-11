import math

import numpy as np
from skspatial.objects import Line, Plane

import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.utils import periodic_distance, angle_between, normalized
from Altendorf_Jeulin_Model.Fiber import Fiber, Ball

MIN_REPULSION_DISTANCE = 5
X_S = 0.05
X_E = 0.1
ALPHA_S = 0.1 * np.pi / 180
ALPHA_E = 0.2 * np.pi / 180


def calculate_forces(grid: sh, fiber_system: list[Fiber], rho: float = 0.2):
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
            total_angle_diff = max(total_angle_diff, ball.angle_diff)
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


def calculate_repulsion_force(i: int, ball: Ball, cell: list[Ball], grid: sh):
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
    for ball2 in cell[i + 1:]:
        # calculate repulsion forces
        if (ball.fiber_label != ball2.fiber_label
                or abs(ball.ball_label - ball2.ball_label) >= MIN_REPULSION_DISTANCE):
            dist, dir = periodic_distance(ball.coordinate, ball2.coordinate, grid.image_size) #TODO in add_repulsion_force ziehen
            add_repulsion_force(ball, ball2, dist, dir)
    # compare with neighbor cells
    cell_index_ball = grid.get_cell_index_of_coord(ball.coordinate)
    neighbor_cells = grid.get_younger_neighbor_cell_indices(cell_index_ball)
    for cell_index in neighbor_cells:
        cell_index_short = cell_index[0] + cell_index[1] * grid.division[0] + cell_index[2] * grid.division[1]*grid.division[0]
        cell = grid.cells[cell_index_short]
        for ball2 in cell:
            if (ball.fiber_label != ball2.fiber_label or
                    abs(ball.ball_label - ball2.ball_label) >= MIN_REPULSION_DISTANCE):
                dist, dir = periodic_distance(ball.coordinate, ball2.coordinate, grid.image_size)
                add_repulsion_force(ball, ball2, dist, dir)


def add_repulsion_force(ball: Ball, neighbor: Ball, dist: float, dir: np.ndarray):
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
    """
    doesOverlap = (dist - ball.radius - neighbor.radius < 0)
    if doesOverlap:
        overlap = abs(dist - ball.radius - neighbor.radius)
        ball.force = ball.force - overlap / 2.0 * dir
        ball.overlap = max(ball.overlap, overlap)
        neighbor.force = neighbor.force + overlap / 2.0 * dir
        neighbor.overlap = max(neighbor.overlap, overlap)


def add_recover_force(ball: Ball, force: np.ndarray, dist:float):
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

def add_spring_force(ball: Ball, force: np.ndarray, dist:float):
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

def add_angle_force(ball: Ball, force: np.ndarray, angle_diff:float):
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
    ball.angle_diff = max(ball.angle_diff, abs(angle_diff))


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
    dist_is, dir = normalized(ball2.coordinate - ball1.coordinate)
    # distance to next ball is currently always radius
    # - may need to adapt for different random walks
    dist_should = ball1.radius/2.0 if is_next else ball2.radius/2.0 #TODO
    dist_displaced = dist_is - dist_should
    ratio_displaced = abs(dist_displaced) / dist_should
    #ratio_displaced = dist_displaced / dist_should
    # smoothing_factor
    s_f = smoothing_factor(ratio_displaced, X_S, X_E)
    # add to recoverforce
    spring_force = s_f * rho * dist_displaced * dir
    add_spring_force(ball1, spring_force, dist_is)


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
    alpha0 = ball.angle
    # m: line cutting plane
    line = Line(ball_prev.coordinate, direction=ball_next.coordinate - ball_prev.coordinate)
    plane = Plane(ball.coordinate, normal=ball_prev.coordinate - ball_next.coordinate)
    m = plane.intersect_line(line)

    # z, z0: calculate distances of ball.coordinate to m
    h1 = np.linalg.norm(m - ball_prev.coordinate)
    h2 = np.linalg.norm(m - ball_next.coordinate)
    z = np.linalg.norm(m - ball.coordinate)
    alpha1 = np.atan2(h1, z)
    alpha2 = np.atan2(h2, z)

    #tan_alpha = np.tan(alpha1 + alpha2)
    tan_alpha = z*(h1 + h2)/(z**2 - h1*h2)

    if math.isclose(tan_alpha, 0):
        z0 = 0
    else:
        z0 = (h1 + h2 + np.sqrt((h1 + h2)**2 + 4*h1*h2*tan_alpha**2))/(2*tan_alpha)

    # calculate force
    _, force_dir = normalized(m - ball.coordinate)
    f = smoothing_factor(alpha0 - (alpha1 + alpha2), ALPHA_S, ALPHA_E)
    force = f*(z - z0)*force_dir
    #force = f * force_dir

    add_angle_force(ball, force, alpha0 - (alpha1 + alpha2))

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
            ball.neighbor_dist = ball.radius/2.
            ball.angle_diff = 0
