import numpy as np

import SpatialHashing
from utils import periodic_distance
from Fiber import Ball

MIN_REPULSION_DISTANCE = 5
X_S = 0.05
X_E = 0.1


def calculate_forces(grid: SpatialHashing, fiber_system: list[list[Ball]], rho: float = 0.2):
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
    """

    for cell in grid.cells:
        for i, ball in enumerate(cell):
            calculate_repulsion_force(i, ball, cell, grid)
    for fiber in fiber_system:
        for i, ball in enumerate(fiber):
            if (i + 1 < len(fiber)):
                calculate_spring_force(ball, fiber[i + 1], is_next=True, rho=rho)
            if (i - 1 >= 0):
                calculate_spring_force(fiber[i - 1], ball, is_next=False, rho=rho)


def calculate_repulsion_force(i: int, ball: Ball, cell: list[Ball], grid: SpatialHashing):
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
    for neighbor in cell[i + 1:]:
        # calculate repulsion forces
        if (ball.fiber_label != neighbor.fiber_label or
                abs(ball.ball_label - neighbor.ball_label) >= MIN_REPULSION_DISTANCE):
            # print("fiberlabels = ", ball.fiber_label, ", ", neighbor.fiber_label, " balllabels = ",
            #      ball.ball_label, ", ", neighbor.ball_label)
            dist, dir = periodic_distance(ball.coordinate, neighbor.coordinate, grid.image_size)
            add_repulsion_force(ball, neighbor, dist, dir)
    # compare with neighbor cells
    cell_index = grid.get_cell_index_of_coord(ball.coordinate)
    neighbor_cells = grid.get_younger_neighbor_cell_indices(cell_index)
    for cell_index in neighbor_cells:
        cell_index_short = cell_index[0] + cell_index[1] * grid.division[0] + cell_index[2] * grid.division[1]
        cell = grid.cells[cell_index_short]
        for neighbor in cell:
            if (ball.fiber_label != neighbor.fiber_label or
                    abs(ball.ball_label - neighbor.ball_label) >= MIN_REPULSION_DISTANCE):
                # print("fiberlabels = ", ball.fiber_label, ", ", neighbor.fiber_label, " balllabels = ",
                #      ball.ball_label, ", ", neighbor.ball_label)
                dist, dir = periodic_distance(ball.coordinate, neighbor.coordinate, grid.image_size)
                add_repulsion_force(ball, neighbor, dist, dir)


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
    if (doesOverlap):
        overlap = abs(dist - ball.radius - neighbor.radius)
        ball.force -= overlap / 2.0 * dir
        neighbor.force += overlap / 2.0 * dir


def add_recover_force(ball: Ball, force: np.ndarray):
    """
    Adds a recovery force to a ball

    Attributes
    ---------------------
    :param ball: Ball
        The ball that the force is added to
    :param force: np.ndarray
        The force that is added to the ball
    """
    ball.force += force


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
    Calculates the spring force between 2 balls
    and adds it to corresponding balls

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
    dir = ball2.coordinate - ball1.coordinate  # as original code - no periodicity necessary?
    dist_is = np.linalg.norm(dir)
    dir = dir / dist_is
    # distance to next ball is currently always radius
    # - may need to adapt for different random walks
    dist_should = ball1.radius if is_next else ball2.radius
    dist_displaced = dist_is - dist_should
    ratio_displaced = dist_displaced / dist_should
    # smoothing_factor
    s_f = smoothing_factor(ratio_displaced, X_S, X_E)
    # add to recoverforce
    spring_force = s_f * rho * dist_displaced / 2 * dir
    add_recover_force(ball1, spring_force)


def apply_forces(fiber_system: list[list[Ball]]):
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
        for ball in fiber:
            old_coord = ball.coordinate
            new_coord = old_coord + ball.force
            ball.coordinate = new_coord
            ball.force = np.array([0, 0, 0])
