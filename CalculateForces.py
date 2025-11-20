import numpy as np

import SpatialHashing
from utils import periodic_distance
from Fiber import Ball

MIN_REPULSION_DISTANCE = 5


def calculate_forces(grid: SpatialHashing):
    """
    Calculates forces in the fiber system

    Attributes
    ---------------------
    :param grid: SpatialHashing
        The spatial hashing grid for the model
    """
    for cell in grid.cells:
        for i, ball in enumerate(cell):
            calculate_repulsion_force(i, ball, cell, grid)


def calculate_repulsion_force(i: int, ball: Ball, cell: list[Ball], grid: SpatialHashing):
    """
    Calculates the repulsion force for the whole fiber system

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


def add_repulsion_force(ball: Ball, neighbor: Ball, dist: float, dir:np.ndarray):
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


def apply_forces(fiber_system: list[list[Ball]]):
    """
    Applies forces to the fiber system
    - it adds their forces to their coordinates, thus moves the balls
    - it sets all forces to 0
    :param fiber_system: list[list[Ball]])
        The fiber system that contains all balls
    """
    for fiber in fiber_system:
        for ball in fiber:
            old_coord = ball.coordinate
            new_coord = old_coord + ball.force
            ball.coordinate = new_coord
            ball.force = np.array([0, 0, 0])
