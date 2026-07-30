# cython: language_level=3, infer_type=True
import cython
from libc.stdint cimport int64_t
cimport numpy as np
np.import_array()

import numpy as np
from Altendorf_Jeulin_Model.CalculateForces import (
    apply_forces,
    calculate_forces,
    calculate_forces_endstep,
)

import Altendorf_Jeulin_Model.Fiber as Fiber
import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.io_utils import print_stats, print_stats_row


MIN_REPULSION_DISTANCE = 5



def find_contact_pairs(fs, image_size, boundary_size = 0, is_periodic=True):
    max_radius = max(fiber.get_max_radius() for fiber in fs)
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    if not is_periodic:
        image_size = image_size + 2 * boundary_size_vec
    grid = sh.SpatialHashing(image_size, 2.5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    
    contact_pairs = set()
    for cell in grid.cells:
        if len(cell) > 0:
            neighbor_cells = grid.get_younger_neighbor_cell_indices(
                grid.get_cell_index_of_coord(cell[0].coordinate)
            )
            for i, ball in enumerate(cell):
                contact_set = identify_contact_pairs_it(
                    i, ball, cell, grid, neighbor_cells, is_periodic=is_periodic
                )
                contact_pairs.update(contact_set)
    return contact_pairs

def identify_contact_pairs_it(i, ball, cell, grid, neighbor_cells, is_periodic: bool =True):
    fiber_label: cython.int = ball.fiber_label
    label: cython.int = ball.ball_label
    coord = ball.coordinate

    contact_pairs = set()
    # compare within cell
    for ball2 in cell[i + 1 :]:
        if identify_contact_pairs_within_cell(ball, ball2, is_periodic, coord, grid.image_size):
            pair = ((ball.fiber_label, ball.ball_label), (ball.fiber_label, ball.ball_label))
            contact_pairs.add(pair)
    # compare with neighbor cells
    for cell_index in neighbor_cells:
        cell = grid.cells[cell_index]
        for ball2 in cell:
            if identify_contact_pairs_within_cell(ball, ball2, is_periodic, coord, grid.image_size):
                pair = ((ball.fiber_label, ball.ball_label), (ball2.fiber_label, ball2.ball_label))
                contact_pairs.add(pair)
    return contact_pairs

def identify_contact_pairs_within_cell(ball, ball2, is_periodic: bool, double[:] coord, int64_t[:] image_size, epsi = 0):
    if (
        ball.fiber_label != ball2.fiber_label
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

            if dist - (ball.radius + ball2.radius) <= epsi:
                return True

        else:
            coord2 = ball2.coordinate
            dist: cython.double = np.linalg.norm(coord2 - coord)
            if dist - (ball.radius + ball2.radius) <= epsi:
                return True

    return False