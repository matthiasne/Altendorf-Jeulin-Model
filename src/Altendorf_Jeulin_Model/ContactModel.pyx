# cython: language_level=3, infer_type=True
import cython
from libc.stdint cimport int64_t
cimport numpy as np
np.import_array()

import numpy as np
import networkx as nx
import itertools
from Altendorf_Jeulin_Model.CalculateForces import (
    apply_forces,
    calculate_forces,
    calculate_forces_endstep,
)

import Altendorf_Jeulin_Model.Fiber as Fiber
import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.io_utils import print_stats, print_stats_row


MIN_REPULSION_DISTANCE = 5

def find_contact_areas(fs, image_size, epsi = 0):
    contact_pairs = find_contact_pairs(fs, image_size, epsi=epsi)
    contact_graph = nx.Graph(contact_pairs)

    # extend contact pairs by fiber edges to find connected components
    incident_fiber_edges = set()
    for pair in contact_pairs:
        for node in pair:
            incident_fiber_edges.add(((node), (node[0], node[1] + 1)))
            incident_fiber_edges.add(((node), (node[0], node[1] - 1)))
    joined_edges = incident_fiber_edges.union(contact_pairs)
    ext_contact_graph = nx.Graph(joined_edges)

    # find number of pairwise contact areas
    n_contact_areas = 0
    n_clots = 0
    cc = [ext_contact_graph.subgraph(c).copy() for c in nx.connected_components(ext_contact_graph)]
    for component in cc:
        if len(component) > 2:
            n_clots += 1
        fiber_edges = incident_fiber_edges.intersection(set(component.edges))
        fiber_graph = nx.Graph(fiber_edges)
        fiber_parts = [fiber_graph.subgraph(c).copy() for c in nx.connected_components(fiber_graph)]
        for fiber in fiber_parts:
            contact_partners = {nbr_ball[0] for ball in fiber if contact_graph.has_node(ball)
                for nbr_ball in contact_graph.adj[ball]
            }
            n_contact_areas += len(contact_partners)
    n_contact_areas /= 2
    return len(cc), n_clots, n_contact_areas

def find_contact_candidates(fs, image_size, epsi = 0):
    contact_pairs = find_contact_pairs(fs, image_size, epsi=epsi, is_weighted=True)
    contact_graph = nx.Graph()
    contact_graph.add_weighted_edges_from(contact_pairs)

    # shortlist: iterate over edges and delete all that are not minimum (or first)
    to_remove = set()
    for u, v, data in contact_graph.edges(data=True):
        inc_edges = list(contact_graph.edges(nbunch=[u, v], data=True))
        min_weight = min(data.get("weight") for _, _, data in inc_edges)
        to_remove = to_remove.union(set([(a, b) for a, b, data in inc_edges if data.get("weight") > min_weight]))
    for a, b in to_remove:
        if contact_graph.has_edge(a, b):
            contact_graph.remove_edge(a, b)
    return contact_graph.edges

def find_contact_pairs(fs, image_size, boundary_size = 0, is_periodic=True, epsi = 0, is_weighted=False):
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
                    i, ball, cell, grid, neighbor_cells, is_periodic=is_periodic, epsi=epsi, is_weighted=is_weighted
                )
                contact_pairs.update(contact_set)
    return contact_pairs

def identify_contact_pairs_it(i, ball, cell, grid,
                              neighbor_cells, is_periodic: bool =True, epsi = 0, is_weighted=False):
    fiber_label: cython.int = ball.fiber_label
    label: cython.int = ball.ball_label
    coord = ball.coordinate

    contact_pairs = set()
    # compare within cell
    for ball2 in cell[i + 1 :]:
        is_in_contact, weight = identify_contact_pairs_within_cell(ball, ball2, is_periodic, coord, grid.image_size, epsi)
        if is_in_contact:
            if is_weighted:
                pair = ((ball.fiber_label, ball.ball_label), (ball2.fiber_label, ball2.ball_label), weight)
            else:
                pair = ((ball.fiber_label, ball.ball_label), (ball2.fiber_label, ball2.ball_label))
            contact_pairs.add(pair)
    # compare with neighbor cells
    for cell_index in neighbor_cells:
        cell = grid.cells[cell_index]
        for ball2 in cell:
            is_in_contact, weight = identify_contact_pairs_within_cell(ball, ball2, is_periodic, coord, grid.image_size, epsi)
            if is_in_contact:
                if is_weighted:
                    pair = ((ball.fiber_label, ball.ball_label), (ball2.fiber_label, ball2.ball_label), weight)
                else:
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
            weight = dist - (ball.radius + ball2.radius)
            if weight <= epsi:
                return True, weight

        else:
            coord2 = ball2.coordinate
            dist: cython.double = np.linalg.norm(coord2 - coord)
            weight = dist - (ball.radius + ball2.radius)
            if weight <= epsi:
                return True, weight

    return False, -1
