# cython: language_level=3, infer_type=True

import cython
from Altendorf_Jeulin_Model.SpatialHashing cimport SpatialHashing
from Altendorf_Jeulin_Model.Fiber cimport Ball
from Altendorf_Jeulin_Model.utils cimport cdistance_ball

from libc.math cimport sqrt
from libc.stdint cimport int64_t
cimport numpy as np
np.import_array()

import numpy as np
import networkx as nx

def find_contact_areas(fs, image_size, is_periodic: bool, contact_distance: float = 0):
    """
    Calculates statistics on the contact between fibers

    :param fs:
        list of fibers
    :param image_size:
        image size (in micrometer)
    :param is_periodic: bool
        whether or not the fiber system is periodic
    :param contact_distance: float, default 0
        The maximal distance that balls can have to be considered in contact
    :return n_connected_components, n_clots, n_contact_areas, contact_surface: int, int, int, float
        number of connected components, number of clots (more than two fibers in a contact area),
         number of contact areas, measure of contact surface
    """
    contact_pairs = find_contact_pairs(fs, image_size, is_periodic= is_periodic, contact_distance=contact_distance)
    contact_graph = nx.Graph(contact_pairs)

    # extend contact pairs by fiber edges to find connected components
    incident_fiber_edges = set()
    for pair in contact_pairs:
        for node in pair:
            if node[1] + 1 < fs[node[0]].get_number_of_balls():
                incident_fiber_edges.add(((node), (node[0], node[1] + 1)))
            if node[1] - 1 >= 0:
                incident_fiber_edges.add(((node[0], node[1] - 1), (node)))
    fiber_parts = nx.Graph(incident_fiber_edges)
    ext_contact_graph = nx.compose(fiber_parts, contact_graph)

    # find number of pairwise contact areas
    n_connected_components = 0
    n_contact_areas = 0
    contact_surface = 0
    n_clots = 0
    cc = [ext_contact_graph.subgraph(c).copy() for c in nx.connected_components(ext_contact_graph)]
    for component in cc:
        fiber_edges = incident_fiber_edges.intersection(set(component.edges))
        if len(fiber_edges) == 0:
            continue
        n_connected_components += 1
        fiber_sub_graph = nx.intersection(component, fiber_parts)
        fibers_in_contact = [fiber_sub_graph.subgraph(c).copy() for c in nx.connected_components(fiber_sub_graph)]
        if len(fibers_in_contact) > 2:
            n_clots += 1
        # calculate measure for the contact surface
        for fiber in fibers_in_contact:
            contact_partners = {nbr_ball[0] for ball in fiber if contact_graph.has_node(ball)
                for nbr_ball in contact_graph.adj[ball]
            }
            n_contact_areas += len(contact_partners)
            sub_fiber_length = [fs[ball[0]].balls[ball[1]].neighbor_dist for ball in fiber if fiber_sub_graph.has_node(ball)]
            contact_surface += np.sum(sub_fiber_length)
    return n_connected_components, n_clots, n_contact_areas, contact_surface

def find_contact_candidates(fs, image_size, is_periodic:bool = True, interaction_distance: float = 0):
    """
    finds contact candidates and calculates a shortlist

    :param fs: list of fibers
    :param image_size:
        image size (in micrometers)
    :param is_periodic: bool, default True
        whether or not to use periodic boundary conditions
    :param interaction_distance: float, default = 0
        largest distance between balls to make them contact candidates
    :return:
        shortlist (see Keilmann et al. (2026) or tbd)

    """
    contact_pairs = find_contact_pairs(fs, image_size, is_periodic=is_periodic, contact_distance=interaction_distance,
                                       is_weighted=True)
    contact_graph = nx.Graph()
    contact_graph.add_weighted_edges_from(contact_pairs)

    node_min = {
        node: min(data["weight"] for _, _, data in contact_graph.edges(node, data=True))
        for node in contact_graph
    }
    neighborhood_min = {}

    for node in contact_graph:
        best = node_min[node]

        for neighbor in contact_graph.neighbors(node):
            best = min(best, node_min[neighbor])

        neighborhood_min[node] = best

    to_remove = [
        (u, v)
        for u, v, data in contact_graph.edges(data=True)
        if (
                data["weight"] > neighborhood_min[u]
                or data["weight"] > neighborhood_min[v]
        )
    ]

    contact_graph.remove_edges_from(to_remove)
    return contact_graph.edges


def find_contact_pairs(fs, image_size, boundary_size: int = 0, is_periodic: bool=True,
                       contact_distance: float = 0, is_weighted: bool=False):
    """
    finds ball pairs that have a distance of at most contact_distance. This method is both used to determine
    contact areas and contact candidates/the shortlist

    :param fs:
        list of fibers
    :param image_size:
        size of the image
    :param boundary_size: int, default 0
        boundary size of the image, if it exists
    :param is_periodic: bool, default True
        whether the image is periodic or not
    :param contact_distance: float, default 0
        the maximal distance between balls under which they are considered "in contact"
    :param is_weighted: bool, default False
        whether the returned list of edges should be returned with weights or not
        This is relevant to calculate the shortlist
    :return:
        set of pairs of balls that are "in contact"

    """
    cdef Ball ball
    cdef int len_cell
    cdef int[3] index
    cdef set neighbor_cells, contact_set
    cdef set contact_pairs = set()
    max_radius = max(fiber.get_max_radius() for fiber in fs)
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    if not is_periodic:
        image_size = image_size + 2 * boundary_size_vec
    grid = SpatialHashing(image_size, 2.5 * (max_radius + contact_distance))
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    
    for cell in grid.cells:
        len_cell = len(cell)
        if len_cell > 0:
            index = grid.get_cell_index_of_coord(cell[0].coordinate[0], cell[0].coordinate[1], cell[0].coordinate[2])
            neighbor_cells = grid.get_younger_neighbor_cell_indices(index[0], index[1], index[2], is_periodic)
            for i in range(len_cell):
                ball = cell[i]
                contact_set = identify_contact_partners(
                    i, ball, cell, grid, neighbor_cells, is_periodic=is_periodic, contact_distance=contact_distance, is_weighted=is_weighted
                )
                contact_pairs.update(contact_set)
    return contact_pairs

cdef set identify_contact_partners(int i, Ball ball, list cell, SpatialHashing grid,
                              set neighbor_cells, bint is_periodic =True,
                              double contact_distance = 0, bint is_weighted =False):
    """
    identify neighbors that have a distance of at most contact_distance to ball
    This method is called from find_contact_pairs

    :param i: int
        ball index in its cell of the spatial hashing grid
    :param ball: Ball
        ball whose contact partners are supposed to be found
    :param cell:
        cell in the grid in which ball is stored
    :param grid: SpatialHashing
        spatial hashing grid
    :param neighbor_cells:
        list of neighbor cells
    :param is_periodic: bool, default True
        whether the image is periodic or not
    :param contact_distance: float, default 0
        the maximal distance between balls under which they are considered "in contact"
    :param is_weighted: bool, default False
        whether the returned list of edges should be returned with weights or not
    :return: list
        list of contact pairs in the format ((fiber label, ball label), (fiber label, ball label))
        or ((fiber label, ball label), (fiber label, ball label), distance) if is_weighted == True
    """
    cdef Ball ball2
    cdef list neighbor_cell
    cdef j
    cdef int n = len(cell)
    cdef int[3] image_size = grid.image_size
    cdef int fiber_label = ball.fiber_label
    cdef int ball_label = ball.ball_label
    cdef set contact_pairs = set()
    cdef tuple pair
    cdef bint is_in_contact
    cdef double weight
    # compare within cell
    for j in range(i+1, n):
        ball2 = cell[j]
        if is_periodic:
            is_in_contact, weight = test_in_contact_periodic(ball, ball2, image_size, contact_distance)
        else:
            is_in_contact, weight = test_in_contact_nonperiodic(ball, ball2, contact_distance)
        if is_in_contact:
            if is_weighted:
                pair = ((fiber_label, ball_label), (ball2.fiber_label, ball2.ball_label), weight)
            else:
                pair = ((fiber_label, ball_label), (ball2.fiber_label, ball2.ball_label))
            contact_pairs.add(pair)
    # compare with neighbor cells
    for cell_index in neighbor_cells:
        cell = grid.cells[cell_index]
        for ball2 in cell:
            if is_periodic:
                is_in_contact, weight = test_in_contact_periodic(ball, ball2, image_size, contact_distance)
            else:
                is_in_contact, weight = test_in_contact_nonperiodic(ball, ball2, contact_distance)
            if is_in_contact:
                if is_weighted:
                    pair = ((fiber_label, ball_label), (ball2.fiber_label, ball2.ball_label), weight)
                else:
                    pair = ((fiber_label, ball_label), (ball2.fiber_label, ball2.ball_label))
                contact_pairs.add(pair)
    return contact_pairs

def test_in_contact(ball, ball2, image_size, is_periodic, contact_distance):
    cdef int[3] image_size_loc

    if is_periodic:
        image_size_loc[0] = image_size[0]
        image_size_loc[1] = image_size[1]
        image_size_loc[2] = image_size[2]
        return test_in_contact_periodic(ball, ball2, image_size_loc, contact_distance)
    else:
        return test_in_contact_nonperiodic(ball, ball2, contact_distance)

cdef tuple test_in_contact_periodic(Ball ball, Ball ball2, int[3] image_size,
                double contact_distance = 0) noexcept:
    """
    test whether neighbors have a distance of at most contact_distance in the periodic case
    This function is called by identify_contact_partners

    :param ball: Ball
    :param ball2: Ball
    :param image_size: int64_t[:]
        image size (in micrometer)
    :param contact_distance: float, default 0
        The maximal distance that balls can have to be considered in contact
    :return: bool, float
        whether the neighbors are "in contact", and if so, the distance, otherwise -1
    """
    cdef double dist, displaced, dist_perfect, weight
    cdef double dx, dy, dz, coordx, coordy, coordz, coord2x, coord2y, coord2z
    if (
        ball.fiber_label != ball2.fiber_label
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
        dist_perfect = ball.radius + ball2.radius
        weight = dist - dist_perfect

        if weight <= contact_distance:
            return True, weight

    return False, -1


cdef tuple test_in_contact_nonperiodic(Ball ball, Ball ball2, double contact_distance = 0) noexcept:
    """
    test whether neighbors have a distance of at most contact_distance in the nonperiodic case
    This function is called by identify_contact_partners

    :param ball: Ball
    :param ball2: Ball
    :param contact_distance: float, default 0
        The maximal distance that balls can have to be considered in contact
    :return: bool, float
        whether the neighbors are "in contact", and if so, the distance, otherwise -1
    """
    cdef double dist, weight
    if (
        ball.fiber_label != ball2.fiber_label
    ):
        dist = cdistance_ball(ball2, ball)
        weight = dist - (ball.radius + ball2.radius)
        if weight <= contact_distance:
            return True, weight

    return False, -1