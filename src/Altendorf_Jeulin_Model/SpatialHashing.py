from math import floor, ceil
from itertools import product
import numpy as np
from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from line_profiler import profile


class SpatialHashing:
    """
    Contains the grid for spatial hashing and implements all necessary functions

    Attributes
    ---------------------
    image_size : tuple[int, int, int]
        The size of the image
    division : tuple[int, int, int]
        The number of cells in each direction x, y, and z
    n_cells : int
        The total number of cells in the grid
    cells : list[list[Ball]]
        A list, which represents the cells, which lists the balls it contains
    cell_width : tuple[int, int, int]
        The cell size in each direction x, y, and z
    """

    @profile
    def __init__(self, image_size: tuple[int, int, int], max_cell_size: int):
        """
        Initializes the SpatialHashing object, i.e. a grid for spatial hashing

        Parameters
        ---------------------
        :param image_size: tuple[int, int, int]
            The size of the image that the grid needs to cover
        :param max_cell_size: int
            The maximal size for each cell
        """
        self.image_size = image_size
        self.division = tuple(ceil(size / max_cell_size) for size in image_size)
        self.n_cells = self.division[0] * self.division[1] * self.division[2]
        "TODO: this must be implemented as spheres instead"
        self.cells = [[] for _ in range(self.n_cells)]
        self.cell_width = tuple(ceil(size / div) for size, div in zip(image_size, self.division))

    @profile
    def _in_bounds(self, index: tuple[int, int, int]) -> bool:
        """
        test whether an index is within bounds of the SpatialHashing

        Parameters
        ---------------------
        :param index: tuple[int, int, int]
            The tuple to be tested
        :return: bool
            True: the index is within bounds, False: the index is not within bounds
        """
        return all(0 <= c < m for c, m in zip(index, self.division))

    @profile
    def get_cell_index_of_coord(self, position: np.ndarray) -> tuple[int, int, int]:
        """
        gets the cell index of a specific coordinate

        Parameters
        ---------------------
        :param position: np.ndarray
            The coordinate for which the cell index is sought
        :return: tuple[int, int, int]
            The cell index
        """

        pos_mod = (position[0] % self.image_size[0], position[1] % self.image_size[1],
                   position[2] % self.image_size[2])
        return tuple(floor(coord / width) for coord, width in zip(pos_mod, self.cell_width))

    @profile
    def get_neighbor_cell_indices(self, index: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        """
        Gets the indices of all neighboring cells
        :param index: tuple[int, int, int]
            The original cell index
        :return: list[tuple[int, int, int]]
            The list of neighbor cells (indices)
        """
        if not self._in_bounds(index):
            raise ValueError(f"Index {index} outside of bounds")

        neighbor_cells: list[tuple[int, int, int]] = []
        for di, dj, dk in product((-1, 0, 1), repeat=3):
            if di == dj == dk == 0:
                continue  # skip the cell itself
            neighbor = ((index[0] + di) % self.division[0], (index[1] + dj) % self.division[1],
                        (index[2] + dk) % self.division[2])
            if self._in_bounds(neighbor) and neighbor not in neighbor_cells:
                neighbor_cells.append(neighbor)
        return neighbor_cells

    @profile
    def get_younger_neighbor_cell_indices(self, index: tuple[int, int, int]):
        """
        Gets the indices of all neighboring cells
        :param index: tuple[int, int, int]
            The original cell index
        :return: list[tuple[int, int, int]]
            The list of neighbor cells (indices)
        """

        neighbor_cells = set()
        for di, dj, dk in product((-1, 0, 1), repeat=3):
            if (di, dj, dk) < (0, 0, 0):
                neighbor = ((index[0] + di) % self.division[0] + ((index[1] + dj) % self.division[1])*self.division[0]
                            + ((index[2] + dk) % self.division[2])*self.division[0]*self.division[1])
                neighbor_cells.add(neighbor)
        return neighbor_cells

    @profile
    def add_ball(self, ball: Ball):
        """
        Adds a ball to the SpatialHashing

        Parameters
        ---------------------
        :param ball: Ball
            The ball that is to be added
        """
        coord = ball.coordinate
        index = self.get_cell_index_of_coord(coord % self.image_size)
        idx = index[0] + index[1] * self.division[0] + index[2] * self.division[0] * self.division[1]
        self.cells[idx].append(ball)

    @profile
    def add_fiber(self, fiber: Fiber):
        """
        Adds a fiber to the SpatialHashing

        Parameters
        ---------------------
        :param fiber: list[Ball]
            The fiber to be added
        """
        for ball in fiber.balls:
            self.add_ball(ball)

    @profile
    def add_fiber_system(self, fiber_system: list[Fiber]):
        """
        Adds a fiber system to the SpatialHashing

        Parameters
        ---------------------
        :param fiber_system: list[list[Ball]]
        """
        for fiber in fiber_system:
            self.add_fiber(fiber)
