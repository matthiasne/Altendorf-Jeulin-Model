# cython: language_level=3, infer_type=True
import cython
from math import ceil, floor

from Altendorf_Jeulin_Model.Fiber cimport Ball
from Altendorf_Jeulin_Model.Fiber import Fiber


cdef class SpatialHashing:
    """
    Contains the grid for spatial hashing and implements all necessary functions

    :param image_size : tuple[int, int, int]
        The size of the image
    :param division : tuple[int, int, int]
        The number of cells in each direction x, y, and z
    :param n_cells : int
        The total number of cells in the grid
    :param cells : list[list[Ball]]
        A list, which represents the cells, which lists the balls it contains
    :param cell_width : tuple[int, int, int]
        The cell size in each direction x, y, and z
    """

    def __cinit__(self, image_size, max_cell_size: int):
        """
        Initializes the SpatialHashing object, i.e. a grid for spatial hashing

        :param image_size: tuple[int, int, int]
            The size of the image that the grid needs to cover
        :param max_cell_size: int
            The maximal size for each cell
        """
        self.image_size[0] = image_size[0]
        self.image_size[1] = image_size[1]
        self.image_size[2] = image_size[2]
        self.division[0] = ceil(image_size[0] / max_cell_size)
        self.division[1] = ceil(image_size[1] / max_cell_size)
        self.division[2] = ceil(image_size[2] / max_cell_size)
        self.n_cells = self.division[0] * self.division[1] * self.division[2]
        self.cells = [[] for i in range(self.n_cells)]
        self.cell_width[0] = ceil(image_size[0] / self.division[0])
        self.cell_width[1] = ceil(image_size[1] / self.division[1])
        self.cell_width[2] = ceil(image_size[2] / self.division[2])


    cdef inline tuple get_cell_index_of_coord(
        self, double posx, double posy, double posz
    ):
        """
        gets the cell index of a specific coordinate

        :param position: np.ndarray
            The coordinate for which the cell index is sought
        :return: tuple[int, int, int]
            The cell index
        """
        return (floor(posx/self.cell_width[0]),
                     floor(posy/self.cell_width[1]),
                     floor(posz/self.cell_width[2]))


    cdef set get_younger_neighbor_cell_indices(
        self, int i, int j, int k, bint is_periodic
    ):
        """
        Gets the indices of all neighboring cells
        :param index: tuple[int, int, int]
            The original cell index
        :return: list[tuple[int, int, int]]
            The list of neighbor cells (indices)
        """

        cdef set neighbor_cells = set()
        cdef int di, dj, dk
        cdef int ni, nj, nk
        cdef int neighbor
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di > 0:
                        continue
                    if di == 0 and dj > 0:
                        continue
                    if di == 0 and dj == 0 and dk >= 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    nk = k + dk

                    if is_periodic:
                        ni %= self.division[0]
                        nj %= self.division[1]
                        nk %= self.division[2]
                    else:
                        if (
                                ni < 0 or ni >= self.division[0]
                                or nj < 0 or nj >= self.division[1]
                                or nk < 0 or nk >= self.division[2]
                        ):
                            continue

                    neighbor = (
                            ni
                            + nj * self.division[0]
                            + nk * self.division[0] * self.division[1]
                    )

                    neighbor_cells.add(neighbor)

        return neighbor_cells

    cdef inline void add_ball(self, Ball ball, bint is_periodic):
        """
        Adds a ball to the SpatialHashing

        :param ball: Ball
            The ball that is to be added
        """
        cdef double x = ball.coordinate[0]
        cdef double y = ball.coordinate[1]
        cdef double z = ball.coordinate[2]
        cdef int ix, iy, i, idx
        if is_periodic:
            x = x % self.image_size[0]
            y = y % self.image_size[1]
            z = z % self.image_size[2]

        ix = floor(x / self.cell_width[0])
        iy = floor(y / self.cell_width[1])
        iz = floor(z / self.cell_width[2])
        idx = (
            ix
            + iy * self.division[0]
            + iz * self.division[0] * self.division[1]
        )
        self.cells[idx].append(ball)

    cdef void add_fiber(self, object fiber, bint is_periodic):
        """
        Adds a fiber to the SpatialHashing

        :param fiber: Fiber
            The fiber to be added
        """
        cdef Ball ball
        for ball in fiber.balls:
            self.add_ball(ball, is_periodic)

    def add_fiber_system(self, list[object] fiber_system, bint is_periodic = True):
        """
        Adds a fiber system to the SpatialHashing

        :param fiber_system: list[Fiber]
        """
        for fiber in fiber_system:
            self.add_fiber(fiber, is_periodic)
