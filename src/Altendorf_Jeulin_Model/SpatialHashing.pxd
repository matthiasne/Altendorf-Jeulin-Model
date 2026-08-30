# cython: language_level=3, infer_type=True
import cython
from itertools import product
from math import ceil, floor
from libc.stdint cimport int64_t

import numpy as np
from Altendorf_Jeulin_Model.Fiber cimport Ball


cdef class SpatialHashing:
    cdef public int image_size[3]
    cdef public int n_cells
    cdef public list[list[Ball]] cells
    cdef public int division[3]
    cdef public int cell_width[3]

    cdef inline tuple get_cell_index_of_coord(
        self, double posx, double posy, double posz
    )
    cdef inline void add_ball(self, Ball ball, bint is_periodic)
    cdef void add_fiber(self, object fiber, bint is_periodic)
    cdef set get_younger_neighbor_cell_indices(
        self, int i, int j, int k, bint is_periodic
    )