# cython: language_level=3, infer_type=True, exception_check=False, cdivision=True
import numpy as np
import cython
from libc.stdint cimport int64_t
cimport numpy as np
np.import_array()

cdef class Ball:
    """
    Ball contains balls used in the fiber model

    :param coordinate: np.ndarray
        coordinate of the center of the ball
    :param radius: float
        radius of the ball
    :param fiber_label: int
        label of the fiber that the ball belongs to
    :param ball_label: int
        index of the ball within the fiber
    :param angle: float
        angle between the incident edges of the ball
    """
    cdef public double radius
    cdef public int fiber_label
    cdef public int ball_label
    cdef public double coordinate[3]
    cdef public double force[3]
    cdef public double overlap
    cdef public double angle
    cdef public double neighbor_dist
    cdef public double angle_diff
    cdef public double optim_sum

