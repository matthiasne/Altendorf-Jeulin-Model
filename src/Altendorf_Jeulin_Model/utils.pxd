# cython: boundscheck=False
import cython
from Altendorf_Jeulin_Model.Fiber cimport Ball


cdef double cdistance_ball(Ball a, Ball b) noexcept nogil
cdef double cdistance3(double ax, double ax, double az, double bx, double by, double bz) noexcept nogil
cdef double cdirection(Ball a, Ball b, double* dir_x, double* dir_y, double* dir_z) noexcept