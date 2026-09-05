# cython: boundscheck=False
import cython
from Altendorf_Jeulin_Model.Fiber cimport Ball

cdef void cartesian_to_spherical(double x, double y, double z, double *r, double *theta, double *phi) noexcept nogil

cdef void rot(double mu_x, double mu_y, double mu_z, double n_x, double n_y, double n_z, double alpha,
        double* rot_x, double* rot_y, double* rot_z) noexcept nogil
cdef double cnormalized(double* ax, double* ay, double* az) noexcept nogil
cdef double cdistance_ball(Ball a, Ball b) noexcept nogil
cdef double cdistance3(double ax, double ax, double az, double bx, double by, double bz) noexcept nogil
cdef double cdirection(Ball a, Ball b, double* dir_x, double* dir_y, double* dir_z) noexcept
