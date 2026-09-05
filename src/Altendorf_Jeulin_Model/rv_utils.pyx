import copy

import numpy as np
from numpy.random import default_rng
from Cython.Shadow import returns
from line_profiler import profile
import cython
from libc.math cimport log, exp, sqrt, asin
cdef double PI = 3.141592653589793


cimport numpy as np
np.import_array()
from libc.math cimport sin, cos, sqrt
from Altendorf_Jeulin_Model.utils cimport cartesian_to_spherical

#TODO get rid of numpy rng
def vonmises_fisher(double kappa, double mu_x, double mu_y, double mu_z, rng):
    cdef double mean_r, mean_theta, mean_phi, dx, dy, dz
    cdef double dir_x, dir_y, dir_z
    cdef double r1 = rng.random()
    cdef double r2 = rng.random()
    cdef double lambd = exp(-2.0*kappa)

    cdef double theta = 2.0*asin(sqrt(-log(r1*(1.0-lambd)+lambd)/(2.0*kappa)))
    cdef double phi = 2.0*PI*r2

    if mu_x == 0 and mu_y == 0 and mu_z ==1:
        return
    cartesian_to_spherical(mu_x, mu_y, mu_z, &mean_r, &mean_theta, &mean_phi)
    dx = sin(theta) * cos(phi)
    dy = sin(theta) * sin(phi)
    dz = cos(theta)
    dir_x = cos(mean_theta) * cos(mean_phi)* dx - sin(mean_phi)*dy + sin(mean_theta)*cos(mean_phi)*dz
    dir_y = cos(mean_theta)*sin(mean_phi)*dx + cos(mean_phi)*dz + sin(mean_theta)*sin(mean_phi)*dz
    dir_z = cos(mean_theta)*dz-sin(mean_theta)*dx
    return dir_x, dir_y, dir_z
    