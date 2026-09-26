import copy

import numpy as np
from numpy.random import default_rng
from scipy.linalg import cholesky
from scipy.stats import norm
from Altendorf_Jeulin_Model.utils import normalized
import cython
from libc.math cimport log, exp, sqrt, asin, acos
cdef double PI = 3.141592653589793


cimport numpy as np
np.import_array()
from libc.math cimport sin, cos, sqrt
from Altendorf_Jeulin_Model.utils cimport cartesian_to_spherical

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
    dir_x = cos(mean_theta)*cos(mean_phi)*dx - sin(mean_phi)*dy + sin(mean_theta)*cos(mean_phi)*dz
    dir_y = cos(mean_theta)*sin(mean_phi)*dx + cos(mean_phi)*dy + sin(mean_theta)*sin(mean_phi)*dz
    dir_z = cos(mean_theta)*dz-sin(mean_theta)*dx
    return dir_x, dir_y, dir_z


def acg_distribution(param_matrix, rng):
    """
    generates a direction following the Angular Central Gaussian (ACG) distribution

    :param param_matrix: np.ndarray
        parameter matrix of the ACG distribution
    :param rng: random state
    :return: np.ndarray
        direction vector
    """
    r = norm.rvs(size=3, random_state=rng)
    L = cholesky(param_matrix)
    _, r_acg = normalized(np.dot(L, r))
    return r_acg

def schladitz_distribution(beta: float, rng):
    """
    generate random direction following the Schladitz distribution

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    :param beta: float
        beta parameter of the Schladitz distribution
    :param rng: random state
    :return: np.ndarray, float, float
        direction vector, polar coordinates of direction vector
    """
    u1 = rng.random()
    u2 = rng.random()
    phi0 = PI * 2 * u1
    theta0 = acos(
        (1 - 2 * u2) / sqrt(beta*beta - (beta*beta - 1) * (1 - 2 * u2) ** 2)
    )
    return sin(theta0)*cos(phi0), sin(theta0)*sin(phi0), cos(theta0)