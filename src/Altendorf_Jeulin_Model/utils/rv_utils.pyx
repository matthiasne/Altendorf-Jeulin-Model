import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm
from Altendorf_Jeulin_Model.utils.utils import normalized
from libc.math cimport log, exp, sqrt, asin, acos
cdef double PI = 3.141592653589793
from libc.float cimport DBL_EPSILON

cimport numpy as np
np.import_array()
from libc.math cimport sin, cos, sqrt
from Altendorf_Jeulin_Model.utils.utils cimport cartesian_to_spherical

def vonmises_fisher(double kappa, double mu_x, double mu_y, double mu_z, rng):
    """
    Sample a unit direction from the von Mises–Fisher distribution.

    Uses the Wood (1994) / Ulrich (1984) rejection-free method: a polar
    angle theta is drawn from the marginal distribution on [0, pi] and an
    azimuthal angle phi is drawn uniformly on [0, 2*pi], giving a sample
    concentrated around the north pole (0, 0, 1).  The result is then
    rotated to the requested mean direction mu via the rotation matrix
    Rz(phi_mu) @ Ry(theta_mu).

    Special cases
    -------------
    * kappa < DBL_EPSILON (effectively zero):
        The distribution collapses to the uniform distribution on the
        sphere; theta is drawn from an arccosine-uniform distribution
        instead of the vMF marginal.
    * mu close to the z-axis (|mu_x| < DBL_EPSILON and |mu_y| < DBL_EPSILON):
        The rotation matrix is the identity, so the north-pole sample is
        returned directly without applying the rotation.

    :param kappa: float
        Concentration parameter (kappa >= 0).  kappa = 0 gives a uniform
        distribution; larger values concentrate mass around mu.
    :param mu_x: float
        x-component of the mean direction unit vector.
    :param mu_y: float
        y-component of the mean direction unit vector.
    :param mu_z: float
        z-component of the mean direction unit vector.
    :param rng: numpy.random.Generator
        Random number generator (e.g. numpy.random.default_rng()).
    :return: tuple[float, float, float]
        (dir_x, dir_y, dir_z) — Cartesian components of the sampled unit
        direction vector.
    """
    cdef double mean_r, mean_theta, mean_phi, dx, dy, dz
    cdef double dir_x, dir_y, dir_z
    cdef double r1 = rng.random()
    cdef double r2 = rng.random()
    cdef double lambd = exp(-2.0*kappa)

    cdef double theta
    cdef double phi = 2.0*PI*r2
    if kappa < DBL_EPSILON:
        theta = acos(1 - 2*r1)
    else:
        theta = 2.0*asin(sqrt(-log(r1*(1.0-lambd)+lambd)/(2.0*kappa)))

    cartesian_to_spherical(mu_x, mu_y, mu_z, &mean_r, &mean_theta, &mean_phi)
    dx = sin(theta) * cos(phi)
    dy = sin(theta) * sin(phi)
    dz = cos(theta)
    if abs(mu_x) < DBL_EPSILON and abs(mu_y) < DBL_EPSILON:
        return dx, dy, dz
    dir_x = cos(mean_theta)*cos(mean_phi)*dx - sin(mean_phi)*dy + sin(mean_theta)*cos(mean_phi)*dz
    dir_y = cos(mean_theta)*sin(mean_phi)*dx + cos(mean_phi)*dy + sin(mean_theta)*sin(mean_phi)*dz
    dir_z = cos(mean_theta)*dz-sin(mean_theta)*dx
    return dir_x, dir_y, dir_z


def acg_distribution(param_matrix, rng):
    """
    Sample a unit direction from the Angular Central Gaussian (ACG)
    distribution.

    A standard normal vector r is drawn and multiplied by the lower
    Cholesky factor L of *param_matrix*, then normalised to unit length.
    The resulting direction is uniformly distributed on the unit sphere
    when *param_matrix* is a multiple of the identity, and is
    concentrated toward the dominant eigenvector(s) of *param_matrix*
    otherwise.

    :param param_matrix: array_like, shape (3, 3)
        Parameter matrix of the ACG distribution.  Must be symmetric and
        positive-definite so that the Cholesky decomposition exists.
    :param rng: numpy.random.Generator
        Random number generator passed as *random_state* to
        ``scipy.stats.norm.rvs``.
    :return: np.ndarray, shape (3,)
        Unit direction vector sampled from the ACG distribution.
    """
    r = norm.rvs(size=3, random_state=rng)
    L = cholesky(param_matrix)
    _, r_acg = normalized(np.dot(L, r))
    return r_acg

def schladitz_distribution(beta: float, rng):
    """
    Sample a unit direction from the Schladitz distribution.

    :param beta: float
        Shape parameter of the Schladitz distribution (beta > 0).
        beta = 1 gives an isotropic (uniform) distribution;
        beta < 1 concentrates fibers toward the z-axis (planar tendency);
        beta > 1 concentrates fibers in the xy-plane (planar isotropy).
    :param rng: numpy.random.Generator
        Random number generator exposing a ``.random()`` method (e.g.
        ``numpy.random.default_rng()``).
    :return: tuple[float, float, float]
        (dir_x, dir_y, dir_z) — Cartesian components of the sampled unit
        direction vector, equivalent to
        (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)).
    :raises ValueError:
        If *beta* is less than DBL_EPSILON (effectively zero or negative),
        because the distribution is undefined for beta <= 0.
    """
    if beta < DBL_EPSILON:
        raise ValueError("beta must be positive")
    u1 = rng.random()
    u2 = rng.random()
    phi0 = PI * 2 * u1
    theta0 = acos(
        (1 - 2 * u2) / sqrt(beta*beta - (beta*beta - 1) * (1 - 2 * u2) ** 2)
    )
    return sin(theta0)*cos(phi0), sin(theta0)*sin(phi0), cos(theta0)