import numpy as np
import cython
from numpy.random import default_rng
from scipy.stats import poisson, vonmises_fisher
from libc.math cimport sin, cos, sqrt, atan2, acos

cimport numpy as np
np.import_array()
from Altendorf_Jeulin_Model.rv_utils import vonmises_fisher
cdef double PI = 3.141592653589793

from Altendorf_Jeulin_Model.utils import (
    is_in_image,
    set_value,
)
from Altendorf_Jeulin_Model.utils cimport cartesian_to_spherical, cnormalized, clip



def simulate_poisson_lines(
    mu: float,
    R,
    direction_distribution,
    image_size,
    seed: int = None,
    is_poisson: bool = True,
    volume_fraction_should: float = 1.0,
):
    """
    initializes a fiber system of endless fibers, where fibers still overlap.
    This method follows the initial fiber system by Prakash Easwaran

    :param mu: float
        The mean number of fibers
    :param R: float or random variable
        radius of the fiber
    :param beta: float
        direction parameter for the Schladitz distribution
    :param image_size: tuple[int, int, int]
    :param seed: int, default None
        seed for the random variables, default is system time
    :param has_beta: bool, default True
        whether to use the beta parameter for the Schladitz distribution or A for the ACG distribution
    :param is_poisson: bool, default True
        whether to sample the number of fibers from a Poisson distribution (Poisson line process)
    :param volume_fraction_should: float, default 1.0
        volume fraction that should not be exceeded. When it is set to 1.0, the volume fraction is not tested.
        In general, the number of fibers will not be exceeded.
    :return: list[Fiber]
        the generated fiber system
    """
    rng = default_rng(seed)
    if is_poisson:
        n = poisson(mu).rvs(random_state=rng)
    else:
        n = mu
    line_system = []
    vol = 0
    image_vol = image_size[0] * image_size[1] * image_size[2]
    for i in range(0, n):
        # 1. Simulate the radius of the ith fiber
        r_line = set_value(R, rng)
        # 2. Generate Poisson line
        pos1, pos2, _, _, length = generate_poisson_line(
            rng, direction_distribution, image_size
        )
        pos1 *= np.array(image_size)
        pos2 *= np.array(image_size)

        # create list of end points and radius
        if length > -1:
            line_system.append([pos1, pos2, r_line])
            vol += length * r_line**2 * np.pi * image_size[0]
            volume_fraction_is = vol / image_vol
            if volume_fraction_is > volume_fraction_should:
                break
    print("Volume fraction ", str(vol / image_vol))
    return line_system


def generate_poisson_line(
    rng, direction_distribution, image_size
):
    """
    generates Poisson line within the observation window

    :param rng: random number generator state
    :param beta: float
        parameter of the Schladitz distribution
    :param image_size: tuple[int, int, int]
        size of the observation window/image
    :return: np.ndarray, np.ndarray
        the line's position (center of the line) and direction
    """
    cdef double r, theta0, phi0
    # 2. Simulate the mean orientation (Schladitz distribution)
    mu0_x, mu0_y, mu0_z = direction_distribution(rng)
    mu0 = np.array([mu0_x, mu0_y, mu0_z])
    cartesian_to_spherical(mu0_x, mu0_y, mu0_z, &r, &theta0, &phi0)
    # 3. Generate Poisson line
    r = 2
    while r > 3 / 2:
        r = 3 / 2 * np.sqrt(abs(rng.random()*2-1))
        phi = 2 * PI * (rng.random()*2-1)
        u1 = r * cos(phi)
        u2 = r * sin(phi)
    lp_x = cos(theta0) * cos(phi0)* u1 - sin(phi0)*u2 + 0.5
    lp_y = cos(theta0)*sin(phi0)*u1 + cos(phi0)*u2 + 0.5
    lp_z = 0.5 - sin(theta0)*u1
    linepos = np.array([lp_x, lp_y, lp_z])
    # 4. if it cuts 2 planes, calculate midpoint
    is_cut, start_pos, end_pos, mid_pos, length = line_cut_cube(linepos, mu0)
    mid_pos = mid_pos * np.array(image_size)
    return start_pos, end_pos, mid_pos, mu0, length


def line_cut_sphere_length(linepos: np.ndarray, mu0: np.ndarray):
    """
    calculates whether a given line cuts a unit cube and the line's midpoint

    :param linepos: np.ndarray
        position of the line
    :param mu0: np.ndarray
        direction of the line
    :return: Bool, np.ndarray
        True, midpoint   or   False, 0
    """

    B = 2 * np.dot(linepos, mu0)
    C = np.dot(linepos, linepos) - 3 / 4
    if B**2 - 4 * C < 0:
        return -1
    lambd1 = -B + np.sqrt(B**2 - 4 * C) / 2
    p1 = linepos + lambd1 * mu0
    lambd2 = -B - np.sqrt(B**2 - 4 * C) / 2
    p2 = linepos + lambd2 * mu0
    return np.linalg.norm(p2 - p1)


def line_cut_cube(linepos: np.ndarray, mu0):
    """
    calculates whether a given line cuts a unit cube and the line's midpoint

    :param linepos: np.ndarray
        position of the line
    :param mu0: np.ndarray
        direction of the line
    :return: Bool, np.ndarray
        True, midpoint   or   False, 0
    """
    face_normals = [
        np.array([1, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([0, 0, 1]),
    ]
    intersections = []
    for i, normal in enumerate(face_normals):
        dist = 1 if i % 2 == 0 else 0
        intersection = line_cut_face(normal, dist, linepos, mu0)
        if (
            intersection[0] >= 0
            and intersection[0] <= 1
            and intersection[1] >= 0
            and intersection[1] <= 1
            and intersection[2] >= 0
            and intersection[2] <= 1
        ):
            intersections.append(intersection)
    if len(intersections) == 2:
        return (
            True,
            intersections[0],
            intersections[1],
            (intersections[0] + intersections[1]) / 2,
            np.linalg.norm(intersections[0] - intersections[1]),
        )
    else:
        return False, np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), -1


def line_cut_face(
    face_normal: np.ndarray, face_dist: int, linepos: np.ndarray, mu0: np.ndarray
):
    """
    calculates whether a given line cuts a given face of a cube

    :param face_normal: np.ndarray
        normal of the face, e.g. (1, 0, 0)
    :param face_dist: int
        -1 or 0
    :param linepos: np.ndarray
        position of the line
    :param mu0: np.ndarray
        direction of the line
    :return: np.ndarray
        intersection of line and face
    """
    cdef double cos = face_normal[0]*mu0[0] + face_normal[1]*mu0[1] + face_normal[2]*mu0[2]
    cdef double other_dot, lambd
    if np.isclose(cos, 0):
        return np.array([2, 2, 2])  # TODO ok?
    else:
        other_dot = face_normal[0]*linepos[0] + face_normal[1]*linepos[1] + face_normal[2]*linepos[2]
        lambd = (face_dist - other_dot) / cos
        return linepos + lambd * mu0


def generate_half_fiber(
    mu0: np.ndarray,
    mid_pos: np.ndarray,
    image_size,
    kappa1: cython.float,
    kappa2: cython.float,
    r_fiber: cython.float,
    coords: list[np.ndarray],
    rng,
    is_forward: bool,
):
    """
    generates one half of and endless fiber; which half is determined by is_forward

    :param mu0: np.ndarray
        the fiber's original direction
    :param mid_pos: np.ndarray
        its center position in the observation window/image
    :param image_size: size of the observation window/image
    :param kappa1: float
        curvature parameter 1
    :param kappa2: float
        curvature parameter 2
    :param r_fiber: float
        radius of the fiber
    :param coords: list[np.ndarray]
        coordinates of the fiber
    :param rng: random number generator state
    :param is_forward: bool
        the fiber is generated "forwards" (in direction of mu0) or "backwards" (in direction of -mu0)
    :return: list[np.ndarray]
        coordinates of the fiber
    """
    cdef double mu0_x, mu0_y, mu0_z, mu_old_x, mu_old_y, mu_old_z, mu_new_x, mu_new_y, mu_new_z
    cdef double current_pos_x, current_pos_y, current_pos_z
    cdef int sign
    if is_forward:
        sign = 1
    else:
        sign = -1
    mu0_x = mu0[0]
    mu0_y = mu0[1]
    mu0_z = mu0[2]
    mu_old_x = sign*mu0_x
    mu_old_y = sign*mu0_y
    mu_old_z = sign*mu0_z
    current_pos_x = mid_pos[0]
    current_pos_y = mid_pos[1]
    current_pos_z = mid_pos[2]

    while is_in_image(np.array([current_pos_x, current_pos_y, current_pos_z]), image_size, 0):
        mu_new_x = sign*kappa1*mu0_x + kappa2*mu_old_x
        mu_new_y = sign*kappa1*mu0_y + kappa2*mu_old_y
        mu_new_z = sign*kappa1*mu0_z + kappa2*mu_old_z
        kappa_new = cnormalized(&mu_new_x, &mu_new_y, &mu_new_z)
        mu_old_x, mu_old_y, mu_old_z = vonmises_fisher(kappa_new, mu_new_x, mu_new_y, mu_new_z, rng)
        current_pos_x += r_fiber * mu_old_x / 2
        current_pos_y += r_fiber * mu_old_y / 2
        current_pos_z += r_fiber * mu_old_z / 2
        if is_forward:
            coords.append(np.array([current_pos_x, current_pos_y, current_pos_z]))
        else:
            coords.insert(0, np.array([current_pos_x, current_pos_y, current_pos_z]))
    return coords
