import numpy as np
import cython
from numbers import Real
from libc.math cimport sin, cos, sqrt, atan2, acos

cimport numpy as np
np.import_array()
from numpy.random import default_rng
from scipy.stats import poisson, uniform
from Altendorf_Jeulin_Model.rv_utils import vonmises_fisher, acg_distribution, schladitz_distribution
cdef double PI = 3.141592653589793


from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.utils cimport cartesian_to_spherical, cnormalized

from Altendorf_Jeulin_Model.utils import (
    is_in_image,
    normalized,
    spherical_to_matrix,
)


class FiberModel:
    def __init__(self, initial_fiber_system):
        if not (
            isinstance(initial_fiber_system, list)
            and all(isinstance(x, Fiber) for x in initial_fiber_system)
        ):
            raise TypeError("Initial_fiber_system must be a list of fibers")


def initialize_fiber_system(
    intensity,
    L,
    R,
    direction_distribution,
    image_size,
    kappa1: float|int,
    kappa2: float|int,
    seed: int = None,
):
    """
    initializes a fiber system, where fibers still overlap. This method follows the initial fiber system by
    Altendorf&Jeulin (2011), further systems tbd

    :param intensity: float
        expected number of fibers
    :param L: float or random variable
        length of the fiber
    :param R: float or random variable
        radius of the fiber
    :param beta: float
        direction parameter for the Schladitz distribution
    :param image_size: tuple[int, int, int]
    :param kappa1: float
        curvature parameter for the random walk
    :param kappa2: float
        curvature parameter for the random walk
    :param seed: int, default 42
        seed for the random variables
    :param is_poisson: bool, default True
        whether to sample the number of fibers from a Poisson distribution (Poisson line process)
    :param volume_fraction_should: float, default 1.0
        volume fraction that should not be exceeded. When it is set to 1.0, the volume fraction is not tested.
        In general, the number of fibers will not be exceeded.
    :return: list[Fiber]
        the generated fiber system
    """
    cdef double r, theta, phi
    cdef double mu0_x, mu0_y, mu0_z, rot_x, rot_y, rot_z
    cdef double mu_old_x, mu_old_y, mu_old_z, mu_new_x, mu_new_y, mu_new_z
    cdef double mu_bar_x, mu_bar_y, mu_bar_z, n_axis_x, n_axis_y, n_axis_z
    cdef double dir_prev_x, dir_prev_y, dir_prev_z, dir_next_x, dir_next_y, dir_next_z
    rng = default_rng(seed)
    N = set_value(intensity, rng)


    #TODO (5) utilize ball as class/struct
    fiber_system = []
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius (for now only constant) TODO (4)
        l_fiber = set_value(L, rng)
        r_fiber = set_value(R, rng)
        l_fiber_discrete = int(2 * ((l_fiber-2*r_fiber )/ r_fiber))
        # 2. Simulate the mean orientation
        mu0_x, mu0_y, mu0_z = direction_distribution(rng)

        # 3. Simulating a random walk for the fiber system
        coord = np.zeros((l_fiber_discrete, 3))
        coord[0, 0] = image_size[0] * rng.random()
        coord[0, 1] = image_size[1] * rng.random()
        coord[0, 2] = image_size[2] * rng.random()

        cnt = 1
        mu_old_x = mu0_x
        mu_old_y = mu0_y
        mu_old_z = mu0_z
        while cnt < l_fiber_discrete:
            mu_new_x = kappa1*mu0_x + kappa2*mu_old_x
            mu_new_y = kappa1*mu0_y + kappa2*mu_old_y
            mu_new_z = kappa1*mu0_z + kappa2*mu_old_z
            kappa_new = cnormalized(&mu_new_x, &mu_new_y, &mu_new_z)
            mu_old_x, mu_old_y, mu_old_z = vonmises_fisher(kappa_new, mu_new_x, mu_new_y, mu_new_z, rng)
            coord[cnt, 0] = coord[cnt - 1, 0] + r_fiber/2 * mu_old_x
            coord[cnt, 1] = coord[cnt - 1, 1] + r_fiber/2 * mu_old_y
            coord[cnt, 2] = coord[cnt - 1, 2] + r_fiber/2 * mu_old_z
            cnt = cnt + 1

        # 4. Adjusting the fibers such that the mean orientation is maintained
        mu_bar_x = coord[l_fiber_discrete-1, 0] - coord[0,0]
        mu_bar_y = coord[l_fiber_discrete-1, 1] - coord[0,1]
        mu_bar_z = coord[l_fiber_discrete-1, 2] - coord[0,2]
        cnormalized(&mu_bar_x, &mu_bar_y, &mu_bar_z)
        n_axis_x = mu0_y*mu_bar_z - mu0_z*mu_bar_y
        n_axis_y = mu0_z*mu_bar_x - mu0_x*mu_bar_z
        n_axis_z = mu0_x*mu_bar_y - mu0_y*mu_bar_x
        cnormalized(&n_axis_x, &n_axis_y, &n_axis_z)
        alpha = 2*PI - acos(mu0_x*mu_bar_x + mu0_y*mu_bar_y + mu0_z*mu_bar_z)

        if alpha > 0:
            cos_alpha = cos(alpha)
            sin_alpha = sin(alpha)
            for j in range(1, l_fiber_discrete):
                rot(coord[j,0] - coord[0,0], coord[j,1] - coord[0,1],
                    coord[j,2] - coord[0,2], n_axis_x, n_axis_y, n_axis_z,
                    cos_alpha, sin_alpha, &rot_x, &rot_y, &rot_z)
                coord[j, 0] = coord[0, 0] + rot_x
                coord[j, 1] = coord[0, 1] + rot_y
                coord[j, 2] = coord[0, 2] + rot_z

        save_balls_in_fiber_system(fiber_system, coord, i, r_fiber)

    return fiber_system

"""
def initialize_fiber_system_endless(
    mu: float|int,
    R,
    beta,
    image_size,
    boundary_size: int,
    kappa1: float|int,
    kappa2: float|int,
    seed: int = None,
    has_beta: bool = True,
    is_poisson: bool = True,
    volume_fraction_should: float|int = 1.0,
):"""

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
    :param boundary_size: int
    :param kappa1: float
        curvature parameter for the random walk
    :param kappa2: float
        curvature parameter for the random walk
    :param seed: int, default 42
        seed for the random variables
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
"""rng = default_rng(seed)
    if is_poisson:
        n = poisson(mu).rvs(random_state=rng)
    else:
        n = mu
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    ext_image_size = image_size + 2 * boundary_size_vec
    print("image size ", ext_image_size)
    fiber_system = []
    n_lines = 0
    for i in range(0, n):
        # 1. Simulate the radius of the ith fiber
        r_fiber = set_value(R, rng)
        # 2. Generate Poisson line
        mid_pos, mu0, length = generate_poisson_line(rng, beta, ext_image_size, has_beta)
        if length > -1:
            # 3. Simulating a random walk for the fiber system
            coords = [mid_pos + boundary_size_vec]
            coords = generate_half_fiber(
                mu0,
                mid_pos + boundary_size_vec,
                ext_image_size,
                kappa1,
                kappa2,
                r_fiber,
                coords,
                rng,
                True,
            )
            coords = generate_half_fiber(
                mu0,
                mid_pos + boundary_size_vec,
                ext_image_size,
                kappa1,
                kappa2,
                r_fiber,
                coords,
                rng,
                False,
            )

            # 4. Adjusting the fibers such that the mean orientation is maintained
            l_fiber_discrete = len(coords)
            if l_fiber_discrete < 2:
                continue
            else:
                n_lines += 1
                save_balls_in_fiber_system(fiber_system, coords, n_lines - 1, r_fiber)
        # test for volume fraction starting late and only every tenth trial to save time
        if volume_fraction_should != 1 and i > 3 / 4 * n and i % 10 == 0:
            volume_fraction_is = (
                len(fiber_system) * mean_length(fiber_system) * R**2 * np.pi
            )
            volume_fraction_is /= (
                ext_image_size[0] * ext_image_size[1] * ext_image_size[2]
            )
            if volume_fraction_is > volume_fraction_should:
                break
    return fiber_system
"""

cdef inline void rot(double mu_x, double mu_y, double mu_z, double n_x, double n_y, double n_z,
        double cos_alpha, double sin_alpha, double* cross_x, double* cross_y, double* cross_z) noexcept:
    # Rodrigues' formula
    cdef double dot = n_x*mu_x + n_y*mu_y + n_z*mu_z
    cross_x[0] = n_y*mu_z - n_z*mu_y
    cross_y[0] = n_z*mu_x - n_x*mu_z
    cross_z[0] = n_x*mu_y - n_y*mu_x

    cross_x[0] *= sin_alpha
    cross_x[0] += mu_x*cos_alpha + n_x*dot*(1.0 - cos_alpha)
    cross_y[0] *= sin_alpha
    cross_y[0] += mu_y*cos_alpha + n_y*dot*(1.0 - cos_alpha)
    cross_z[0] *= sin_alpha
    cross_z[0] += mu_z*cos_alpha + n_z*dot*(1.0 - cos_alpha)


def set_value(input_value, rng):
    """
    sets values that could be a constant scalar or a realization of a random variable

    :param input_value: constant scalar or random variable
    :param rng: random number Generator providing the random state
    :return: float
        value that the variable should be set to
    """
    if isinstance(input_value, Real):
        return input_value
    rvs = getattr(input_value, "rvs", None)
    if callable(rvs):
        return rvs(random_state=rng)
    raise ValueError(
        "Input must be a float/int or a distribution object with an 'rvs' method."
    )


def save_balls_in_fiber_system(
    fiber_system: list[Fiber], coords, i: int, r_fiber: float|int
):
    """
    saves balls in fiber system

    :param fiber_system: list[Fiber]
    :param coords: list[np.ndarray]
        coordinates of balls
    :param i: int
        fiber index within the fiber system
    :param r_fiber: float
        fiber radius
    """
    cdef double dir_prev_x, dir_prev_y, dir_prev_z, dir_next_x, dir_next_y, dir_next_z, angle
    cdef int l_fiber_discrete = len(coords)
    fiber_system.append(Fiber(Ball(coords[0], r_fiber, i, 0)))
    for j in range(1, l_fiber_discrete):
        angle = PI
        if j < l_fiber_discrete - 1:
            dir_prev_x = coords[j, 0] - coords[j-1, 0]
            dir_prev_y = coords[j, 1] - coords[j-1, 1]
            dir_prev_z = coords[j, 2] - coords[j-1, 2]
            cnormalized(&dir_prev_x, &dir_prev_y, &dir_prev_z)
            dir_next_x = coords[j+1, 0] - coords[j, 0]
            dir_next_y = coords[j+1, 1] - coords[j, 1]
            dir_next_z = coords[j+1, 2] - coords[j, 2]
            cnormalized(&dir_next_x, &dir_next_y, &dir_next_z)
            angle = PI - acos(dir_prev_x*dir_next_x + dir_prev_y*dir_next_y + dir_prev_z*dir_next_z)
        fiber_system[i].add_ball(Ball(coords[j], r_fiber, i, j, angle))


####### Poisson line generation #####################################
#def generate_poisson_line(
#    rng, beta, image_size, has_beta: bool = True
#):
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
    """U = uniform(loc=-1, scale=2)
    # 2. Simulate the mean orientation (Schladitz distribution)
    if has_beta:
        mu0, theta0, phi0 = schladitz_distribution(beta, rng)
    else:
        mu0 = acg_distribution(beta, rng)
        _, theta0, phi0 = cartesian_to_spherical(mu0[0], mu0[1], mu0[2])
        # print(mu0, theta0, phi0, spherical_to_cartesian(1, theta0, phi0))
    # 3. Generate Poisson line
    M = spherical_to_matrix(theta0, phi0)
    r = 2
    while r > 3 / 2:
        r = 3 / 2 * np.sqrt(abs(U.rvs(random_state=rng)))
        phi = 2 * np.pi * U.rvs(random_state=rng)
        u1 = r * np.cos(phi)
        u2 = r * np.sin(phi)
    linepos = np.array([u1, u2, 0])
    linepos = np.matvec(M, linepos) + np.array([0.5, 0.5, 0.5])
    # 4. if it cuts 2 planes, calculate midpoint
    is_cut, mid_pos, length = line_cut_cube(linepos, mu0)
    mid_pos = linepos * np.array(image_size)
    return mid_pos, mu0, length
"""

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
    # print("length ", np.linalg.norm(p2 - p1))
    return np.linalg.norm(p2 - p1)


def line_cut_cube(linepos: np.ndarray, mu0: np.ndarray):
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
            (intersections[0] + intersections[1]) / 2,
            np.linalg.norm(intersections[0] - intersections[1]),
        )
    else:
        return False, np.array([0, 0, 0]), -1


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
    cos = np.dot(face_normal, mu0)
    if np.isclose(cos, 0):
        return np.array([2, 2, 2])  # TODO ok?
    else:
        lambd = (face_dist - np.dot(face_normal, linepos)) / cos
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
    if is_forward:
        sign = 1
    else:
        sign = -1
    mu_old = sign * mu0
    current_pos = mid_pos
    while is_in_image(current_pos, image_size, 0):
        old_pos = current_pos
        kappa_new: cython.double = np.linalg.norm(sign * kappa1 * mu0 + kappa2 * mu_old)
        mu_new = (sign * kappa1 * mu0 + kappa2 * mu_old) / kappa_new
        vmf = vonmises_fisher(mu_new, kappa_new)

        direction = vmf.rvs(random_state=rng)[0]

        current_pos = old_pos + r_fiber * direction / 2
        if is_forward:
            coords.append(current_pos)
        else:
            coords.insert(0, current_pos)
        mu_old = direction
    return coords


####### end Poisson line generation #####################################
