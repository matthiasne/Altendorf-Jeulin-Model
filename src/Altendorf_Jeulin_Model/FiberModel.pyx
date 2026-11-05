import numpy as np
import cython

from Altendorf_Jeulin_Model.Statistics import volume_fraction
from libc.math cimport sin, cos, sqrt, atan2, acos

cimport numpy as np
np.import_array()
from numpy.random import default_rng
from Altendorf_Jeulin_Model.rv_utils import vonmises_fisher, acg_distribution, schladitz_distribution
cdef double PI = 3.141592653589793


from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.utils cimport cartesian_to_spherical, cnormalized, clip
from Altendorf_Jeulin_Model.PoissonLines import (
    generate_half_fiber,
    generate_poisson_line,
)


from Altendorf_Jeulin_Model.utils import (
    is_in_image,
    set_value
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
    :param direction_distribution: function
        a direction distribution that takes rng as input and returns the direction vector
    :param image_size:
    :param kappa1: float
        curvature parameter for the random walk
    :param kappa2: float
        curvature parameter for the random walk
    :param seed: int, default 42
        seed for the random variables
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

    fiber_system = []
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius
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
            coord[cnt, 0] = coord[cnt - 1, 0] + r_fiber/2 * mu_old_x
            coord[cnt, 1] = coord[cnt - 1, 1] + r_fiber/2 * mu_old_y
            coord[cnt, 2] = coord[cnt - 1, 2] + r_fiber/2 * mu_old_z
            cnt = cnt + 1

        save_balls_in_fiber_system(fiber_system, coord, i, r_fiber)

    return fiber_system

def initialize_fiber_system_AJ(
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
    :param direction_distribution: function
        a direction distribution that takes rng as input and returns the direction vector
    :param image_size:
    :param kappa1: float
        curvature parameter for the random walk
    :param kappa2: float
        curvature parameter for the random walk
    :param seed: int, default 42
        seed for the random variables
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

    fiber_system = []
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius
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
        alpha = 2*PI - acos(clip(mu0_x*mu_bar_x + mu0_y*mu_bar_y + mu0_z*mu_bar_z))

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


def initialize_fiber_system_endless(
    mu: float|int,
    R,
    direction_distribution,
    image_size,
    boundary_size: int,
    kappa1: float|int,
    kappa2: float|int,
    seed: int = None
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
    rng = default_rng(seed)
    n = int(set_value(mu, rng))
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    ext_image_size = image_size + 2 * boundary_size_vec
    print("image size ", ext_image_size)
    fiber_system = []
    n_lines = 0
    for i in range(0, n):
        # 1. Simulate the radius of the ith fiber
        r_fiber = set_value(R, rng)
        # 2. Generate Poisson line
        _, _, mid_pos, mu0, length = generate_poisson_line(rng, direction_distribution, ext_image_size)
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
                save_balls_in_fiber_system_endless(fiber_system, coords, n_lines - 1, r_fiber)
    return fiber_system


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
            angle = PI - acos(clip(dir_prev_x*dir_next_x + dir_prev_y*dir_next_y + dir_prev_z*dir_next_z))
        fiber_system[i].add_ball(Ball(coords[j], r_fiber, i, j, angle))

def save_balls_in_fiber_system_endless(
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
            dir_prev_x = coords[j][0] - coords[j-1][0]
            dir_prev_y = coords[j][1] - coords[j-1][1]
            dir_prev_z = coords[j][2] - coords[j-1][2]
            cnormalized(&dir_prev_x, &dir_prev_y, &dir_prev_z)
            dir_next_x = coords[j+1][0] - coords[j][0]
            dir_next_y = coords[j+1][1] - coords[j][1]
            dir_next_z = coords[j+1][2] - coords[j][2]
            cnormalized(&dir_next_x, &dir_next_y, &dir_next_z)
            angle = PI - acos(clip(dir_prev_x*dir_next_x + dir_prev_y*dir_next_y + dir_prev_z*dir_next_z))
        fiber_system[i].add_ball(Ball(np.array([coords[j][0], coords[j][1], coords[j][2]]), r_fiber, i, j, angle))


