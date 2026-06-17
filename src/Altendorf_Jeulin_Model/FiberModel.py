import numpy as np
import scipy
from numpy.random import default_rng
from scipy.stats import uniform, vonmises_fisher

from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.utils import (
    acg_distribution,
    cartesian_to_spherical,
    is_in_image,
    normalized,
    schladitz_distribution,
    spherical_to_matrix,
)

class FiberModel:
    def __init__(self, initial_fiber_system):
        if not (isinstance(initial_fiber_system, list) and all(isinstance(x, Fiber) for x in initial_fiber_system)):
            raise TypeError('Initial_fiber_system must be a list of fibers')


def initialize_fiber_system(N: int, L, R, beta: float, image_size: tuple[int, int, int],
                            kappa1: float, kappa2: float, seed: int = 42, has_beta: bool = True):
    """
    initializes a fiber system, where fibers still overlap. This method follows the initial fiber system by
    Altendorf&Jeulin (2011), further systems tbd

    Parameters
    ---------------------
    :param N: int
        number of fibers
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
    :return: list[Fiber]
        the generated fiber system
    """
    if beta < 0:
        raise TypeError('beta must be non-negative')

    rng = default_rng(seed)
    U = uniform(loc=0, scale=1)

    fiber_system = []
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius (for now only constant) TODO
        l_fiber = set_value(L, rng)
        r_fiber = set_value(R, rng)
        l_fiber_discrete = int(2*l_fiber/r_fiber + 1)
        # 2. Simulate the mean orientation
        if has_beta:
            mu0, theta0, phi0 = schladitz_distribution(beta, rng)
        else:
            mu0 = acg_distribution(beta, rng)
            _, theta0, phi0 = cartesian_to_spherical(mu0[0], mu0[1], mu0[2])

        # 3. Simulating a random walk for the fiber system
        coord = np.zeros((l_fiber_discrete, 3))
        coord[0, 0] = image_size[0] * U.rvs(random_state=rng)
        coord[0, 1] = image_size[1] * U.rvs(random_state=rng)
        coord[0, 2] = image_size[2] * U.rvs(random_state=rng)

        cnt = 1
        mu_old = mu0
        while cnt < l_fiber_discrete:
            mu_new = (kappa1 * mu0 + kappa2 * mu_old)
            kappa_new, mu_new = normalized(mu_new)
            vmf = vonmises_fisher(mu_new, kappa_new)

            direction = vmf.rvs(random_state=rng)[0]

            coord[cnt] = coord[cnt - 1] + r_fiber * direction / 2
            mu_old = direction
            cnt = cnt + 1

        # 4. Adjusting the fibers such that the mean orientation is maintained
        _, mu_bar = normalized(coord[l_fiber_discrete - 1] - coord[0])
        _, n_axis = normalized(np.cross(mu0, mu_bar))
        alpha = 2*np.pi - np.arccos(np.dot(mu0, mu_bar))

        for j in range(1, l_fiber_discrete):
            if alpha > 0:
                coord[j] = coord[0] + rot(coord[j] - coord[0], n_axis, alpha)
        _, mu_bar2 = normalized(coord[l_fiber_discrete - 1] - coord[0])

        save_balls_in_fiber_system(fiber_system, coord, i, r_fiber)

    return fiber_system

def initialize_fiber_system_endless(mu: float, R, beta, image_size: tuple[int, int, int],
                                    boundary_size: int,
                                    kappa1: float, kappa2: float, seed: int = 42, has_beta: bool = True):
    """
    initializes a fiber system of endless fibers, where fibers still overlap.
    This method follows the initial fiber system by Prakash Easwaran

    Parameters
    ---------------------
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
    :return: list[Fiber]
        the generated fiber system
    """
    rng = default_rng(seed)
    n = int(scipy.stats.poisson(mu).rvs(random_state=rng))
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    ext_image_size = image_size + 2 * boundary_size_vec
    print("image size ", ext_image_size)
    fiber_system = []
    n_lines = 0
    for i in range(0, n):
        # 1. Simulate the radius of the ith Fiber
        r_fiber = set_value(R, rng)
        # 2. Generate Poisson line
        mid_pos, mu0, length = generate_poisson_line(rng, beta, image_size, has_beta)
        if length > -1:
            # 3. Simulating a random walk for the fiber system
            coords = [mid_pos + boundary_size_vec]
            coords = generate_half_fiber(mu0, mid_pos + boundary_size_vec, ext_image_size, kappa1, kappa2, r_fiber,
                                         coords,
                                         rng, True)
            coords = generate_half_fiber(mu0, mid_pos + boundary_size_vec, ext_image_size, kappa1, kappa2, r_fiber,
                                         coords,
                                         rng, False)

            # 4. Adjusting the fibers such that the mean orientation is maintained
            l_fiber_discrete = len(coords)
            if l_fiber_discrete < 2:
                continue
            else:
                n_lines += 1
                save_balls_in_fiber_system(fiber_system, coords, n_lines - 1, r_fiber)
    return fiber_system


def rot(mu, n, alpha):
    return np.dot(n, mu) * n + np.cos(alpha) * np.cross(np.cross(n, mu), n) + np.sin(alpha) * np.cross(n, mu)


def set_value(input_value, rng):
    """
    sets values that could be a constant scalar or a realization of a random variable

    Parameters
    ---------------------
    :param input_value: constant scalar or random variable
    :param rng: random number Generator providing the random state
    :return: float
        value that the variable should be set to
    """
    if isinstance(input_value, float) or isinstance(input_value, int):
        # L is a constant number
        result = input_value
    elif hasattr(input_value, 'rvs'):
        # L is a Poisson generator (or any similar object with an rvs method)
        result = input_value.rvs(random_state=rng)
    else:
        raise ValueError("Input must be a float/int or a distribution object with an 'rvs' method.")

    return result

def save_balls_in_fiber_system(fiber_system: list[Fiber], coords: list[np.ndarray], i: int, r_fiber: float):
    """
    saves balls in fiber system

    Parameters
    ---------------------
    :param fiber_system: list[Fiber]
    :param coords: list[np.ndarray]
        coordinates of balls
    :param i: int
        fiber index within the fiber system
    :param r_fiber: float
        fiber radius
    """
    l_fiber_discrete = len(coords)
    fiber_system.append(Fiber(Ball(coords[0], r_fiber, i, 0)))
    for j in range(1, l_fiber_discrete):
        angle = np.pi
        if j < l_fiber_discrete - 1:
            _, dir_prev = normalized(coords[j] - coords[j - 1])
            _, dir_next = normalized(coords[j + 1] - coords[j])
            angle = np.pi - np.arccos(np.dot(dir_prev, dir_next))
        fiber_system[i].add_ball(Ball(coords[j], r_fiber, i, j, angle))

####### Poisson line generation #####################################
def generate_poisson_line(rng, beta: float, image_size: tuple[int, int, int], has_beta: bool = True):
    """
    generates Poisson line within the observation window

    Parameters
    ---------------------
    :param rng: random number generator state
    :param beta: float
        parameter of the Schladitz distribution
    :param image_size: tuple[int, int, int]
        size of the observation window/image
    :return: np.ndarray, np.ndarray
        the line's position (center of the line) and direction
    """
    U = uniform(loc=-1, scale=2)
    # 2. Simulate the mean orientation (Schladitz distribution)
    if has_beta:
        mu0, theta0, phi0 = schladitz_distribution(beta, rng)
    else:
        mu0 = acg_distribution(beta, rng)
        _, theta0, phi0 = cartesian_to_spherical(mu0[0], mu0[1], mu0[2])
        #print(mu0, theta0, phi0, spherical_to_cartesian(1, theta0, phi0))
    # 3. Generate Poisson line
    M = spherical_to_matrix(theta0, phi0)
    r = 2
    while r > 3 / 2:
        r = 3/2 * np.sqrt(abs(U.rvs(random_state=rng)))
        phi = 2*np.pi*U.rvs(random_state=rng)
        u1 = r*np.cos(phi)
        u2 = r*np.sin(phi)
    linepos = np.array([u1, u2, 0])
    linepos = np.matvec(M, linepos) + np.array([0.5, 0.5, 0.5])
    # 4. if it cuts 2 planes, calculate midpoint
    is_cut, mid_pos, length = line_cut_cube(linepos, mu0)
    mid_pos = linepos * np.array(image_size)
    return mid_pos, mu0, length


def line_cut_sphere_length(linepos: np.ndarray, mu0: np.ndarray):
    """
    calculates whether a given line cuts a unit cube and the line's midpoint

    Parameters
    ---------------------
    :param linepos: np.ndarray
        position of the line
    :param mu0: np.ndarray
        direction of the line
    :return: Bool, np.ndarray
        True, midpoint   or   False, 0
    """

    B = 2 * np.dot(linepos, mu0)
    C = np.dot(linepos, linepos) - 3 / 4
    if B ** 2 - 4 * C < 0:
        return -1
    lambd1 = -B + np.sqrt(B ** 2 - 4 * C) / 2
    p1 = linepos + lambd1 * mu0
    lambd2 = -B - np.sqrt(B ** 2 - 4 * C) / 2
    p2 = linepos + lambd2 * mu0
    # print("length ", np.linalg.norm(p2 - p1))
    return np.linalg.norm(p2 - p1)


def line_cut_cube(linepos: np.ndarray, mu0: np.ndarray):
    """
    calculates whether a given line cuts a unit cube and the line's midpoint

    Parameters
    ---------------------
    :param linepos: np.ndarray
        position of the line
    :param mu0: np.ndarray
        direction of the line
    :return: Bool, np.ndarray
        True, midpoint   or   False, 0
    """
    face_normals = [np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 1, 0]),
                    np.array([0, 0, 1]), np.array([0, 0, 1])]
    intersections = []
    for i, normal in enumerate(face_normals):
        dist = 1 if i % 2 == 0 else 0
        intersection = line_cut_face(normal, dist, linepos, mu0)
        if (intersection[0] >= 0 and intersection[0] <= 1
                and intersection[1] >= 0 and intersection[1] <= 1
                and intersection[2] >= 0 and intersection[2] <= 1):
            intersections.append(intersection)
    if len(intersections) == 2:
        return True, (intersections[0] + intersections[1]) / 2, np.linalg.norm(intersections[0] - intersections[1])
    else:
        return False, np.array([0, 0, 0]), -1


def line_cut_face(face_normal: np.ndarray, face_dist: int, linepos: np.ndarray, mu0: np.ndarray):
    """
    calculates whether a given line cuts a given face of a cube

    Parameters
    ---------------------
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


def generate_half_fiber(mu0: np.ndarray, mid_pos: np.ndarray, image_size: tuple[int, int, int],
                        kappa1: float, kappa2: float, r_fiber: float, coords: list[np.ndarray], rng, is_forward: bool):
    """
    generates one half of and endless fiber; which half is determined by is_forward

    Parameters
    ---------------------
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
        kappa_new = np.linalg.norm(sign * kappa1 * mu0 + kappa2 * mu_old)
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