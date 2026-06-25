import copy

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm, uniform

import Altendorf_Jeulin_Model.Fiber as Fiber


def periodic_distance(coord1: np.ndarray, coord2: np.ndarray, image_size: tuple[int, int, int]):
    """
    Calculates the periodic distance between two coordinates and the normalized direction vector between them

    Attributes
    ---------------------
    :param coord1: np.ndarray
        The first coordinate, start of the direction vector
    :param coord2: np.ndarray
        The second coordinate, "end" of the direction vector
    :param image_size: tuple[int, int, int]
        The size of the image, thus, the periodicity
    :return: distance between coordinates, direction vector
    """
    coord1mod = coord1 % image_size
    coord2mod = coord2 % image_size
    dist_orig = np.linalg.norm(coord2mod - coord1mod)
    delta = coord2mod - coord1mod
    for i in range(3):
        if (abs(delta[i]) > image_size[i] / 2.):
            if (delta[i] > 0):
                coord2mod[i] -= image_size[i]
            else:
                coord2mod[i] += image_size[i]

    dist, dir = normalized(coord2mod - coord1mod)

    if (np.linalg.norm(coord2mod - coord1mod) > dist_orig):
        raise ValueError("There is an issue in the periodic distance calculation")
    else:
        return dist, dir


def angle_between(v1: np.ndarray, v2: np.ndarray):
    """
    Calculates the angle between two vectors v1 and v2

    Attributes
    ---------------------
    :param v1: np.ndarray
    :param v2: np.ndarray
    :return: float
        angle between the two vectors v1, v2
    """
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    if l1 == 0:
        raise ValueError("v1 has length 0")
    if l2 == 0:
        raise ValueError("v2 has length 0")
    cos_angle = np.dot(v1, v2) / (l1 * l2)
    return np.acos(np.clip(cos_angle, -1.0, 1.0))


def normalized(v: np.ndarray):
    """
    normalizes a vector and returns both the original length and the normalized vector

    Attributes
    ---------------------
    :param v: np.ndarray
        vector to be normalized
    :return: float, np.ndarray
        original length, normalized vector
    """
    if np.linalg.norm(v) == 0:
        return 0, v
    v_length = np.linalg.norm(v)
    return v_length, v / v_length


def cartesian_to_spherical(x, y, z):
    """
    transform cartesian coordinates to spherical coordinates

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    Attributes
    ---------------------
    :param x: float
        cartesian x coordinate
    :param y: float
        cartesian y coordinate
    :param z: float
        cartesian z coordinate
    :return: float, float, float
        radius, theta angle, phi angle in radian
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if r == 0:
        return 0, 0, 0
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z / r, -1, 1))  # avoid domain errors
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """
    transform spherical coordinates (in radian) to cartesian coordinates

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    Attributes
    ---------------------
    :param r: float
        radius
    :param theta: float
        polar theta angle (down from z-axis)
    :param phi: float
        polar phi angle (within x-y-plane)
    :return: float, float, float
        cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def spherical_to_matrix(theta: float, phi: float):
    """
    tranforms spherical coordinates to a rotation matrix

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    Attributes
    ---------------------
    :param theta: float
        first rotation angle (down from z-axis)
    :param phi: float
        second rotation angle (within x-y-plane)
    :return: np.ndarray
        rotation matrix
    """
    return np.array([[np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta) * np.cos(phi)],
                     [np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)],
                     [-np.sin(theta), 0, np.cos(theta)]])


def rot(mu: np.ndarray, n: np.ndarray, alpha: float) -> np.ndarray:
    """
    rotation of a vector mu around normal n and angle alpha

    Attributes
    ---------------------
    :param mu: np.ndarray
        vector to be rotated
    :param n: np.ndarray
        vector to be rotated around
    :param alpha: float
        angle to be rotated by
    :return: np.ndarray
        rotated vector
    """
    return np.dot(n, mu) * n + np.cos(alpha) * np.cross(np.cross(n, mu), n) + np.sin(alpha) * np.cross(n, mu)


def is_in_image(pos: np.ndarray, image_size: tuple[int, int, int], buffer: int = 100) -> bool:
    """
    calculates whether a coordinate lies within an image/observation window

    Attributes
    ---------------------
    :param pos: np.ndarray
        position to be checked
    :param image_size: tuple[int, int, int]
        image size
    :param buffer: int
        size of image extension
    :return: bool
        is in image or not
    """
    if pos[0] < -buffer or pos[1] < -buffer or pos[2] < -buffer:
        return False
    elif pos[0] > image_size[0] + buffer or pos[1] > image_size[1] + buffer or pos[2] > image_size[2] + buffer:
        return False
    else:
        return True


def cut_border(fs: list[Fiber], image_size, boundary_size: int) -> list[Fiber]:
    """
    cuts fibers when they cross image/observation window borders

    Attributes
    ---------------------
    :param fs: list[Fiber]
        fiber system to be cut
    :param image_size: tuple[int, int, int]
        image size
    :param boundary_size: int
        size of the boundary to each side
    :return: list[Fiber]
        cut fiber system
    """
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    fs_cut = copy.deepcopy(fs)
    for fiber in fs_cut:
        j_start = -1
        j_end = len(fiber.balls) - 1
        for j in range(0, len(fiber.balls)):
            fiber.balls[j].coordinate = fiber.balls[j].coordinate - boundary_size_vec
            if j_start == j - 1 and not is_in_image(fiber.balls[j].coordinate, image_size, 0):
                j_start = j
            elif j_start < j and not is_in_image(fiber.balls[j].coordinate, image_size, 0):
                j_end = j + 1
                break
        fiber.balls = fiber.balls[j_start + 1:j_end - 1]
    return fs_cut


def schladitz_distribution(beta: float, rng):
    """
    generate random direction following the Schladitz distribution

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    Attributes
    ---------------------
    :param beta: float
        beta parameter of the Schladitz distribution
    :param rng: random state
    :return: np.ndarray, float, float
        direction vector, polar coordinates of direction vector
    """
    U = uniform(loc=0, scale=1)
    u1 = U.rvs(random_state=rng)
    u2 = U.rvs(random_state=rng)
    phi0 = np.pi * 2 * u1
    theta0 = np.arccos((1 - 2 * u2) / np.sqrt(beta ** 2 - (beta ** 2 - 1) * (1 - 2 * u2) ** 2))
    mu0 = np.array(spherical_to_cartesian(1, theta0, phi0))
    return mu0, theta0, phi0


def acg_distribution(param_matrix, rng):
    """
    generates a direction following the Angular Central Gaussian (ACG) distribution

    Attributes
    ---------------------
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

def discretize_spheres_periodic(coordinates: np.ndarray, radii: np.ndarray,
                                min_coordinates: np.ndarray, max_coordinates: np.ndarray):
    """
    Discretize spheres to an image with periodic boundary conditions
    :param coordinates: np.ndarray
        The sphere coordinates
    :param radii: np.ndarray
        The sphere radii
    :param min_coordinates: np.ndarray
        The smallest coordinate of spheres
    :param max_coordinates: np.ndarray
        The largest coordinate of spheres
    :return: np.ndarray
        The image containing spheres
    """
    L = max_coordinates - min_coordinates
    coordinates = coordinates - min_coordinates
    image = np.zeros(L, 'uint16')

    for iota in range(len(coordinates)):
        r_square = radii[iota] ** 2
        for i in range(int(coordinates[iota, 0] - radii[iota]) - 1, int(coordinates[iota, 0] + radii[iota]) + 1):
            i_corr = i
            if i < 0:
                i_corr = L[0] + i
            if i >= L[0]:
                i_corr = i - L[0]
            delta_i = (i - coordinates[iota, 0]) ** 2
            for j in range(int(coordinates[iota, 1] - radii[iota]) - 1, int(coordinates[iota, 1] + radii[iota]) + 1):
                j_corr = j
                if j < 0:
                    j_corr = L[1] + j
                if j >= L[1]:
                    j_corr = j - L[1]
                delta_ij = delta_i + (j - coordinates[iota, 1]) ** 2
                for k in range(int(coordinates[iota, 2] - radii[iota]) - 1,
                               int(coordinates[iota, 2] + radii[iota]) + 1):
                    k_corr = k
                    if k < 0:
                        k_corr = L[2] + k
                    if k >= L[2]:
                        k_corr = k - L[2]
                    delta_ijk = delta_ij + (k - coordinates[iota, 2]) ** 2
                    if delta_ijk <= r_square:
                        image[i_corr, j_corr, k_corr] = 1
    return image


def discretize_spheres_nonperiodic(coordinates: np.ndarray, radii: np.ndarray,
                                   min_coordinates: np.ndarray, max_coordinates: np.ndarray):
    """
    Discretize spheres to an image (no periodic boundary conditions)
    :param coordinates: np.ndarray
        The sphere coordinates
    :param radii: np.ndarray
        The sphere radii
    :param min_coordinates: np.ndarray
        The smallest coordinate of spheres
    :param max_coordinates: np.ndarray
        The largest coordinate of spheres
    :return: np.ndarray
        The image containing spheres
    """
    L = max_coordinates - min_coordinates
    image = np.zeros(L, 'uint16')

    for iota in range(len(coordinates)):
        r_square = radii[iota] ** 2
        for i in range(int(coordinates[iota, 0] - radii[iota]) - 1, int(coordinates[iota, 0] + radii[iota]) + 1):
            if i < 0 or i >= L[0]:
                continue
            delta_i = (i - coordinates[iota, 0]) ** 2
            for j in range(int(coordinates[iota, 1] - radii[iota]) - 1, int(coordinates[iota, 1] + radii[iota]) + 1):
                if j < 0 or j >= L[1]:
                    continue
                delta_ij = delta_i + (j - coordinates[iota, 1]) ** 2
                for k in range(int(coordinates[iota, 2] - radii[iota]) - 1,
                               int(coordinates[iota, 2] + radii[iota]) + 1):
                    if k < 0 or k >= L[2]:
                        continue
                    delta_ijk = delta_ij + (k - coordinates[iota, 2]) ** 2
                    if delta_ijk <= r_square:
                        image[i, j, k] = 1
    return image
