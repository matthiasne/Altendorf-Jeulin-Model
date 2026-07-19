import numpy as np
from numpy.random import default_rng
from scipy.stats import poisson, uniform, vonmises_fisher

from Altendorf_Jeulin_Model.utils import (
    acg_distribution,
    cartesian_to_spherical,
    is_in_image,
    schladitz_distribution,
    set_value,
    spherical_to_matrix,
)


def simulate_poisson_lines(
    mu: float,
    R,
    beta,
    image_size: tuple[int, int, int],
    seed: int = None,
    has_beta: bool = True,
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
            rng, beta, image_size, has_beta
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
    rng, beta: float, image_size: tuple[int, int, int], has_beta: bool = True
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
    U = uniform(loc=-1, scale=2)
    # 2. Simulate the mean orientation (Schladitz distribution)
    if has_beta:
        mu0, theta0, phi0 = schladitz_distribution(beta, rng)
    else:
        mu0 = acg_distribution(beta, rng)
        _, theta0, phi0 = cartesian_to_spherical(mu0[0], mu0[1], mu0[2])
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
    is_cut, pos1, pos2, length = line_cut_cube(linepos, mu0)
    mid_pos = linepos * np.array(image_size)
    return pos1, pos2, mid_pos, mu0, length


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
            intersections[0],
            intersections[1],
            np.linalg.norm(intersections[0] - intersections[1]),
        )
    else:
        return False, np.array([0, 0, 0]), np.array([0, 0, 0]), -1


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
        return np.array([2, 2, 2])
    else:
        lambd = (face_dist - np.dot(face_normal, linepos)) / cos
        return linepos + lambd * mu0


def generate_half_fiber(
    mu0: np.ndarray,
    mid_pos: np.ndarray,
    image_size: tuple[int, int, int],
    kappa1: float,
    kappa2: float,
    r_fiber: float,
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
