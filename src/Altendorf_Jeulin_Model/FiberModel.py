import numpy as np
from numpy.random import default_rng
from scipy.stats import poisson, uniform, vonmises_fisher

from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.PoissonLines import (
    generate_half_fiber,
    generate_poisson_line,
)
from Altendorf_Jeulin_Model.Statistics import mean_length
from Altendorf_Jeulin_Model.utils import (
    acg_distribution,
    cartesian_to_spherical,
    normalized,
    schladitz_distribution,
    set_value,
)


class FiberModel:
    def __init__(self, initial_fiber_system):
        if not (
            isinstance(initial_fiber_system, list)
            and all(isinstance(x, Fiber) for x in initial_fiber_system)
        ):
            raise TypeError("Initial_fiber_system must be a list of fibers")


def initialize_fiber_system(
    intensity: float,
    L,
    R,
    beta: float,
    image_size: tuple[int, int, int],
    kappa1: float,
    kappa2: float,
    seed: int = None,
    has_beta: bool = True,
    is_poisson: bool = True,
    volume_fraction_should: float = 1.0,
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
    rng = default_rng(seed)
    U = uniform(loc=0, scale=1)
    if is_poisson:
        N = poisson(intensity).rvs(random_state=rng)
    else:
        N = intensity

    fiber_system = []
    volume = 0
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius (for now only constant) TODO
        l_fiber = set_value(L, rng)
        r_fiber = set_value(R, rng)
        l_fiber_discrete = int(2 * l_fiber / r_fiber + 1)
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
            mu_new = kappa1 * mu0 + kappa2 * mu_old
            kappa_new, mu_new = normalized(mu_new)
            vmf = vonmises_fisher(mu_new, kappa_new)

            direction = vmf.rvs(random_state=rng)[0]

            coord[cnt] = coord[cnt - 1] + r_fiber * direction / 2
            mu_old = direction
            cnt = cnt + 1

        # 4. Adjusting the fibers such that the mean orientation is maintained
        _, mu_bar = normalized(coord[l_fiber_discrete - 1] - coord[0])
        _, n_axis = normalized(np.cross(mu0, mu_bar))
        alpha = 2 * np.pi - np.arccos(np.dot(mu0, mu_bar))

        for j in range(1, l_fiber_discrete):
            if alpha > 0:
                coord[j] = coord[0] + rot(coord[j] - coord[0], n_axis, alpha)
        _, mu_bar2 = normalized(coord[l_fiber_discrete - 1] - coord[0])

        save_balls_in_fiber_system(fiber_system, coord, i, r_fiber)

        volume += l_fiber * r_fiber**2 * np.pi
        volume_fraction_is = volume / (image_size[0] * image_size[1] * image_size[2])
        if volume_fraction_is > volume_fraction_should:
            break
    print(
        "number of fibers ", len(fiber_system), " volume fraction ", volume_fraction_is
    )
    return fiber_system


def initialize_fiber_system_endless(
    mu: float,
    R,
    beta,
    image_size: tuple[int, int, int],
    boundary_size: int,
    kappa1: float,
    kappa2: float,
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
        _, _, mid_pos, mu0, length = generate_poisson_line(rng, beta, image_size, has_beta)
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


def rot(mu, n, alpha):
    return (
        np.dot(n, mu) * n
        + np.cos(alpha) * np.cross(np.cross(n, mu), n)
        + np.sin(alpha) * np.cross(n, mu)
    )




def save_balls_in_fiber_system(
    fiber_system: list[Fiber], coords: list[np.ndarray], i: int, r_fiber: float
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
    l_fiber_discrete = len(coords)
    fiber_system.append(Fiber(Ball(coords[0], r_fiber, i, 0)))
    for j in range(1, l_fiber_discrete):
        angle = np.pi
        if j < l_fiber_discrete - 1:
            _, dir_prev = normalized(coords[j] - coords[j - 1])
            _, dir_next = normalized(coords[j + 1] - coords[j])
            angle = np.pi - np.arccos(np.dot(dir_prev, dir_next))
        fiber_system[i].add_ball(Ball(coords[j], r_fiber, i, j, angle))


