from Altendorf_Jeulin_Model.Fiber import Fiber, Ball
from Altendorf_Jeulin_Model.utils import normalized, spherical_to_cartesian
from scipy.stats import uniform
from scipy.stats import vonmises_fisher
import numpy as np
from numpy.random import default_rng


class FiberModel:
    def __init__(self, initial_fiber_system):
        if not (isinstance(initial_fiber_system, list) and all(isinstance(x, Fiber) for x in initial_fiber_system)):
            raise TypeError('Initial_fiber_system must be a list of fibers')


def initialize_fiber_system(N: int, L, R, beta: float, image_size: tuple[int, int, int],
                            kappa1: float, kappa2: float, seed: int = 42):
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

    fiber_system = [None] * N
    beta_sq = np.square(beta)
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius (for now only constant) TODO
        l_fiber = set_value(L, rng)
        r_fiber = set_value(R, rng)
        l_fiber_discrete = int(2*l_fiber/r_fiber + 1)
        # 2. Simulate the mean orientation
        u1 = U.rvs(random_state=rng)
        u2 = U.rvs(random_state=rng)
        phi0 = np.pi * 2 * u1
        theta0 = np.arccos((1 - 2 * u2) / np.sqrt(beta_sq - (beta_sq - 1) * np.square(1 - 2 * u2)))
        mu0 = np.array(spherical_to_cartesian(1, theta0, phi0))
        # 3. Simulating a random walk for the fiber system
        coord = np.zeros((l_fiber_discrete, 3))
        coord[0, 0] = image_size[0] * U.rvs(random_state=rng)
        coord[0, 1] = image_size[1] * U.rvs(random_state=rng)
        coord[0, 2] = image_size[2] * U.rvs(random_state=rng)
        fiber_system[i] = Fiber(Ball(coord[0], r_fiber, i, 0))

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
        #print("mu0 ", mu0, " mubar ", mu_bar, " mubar2 ", mu_bar2)

        # saving balls in fiber_system
        for j in range(1, l_fiber_discrete):
            angle = np.pi
            if j < l_fiber_discrete - 1:
                _, dir_prev = normalized(coord[j] - coord[j - 1])
                _, dir_next = normalized(coord[j + 1] - coord[j])
                angle = np.pi - np.arccos(np.dot(dir_prev, dir_next))
            fiber_system[i].add_ball(Ball(coord[j], r_fiber, i, j, angle))

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
