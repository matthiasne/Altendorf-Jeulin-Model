from Altendorf_Jeulin_Model.Fiber import Fiber, Ball
from scipy.stats import uniform
from scipy.stats import vonmises_fisher
import numpy as np
from numpy.random import default_rng


# TODO: normalize vectors function

class FiberModel:
    def __init__(self, initial_fiber_system):
        if not (isinstance(initial_fiber_system, list) and all(isinstance(x, Fiber) for x in initial_fiber_system)):
            raise TypeError('Initial_fiber_system must be a list of fibers')


def initialize_fiber_system(N: int, L, R, beta: float, image_size: tuple[int, int, int],
                            kappa1: float, kappa2: float, seed: int = 42):
    if beta < 0:
        raise TypeError('beta must be non-negative')

    rng = default_rng(seed)
    U = uniform(loc=0, scale=1)

    # TODO
    Fiber_System = [None] * N
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius (for now only constant) TODO
        l_fiber = set_value(L, rng)
        r_fiber = set_value(R, rng)
        # 2. Simulate the mean orientation TODO: put this code into its own function
        u1 = U.rvs(random_state=rng)
        u2 = U.rvs(random_state=rng)
        phi0 = np.pi * 2 * u1
        theta0 = np.arccos((1 - 2 * u2) / np.sqrt(beta ** 2 - (beta ** 2 - 1) * (1 - 2 * u2) ** 2))
        mu0 = np.array(spherical_to_cartesian(1, theta0, phi0))
        # 3. Simulating a random walk for the fiber system
        coord = np.zeros((l_fiber, 3))
        coord[0, 0] = image_size[0] * U.rvs(random_state=rng)
        coord[0, 1] = image_size[1] * U.rvs(random_state=rng)
        coord[0, 2] = image_size[2] * U.rvs(random_state=rng)
        Fiber_System[i] = Fiber(Ball(coord[0], r_fiber, i, 0))

        cnt = 1
        mu_old = mu0
        while cnt < l_fiber:
            kappa_new = np.linalg.norm(kappa1 * mu0 + kappa2 * mu_old)
            mu_new = (kappa1 * mu0 + kappa2 * mu_old) / kappa_new
            vmf = vonmises_fisher(mu_new, kappa_new)

            direction = vmf.rvs(random_state=rng)[0]

            coord[cnt] = coord[cnt - 1] + r_fiber * direction / 2
            mu_old = direction
            cnt = cnt + 1

        # 4. Adjusting the fibers such that the mean orientation is maintained
        mu_bar = (coord[l_fiber - 1] - coord[0]) / np.linalg.norm(coord[l_fiber - 1] - coord[0])
        n_axis = np.cross(mu0, mu_bar) / np.linalg.norm(np.cross(mu0, mu_bar))
        alpha = np.arccos(np.dot(mu0, mu_bar))

        for j in range(1, l_fiber):
            if alpha > 0:
                coord[j] = coord[0] + rot(coord[j] - coord[0], n_axis, alpha)
            angle = np.pi
            if j < l_fiber - 1:
                dir_prev = (coord[j] - coord[j - 1]) / np.linalg.norm(coord[j] - coord[j - 1])
                dir_next = (coord[j + 1] - coord[j]) / np.linalg.norm(coord[j + 1] - coord[j])
                angle = np.pi - np.arccos(np.dot(dir_prev, dir_next))
            Fiber_System[i].add_ball(Ball(coord[j], r_fiber, i, j, angle))

    return Fiber_System


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z / r, -1, 1))  # avoid domain errors
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def rot(mu, n, alpha):
    return np.dot(n, mu) * n + np.cos(alpha) * np.cross(np.cross(n, mu), n) + np.sin(alpha) * np.cross(n, mu)


def set_value(input_value, rng):
    if isinstance(input_value, float) or isinstance(input_value, int):
        # L is a constant number
        result = input_value
    elif hasattr(input_value, 'rvs'):
        # L is a Poisson generator (or any similar object with an rvs method)
        result = input_value.rvs(random_state=rng)
    else:
        raise ValueError("Input must be a float/int or a distribution object with an 'rvs' method.")

    return result
