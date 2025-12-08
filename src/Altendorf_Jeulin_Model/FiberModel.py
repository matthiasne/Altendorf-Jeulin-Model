from Altendorf_Jeulin_Model.Fiber import Fiber, Ball
from scipy.stats import rv_discrete
from scipy.stats import rv_continuous
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
                            kappa1: float, kappa2: float, seed:int = 42):
    #if not isinstance(L, rv_discrete):
    #    raise TypeError('L must be a discrete random variables modeling the length of each Fiber')
    #if not isinstance(R, rv_continuous):
    #    raise TypeError('L must be a continuous random variables modeling the radius of each Fiber')
    if beta < 0:
        raise TypeError('beta must be non-negative')

    rng = default_rng(seed)
    U = uniform(loc=0, scale=1)

    # TODO
    Fiber_System = [None] * N
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius (for now only constant) TODO
        l = set_value(L, rng)
        r = set_value(R, rng)
        # 2. Simulate the mean orientation TODO: put this code into its own function
        u1 = U.rvs(random_state=rng)
        u2 = U.rvs(random_state=rng)
        phi0 = np.pi * 2 * u1
        theta0 = np.arccos((1 - 2 * u2)/np.sqrt(beta**2 - (beta**2 - 1) * (1 - 2*u2)**2))
        mu0 = np.array(spherical_to_cartesian(1,theta0, phi0))
        # 3. Simulating a random walk for the fiber system
        coord = np.zeros((l, 3))
        coord[0, 0] = image_size[0] * U.rvs(random_state=rng)
        coord[0, 1] = image_size[1] * U.rvs(random_state=rng)
        coord[0, 2] = image_size[2] * U.rvs(random_state=rng)
        Fiber_System[i] = [Ball(coord[0], r, i, 0)]

        cnt = 1
        mu_old = mu0
        while cnt < l:
            kappa_new = np.linalg.norm(kappa1 * mu0 + kappa2 * mu_old)
            mu_new = (kappa1 * mu0 + kappa2 * mu_old) / kappa_new
            vmf = vonmises_fisher(mu_new, kappa_new)

            direction = vmf.rvs(random_state=rng)[0]

            coord[cnt] = coord[cnt - 1] + r * direction / 2
            mu_old = direction
            cnt = cnt + 1

        # 4. Adjusting the fibers such that the mean orientation is maintained
        mu_bar = (coord[l - 1] - coord[0]) / np.linalg.norm(coord[l - 1] - coord[0])
        n_axis = np.cross(mu0, mu_bar) / np.linalg.norm(np.cross(mu0, mu_bar))
        alpha = np.arccos(np.dot(mu0, mu_bar))

        for j in range(1,l):
            if alpha > 0:
                coord[j] = coord[0] + rot(coord[j] - coord[0], n_axis, alpha)
            angle = np.pi
            if j < l-1:
                dir_prev = (coord[j] - coord[j-1])/np.linalg.norm(coord[j] - coord[j-1])
                dir_next = (coord[j+1] - coord[j]) / np.linalg.norm(coord[j+1] - coord[j])
                angle = np.pi - np.arccos(np.dot(dir_prev, dir_next))
            Fiber_System[i].append(Ball(coord[j], r, i, j, angle))

    return Fiber_System


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
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