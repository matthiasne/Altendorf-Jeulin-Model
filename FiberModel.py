from Fiber import Fiber, Ball
from scipy.stats import rv_discrete
from scipy.stats import rv_continuous
from scipy.stats import uniform
import numpy as np
from numpy.random import default_rng

class FiberModel:
    def __init__(self, initial_fiber_system):
        if not (isinstance(initial_fiber_system, list) and all(isinstance(x, Fiber) for x in initial_fiber_system)):
            raise TypeError('Initial_fiber_system must be a list of fibers')


def initialize_fiber_system(N: int, L: float, R: float, beta: float,
                            kappa1: float, kappa2: float, seed:int = 42):
    #if not isinstance(L, rv_discrete):
    #    raise TypeError('L must be a discrete random variables modeling the length of each Fiber')
    #if not isinstance(R, rv_continuous):
    #    raise TypeError('L must be a continuous random variables modeling the radius of each Fiber')
    if beta < 0:
        raise TypeError('beta must be non-negative')

    rng = default_rng(seed)
    U = uniform(loc=0, scale=1)

    Fiber_System = [None] * N
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius (for now only constant)
        l = L
        r = R
        # 2. Simulate the mean orientation
        u1 = U.rvs(random_state=rng)
        u2 = U.rvs(random_state=rng)
        phi0 = np.pi * 2 * u1
        theta0 = np.arccos((1 - 2 * u2)/np.sqrt(beta**2 - (beta**2 - 1) * (1 - 2*u2)**2))
        mu0 = np.array(spherical_to_cartesian(1,theta0, phi0))
        # 3. Simulating a random walk for the fiber system

        # very preliminary version
        start = np.array([30, 30, 30])
        Fiber_System[i] = [Ball(start, r, i, 0)]
        for j in range(1, int(l/r)):
            pos = start + r*mu0*j
            Fiber_System[i].append(Ball(pos, r, i, j))

        # 4. Adjusting the fibers such that the mean orientation is maintained


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