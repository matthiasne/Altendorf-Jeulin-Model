import numpy as np

from libc.math cimport sin, cos, sqrt, atan2, acos

cimport numpy as np
np.import_array()
from numpy.random import default_rng
cdef double PI = 3.141592653589793

from Altendorf_Jeulin_Model.utils.rv_utils import vonmises_fisher
from Altendorf_Jeulin_Model.Fiber import Ball, Fiber
from Altendorf_Jeulin_Model.utils.utils cimport cartesian_to_spherical, cnormalized, clip, rot
from Altendorf_Jeulin_Model.PoissonLines import (
    generate_half_fiber,
    generate_poisson_line,
)


from Altendorf_Jeulin_Model.utils.utils import (
    is_in_image,
    set_value
)


class FiberModel:
    """
    Container for a fiber system managed by the Altendorf–Jeulin model.

    :param initial_fiber_system: list[Fiber]
        the starting fiber system; must be a non-empty list of Fiber objects
    :raises TypeError:
        if initial_fiber_system is not a list of Fiber instances
    """
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
    Generate a fiber system of straight fibers placed at random positions.

    Each fiber follows its mean direction exactly (no curvature).  Ball
    centers are spaced r_fiber/2 apart along the mean direction.  Starting
    positions are drawn uniformly within image_size.  This is a simplified
    initialiser; for curved fibers use initialize_fiber_system_AJ.

    :param intensity: float or random variable
        expected number of fibers; passed to set_value, so it may be a
        scipy distribution or a plain number
    :param L: float or random variable
        fiber length; passed to set_value
    :param R: float or random variable
        fiber radius; passed to set_value
    :param direction_distribution: callable
        function (rng) -> (mu_x, mu_y, mu_z) returning a unit direction
        vector for each fiber
    :param image_size: array-like of shape (3,)
        dimensions of the domain (x, y, z); starting positions are drawn
        uniformly in [0, image_size[d]) for each dimension d
    :param seed: int or None, optional
        seed for the random number generator; default None (random)
    :return: list[Fiber]
        generated fiber system with overlapping fibers
    """
    cdef double r, theta, phi
    cdef double mu0_x, mu0_y, mu0_z, rot_x, rot_y, rot_z
    cdef double mu_old_x, mu_old_y, mu_old_z, mu_new_x, mu_new_y, mu_new_z
    cdef double mu_bar_x, mu_bar_y, mu_bar_z
    cdef double dir_prev_x, dir_prev_y, dir_prev_z, dir_next_x, dir_next_y, dir_next_z
    rng = default_rng(seed)
    N = set_value(intensity, rng)

    fiber_system = []
    for i in range(0, N):
        # 1. Simulate the length of the ith Fiber and its radius
        l_fiber = set_value(L, rng)
        r_fiber = set_value(R, rng)
        l_fiber_discrete = max(0, int(2 * ((l_fiber-2*r_fiber )/ r_fiber)))
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
    Generate a fiber system using the Altendorf–Jeulin (2011) random-walk model.

    Each fiber is grown by a von Mises–Fisher random walk whose local mean
    direction is a weighted combination of the global mean direction (kappa1)
    and the previous step direction (kappa2).  After the walk, the fiber is
    rigidly rotated so that its end-to-end direction matches the sampled mean
    direction mu0.

    Ball centers are spaced r_fiber/2 apart.  Starting positions are drawn
    uniformly within image_size.

    :param intensity: float or random variable
        expected number of fibers; passed to set_value
    :param L: float or random variable
        fiber length; passed to set_value
    :param R: float or random variable
        fiber radius; passed to set_value
    :param direction_distribution: callable
        function (rng) -> (mu_x, mu_y, mu_z) returning a unit mean
        direction vector for each fiber
    :param image_size: array-like of shape (3,)
        dimensions of the domain (x, y, z)
    :param kappa1: float
        weight of the global mean direction mu0 in the random-walk step;
        higher values pull each step toward mu0
    :param kappa2: float
        weight of the previous step direction in the random-walk step;
        higher values increase local persistence (straightness)
    :param seed: int or None, optional
        seed for the random number generator; default None (random)
    :return: list[Fiber]
        generated fiber system with overlapping fibers
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
        l_fiber_discrete = max(0, int(2 * ((l_fiber-2*r_fiber )/ r_fiber)))
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
        n_axis_x = mu_bar_y*mu0_z - mu_bar_z*mu0_y
        n_axis_y = mu_bar_z*mu0_x - mu_bar_x*mu0_z
        n_axis_z = mu_bar_x*mu0_y - mu_bar_y*mu0_x
        cnormalized(&n_axis_x, &n_axis_y, &n_axis_z)
        alpha = acos(clip(mu0_x*mu_bar_x + mu0_y*mu_bar_y + mu0_z*mu_bar_z))

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
    Generate a system of endless (domain-spanning) fibers following
    Prakash Easwaran's method.

    Fibers are grown from Poisson line midpoints in both directions using
    generate_half_fiber.  An extended domain of size image_size +
    2*boundary_size is used so that fibers entering from outside the
    boundary are captured.  Fibers shorter than 2 balls after cropping
    are discarded.

    :param mu: float or random variable
        mean number of fibers to attempt; passed to set_value
    :param R: float or random variable
        fiber radius; passed to set_value
    :param direction_distribution: callable
        function (rng) -> (mu_x, mu_y, mu_z) returning a unit mean
        direction vector for each fiber
    :param image_size: np.ndarray of shape (3,)
        dimensions of the core domain (x, y, z), excluding boundary
    :param boundary_size: int
        width of the boundary layer added on each side of the domain
    :param kappa1: float
        weight of the global mean direction in the random-walk step
    :param kappa2: float
        weight of the previous step direction in the random-walk step
    :param seed: int or None, optional
        seed for the random number generator; default None (random)
    :return: list[Fiber]
        generated fiber system of endless fibers, still overlapping
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


def save_balls_in_fiber_system(
        fiber_system: list[Fiber], coords, i: int, r_fiber: float|int
):
    """
    Append a new Fiber built from a dense coordinate array to the fiber system.

    For each interior ball (index 1 to n-2) the bending angle is computed
    from the directions of the incoming and outgoing segments and stored in
    ball.angle.  The first and last balls receive angle = pi (straight).

    :param fiber_system: list[Fiber]
        the fiber system to append to; modified in place
    :param coords: np.ndarray, shape (n, 3)
        ball center coordinates in order along the fiber
    :param i: int
        fiber index used as fiber_label for all balls; must equal
        len(fiber_system) before the call
    :param r_fiber: float
        radius assigned to every ball in the fiber
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
    Append a new Fiber built from a list of coordinate arrays to the fiber system.

    Equivalent to save_balls_in_fiber_system but accepts coords as a Python
    list of 1-D arrays (the format returned by generate_half_fiber) rather
    than a 2-D numpy array.

    For each interior ball the bending angle is computed from the directions
    of the incoming and outgoing segments and stored in ball.angle.  The
    first and last balls receive angle = pi (straight).

    :param fiber_system: list[Fiber]
        the fiber system to append to; modified in place
    :param coords: list[array-like]
        ball center coordinates in order along the fiber, each of length 3
    :param i: int
        fiber index used as fiber_label for all balls; must equal
        len(fiber_system) before the call
    :param r_fiber: float
        radius assigned to every ball in the fiber
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


