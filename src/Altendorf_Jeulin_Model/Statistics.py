import numpy as np

import Altendorf_Jeulin_Model.Fiber as Fiber
from Altendorf_Jeulin_Model.utils import (
    cartesian_to_spherical,
    discretize_spheres_nonperiodic,
    discretize_spheres_periodic,
    normalized,
)


def mean_radius(fs: list[Fiber]):
    """
    Calculates the mean radius of the fiber system.

    Attributes
    ---------------------
    :param fs: list[Fiber]
        the fiber system
    :return: float mean radius
    """
    mean_radius = np.mean([fiber.get_mean_radius() for fiber in fs])
    return mean_radius


def mean_length(fs: list[Fiber]):
    """
    Calculates the mean length of the fiber system.

    Attributes
    ---------------------
    :param fs: list[Fiber]
        the fiber system
    :return: float mean length
    """
    if len(fs) == 0:
        return 0
    mean_length = np.mean([fiber.get_length() for fiber in fs])
    return mean_length


def mean_angle_error(fs: list[Fiber]):
    """
    Calculates the mean angle error of the fiber system
    in comparison to the original fiber system upon generation.

    Attributes
    ---------------------
    :param fs: list[Fiber]
        the fiber system
    :return: float mean angle error
    """
    angle_errors = []

    for fiber in fs:
        angles = []
        balls = fiber.balls

        for i in range(1, len(balls) - 1):
            ball = balls[i]
            alpha0 = ball.angle
            _, dir_prev = normalized(ball.coordinate - balls[i - 1].coordinate)
            _, dir_next = normalized(balls[i + 1].coordinate - ball.coordinate)
            alpha = np.pi - np.arccos(np.dot(dir_prev, dir_next))
            angles.append(abs(alpha - alpha0))

        # angle differences for the current fiber
        angle_diff = np.mean(angles) if angles else 0
        angle_errors.append(angle_diff)

    # mean angle error across all fibers
    return np.mean(angle_errors) if angle_errors else 0


def estimate_beta(fs: list[Fiber], beta: float):
    """
    Estimates the beta parameter of the Schladitz distribution following Franke et al. 2016

    Attributes
    ---------------------
    :param fs: list[Fiber]
        the fiber system
    :param beta: float
        start value for the beta estimation
    :return:
    """
    accuracy = beta / 100.0
    beta_0 = beta
    thetas = [cartesian_to_spherical(*fiber.get_direction())[1] for fiber in fs]

    beta_last = beta_0 + 2 * accuracy
    beta_next = beta_0

    while abs(beta_next - beta_last) > accuracy:
        beta_last = beta_next
        beta_next = beta_last - h(beta_last, thetas) / h_dif(beta_last, thetas)
    return beta_next


def h(beta: float, thetas: list):
    """
    helper function for estimate_beta

    Attributes
    ---------------------
    :param beta: float
        last value in the beta estimation
    :param thetas: list
        list of polar angle theta of fibers
    :return:
    """
    sum_h = 0
    beta_sq = np.square(beta)
    for theta in thetas:
        sum_h += np.square(np.cos(theta)) / (
            1 + (beta_sq - 1) * np.square(np.cos(theta))
        )
    return -len(thetas) + 3 * beta_sq * sum_h


def h_dif(beta: float, thetas: list):
    """
    helper function for estimate_beta

    Attributes
    ---------------------
    :param beta: float
        last value in the beta estimation
    :param thetas: list
        list of polar angle theta of fibers
    :return:
    """
    sum_h = 0
    beta_sq = np.square(beta)
    for theta in thetas:
        sum_h += np.square(
            np.sin(2 * theta) / (1 + (beta_sq - 1) * np.square(np.cos(theta)))
        )
    return 3 / 2 * beta * sum_h


def volume_fraction(
    fiber_system: list[Fiber], shape: tuple[int, int, int], is_periodic: bool = True
):
    """
    calculates the volume fraction of the fiber system in an image

    Attributes
    ---------------------
    :param fiber_system: list[Fiber]
        the fiber system
    :param shape: tuple[int, int, int]
        the shape of the image
    :param is_periodic: bool
        boundary conditions of the image
    :return: float
        the volume fraction of the fiber system
    """
    coords = []
    radii = []
    for fiber in fiber_system:
        for ball in fiber.balls:
            coords.append(ball.coordinate)
            radii.append(ball.radius)
    coords = np.array(coords)
    radii = np.array(radii)

    min_coordinates = np.array([0, 0, 0])
    max_coordinates = np.array(shape)

    if is_periodic:
        image = discretize_spheres_periodic(
            coords, radii, min_coordinates, max_coordinates
        )
    else:
        image = discretize_spheres_nonperiodic(
            coords, radii, min_coordinates, max_coordinates
        )
    return np.mean(image)


def estimate_kappa1(fs: list[Fiber]):
    """
    Calculates the curvature parameter kappa1 for the whole fiber system.

    Attributes
    ---------------------
    :param fs: list[Fiber]
        the fiber system
    :return: float
        estimate for kappa1
    """
    diff_sum = []

    for fiber in fs:
        diff = []
        balls = fiber.balls
        _, dir_main = normalized(balls[-1].coordinate - balls[0].coordinate)
        for i in range(0, len(balls) - 1):
            _, dir_current = normalized(balls[i + 1].coordinate - balls[i].coordinate)
            diff.append(
                np.square(dir_current[0] - dir_main[0])
                + np.square(dir_current[1] - dir_main[1])
                + np.square(dir_current[2] - dir_main[2])
            )

        # angle differences for the current fiber
        diff_inner_sum = np.mean(diff) if diff else 0
        diff_sum.append(diff_inner_sum)
    # mean angle error across all fibers
    return 1 / np.mean(diff_sum) if diff_sum else 0


def estimate_kappa2(fs: list[Fiber]):
    """
    Calculates the curvature parameter kappa2 for the whole fiber system.

    Attributes
    ---------------------
    :param fs: list[Fiber]
        the fiber system
    :return: float
        estimate for kappa2
    """
    diff_sum = []

    for fiber in fs:
        diff = []
        balls = fiber.balls

        for i in range(1, len(balls) - 1):
            ball = balls[i]
            _, dir_prev = normalized(ball.coordinate - balls[i - 1].coordinate)
            _, dir_next = normalized(balls[i + 1].coordinate - ball.coordinate)
            diff.append(
                np.square(dir_prev[0] - dir_next[0])
                + np.square(dir_prev[1] - dir_next[1])
                + np.square(dir_prev[2] - dir_next[2])
            )

        # angle differences for the current fiber
        diff_inner_sum = np.mean(diff) if diff else 0
        diff_sum.append(diff_inner_sum)

    # mean angle error across all fibers
    return 2 / np.mean(diff_sum) if diff_sum else 0
