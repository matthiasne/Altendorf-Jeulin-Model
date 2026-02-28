import Altendorf_Jeulin_Model.Fiber as Fiber
from Altendorf_Jeulin_Model.utils import normalized, cartesian_to_spherical, discretize_spheres
import numpy as np


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
    accuracy = beta / 100.
    beta_0 = beta
    thetas = [
        cartesian_to_spherical(*fiber.get_direction())[1]
        for fiber in fs
    ]

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
        sum_h += np.square(np.cos(theta)) / (1 + (beta_sq - 1) * np.square(np.cos(theta)))
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
        sum_h += np.square(np.sin(2 * theta) / (1 + (beta_sq - 1) * np.square(np.cos(theta))))
    return 3 / 2 * beta * sum_h

def volume_fraction(fiber_system: list[Fiber], shape: tuple[int, int, int]):
    """
    calculates the volume fraction of the fiber system in an image

    Attributes
    ---------------------
    :param fiber_system: list[Fiber]
        the fiber system
    :param shape: tuple[int, int, int]
        the shape of the image
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

    image = discretize_spheres(coords, radii, min_coordinates, max_coordinates)
    return np.mean(image)