import numpy as np
from libc.stdint cimport int64_t
from libc.math cimport cos
cimport numpy as np
np.import_array()
from scipy.linalg import solve_triangular
from scipy.special import digamma
from scipy.spatial import cKDTree
from libc.float cimport DBL_EPSILON


from Altendorf_Jeulin_Model.Fiber cimport Ball
from Altendorf_Jeulin_Model.Fiber import Fiber
from Altendorf_Jeulin_Model.utils cimport cartesian_to_spherical, discretize_spheres_periodic, clip
from Altendorf_Jeulin_Model.utils import (
    discretize_spheres_nonperiodic,
    normalized,
)


def mean_radius(fs: list[Fiber]):
    """
    Calculates the mean radius of the fiber system.

    :param fs: list[Fiber]
        the fiber system
    :return: float mean radius
    """
    mean_radius = np.mean([fiber.get_mean_radius() for fiber in fs])
    return mean_radius


def mean_length(fs: list[Fiber]):
    """
    Calculates the mean length of the fiber system.

    :param fs: list[Fiber]
        the fiber system
    :return: float mean length
    """
    if len(fs) == 0:
        return 0
    m_length = np.mean([fiber.get_length() for fiber in fs])
    return m_length


def mean_angle_error(fs: list[Fiber]):
    """
    Calculates the mean angle error of the fiber system
    in comparison to the original fiber system upon generation.

    :param fs: list[Fiber]
        the fiber system
    :return: float mean angle error
    """
    cdef Ball ball, ball_prev, ball_next
    angle_errors = []

    for fiber in fs:
        angles = []
        balls = fiber.balls

        for i in range(1, len(balls) - 1):
            ball = balls[i]
            ball_prev = balls[i-1]
            dir_prev = np.array([
                ball.coordinate[0] - ball_prev.coordinate[0],
                ball.coordinate[1] - ball_prev.coordinate[1],
                ball.coordinate[2] - ball_prev.coordinate[2],
                ], dtype=np.float64)
            _, dir_prev = normalized(dir_prev)
            ball_next = balls[i+1]
            dir_next = np.array([
                ball_next.coordinate[0] - ball.coordinate[0],
                ball_next.coordinate[1] - ball.coordinate[1],
                ball_next.coordinate[2] - ball.coordinate[2],
                ], dtype=np.float64)
            _, dir_next = normalized(dir_next)
            alpha = np.pi - np.arccos(clip(np.dot(dir_prev, dir_next)))

            alpha0 = ball.angle
            angles.append(abs(alpha - alpha0))

        # angle differences for the current fiber
        angle_diff = np.mean(angles) if angles else 0
        angle_errors.append(angle_diff)

    # mean angle error across all fibers
    return np.mean(angle_errors) if angle_errors else 0


def estimate_beta(fs: list[Fiber], beta: float):
    """
    Estimates the beta parameter of the Schladitz distribution following Franke et al. 2016

    :param fs: list[Fiber]
        the fiber system
    :param beta: float
        start value for the beta estimation
    :return:
    """
    accuracy = beta / 100.0
    beta_0 = beta
    thetas = [fiber.get_direction_spherical()[1] for fiber in fs]

    beta_last = beta_0 + 2 * accuracy
    beta_next = beta_0

    max_iter = 1000
    i = 0
    while abs(beta_next - beta_last) > accuracy:
        if i > max_iter:
            break
        beta_last = beta_next
        beta_next = beta_last - h(beta_last, thetas) / h_dif(beta_last, thetas)
        i += 1
    return beta_next


def h(beta: float, thetas: list):
    """
    helper function for estimate_beta

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

    :param beta: float
        last value in the beta estimation
    :param thetas: list
        list of polar angle theta of fibers
    :return:
    """
    sum_h = 0
    beta_sq = np.square(beta)
    cos_sq = np.square(np.cos(thetas))
    denom = 1.0 + (beta_sq - 1.0)*cos_sq
    term1 = np.sum(cos_sq/denom)
    term2 = np.sum(np.square(cos_sq/denom))
    return 6.0*beta*term1 - 6.0*beta_sq*beta*term2


def volume_fraction(
    fiber_system: list[Fiber], shape, is_periodic: bool = True
):
    if is_periodic:
        image = discretize_spheres_periodic(fiber_system, shape[0], shape[1], shape[2])
    else:
        image = discretize_spheres_nonperiodic(fiber_system, shape[0], shape[1], shape[2])
    return image.mean()


def estimate_kappa1(fs: list[Fiber]):
    """
    Calculates the curvature parameter kappa1 for the whole fiber system.

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
    mean = np.mean(diff_sum)
    return 1 / mean if mean else 0


def estimate_kappa2(fs: list[Fiber]):
    """
    Calculates the curvature parameter kappa2 for the whole fiber system.

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
    mean = np.mean(diff_sum)
    return 2 / mean if mean else 0


def calculate_fot(fs: list[Fiber]) -> np.ndarray:
    """
    Calculates the fiber orientation tensor (FOT) of second order of the fiber system.
    :param fs: list[Fiber]
        fiber system
    :return: np.ndarray
        FOT of shape (3, 3)
    """
    fot = np.zeros((3, 3))
    if len(fs) == 0:
        return fot
    for fiber in fs:
        _, direction = normalized(fiber.get_direction())
        fot += np.outer(direction, direction)
    fot /= len(fs)
    return fot

def acg_log_density(x, Sigma):
    """
    Log ACG density on S^2, relative to surface area,
    for x with shape (n, 3) and a 3x3 positive-definite Sigma.
    (generated by Luna 6)
    """
    L = np.linalg.cholesky(Sigma)  # Sigma = L @ L.T

    # x.T @ inv(Sigma) @ x, computed via triangular solves
    y = solve_triangular(L, x.T, lower=True)
    quad = np.sum(y * y, axis=0)

    log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
    return -np.log(4.0 * np.pi) - 0.5 * log_det_sigma - 1.5 * np.log(quad)


def kl_to_acg(samples, Sigma, k=5):
    """
    Estimate KL(P || ACG(Sigma)) for samples from unknown P on S^2.
    Returns the estimate in nats. (generated by Luna 6)
    """
    x = np.asarray(samples, dtype=float)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("samples must have shape (n, 3)")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must contain only finite values")

    norms = np.linalg.norm(x, axis=1)
    if np.any(norms < DBL_EPSILON):
        raise ValueError("samples must be nonzero vectors")
    x = x / norms[:, None]

    n = len(x)
    if not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < number of samples")

    # k-th neighbor's geodesic distance on the unit sphere
    distances, _ = cKDTree(x).query(x, k=k + 1)
    chord = np.minimum(distances[:, k], 2.0)
    angle = 2.0 * np.arcsin(chord / 2.0)

    # Spherical-cap area for each neighbor radius
    cap_area = 2.0 * np.pi * (1.0 - np.cos(angle))
    cap_area = np.maximum(cap_area, np.finfo(float).tiny)

    # kNN estimate of differential entropy on S^2
    entropy = digamma(n) - digamma(k) + np.mean(np.log(cap_area))

    # KL(P || Q) = -H(P) - E_P[log q(X)]
    return -entropy - np.mean(acg_log_density(x, Sigma))

def calculate_kl(fs, A):
    dirs = [fiber.get_direction() for fiber in fs]
    return kl_to_acg(dirs, A, k=5)