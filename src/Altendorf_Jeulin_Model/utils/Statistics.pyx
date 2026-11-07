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
from Altendorf_Jeulin_Model.utils.utils cimport cartesian_to_spherical, discretize_spheres_periodic, clip
from Altendorf_Jeulin_Model.utils.utils import (
    discretize_spheres_nonperiodic,
    normalized,
)


def mean_radius(fs: list[Fiber]):
    """
    Calculates the mean radius of the fiber system.
    Returns 0 if the fiber system is empty.

    :param fs: list[Fiber]
        the fiber system
    :return: float
        mean radius across all balls of all fibers
    """
    if len(fs) == 0:
        return 0
    mean_radius = np.mean([fiber.get_mean_radius() for fiber in fs])
    return mean_radius


def mean_length(fs: list[Fiber]):
    """
    Calculates the mean length of the fiber system.
    Returns 0 if the fiber system is empty.

    :param fs: list[Fiber]
        the fiber system
    :return: float
        mean end-to-end arc length across all fibers, or 0 if fs is empty
    """
    if len(fs) == 0:
        return 0
    m_length = np.mean([fiber.get_length() for fiber in fs])
    return m_length


def mean_angle_error(fs: list[Fiber]):
    """
    Calculates the mean angle error of the fiber system.

    For each interior ball the bending angle alpha is computed from the
    directions of the incoming and outgoing segments
    (alpha = pi - arccos(dir_prev^T dir_next)).  The error is the
    absolute difference between alpha and the target bending angle
    stored in ball.angle at generation time.  The per-fiber mean is
    averaged across all fibers.

    Returns 0 if the fiber system is empty or no fiber has interior balls.

    :param fs: list[Fiber]
        the fiber system
    :return: float
        mean bending-angle error in radians
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


def estimate_beta(thetas, beta: float):
    """
    Estimates the beta parameter of the Schladitz distribution
    following Franke et al. (2016) via Newton–Raphson iteration.

    The polar angle theta of each fiber is derived from its endpoint
    direction vector.  Iteration stops when the step size falls below
    an absolute tolerance of 1e-8 or after 1000 iterations.

    :param fs: list[Fiber]
        the fiber system
    :param beta: float
        initial guess for beta (must be positive)
    :return: float
        estimated beta parameter at convergence
    """
    tol = 1e-5
    beta_last = beta + 2 * tol
    beta_next = beta
    max_iter = 1000
    for _ in range(max_iter):
        if abs(beta_next - beta_last) <= tol:
            break
        beta_last = beta_next
        beta_next = beta_last - h(beta_last, thetas) / h_dif(beta_last, thetas)
    return beta_next


def h(beta: float, thetas):
    """
    Score function for the Schladitz MLE of beta.

    Evaluates h(beta) = -N + 3*beta^2 * sum_i cos^2(theta_i) /
    (1 + (beta^2 - 1)*cos^2(theta_i)), whose root is the maximum-
    likelihood estimate of beta.  Used by estimate_beta.

    :param beta: float
        current beta value
    :param thetas: array-like of float
        polar angles theta of the fiber directions (radians)
    :return: float
        value of the score function h at beta
    """
    beta_sq = np.square(beta)
    cos_sq = np.square(np.cos(thetas))
    denom = 1.0 + (beta_sq - 1.0) * cos_sq
    return -len(thetas) + 3.0 * beta_sq * np.sum(cos_sq / denom)


def h_dif(beta: float, thetas):
    """
    Derivative of the score function h with respect to beta.

    Evaluates h'(beta) = 6*beta * sum_i c_i/d_i
                        - 6*beta^3 * sum_i c_i^2/d_i^2
    where c_i = cos^2(theta_i) and d_i = 1 + (beta^2 - 1)*c_i.
    Used as the denominator in the Newton–Raphson step of estimate_beta.

    :param beta: float
        current beta value
    :param thetas: array-like of float
        polar angles theta of the fiber directions (radians)
    :return: float
        value of h'(beta)
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
    """
    Calculates the volume fraction of the fiber system in a periodic image.

    Discretizes all balls of the fiber system onto a binary voxel grid
    using periodic boundary conditions and returns the fraction of
    occupied voxels.

    :param fiber_system: list[Fiber]
        the fiber system
    :param shape: tuple[int, int, int]
        voxel dimensions of the image (x, y, z)
    :param is_periodic: bool
        treats fiber discretization depending on whether or not we use
        periodic boundaries (default: True)
    :return: float
        volume fraction in [0, 1]
    """
    if is_periodic:
        image = discretize_spheres_periodic(fiber_system, shape[0], shape[1], shape[2])
    else:
        image = discretize_spheres_nonperiodic(fiber_system, shape[0], shape[1], shape[2])
    return image.mean()


def estimate_kappa1(fs: list[Fiber]):
    """
    Estimates the global curvature parameter kappa1 of the fiber system.

    For each fiber, kappa1 is related to how much each successive segment
    direction deviates from the overall fiber direction (first ball to last
    ball).  The estimator is 1 / mean(||dir_segment - dir_fiber||^2).

    Returns 0 if the fiber system is empty or if all fibers are perfectly
    straight (mean squared deviation is zero).

    :param fs: list[Fiber]
        the fiber system
    :return: float
        estimate for kappa1, or 0 if undefined
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
    Estimates the local curvature parameter kappa2 of the fiber system.

    For each interior ball, kappa2 is related to how much the outgoing
    segment direction deviates from the incoming segment direction.
    The estimator is 2 / mean(||dir_next - dir_prev||^2).

    Returns 0 if the fiber system is empty or if all fibers are perfectly
    straight (mean squared deviation is zero).

    :param fs: list[Fiber]
        the fiber system
    :return: float
        estimate for kappa2, or 0 if undefined
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
    Calculates the fiber orientation tensor (FOT) of second order.

    :param fs: list[Fiber]
        the fiber system (must be non-empty)
    :return: np.ndarray, shape (3, 3)
        symmetric FOT matrix with trace 1
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
    Log-density of the Angular Central Gaussian (ACG) distribution on S^2.

    :param x: np.ndarray, shape (n, 3)
        unit direction vectors on S^2
    :param Sigma: np.ndarray, shape (3, 3)
        symmetric positive-definite parameter matrix
    :return: np.ndarray, shape (n,)
        log-density values (relative to surface area measure)
    """
    L = np.linalg.cholesky(Sigma)  # Sigma = L @ L.T

    # x.T @ inv(Sigma) @ x, computed via triangular solves
    y = solve_triangular(L, x.T, lower=True)
    quad = np.sum(y * y, axis=0)

    log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
    return -np.log(4.0 * np.pi) - 0.5 * log_det_sigma - 1.5 * np.log(quad)


def kl_to_acg(samples, Sigma, k=5):
    """
    Estimate KL(P || ACG(Sigma)) for directions sampled from an unknown
    distribution P on S^2.

    Uses a k-nearest-neighbour estimator for the differential entropy of P
    (Kozachenko–Leonenko, adapted to the sphere via spherical-cap areas),
    combined with the analytical ACG log-density.

    Note: the kNN entropy estimator has a negative bias for small sample
    sizes, so the returned value may be slightly negative even when the
    true KL divergence is zero.

    :param samples: array-like, shape (n, 3)
        direction vectors (need not be unit vectors; normalised internally)
    :param Sigma: np.ndarray, shape (3, 3)
        symmetric positive-definite parameter matrix of the ACG reference
    :param k: int
        number of nearest neighbours for the entropy estimator (default 5);
        must satisfy 1 <= k < n
    :return: float
        estimated KL divergence in nats
    :raises ValueError:
        if samples does not have shape (n, 3), contains non-finite values,
        contains zero vectors, or k is out of range
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
    """
    Estimate KL(P_fiber || ACG(A)) for a fiber system.

    Extracts the endpoint direction of each fiber, normalises it to a unit
    vector, and passes the directions to kl_to_acg.

    :param fs: list[Fiber]
        the fiber system
    :param A: np.ndarray, shape (3, 3)
        symmetric positive-definite parameter matrix of the ACG reference
    :return: float
        estimated KL divergence in nats (see kl_to_acg for caveats)
    """
    dirs = [fiber.get_direction() for fiber in fs]
    return kl_to_acg(dirs, A, k=5)