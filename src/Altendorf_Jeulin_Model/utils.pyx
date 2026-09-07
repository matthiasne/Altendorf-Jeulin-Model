import copy

cimport numpy as cnp
import numpy as np
from line_profiler import profile
import cython

cimport numpy as np
np.import_array()
from libc.math cimport sin, cos, sqrt, atan2, acos


from Altendorf_Jeulin_Model.Fiber cimport Ball
import Altendorf_Jeulin_Model.Fiber as Fiber


def periodic_distance(
    coord1: np.ndarray, coord2: np.ndarray, image_size: tuple[int, int, int]
):
    """
    Calculates the periodic distance between two coordinates and the normalized direction vector between them

    :param coord1: np.ndarray
        The first coordinate, start of the direction vector
    :param coord2: np.ndarray
        The second coordinate, "end" of the direction vector
    :param image_size: tuple[int, int, int]
        The size of the image, thus, the periodicity
    :return: distance between coordinates, direction vector
    """
    coord1mod = coord1 % image_size
    coord2mod = coord2 % image_size
    dist_orig = np.linalg.norm(coord2mod - coord1mod)
    delta = coord2mod - coord1mod
    for i in range(3):
        if abs(delta[i]) > image_size[i] / 2.0:
            if delta[i] > 0:
                coord2mod[i] -= image_size[i]
            else:
                coord2mod[i] += image_size[i]

    dist, dir = normalized(coord2mod - coord1mod)

    if np.linalg.norm(coord2mod - coord1mod) > dist_orig:
        raise ValueError("There is an issue in the periodic distance calculation")
    else:
        return dist, dir


def angle_between(v1: np.ndarray, v2: np.ndarray):
    """
    Calculates the angle between two vectors v1 and v2

    :param v1: np.ndarray
    :param v2: np.ndarray
    :return: float
        angle between the two vectors v1, v2
    """
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    if l1 == 0:
        raise ValueError("v1 has length 0")
    if l2 == 0:
        raise ValueError("v2 has length 0")
    cos_angle = np.dot(v1, v2) / (l1 * l2)
    return np.acos(np.clip(cos_angle, -1.0, 1.0))


def normalized(v: np.ndarray):
    """
    normalizes a vector and returns both the original length and the normalized vector

    :param v: np.ndarray
        vector to be normalized
    :return: float, np.ndarray
        original length, normalized vector
    """
    if np.linalg.norm(v) == 0:
        return 0, v
    v_length = np.linalg.norm(v)
    return v_length, v / v_length


cdef inline void cartesian_to_spherical(double x, double y, double z, double *r, double *theta, double *phi) noexcept nogil:
    """
    transform cartesian coordinates to spherical coordinates

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    :param x: float
        cartesian x coordinate
    :param y: float
        cartesian y coordinate
    :param z: float
        cartesian z coordinate
    :return: float, float, float
        radius, theta angle, phi angle in radian
    """
    cdef double val_z_r
    r[0] = sqrt(x*x + y*y + z*z)
    if r[0] == 0.0:
        return

    phi[0] = atan2(y, x)
    val_z_r = z/r[0]
    if val_z_r > 1.0:
        val_z_r = 1.0
    elif val_z_r < -1.0:
        val_z_r = -1.0
    theta[0] = acos(val_z_r)  # avoid domain errors


def spherical_to_cartesian(r, theta, phi):
    """
    transform spherical coordinates (in radian) to cartesian coordinates

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    :param r: float
        radius
    :param theta: float
        polar theta angle (down from z-axis)
    :param phi: float
        polar phi angle (within x-y-plane)
    :return: float, float, float
        cartesian coordinates
    """
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return x, y, z


def spherical_to_matrix(theta: float, phi: float):
    """
    tranforms spherical coordinates to a rotation matrix

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    :param theta: float
        first rotation angle (down from z-axis)
    :param phi: float
        second rotation angle (within x-y-plane)
    :return: np.ndarray
        rotation matrix
    """
    return np.array(
        [
            [np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta) * np.cos(phi)],
            [np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

cdef inline void rot(double mu_x, double mu_y, double mu_z, double n_x, double n_y, double n_z, double alpha,
        double* rot_x, double* rot_y, double* rot_z) noexcept nogil:
    """
    rotation of a vector mu around normal n and angle alpha

    :param mu: np.ndarray
        vector to be rotated
    :param n: np.ndarray
        vector to be rotated around
    :param alpha: float
        angle to be rotated by
    :return: np.ndarray
        rotated vector
    """
    cdef double dot = n_x*mu_x + n_y*mu_y + n_z*mu_z
    cdef double n_cross_mu_x = n_y*mu_z - n_z*mu_y
    cdef double n_cross_mu_y = n_z*mu_x - n_x*mu_z
    cdef double n_cross_mu_z = n_x*mu_y - n_y*mu_x

    cdef double cross2_x = n_cross_mu_y*n_z - n_cross_mu_z*n_y
    cdef double cross2_y = n_cross_mu_z*n_x - n_cross_mu_x*n_z
    cdef double cross2_z = n_cross_mu_x*n_y - n_cross_mu_y*n_x

    cdef double cosa = cos(alpha)
    cdef double sina = sin(alpha)

    rot_x[0] = dot*n_x + cosa*cross2_x + sina*n_cross_mu_x
    rot_y[0] = dot*n_y + cosa*cross2_y + sina*n_cross_mu_y
    rot_z[0] = dot*n_z + cosa*cross2_z + sina*n_cross_mu_z


def is_in_image(
    pos, image_size, buffer: int = 100
) -> bool:
    """
    calculates whether a coordinate lies within an image/observation window

    :param pos: np.ndarray
        position to be checked
    :param image_size: tuple[int, int, int]
        image size
    :param buffer: int
        size of image extension
    :return: bool
        is in image or not
    """
    if pos[0] < -buffer or pos[1] < -buffer or pos[2] < -buffer:
        return False
    elif (
        pos[0] > image_size[0] + buffer
        or pos[1] > image_size[1] + buffer
        or pos[2] > image_size[2] + buffer
    ):
        return False
    else:
        return True


def cut_border(fs: list[Fiber], image_size, boundary_size: int) -> list[Fiber]:
    """
    cuts fibers when they cross image/observation window borders

    :param fs: list[Fiber]
        fiber system to be cut
    :param image_size: tuple[int, int, int]
        image size
    :param boundary_size: int
        size of the boundary to each side
    :return: list[Fiber]
        cut fiber system
    """
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    fs_cut = copy.deepcopy(fs)
    for fiber in fs_cut:
        j_start = -1
        j_end = len(fiber.balls) - 1
        for j in range(0, len(fiber.balls)):
            fiber.balls[j].coordinate = fiber.balls[j].coordinate - boundary_size_vec
            if j_start == j - 1 and not is_in_image(
                fiber.balls[j].coordinate, image_size, 0
            ):
                j_start = j
            elif j_start < j and not is_in_image(
                fiber.balls[j].coordinate, image_size, 0
            ):
                j_end = j + 1
                break
        fiber.balls = fiber.balls[j_start + 1 : j_end - 1]
    return fs_cut



cpdef cnp.ndarray[cnp.uint8_t, ndim=3] discretize_spheres_periodic(
    object fiber_system,
    int i_x, int i_y, int i_z
):
    """
    Discretize spheres to an image with periodic boundary conditions
    :param coordinates: np.ndarray
        The sphere coordinates in voxels
    :param radii: np.ndarray
        The sphere radii in voxels
    :param min_coordinates: np.ndarray
        The smallest coordinate of spheres in voxels
    :param image_shape: np.ndarray
        The image size in voxels
    :return: np.ndarray
        The image containing spheres
    """
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] image = \
            np.zeros((i_x, i_y, i_z), dtype=np.uint8)
    cdef object fiber
    cdef Ball ball
    cdef double r, r_square, c_x, c_y, c_z, d_i, delta_i, d_ij, delta_ij, d_k, delta_ijk
    cdef int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max

    for fiber in fiber_system:
        for ball in fiber.balls:
            r = ball.radius
            r_square = r*r
            c_x = ball.coordinate[0]
            c_y = ball.coordinate[1]
            c_z = ball.coordinate[2]
            i_min = int(c_x - r) - 1
            i_max = int(c_x + r) + 1
            j_min = int(c_y - r) - 1
            j_max = int(c_y + r) + 1
            k_min = int(c_z - r) - 1
            k_max = int(c_z + r) + 1
            for i in range(i_min, i_max):
                i_corr = i % i_x
                d_i = i - c_x
                delta_i = d_i*d_i
                for j in range(j_min, j_max):
                    j_corr = j % i_y
                    d_j = j - c_y
                    delta_ij = delta_i + d_j*d_j
                    if delta_ij > r_square:
                        continue
                    for k in range(k_min, k_max):
                        d_z = k - c_z
                        if delta_ij + d_z*d_z <= r_square:
                            k_corr = k % i_z
                            image[i_corr, j_corr, k_corr] = 1
    return image


def discretize_spheres_nonperiodic(
    coordinates: np.ndarray,
    radii: np.ndarray,
    min_coordinates: np.ndarray,
    image_shape: np.ndarray,
):
    """
    Discretize spheres to an image (no periodic boundary conditions)
    :param coordinates: np.ndarray
        The sphere coordinates in voxels
    :param radii: np.ndarray
        The sphere radii in voxels
    :param min_coordinates: np.ndarray
        The smallest coordinate of spheres in voxels
    :param image_shape: np.ndarray
        The image size in voxels
    :return: np.ndarray
        The image containing spheres
    """
    image = np.zeros(image_shape, "uint16")
    coordinates = coordinates - min_coordinates

    for iota in range(len(coordinates)):
        r_square = radii[iota] ** 2
        for i in range(
            int(coordinates[iota, 0] - radii[iota]) - 1,
            int(coordinates[iota, 0] + radii[iota]) + 1,
        ):
            if i < 0 or i >= image_shape[0]:
                continue
            delta_i = (i - coordinates[iota, 0]) ** 2
            for j in range(
                int(coordinates[iota, 1] - radii[iota]) - 1,
                int(coordinates[iota, 1] + radii[iota]) + 1,
            ):
                if j < 0 or j >= image_shape[1]:
                    continue
                delta_ij = delta_i + (j - coordinates[iota, 1]) ** 2
                for k in range(
                    int(coordinates[iota, 2] - radii[iota]) - 1,
                    int(coordinates[iota, 2] + radii[iota]) + 1,
                ):
                    if k < 0 or k >= image_shape[2]:
                        continue
                    delta_ijk = delta_ij + (k - coordinates[iota, 2]) ** 2
                    if delta_ijk <= r_square:
                        image[i, j, k] = 1
    return image

cdef inline double cdistance_ball(Ball a, Ball b) noexcept nogil:
    cdef double dx = b.coordinate[0] - a.coordinate[0]
    cdef double dy = b.coordinate[1] - a.coordinate[1]
    cdef double dz = b.coordinate[2] - a.coordinate[2]

    return sqrt(dx * dx + dy * dy + dz * dz)

@cython.cdivision(True)
cdef inline double cdistance3(double ax, double ay, double az, double bx, double by, double bz) noexcept nogil:
    cdef double dx = ax - bx
    cdef double dy = ay - by
    cdef double dz = az - bz

    return sqrt(dx * dx + dy * dy + dz * dz)

@cython.cdivision(True)
cdef inline double cnormalized(double* ax, double* ay, double* az) noexcept nogil:
    cdef double norm = sqrt (ax[0]*ax[0] + ay[0]*ay[0] + az[0]*az[0])
    ax[0] = ax[0] / norm
    ay[0] = ay[0] / norm
    az[0] = az[0] / norm

    return norm


cdef inline double cdirection(Ball a, Ball b, double* dir_x, double* dir_y, double* dir_z) noexcept:
    cdef double dx = b.coordinate[0] - a.coordinate[0]
    cdef double dy = b.coordinate[1] - a.coordinate[1]
    cdef double dz = b.coordinate[2] - a.coordinate[2]
    cdef double dist = sqrt(dx * dx + dy * dy + dz * dz)
    if dist <= 0:
        dir_x[0] = 0
        dir_y[0] = 0
        dir_z[0] = 0
        return 0
    dir_x[0] = dx/dist
    dir_y[0] = dy/dist
    dir_z[0] = dz/dist
    return dist

cdef inline double clip(double val) noexcept nogil:
    if val > 1:
        val = 1
    if val < -1:
        val = -1
    return val
