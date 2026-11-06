import copy
from numbers import Real

cimport numpy as cnp
cimport numpy as np
import cython
import numpy as np
from skimage.draw import line_nd
from skimage.morphology import isotropic_dilation

np.import_array()
from libc.math cimport  sqrt, atan2, acos
from libc.float cimport DBL_EPSILON

from Altendorf_Jeulin_Model.Fiber cimport Ball
import Altendorf_Jeulin_Model.Fiber as Fiber

cdef double PI = 3.141592653589793

def angle_between(v1: np.ndarray, v2: np.ndarray):
    """
    Calculates the angle between two vectors v1 and v2.

    :param v1: np.ndarray
    :param v2: np.ndarray
    :return: float
        Angle between v1 and v2 in radians, in the range [0, pi]
    :raises ValueError:
        If either v1 or v2 has length 0
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
    Normalizes a vector and returns both the original length and the normalized vector.

    If the zero vector is passed, returns (0, v) unchanged without raising.

    :param v: np.ndarray
        Vector to be normalized
    :return: tuple[float, np.ndarray]
        Original length and normalized vector. If v is the zero vector, returns (0, v).
    """
    v_length = np.linalg.norm(v)
    if v_length == 0:
        return 0, v
    return v_length, v / v_length

cdef inline void cartesian_to_spherical(double x, double y, double z,
                                        double *r, double *theta, double *phi) noexcept nogil:
    """
    Transform cartesian coordinates to spherical coordinates

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    :param x: double
        cartesian x coordinate
    :param y: double
        cartesian y coordinate
    :param z: double
        cartesian z coordinate
    :param r: double* (out)
        radius
    :param theta: double* (out)
        first rotation angle (down from z-axis) in radians, in [0, pi]
    :param phi: double* (out)
        second rotation angle (within x-y-plane) in radians, in (-pi, pi]
    """
    r[0] = sqrt(x * x + y * y + z * z)
    if r[0] == 0.0:
        return

    phi[0] = atan2(y, x)
    theta[0] = acos(clip(z / r[0]))  # avoid domain errors

def spherical_to_matrix(theta: float, phi: float):
    """
    Transforms spherical coordinates to a rotation matrix.

    Spherical coordinates theta and phi are used as the geographical
    coordinates in Fisher et al. (1987).

    The returned matrix M rotates the z-axis onto the direction (theta, phi),
    i.e. its third column is the corresponding unit direction vector.

    :param theta: float
        first rotation angle (down from z-axis) in radians
    :param phi: float
        second rotation angle (within x-y-plane) in radians
    :return: np.ndarray
        3x3 orthogonal rotation matrix
    """
    return np.array(
        [
            [np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta) * np.cos(phi)],
            [np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

cdef inline void rot(double mu_x, double mu_y, double mu_z,
                     double n_x, double n_y, double n_z,
                     double cos_alpha, double sin_alpha,
                     double * cross_x, double * cross_y, double * cross_z) noexcept:
    """
    Rotation of a vector mu around normal n and angle alpha using # Rodrigues' formula

    :param mu_x: double
        component x of vector to be rotated
    :param mu_y: double
        component y of vector to be rotated
    :param mu_z: double
        component z of vector to be rotated
    :param n_x: double
        component x of vector to be rotated around
    :param n_y: double
        component y of vector to be rotated around
    :param n_z: double
        component z of vector to be rotated around
    :param cos_alpha: double
        cosine of rotation angle
    :param sin_alpha: double
        sinus of rotation angle
    :param cross_x: double* (out)
        component x of rotated vector
    :param cross_y: double* (out)
        component y of rotated vector
    :param cross_z: double* (out)
        component z of rotated vector
    """
    cdef double dot = n_x * mu_x + n_y * mu_y + n_z * mu_z
    cross_x[0] = n_y * mu_z - n_z * mu_y
    cross_y[0] = n_z * mu_x - n_x * mu_z
    cross_z[0] = n_x * mu_y - n_y * mu_x

    cross_x[0] *= sin_alpha
    cross_x[0] += mu_x * cos_alpha + n_x * dot * (1.0 - cos_alpha)
    cross_y[0] *= sin_alpha
    cross_y[0] += mu_y * cos_alpha + n_y * dot * (1.0 - cos_alpha)
    cross_z[0] *= sin_alpha
    cross_z[0] += mu_z * cos_alpha + n_z * dot * (1.0 - cos_alpha)

def is_in_image(
        pos, image_size, buffer: int = 100
) -> bool:
    """
    Checks whether a coordinate lies within the extended observation window.

    The valid range per axis i is [-buffer, image_size[i] + buffer] (inclusive).
    A buffer of 0 restricts to the image itself; the default of 100 allows
    coordinates up to 100 voxels outside the image on each side.

    :param pos: np.ndarray
        Position to be checked
    :param image_size: tuple[int, int, int]
        Image size in voxels
    :param buffer: int
        Extension of the valid region beyond each image boundary (default 100)
    :return: bool
        True if pos lies within [-buffer, image_size[i] + buffer] on all axes
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

def set_value(input_value, rng):
    """
    Returns a concrete value from either a scalar constant or a random variable.

    :param input_value: Real or scipy.stats distribution
        A numeric constant (int or float), or any object with an ``rvs(random_state)``
        method such as a scipy.stats distribution
    :param rng: numpy.random.Generator
        Random number generator passed to ``rvs`` when input_value is a distribution;
        ignored for scalar inputs
    :return: Real
        The scalar value, either the constant itself or one draw from the distribution
    :raises ValueError:
        If input_value is neither a Real scalar nor an object with a callable ``rvs`` method
    """
    if isinstance(input_value, Real):
        return input_value
    rvs = getattr(input_value, "rvs", None)
    if callable(rvs):
        return rvs(random_state=rng)
    raise ValueError(
        "Input must be a float/int or a distribution object with an 'rvs' method."
    )

def cut_border(fs: list[Fiber], image_size, boundary_size: int) -> list[Fiber]:
    """
    Cuts fibers when they cross image/observation window borders

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
        fiber.balls = fiber.balls[j_start + 1: j_end - 1]
    return fs_cut

cpdef cnp.ndarray[cnp.uint8_t, ndim=3] discretize_spheres_periodic(
        object fiber_system,
        int i_x, int i_y, int i_z
):
    """
    Rasterizes a fiber system to a uint8 image with periodic boundary conditions.

    Each fiber is represented as a chain of spherical balls. Voxels whose centre
    lies within the radius of any ball are set to 1. Balls that extend beyond the
    image boundary wrap around periodically.

    :param fiber_system: iterable of Fiber
        The fiber system to rasterize; each fiber must expose a ``balls`` attribute
        whose elements have ``coordinate`` (3-element array, in voxels) and ``radius``
        (float, in voxels) attributes
    :param i_x: int
        Image size along the x axis in voxels
    :param i_y: int
        Image size along the y axis in voxels
    :param i_z: int
        Image size along the z axis in voxels
    :return: np.ndarray of shape (i_x, i_y, i_z), dtype uint8
        Binary image; 1 where at least one ball covers the voxel, 0 elsewhere
    """
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] image = \
        np.zeros((i_x, i_y, i_z), dtype=np.uint8)
    cdef object fiber
    cdef Ball ball
    cdef double r, r_square, c_x, c_y, c_z, d_i, delta_i, d_ij, delta_ij
    cdef int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max

    for fiber in fiber_system:
        for ball in fiber.balls:
            r = ball.radius
            if r < DBL_EPSILON:
                continue
            r_square = r * r
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
                i_corr = ((i % i_x) + i_x) % i_x
                d_i = i - c_x
                delta_i = d_i * d_i
                for j in range(j_min, j_max):
                    j_corr = ((j % i_y) + i_y) % i_y
                    d_j = j - c_y
                    delta_ij = delta_i + d_j * d_j
                    if delta_ij > r_square:
                        continue
                    for k in range(k_min, k_max):
                        d_z = k - c_z
                        if delta_ij + d_z * d_z <= r_square:
                            k_corr = ((k % i_z) + i_z) % i_z
                            image[i_corr, j_corr, k_corr] = 1
    return image

cpdef cnp.ndarray[cnp.uint8_t, ndim=3] discretize_spheres_nonperiodic(
        object fiber_system, int i_x, int i_y, int i_z
):
    """
    Rasterizes a fiber system to a uint8 image without periodic boundary conditions.

    Each fiber is represented as a chain of spherical balls. Voxels whose centre
    lies within the radius of any ball are set to 1. Voxels outside the image
    bounds are clipped and not written.

    :param fiber_system: iterable of Fiber
        The fiber system to rasterize; each fiber must expose a ``balls`` attribute
        whose elements have ``coordinate`` (3-element array, in voxels) and ``radius``
        (float, in voxels) attributes
    :param i_x: int
        Image size along the x axis in voxels
    :param i_y: int
        Image size along the y axis in voxels
    :param i_z: int
        Image size along the z axis in voxels
    :return: np.ndarray of shape (i_x, i_y, i_z), dtype uint8
        Binary image; 1 where at least one ball covers the voxel, 0 elsewhere
    """
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] image = \
        np.zeros((i_x, i_y, i_z), dtype=np.uint8)
    cdef object fiber
    cdef Ball ball
    cdef double r, r_square, c_x, c_y, c_z, d_i, delta_i, d_ij, delta_ij, d_z
    cdef int i, j, k, i_min, i_max, j_min, j_max, k_min, k_max

    for fiber in fiber_system:
        for ball in fiber.balls:
            r = ball.radius
            if r < DBL_EPSILON:
                continue
            r_square = r * r
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
                if i < 0 or i >= i_x:
                    continue
                d_i = i - c_x
                delta_i = d_i * d_i
                for j in range(j_min, j_max):
                    if j < 0 or j >= i_y:
                        continue
                    d_j = j - c_y
                    delta_ij = delta_i + d_j * d_j
                    if delta_ij > r_square:
                        continue
                    for k in range(k_min, k_max):
                        if k < 0 or k >= i_z:
                            continue
                        d_z = k - c_z
                        if delta_ij + d_z * d_z <= r_square:
                            image[i, j, k] = 1
    return image

def discretize_lines(line_system, image_shape):
    """
    Discretize lines to an image

    :param line_system:
        lines to be saved (in voxels), each represented as a start point, an end point, and a radius
    :param image_shape: np.ndarray
        The image size in voxels
    :return: np.ndarray
        The image containing lines
    """
    image = np.zeros(image_shape, "uint16")
    radii = [radius for _, _, radius in line_system]
    if len(radii) == 0:
        return image
    pad = int(np.ceil(max(radii)) + 2)
    boundary_size_vec = np.array([pad, pad, pad])
    ext_image_size = image_shape + 2 * boundary_size_vec
    tmp_image = np.empty(ext_image_size, dtype=np.uint16)

    for start, end, radius in line_system:
        coords = line_nd(start + boundary_size_vec, end + boundary_size_vec, endpoint=False)
        tmp_image.fill(0)
        tmp_image[coords] = 1
        tmp_image = isotropic_dilation(tmp_image, radius)
        np.maximum(image, tmp_image[pad:-pad, pad:-pad, pad:-pad], out=image)

    return image

cdef inline double cdistance_ball(Ball a, Ball b) noexcept nogil:
    """
    Computes the Euclidean distance between two Ball objects.

    Accesses ball.coordinate directly; safe to call without the GIL.

    :param a: Ball
    :param b: Ball
    :return: double
        Euclidean distance between a.coordinate and b.coordinate
    """
    cdef double dx = b.coordinate[0] - a.coordinate[0]
    cdef double dy = b.coordinate[1] - a.coordinate[1]
    cdef double dz = b.coordinate[2] - a.coordinate[2]

    return sqrt(dx * dx + dy * dy + dz * dz)

@cython.cdivision(True)
cdef inline double cdistance3(double ax, double ay, double az,
                              double bx, double by, double bz) noexcept nogil:
    """
    Computes the Euclidean distance between two 3D points given as scalar components.
    
    Integer division warnings are suppressed via cdivision(True). Safe to call
    without the GIL.
    
    :param ax: double  x component of point a
    :param ay: double  y component of point a
    :param az: double  z component of point a
    :param bx: double  x component of point b
    :param by: double  y component of point b
    :param bz: double  z component of point b
    :return: double
        Euclidean distance between a and b
    """
    cdef double dx = ax - bx
    cdef double dy = ay - by
    cdef double dz = az - bz

    return sqrt(dx * dx + dy * dy + dz * dz)

cdef inline double cnormalized(double * ax, double * ay, double * az) noexcept nogil:
    """
    Normalizes a 3D vector in-place and returns its original norm.

    Components are modified through the input pointers. If the norm is below 
    DBL_EPSILON the pointers are left unchanged and 0.0 is returned — callers
    should check the return value before using the direction.

    :param ax: double*  x component, overwritten with normalized value
    :param ay: double*  y component, overwritten with normalized value
    :param az: double*  z component, overwritten with normalized value
    :return: double
        Original Euclidean norm of the vector, or 0.0 if the vector was zero-length
    """
    cdef double norm = sqrt(ax[0] * ax[0] + ay[0] * ay[0] + az[0] * az[0])
    if norm < DBL_EPSILON:
        return 0
    ax[0] = ax[0] / norm
    ay[0] = ay[0] / norm
    az[0] = az[0] / norm

    return norm

cdef inline double cdirection(Ball a, Ball b,
                              double * dir_x, double * dir_y, double * dir_z) noexcept:
    """
    Computes the unit direction vector and distance from Ball a to Ball b.

    Writes the unit direction into the output pointers. If a and b are coincident
    (distance <= 0), all direction components are set to 0 and 0.0 is returned.

    :param a: Ball
        Source ball
    :param b: Ball
        Target ball
    :param dir_x: double*
        Output: x component of the unit direction vector from a to b
    :param dir_y: double*
        Output: y component of the unit direction vector from a to b
    :param dir_z: double*
        Output: z component of the unit direction vector from a to b
    :return: double
        Euclidean distance between a and b, or 0.0 if they are coincident
    """
    cdef double dx = b.coordinate[0] - a.coordinate[0]
    cdef double dy = b.coordinate[1] - a.coordinate[1]
    cdef double dz = b.coordinate[2] - a.coordinate[2]
    cdef double dist = sqrt(dx * dx + dy * dy + dz * dz)
    if dist <= 0:
        dir_x[0] = 0
        dir_y[0] = 0
        dir_z[0] = 0
        return 0
    dir_x[0] = dx / dist
    dir_y[0] = dy / dist
    dir_z[0] = dz / dist
    return dist

cdef inline double clip(double val) noexcept nogil:
    """
    Clamps a value to the range [-1, 1].

    Used to guard the domain of acos and avoid floating-point values
    marginally outside [-1, 1] that would produce NaN. Safe to call without the GIL.

    :param val: double
        Value to clamp
    :return: double
        val clamped to [-1, 1]
    """
    if val > 1:
        val = 1
    if val < -1:
        val = -1
    return val
