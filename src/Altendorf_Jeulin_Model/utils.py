import numpy as np


def periodic_distance(
    coord1mod: np.ndarray, coord2: np.ndarray, image_size
):  # tuple[int, int, int]):
    """
    Calculates the periodic distance between two coordinates and the normalized direction vector between them
    TODO: rename variables and fix documentation
    Attributes
    ---------------------
    :param coord1: np.ndarray
        The first coordinate, start of the direction vector
    :param coord2: np.ndarray
        The second coordinate, "end" of the direction vector
    :param image_size: tuple[int, int, int]
        The size of the image, thus, the periodicity
    :return: distance between coordinates, direction vector
    """
    coord2mod = coord2 % image_size

    for i in range(3):
        disp = coord2mod[i] - coord1mod[i]
        if abs(disp) > image_size[i] / 2.0:
            if disp > 0:
                coord2mod[i] -= image_size[i]
            else:
                coord2mod[i] += image_size[i]
        coord2mod[i] -= coord1mod[i]

    return np.linalg.norm(coord2mod), coord2mod  # dist, dir


def angle_between(v1: np.ndarray, v2: np.ndarray):
    """
    Calculates the angle between two vectors v1 and v2

    Attributes
    ---------------------
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

    Attributes
    ---------------------
    :param v: np.ndarray
        vector to be normalized
    :return: float, np.ndarray
        original length, normalized vector
    """
    v_length = np.linalg.norm(v)
    if v_length == 0:
        return 0, v
    return v_length, v / v_length


def cartesian_to_spherical(x, y, z):
    """
    transform cartesian coordinates to spherical coordinates

    Attributes
    ---------------------
    :param x: float
        cartesian x coordinate
    :param y: float
        cartesian y coordinate
    :param z: float
        cartesian z coordinate
    :return: float, float, float
        radius, theta angle, phi angle in radian
    """
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z / r, -1, 1))  # avoid domain errors
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """
    transform spherical coordinates (in radian) to cartesian coordinates

    Attributes
    ---------------------
    :param r: float
        radius
    :param theta: float
        polar theta angle
    :param phi: float
        polar phi angle
    :return: float, float, float
        cartesian coordinates
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def discretize_spheres(
    coordinates: np.ndarray,
    radii: np.ndarray,
    min_coordinates: np.ndarray,
    max_coordinates: np.ndarray,
):
    """
    discretizes spheres to an image
    :param coordinates: np.ndarray
        sphere coordinates
    :param radii: np.ndarray
        sphere radii
    :param min_coordinates: np.ndarray
        smallest cooridnate of spheres
    :param max_coordinates: np.ndarray
        largest cooridnate of spheres
    :return: np.ndarray
        image containing spheres
    """
    L = max_coordinates - min_coordinates
    coordinates = coordinates - min_coordinates
    image = np.zeros(L, "uint16")

    for iota in range(len(coordinates)):
        r_square = radii[iota] ** 2
        for i in range(
            int(coordinates[iota, 0] - radii[iota]) - 1,
            int(coordinates[iota, 0] + radii[iota]) + 1,
        ):
            i_corr = i
            if i < 0:
                i_corr = L[0] + i
            if i >= L[0]:
                i_corr = i - L[0]
            delta_i = (i - coordinates[iota, 0]) ** 2
            for j in range(
                int(coordinates[iota, 1] - radii[iota]) - 1,
                int(coordinates[iota, 1] + radii[iota]) + 1,
            ):
                j_corr = j
                if j < 0:
                    j_corr = L[1] + j
                if j >= L[1]:
                    j_corr = j - L[1]
                delta_ij = delta_i + (j - coordinates[iota, 1]) ** 2
                for k in range(
                    int(coordinates[iota, 2] - radii[iota]) - 1,
                    int(coordinates[iota, 2] + radii[iota]) + 1,
                ):
                    k_corr = k
                    if k < 0:
                        k_corr = L[2] + k
                    if k >= L[2]:
                        k_corr = k - L[2]
                    delta_ijk = delta_ij + (k - coordinates[iota, 2]) ** 2
                    if delta_ijk <= r_square:
                        image[i_corr, j_corr, k_corr] = 1
    return image
