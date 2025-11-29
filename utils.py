import numpy as np


def periodic_distance(coord1: np.ndarray, coord2: np.ndarray, image_size: tuple[int, int, int]):
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
    coord1mod = coord1%image_size
    coord2mod = coord2%image_size
    dist = np.linalg.norm(coord2mod - coord1mod)
    delta = coord2mod - coord1mod
    for i in range(3):
        if(abs(delta[i]) > image_size[i]/2.):
            if(delta[i] > 0):
                coord2mod[i] -= image_size[i]
            else:
                coord2mod[i] += image_size[i]

    if(np.all(coord1mod == coord2mod)):
        dir = np.array([1,1,1])
    else:
        dir = coord2mod - coord1mod
        dir /= np.linalg.norm(dir)

    if(np.linalg.norm(coord2mod - coord1mod) > dist):
        raise ValueError("There is an issue in the periodic distance calculation")
    else:
        dist = np.linalg.norm(coord2mod - coord1mod)
        return dist, dir

def angle_between(v1:np.ndarray, v2:np.ndarray):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.acos(np.clip(cos_angle, -1.0, 1.0))