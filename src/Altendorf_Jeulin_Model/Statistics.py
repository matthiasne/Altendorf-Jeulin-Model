import Altendorf_Jeulin_Model.Fiber as Fiber
import numpy as np

def mean_radius(fs:list[Fiber]):
    mean_radius = np.mean([fiber.get_mean_radius() for fiber in fs])
    return mean_radius

def mean_length(fs:list[Fiber]):
    mean_length = np.mean([fiber.get_length() for fiber in fs])
    return mean_length