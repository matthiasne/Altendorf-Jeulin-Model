import csv

import numpy as np
import tifffile

import Altendorf_Jeulin_Model.FiberModel as FiberModel
import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.Fiber import Fiber
from Altendorf_Jeulin_Model.utils import (
    discretize_spheres_nonperiodic,
    discretize_spheres_periodic,
)


def print_fiber_positions(fiber_system: FiberModel,
                          max_fibers: int = 10, max_balls: int = 10):
    """
    Print fibers as positions.

    Parameters
    ---------------------
    :param fiber_system: list[Fiber]
        A list of fibers, each represented as a list of balls
    :param max_fibers: int, optional
        maximal number of fibers to be printed
    :param max_balls:  int, optional
        maximal number of spheres to be printed
    """
    for i, fiber in enumerate(fiber_system[:max_fibers]):
        coords = ' '.join(f"[{ball.coordinate[0]:.2f},"
                          f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
                          for ball in fiber.balls[:max_balls])
        print("Fiber ", i, ":", coords)

def save_fibers_as_tif(fiber_system: list[Fiber],
                       shape: tuple[int, int, int],
                       path: str = "spheres.tif", scale:float = 1, is_periodic:bool = True):
    """
    Save fibers as tif-image

    Parameters
    ---------------------
    :param fiber_system: list[list[Ball]]
        A list of fibers, each represented as a list of 3D np.arrays
    :param shape: tuple(int, int, int)
        z,y,x coordinate of the image
    :param path: string optional
        The path where the tif image will be saved
        (default: "spheres.tif")
    :param scale: float optional
        The scale of the image, e.g., when the data is given in um and the image has voxel size 4um, the scale is 4
        (default: 1)
    :param is_periodic: bool optional
        Whether the system is periodic or not (default: True)
    """
    coords = []
    radii = []
    for fiber in fiber_system:
        for ball in fiber.balls:
            coords.append(ball.coordinate)
            radii.append(ball.radius)
    coords = np.array(coords)/scale
    radii = np.array(radii)/scale

    min_coordinates = np.array([0, 0, 0])
    max_coordinates = np.array(shape)

    if is_periodic:
        image = discretize_spheres_periodic(coords, radii, min_coordinates, max_coordinates)
    else:
        image = discretize_spheres_nonperiodic(coords, radii, min_coordinates, max_coordinates)
    tifffile.imwrite(path, image, photometric='minisblack')


def print_grid(grid: sh):
    """
    Print cells of a SpatialHashing

    Parameters
    ---------------------
    :param grid: SpatialHashing
    A list of fibers, each represented as a list of Balls
    """
    for i, cell in enumerate(grid.cells):
        coords = ' '.join(f"[{ball.coordinate[0]:.2f},"
                          f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
                          for ball in cell)
        print("Cell ", i, ":", coords)


def print_stats(output_file: str, rows, has_beta:bool = True):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        if has_beta:
            writer.writerow(['Step', '#Fibers', 'Beta', 'EstimatedBeta', 'MeanRadius', 'MeanLength', 'MeanAngleError',
                             "Kappa1Estimate", "Kappa2Estimate", 'MaxOverlap', 'ForceStrength'])  # Header
        else:
            writer.writerow(['Step', '#Fibers', 'MeanRadius', 'MeanLength', 'MeanAngleError',
                             "Kappa1Estimate", "Kappa2Estimate", 'MaxOverlap', 'ForceStrength'])
        writer.writerows(rows)
