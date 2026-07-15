import csv
from pathlib import Path

import numpy as np
import tifffile

import Altendorf_Jeulin_Model.FiberModel as FiberModel
import Altendorf_Jeulin_Model.SpatialHashing as sh
from Altendorf_Jeulin_Model.Fiber import Fiber
from Altendorf_Jeulin_Model.utils import (
    discretize_spheres_nonperiodic,
    discretize_spheres_periodic,
    normalized,
)


def print_fiber_positions(
    fiber_system: FiberModel, max_fibers: int = 10, max_balls: int = 10
):
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
        coords = " ".join(
            f"[{ball.coordinate[0]:.2f},"
            f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
            for ball in fiber.balls[:max_balls]
        )
        print("Fiber ", i, ":", coords)


def print_fiber_positions_to_file(fiber_system: FiberModel, output_file: str):
    """
    Print fibers as positions.

    Parameters
    ---------------------
    :param fiber_system: list[Fiber]
        A list of fibers, each represented as a list of balls
    :param output_file: str
        path for the output file
    """
    with open(output_file, mode="w", newline="") as file:
        for i, fiber in enumerate(fiber_system):
            coords = " ".join(
                f"[{ball.coordinate[0]:.2f},"
                f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
                for ball in fiber.balls
            )
            print("Fiber ", i, ":", coords, file=file)


def save_fibers_as_tif(
    fiber_system: list[Fiber],
    domain: tuple[int, int, int],
    boundary: tuple[int, int, int] = (0, 0, 0),
    path: str = "spheres.tif",
    scale: float = 1,
    is_periodic: bool = True,
):
    """
    Save fibers as tif-image

    Parameters
    ---------------------
    :param fiber_system: list[list[Ball]]
        A list of fibers, each represented as a list of 3D np.arrays
    :param domain: tuple(int, int, int)
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
    coords = np.array(coords) / scale
    radii = np.array(radii) / scale

    min_coordinates = np.array(boundary) / scale
    image_shape = np.array([d / scale for d in domain], dtype=int)

    if is_periodic:
        image = discretize_spheres_periodic(coords, radii, min_coordinates, image_shape)
    else:
        image = discretize_spheres_nonperiodic(
            coords, radii, min_coordinates, image_shape
        )
    image = np.transpose(image, (2, 1, 0))
    tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": "XYZ"})


def print_grid(grid: sh):
    """
    Print cells of a SpatialHashing

    Parameters
    ---------------------
    :param grid: SpatialHashing
    A list of fibers, each represented as a list of Balls
    """
    for i, cell in enumerate(grid.cells):
        coords = " ".join(
            f"[{ball.coordinate[0]:.2f},"
            f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
            for ball in cell
        )
        print("Cell ", i, ":", coords)


def print_stats(output_file: str, rows, has_beta: bool = True):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        if has_beta:
            writer.writerow(
                [
                    "Step",
                    "#Fibers",
                    "Beta",
                    "EstimatedBeta",
                    "MeanRadius",
                    "MeanLength",
                    "MeanAngleError",
                    "Kappa1Estimate",
                    "Kappa2Estimate",
                    "MaxOverlap",
                    "ForceStrength",
                ]
            )  # Header
        else:
            writer.writerow(
                [
                    "Step",
                    "#Fibers",
                    "MeanRadius",
                    "MeanLength",
                    "MeanAngleError",
                    "Kappa1Estimate",
                    "Kappa2Estimate",
                    "MaxOverlap",
                    "ForceStrength",
                ]
            )
        writer.writerows(rows)


def save_fibers_as_graph(file_path: str, fs: sh):
    """
    Prints the fiber system according to input file format of
    TexMathVisualizer. Each edge is the original AJ edge.

    Parameters
    ---------------------
    :param file_path: string
        base path for the output file
    :param fs: list[Fiber]
        fiber system to be printed
    """
    base = Path(file_path)
    nod_file = base.with_suffix(base.suffix + ".fft.nod")
    elm_file = base.with_suffix(base.suffix + ".fft.elm")
    thread_file = base.with_suffix(base.suffix + ".fft.threads")
    numnod_file = base.with_suffix(base.suffix + ".fft.ndn")
    numelm_file = base.with_suffix(base.suffix + ".fft.eln")
    numthread_file = base.with_suffix(base.suffix + ".fft.threads.number")

    node_label = 0
    edge_label = 0
    with (
        nod_file.open("w", newline="") as fnod,
        elm_file.open("w", newline="") as felm,
        thread_file.open("w", newline="") as fthreads,
    ):
        for fiber in fs:
            parts = ["acyclic", str(len(fiber.balls) - 1)]
            if len(fiber.balls) < 3:
                continue
            for i, ball in enumerate(fiber.balls):
                x, y, z = ball.coordinate
                print(x, " ", y, " ", z, file=fnod)
                parts.append(str(node_label))
                if i < len(fiber.balls) - 1:
                    print(node_label, " ", node_label + 1, " ", ball.radius, file=felm)
                    parts.append(str(edge_label))
                    edge_label += 1
                node_label += 1
            print(" ".join(parts), file=fthreads)

    with numnod_file.open("w", newline="") as fn:
        print(node_label, file=fn)

    with numelm_file.open("w", newline="") as fe:
        print(edge_label, file=fe)

    with numthread_file.open("w", newline="") as ft:
        print(len(fs), file=ft)


def save_fibers_as_small_graph(file_path, fs):
    """
    Prints the fiber system according to input file format of
    TexMathVisualizer but removes edges for a maximal curvature of
    0.2 radian

    Parameters
    ---------------------
    :param file_path: string
        base path for the output file
    :param fs: list[Fiber]
        fiber system to be printed
    """
    base = Path(file_path)
    nod_file = base.with_suffix(base.suffix + ".fft.nod")
    elm_file = base.with_suffix(base.suffix + ".fft.elm")
    thread_file = base.with_suffix(base.suffix + ".fft.threads")
    numnod_file = base.with_suffix(base.suffix + ".fft.ndn")
    numelm_file = base.with_suffix(base.suffix + ".fft.eln")
    numthread_file = base.with_suffix(base.suffix + ".fft.threads.number")

    node_label = 0
    edge_label = 0
    with (
        nod_file.open("w", newline="") as fnod,
        elm_file.open("w", newline="") as felm,
        thread_file.open("w", newline="") as fthreads,
    ):
        for fiber in fs:
            # TODO replace this part
            if len(fiber.balls) < 3:
                continue
            prev_node_label = node_label
            x, y, z = fiber.balls[0].coordinate
            print(x, " ", y, " ", z, file=fnod)
            print(
                node_label, " ", node_label + 1, " ", fiber.balls[0].radius, file=felm
            )
            parts = [str(node_label), str(edge_label)]
            node_label += 1
            edge_label += 1

            i_end = len(fiber.balls) - 1
            i = find_next_node(fiber, 0, i_end)
            while i < i_end - 1:
                x, y, z = fiber.balls[i].coordinate
                print(x, " ", y, " ", z, file=fnod)
                print(
                    node_label,
                    " ",
                    node_label + 1,
                    " ",
                    fiber.balls[i].radius,
                    file=felm,
                )
                parts.append(str(node_label))
                parts.append(str(edge_label))
                node_label += 1
                edge_label += 1
                i = find_next_node(fiber, i, i_end)
            x, y, z = fiber.balls[i].coordinate
            print(x, " ", y, " ", z, file=fnod)
            parts.append(str(node_label))
            node_label += 1
            parts.insert(0, str(node_label - prev_node_label - 1))
            parts.insert(0, "acyclic")
            print(" ".join(parts), file=fthreads)

    with numnod_file.open("w", newline="") as fn:
        print(node_label, file=fn)

    with numelm_file.open("w", newline="") as fe:
        print(edge_label, file=fe)

    with numthread_file.open("w", newline="") as ft:
        print(len(fs), file=ft)


def find_next_node(fiber, i_start: int, i_end: int) -> int:
    """
    Helper function for save_fibers_as_small_graph: finds next node to remove
    superfluous edges indirectly

    Parameters
    ----------
    :param fiber:   list[Ball]
    :param i_start: int
        index to start searching from
    :param i_end: int
        last index in fiber
    :return: int
        index of next node

    """
    curvature = 0
    i = i_start
    while i + 2 <= i_end and curvature < 0.2:
        _, dir_prev = normalized(
            fiber.balls[i + 1].coordinate - fiber.balls[i].coordinate
        )
        _, dir_next = normalized(
            fiber.balls[i + 2].coordinate - fiber.balls[i + 1].coordinate
        )
        angle = np.arccos(np.dot(dir_prev, dir_next))
        curvature += angle
        i += 1
    if i == i_start:
        return i + 1
    return i
