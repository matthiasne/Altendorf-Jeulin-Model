import csv
import cython
from pathlib import Path

import Altendorf_Jeulin_Model.FiberModel as FiberModel
import numpy as np
import tifffile
import SimpleITK as sitk
import json
import platform
from datetime import datetime
from Altendorf_Jeulin_Model.utils.utils cimport discretize_spheres_periodic, discretize_spheres_nonperiodic, clip
from Altendorf_Jeulin_Model.utils.utils import (
    normalized,
    discretize_lines
)

import Altendorf_Jeulin_Model.utils.SpatialHashing as sh
from Altendorf_Jeulin_Model.Fiber import Fiber
from Altendorf_Jeulin_Model.utils.Statistics import mean_angle_error, volume_fraction, calculate_kl


def print_fiber_positions(
    fiber_system: list[Fiber], max_fibers: int = 10, max_balls: int = 10
):
    """
    Print fiber positions to stdout.

    :param fiber_system: list[Fiber]
        A list of fibers, each represented as a list of balls
    :param max_fibers: int, optional
        Maximum number of fibers to print (default: 10)
    :param max_balls: int, optional
        Maximum number of balls per fiber to print (default: 10)
    """
    for i, fiber in enumerate(fiber_system[:max_fibers]):
        coords = " ".join(
            f"[{ball.coordinate[0]:.2f},"
            f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
            for ball in fiber.balls[:max_balls]
        )
        print("Fiber ", i, ":", coords)


def print_fiber_positions_to_file(fiber_system: list[Fiber], output_file: str):
    """
    Write fiber positions to a text file, one fiber per line.

    :param fiber_system: list[Fiber]
        A list of fibers, each represented as a list of balls
    :param output_file: str
        Path for the output file
    """
    with open(output_file, mode="w", newline="") as file:
        for i, fiber in enumerate(fiber_system):
            coords = " ".join(
                f"[{ball.coordinate[0]:.2f},"
                f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
                for ball in fiber.balls
            )
            print("Fiber ", i, ":", coords, file=file)


def save_fibers_as_image(
    fiber_system: list[Fiber],
    domain,
    path: str = "spheres.tif",
    scale: float = 1,
    is_periodic: bool = True,
    spacing=(1.0, 1.0, 1.0)
):
    """
    Save a fiber system as a 3D image file (TIF or MHA/MHD).

    Fibers are discretized as chains of spheres. The image is saved with axes
    transposed to (z, y, x) order as expected by tifffile and SimpleITK.

    :param fiber_system: list[Fiber]
        A list of fibers, each represented as a list of balls
    :param domain: tuple[int, int, int]
        Size of the domain in physical units (x, y, z)
    :param path: str, optional
        Output file path; must end with .tif, .tiff, .mha, or .mhd (default: "spheres.tif")
    :param scale: float, optional
        Voxel size in physical units, e.g. 4 if coordinates are in µm and voxels are 4 µm
        (default: 1)
    :param is_periodic: bool, optional
        Whether to apply periodic boundary conditions when discretizing (default: True)
    :param spacing: tuple[float, float, float], optional
        Physical voxel spacing written into MHA/MHD metadata (default: (1.0, 1.0, 1.0))
    :raises ValueError:
        If the file extension is not .tif, .tiff, .mha, or .mhd
    """
    image_shape = np.array([d / scale for d in domain], dtype=int)

    if is_periodic:
        image = discretize_spheres_periodic(fiber_system, image_shape[0], image_shape[1], image_shape[2])
    else:
        image = discretize_spheres_nonperiodic(fiber_system, image_shape[0], image_shape[1], image_shape[2])

    image = np.transpose(image, (2, 1, 0))
    if path.endswith(".tif") or path.endswith(".tiff"):
        tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": "XYZ"})
    elif path.endswith(".mha") or path.endswith(".mhd"):
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(spacing[::-1])
        sitk.WriteImage(sitk_image, path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def save_lines_as_image(
    line_system,
    domain: tuple[int, int, int],
    path: str = "lines.tif",
    scale: float = 1.0,
    spacing=(1.0, 1.0, 1.0)
):
    """
    Save a line system as a 3D image file (TIF or MHA/MHD).

    Lines are discretized using isotropic dilation. The image is saved with axes
    transposed to (z, y, x) order as expected by tifffile and SimpleITK.

    :param line_system: list[tuple[np.ndarray, np.ndarray, float]]
        A list of lines, each represented as (start, end, radius) in physical units
    :param domain: tuple[int, int, int]
        Size of the domain in physical units (x, y, z)
    :param path: str, optional
        Output file path; must end with .tif, .tiff, .mha, or .mhd (default: "lines.tif")
    :param scale: float, optional
        Voxel size in physical units (default: 1.0)
    :param spacing: tuple[float, float, float], optional
        Physical voxel spacing written into MHA/MHD metadata (default: (1.0, 1.0, 1.0))
    :raises ValueError:
        If the file extension is not .tif, .tiff, .mha, or .mhd
    :raises ValueError:
        If line_system is empty (max() on empty sequence); guard against this before calling
    """
    line_system = [
        (
            np.floor(line[0] / scale) - np.array([1, 1, 1]),
            np.floor(line[1] / scale) - np.array([1, 1, 1]),
            line[2] / scale,
        )
        for line in line_system
    ]
    image_shape = np.array([d / scale for d in domain], dtype=int)
    image = discretize_lines(line_system, image_shape)
    image = np.transpose(image, (2, 1, 0))
    if path.endswith(".tif") or path.endswith(".tiff"):
        tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": "XYZ"})
    elif path.endswith(".mha") or path.endswith(".mhd"):
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(spacing[::-1])
        sitk.WriteImage(sitk_image, path)
    else:
        raise ValueError(f"Unsupported file format: {path}")



def print_grid(grid: sh):
    """
    Print the contents of each cell in a SpatialHashing grid to stdout.

    :param grid: SpatialHashing
        The spatial hashing grid whose cells will be printed
    """
    for i, cell in enumerate(grid.cells):
        coords = " ".join(
            f"[{ball.coordinate[0]:.2f},"
            f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
            for ball in cell
        )
        print("Cell ", i, ":", coords)


def print_stats(output_file: str, rows):
    """
    Write packing statistics to a CSV file.

    :param output_file: str
        Path to the output file
    :param rows: list
        Rows to be written; each row should contain values for the columns:
        "Step", "#Fibers", "VF", "MAE", "meanLength", "KL",
        "MaxOverlap", "ContactDistances", "change"
    """
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Step",
                "#Fibers",
                "VF",
                "MAE",
                "meanLength",
                "KL",
                "MaxOverlap",
                "ContactDistances",
                "change"
            ]
        )  # Header
        writer.writerows(rows)
        

def print_stats_row(
    fs: list[Fiber], i, overlap: float, neighbor_dist: float, contact_distance:float,
        objective:float, change:float, image_size, dir_matrix, is_periodic: bool
):
    """
    Compute and print packing statistics for the current iteration step.

    Prints a summary line to stdout and returns the statistics as a list
    suitable for passing to print_stats.

    :param fs: list[Fiber]
        Current fiber system
    :param i: int
        Current iteration step of the force-biased algorithm
    :param overlap: float
        Maximum overlap between balls at this step
    :param neighbor_dist: float
        Maximum distance between neighbouring balls at this step
    :param contact_distance: float
        Maximum contact distance at this step
    :param objective: float
        Current value of the packing objective function
    :param change: float
        Relative change of the objective function since the previous step
    :param image_size: tuple[int, int, int]
        Domain size, used to compute the volume fraction
    :param dir_matrix: np.ndarray
        3x3 direction matrix used to compute the KL divergence
    :param is_periodic: bool
        Whether the fiber system uses periodic boundary conditions
    :return: list
        Row containing: Step, #Fibers, VF, MAE, meanLength, KL,
        MaxOverlap, ContactDistances, change
    """
    mae = mean_angle_error(fs)
    mean_length = np.mean([fiber.get_length() for fiber in fs])
    vf = volume_fraction(fs, image_size, is_periodic=is_periodic)
    kl = calculate_kl(fs, dir_matrix)
    print(
        f"step {i} objective {objective:.3f} change {change:.5f} max overlap {overlap:.3f} "
        f"MAE {mae*180/np.pi:.3f}° max neighbor dist {neighbor_dist:.3f} "
        f"KL divergence {kl:.3f} "
        f"contact distance {contact_distance:.3f} "
    )
    return [
        i,
        len(fs),
        vf,
        mae,
        mean_length,
        kl,
        overlap,
        contact_distance,
        change
    ]


def print_contact_stats(output_file: str, rows):
    """
    Write contact statistics to a CSV file.

    :param output_file: str
        Path to the output file
    :param rows: list
        Rows to be written; each row should contain values for the columns:
        "#connected_components", "#clots", "#contacts", "contact_surface"
    """
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "#connected_components", "#clots", "#contacts", "contact_surface"
            ]
        )  # Header
        writer.writerows(rows)



def save_fibers_as_graph(file_path: str, fs: sh):
    """
    Write a fiber system to the TexMathVisualizer graph format.

    Produces six files at the given base path:
    - .fft.nod            — node coordinates (one per line: x y z)
    - .fft.elm            — edges (one per line: node_a node_b radius)
    - .fft.threads        — thread (fiber) descriptions
    - .fft.ndn            — total node count
    - .fft.eln            — total edge count
    - .fft.threads.number — total thread (fiber) count

    Fibers with fewer than 3 balls are skipped.
    Each ball becomes a node and each consecutive ball pair becomes an edge.

    :param file_path: str
        Base path for the output files (without extension)
    :param fs: list[Fiber]
        Fiber system to be written
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
    Write a simplified fiber system to the TexMathVisualizer FFT graph format.

    Produces the same six files as save_fibers_as_graph, but simplifies each
    fiber by merging consecutive balls whose cumulative curvature stays below
    0.2 radians. This reduces the number of nodes and edges for nearly straight
    fibers. Fibers with fewer than 3 balls are skipped.

    :param file_path: str
        Base path for the output files (without extension)
    :param fs: list[Fiber]
        Fiber system to be written
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
    Find the index of the next node to retain when simplifying a fiber.

    Starting from i_start, advances along the fiber accumulating curvature
    (sum of turning angles between consecutive ball-to-ball directions) until
    the cumulative curvature reaches or exceeds 0.2 radians or the end of the
    fiber is reached. Returns the index of the last ball visited, which is
    the next node to keep in the simplified graph.

    If only one step ahead is available, returns i_start + 1 unconditionally.

    :param fiber: Fiber
        The fiber to traverse
    :param i_start: int
        Index of the current node to start from
    :param i_end: int
        Index of the last ball in the fiber
    :return: int
        Index of the next node to retain, always > i_start and <= i_end
    """
    curvature = 0
    i = i_start
    while i + 2 <= i_end and curvature < 0.2:

        _, dir_prev = normalized(
            np.array(fiber.balls[i + 1].coordinate) - np.array(fiber.balls[i].coordinate)
        )
        _, dir_next = normalized(
            np.array(fiber.balls[i + 2].coordinate) - np.array(fiber.balls[i + 1].coordinate)
        )
        angle = np.arccos(clip(np.dot(dir_prev, dir_next)))
        curvature += angle
        i += 1
    if i == i_start:
        return i + 1
    return i


def read_gad(path: str, scale: int = 1000000):
    """
    Read a fiber system from a GeoDict GAD file.

    :param path: str
        Path to the GAD file (JSON format)
    :param scale: int, optional
        Conversion factor from the file's unit (metres) to the working unit
        (micrometres by default); coordinates and radii are multiplied by this
        value (default: 1000000)
    :return: tuple[list[Fiber], tuple[int, int, int]]
        The fiber system and the domain size in working units (x, y, z)
    """
    fiber_system = []
    with open(path, "r") as f:
        gad = json.load(f)
        n_fibers = gad["NumberOfObjects"]
        window = gad["Domain"]
        shape = (
            int(np.ceil(window["LengthX"][0] * scale)),
            int(np.ceil(window["LengthY"][0] * scale)),
            int(np.ceil(window["LengthZ"][0] * scale)),
        )
        for i in range(1, n_fibers + 1):
            fiber = gad["Object" + str(i)]
            radius = fiber["Diameter"] / 2 * scale
            n_balls = fiber["NumberOfSegments"] + 1
            coords = []
            for j in range(1, n_balls + 1):
                coords.append(np.array(fiber["Point" + str(j)][0]) * scale)
            FiberModel.save_balls_in_fiber_system_endless(fiber_system, coords, i - 1, radius)
    return fiber_system, shape


def write_gad(
    fiber_system,
    path: str,
    shape,
    voxel_size: float | int = 1e-06,
    is_periodic: bool = True,
    scale: int = 1000000,
):
    """
    Write a fiber system to a GeoDict GAD file.

    Coordinates and radii are converted from working units (micrometres) to
    metres by dividing by scale. The voxel_size field in the output records
    the physical size of one voxel in metres and is independent of the
    coordinate scaling.

    :param fiber_system: list[Fiber]
        Fiber system to be written
    :param path: str
        Path to the output GAD file
    :param shape: tuple[int, int, int]
        Domain size in working units (x, y, z)
    :param voxel_size: float, optional
        Physical size of one voxel in metres, written into the GAD header
        (default: 1e-6)
    :param is_periodic: bool, optional
        Whether the domain is periodic in all three axes (default: True)
    :param scale: int, optional
        Conversion factor from working units to metres; coordinates and
        radii are divided by this value (default: 1000000)
    """
    system_dict = dict()

    header = dict()
    now = datetime.now()
    header["Release"] = now.strftime("%Y")
    header["Revision"] = 0
    header["BuildDate"] = now.strftime("%d %b %Y")
    header["CreationDate"] = now.strftime("%d %b %Y")
    header["CreationTime"] = now.strftime("%H:%M:%S")
    header["Creator"] = "Altendorf_Jeulin_Model"
    header["Platform"] = platform.system()

    system_dict["Header"] = header
    system_dict["NumberOfObjects"] = len(fiber_system)
    system_dict["Description"] = "Generated with Altendorf_Jeulin_Model"

    domain = dict()
    domain["PeriodicX"] = is_periodic
    domain["PeriodicY"] = is_periodic
    domain["PeriodicZ"] = is_periodic
    domain["DomainMode"] = "Length"
    domain["OriginX"] = [0, "m"]
    domain["OriginY"] = [0, "m"]
    domain["OriginZ"] = [0, "m"]
    domain["VoxelLength"] = [voxel_size, "m"]
    domain["LengthX"] = [shape[0]/scale, "m"]
    domain["LengthY"] = [shape[1]/scale, "m"]
    domain["LengthZ"] = [shape[2]/scale, "m"]
    domain["OverlapMode"] = "NewMaterial"
    domain["HollowMaterialID"] = 0

    post_processing = dict()
    post_processing["ResolveOverlap"] = True
    post_processing["MarkContactVoxels"] = False
    post_processing["ContactMaterialID"] = 3
    domain["PostProcessing"] = post_processing

    system_dict["Domain"] = domain

    for i, fiber in enumerate(fiber_system):
        fiber_dict = dict()
        fiber_dict["MaterialIDModel"] = "Constant"
        fiber_dict["MaterialID"] = 1
        fiber_dict["Type"] = "CurvedCircularFiber"
        fiber_dict["NumberOfSegments"] = (fiber.get_number_of_balls() - 1)
        fiber_dict["FiberEndType1"] = "Rounded"
        fiber_dict["FiberEndType2"] = "Rounded"
        fiber_dict["CuttingPlane1"] = [0.0, 0.0, 0.0]
        fiber_dict["CuttingPlane2"] = [0.0, 0.0, 0.0]
        fiber_dict["Diameter"] = fiber.get_mean_radius() * 2/scale
        for j, ball in enumerate(fiber.balls):
            coord = np.array([ball.coordinate[0], ball.coordinate[1], ball.coordinate[2]])
            fiber_dict["Point" + str(j + 1)] = [(coord/scale).tolist(), "m"]
        system_dict["Object" + str(i + 1)] = fiber_dict
    with open(path, "w") as f:
        json.dump(system_dict, f, indent=2)