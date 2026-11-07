# cython: language_level=3, infer_type=True
import numpy as np
from typing import Callable
from Altendorf_Jeulin_Model.CalculateForces import (
    apply_forces,
    calculate_forces,
    calculate_forces_endstep
)

import Altendorf_Jeulin_Model.Fiber as Fiber
import Altendorf_Jeulin_Model.utils.SpatialHashing as sh

from Altendorf_Jeulin_Model.ContactModel import test_in_contact
from Altendorf_Jeulin_Model.utils.Statistics import  calculate_kl
from Altendorf_Jeulin_Model.utils.io_utils import print_stats, print_stats_row, write_gad


def run_force_biased(
    fs: list[Fiber],
    image_size,
    use_end_step_radius: bool = False,
    use_end_step_repulsion: bool = False,
    output_path: str = "examples/outputs/",
    verbose: bool = True,
    is_periodic: bool = True,
    shortlist = [],
    hardcore_ratio: float = 1.0,
    contact_distance: float = 1.0,
    method: str = "AJ++",
    boundary_size: int = 0,
    output_step_size: int = 100,
    max_steps: int = 1000,
    tol: float = 1e-3,
    dir_matrix = np.diag([1,1,1]),
    tau: float = 0.25,
    rho: float = 0.25,
    alpha_s: float = 1*np.pi/180, alpha_e: float = 2*np.pi/180,
    x_s: float = 0.25, x_e: float = 0.5
):
    """
    Dispatcher for all force-biased packing variants.

    Selects the appropriate packing algorithm based on *method* and forwards
    all remaining parameters.  For non-periodic systems, *boundary_size* voxels
    are added to each side of *image_size* before the algorithm runs.

    :param fs: list[Fiber]
        Fiber system to be packed.
    :param image_size: array-like of int, shape (3,)
        Size of the image domain in each spatial dimension (micrometers).
    :param use_end_step_radius: bool (default: False)
        If ``True`` and *method* is ``"AJ"``, reduces ball radii at the end
        of the run to eliminate residual overlaps.
    :param use_end_step_repulsion: bool (default: False)
        If ``True`` and *method* is ``"AJ"``, applies a pure repulsion step
        at the end of the run to eliminate residual overlaps.
    :param output_path: str (default: ``"examples/outputs/"``)
        Directory path for intermediate ``.gad`` snapshots of the current
        fiber system and the final ``results.csv`` statistics file.
    :param verbose: bool (default: True)
        If ``True``, prints per-step statistics and writes intermediate
        ``.gad`` snapshots of the current fiber system
    :param is_periodic: bool (default: True)
        Whether to apply periodic boundary conditions.
    :param shortlist: list (default: [])
        Pre-computed contact shortlist (ball-pair edges).  Only used by
        the ``"Contact++"`` method.
    :param hardcore_ratio: float (default: 1.0)
        Fraction of each ball's radius that may not be penetrated.
    :param contact_distance: float (default: 1.0)
        Maximum surface-to-surface distance for two balls to be considered
        in contact.  Only used by the ``"Contact++"`` method.
    :param method: str (default: ``"AJ++"``)
        Packing algorithm to use.  One of:

        * ``"AJ"``        – original Altendorf–Jeulin algorithm.
        * ``"AJ++"``      – refined AJ with relative-change stop criterion.
        * ``"AJ++fast"``  – AJ++ stopping as soon as overlap reaches zero.
        * ``"Contact++"`` – contact-extension variant with shortlist forces.
        * any other str   – AJ++
    :param boundary_size: int (default: 0)
        Width of the boundary layer (voxels) added around the domain when
        *is_periodic* is ``False``.
    :param output_step_size: int (default: 100)
        Number of iteration steps between statistics snapshots and saving the current fiber system.
    :param max_steps: int (default: 1000)
        Maximum number of packing iterations.
    :param tol: float (default: 1e-3)
        Relative-change tolerance for the AJ++ and Contact++ stop criteria.
    :param dir_matrix: array-like, shape (3, 3), default ``np.diag([1, 1, 1])``
        Direction matrix used for orientation statistics in the verbose output.
    :param tau: float (default: 0.25)
        Step size of the repulsion force
    :param rho: float (default: 0.25)
        Step size of the recovery forces
    :param alpha_s: float (default: pi/180)
        Start angle for the angle-force cut-off smoothing window (radians).
    :param alpha_e: float (default: 2pi/180)
        End angle for the angle-force cut-off smoothing window (radians).
    :param x_s: float (default: 0.25)
        Start distance ratio for the spring-force cut-off smoothing window.
    :param x_e: float (default: 0.5)
        End distance ratio for the spring-force cut-off smoothing window.
    """
    boundary_size_vec = np.array([boundary_size, boundary_size, boundary_size])
    if not is_periodic:
        image_size = image_size + 2 * boundary_size_vec
    if method == "AJ":
        run_force_biased_AJ(fs, image_size, use_end_step_radius, use_end_step_repulsion, output_path, verbose,
                            is_periodic, hardcore_ratio, output_step_size, max_steps, dir_matrix=dir_matrix,
                            tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e)
    elif method == "AJ++":
        run_force_biased_AJpp(fs, image_size, output_path, verbose, is_periodic, hardcore_ratio = hardcore_ratio,
                              output_step_size=output_step_size, max_steps=max_steps, tol=tol, dir_matrix=dir_matrix,
                              tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e)
    elif method == "AJ++fast":
        run_force_biased_AJppfast(fs, image_size, output_path, verbose, is_periodic, hardcore_ratio = hardcore_ratio,
                              output_step_size=output_step_size, max_steps=max_steps, dir_matrix=dir_matrix,
                                  tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e)
    elif method == "Contact++":
        run_force_biased_contactpp(fs, image_size, output_path, verbose, is_periodic, shortlist,
                                   hardcore_ratio = hardcore_ratio, contact_distance=contact_distance,
                                   output_step_size=output_step_size, max_steps=max_steps, dir_matrix=dir_matrix,
                                   tol=tol, tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e)
    else:
        run_force_biased_AJpp(fs, image_size, output_path, verbose, is_periodic, hardcore_ratio = hardcore_ratio,
                              output_step_size=output_step_size, max_steps=max_steps, tol=tol, dir_matrix=dir_matrix,
                              tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e)

def run_force_biased_AJ(
    fs: list[Fiber],
    image_size,
    use_end_step_radius: bool = False,
    use_end_step_repulsion: bool = False,
    output_path: str = "examples/outputs/",
    verbose: bool = True,
    is_periodic: bool = True,
    hardcore_ratio: float = 1.0,
    output_step_size: int = 100,
    max_steps: int = 1000,
    dir_matrix = np.diag([1,1,1]),
    tau: float = 0.25,
    rho: float = 0.25,
    alpha_s: float = 1*np.pi/180, alpha_e: float = 2*np.pi/180,
    x_s: float = 0.25, x_e: float = 0.5
):
    """
    Run the force-biased packing by Altendorf & Jeulin, using the original end criteria

    :param fs: list[Fiber]
        Fiber system to be packed.
    :param image_size: array-like of int, shape (3,)
        Size of the image domain in each spatial dimension (micrometers).
    :param use_end_step_radius: bool (default: False)
        If ``True`` and *method* is ``"AJ"``, reduces ball radii at the end
        of the run to eliminate residual overlaps.
    :param use_end_step_repulsion: bool (default: False)
        If ``True`` and *method* is ``"AJ"``, applies a pure repulsion step
        at the end of the run to eliminate residual overlaps.
    :param output_path: str (default: ``"examples/outputs/"``)
        Directory path for intermediate ``.gad`` snapshots of the current
        fiber system and the final ``results.csv`` statistics file.
    :param verbose: bool (default: True)
        If ``True``, prints per-step statistics and writes intermediate
        ``.gad`` snapshots of the current fiber system
    :param is_periodic: bool (default: True)
        Whether to apply periodic boundary conditions.
    :param hardcore_ratio: float (default: 1.0)
        Fraction of each ball's radius that may not be penetrated.
    :param boundary_size: int (default: 0)
        Width of the boundary layer (voxels) added around the domain when
        *is_periodic* is ``False``.
    :param output_step_size: int (default: 100)
        Number of iteration steps between statistics snapshots and saving the current fiber system.
    :param max_steps: int (default: 1000)
        Maximum number of packing iterations.
    :param dir_matrix: array-like, shape (3, 3), default ``np.diag([1, 1, 1])``
        Direction matrix used for orientation statistics in the verbose output.
    :param tau: float (default: 0.25)
        Step size of the repulsion force
    :param rho: float (default: 0.25)
        Step size of the recovery forces
    :param alpha_s: float (default: pi/180)
        Start angle for the angle-force cut-off smoothing window (radians).
    :param alpha_e: float (default: 2pi/180)
        End angle for the angle-force cut-off smoothing window (radians).
    :param x_s: float (default: 0.25)
        Start distance ratio for the spring-force cut-off smoothing window.
    :param x_e: float (default: 0.5)
        End distance ratio for the spring-force cut-off smoothing window.
    """
    rows = []

    max_radius = np.max([fiber.get_max_radius() for fiber in fs])
    min_radius = np.min([fiber.get_min_radius() for fiber in fs])

    grid = sh.SpatialHashing(image_size, 5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
        = calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio, repulsion_factor=1,
        tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
    )
    prev_optim_sum = optim_sum*optim_sum
    if verbose:
        rows.append(print_stats_row(fs, 0, overlap, neighbor_dist,
                                    0, optim_sum, 0, image_size, dir_matrix, is_periodic))
    print("We run the force-biased algorithm:")
    end_force_biased = 0.002 * max(image_size) * max(1, len(fs))

    for i in range(1, max_steps):
        if total_force_strength < end_force_biased and overlap < 0.1 * min_radius:
            break

        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        prev_optim_sum = optim_sum
        total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
            = calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
            repulsion_factor=1, tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
        )
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, overlap, neighbor_dist,
                                        0, optim_sum, abs(optim_sum - prev_optim_sum)/optim_sum,
                                        image_size, dir_matrix, is_periodic))
            write_gad(
                fs,
                output_path + "model.gad",
                image_size,
                is_periodic=is_periodic,
                )
    if use_end_step_radius and is_periodic:
        end_step_radius(fs, overlap, 0.1 * min_radius)
    if use_end_step_repulsion:
        end_step_repulsion(fs, max_radius, overlap, image_size)

    if verbose:
        rows.append(print_stats_row(fs, i, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum, abs(optim_sum - prev_optim_sum)/optim_sum,
                                    image_size, dir_matrix, is_periodic))
        write_gad(
            fs,
            output_path + "model.gad",
            image_size,
            is_periodic=is_periodic,
            )
        print_stats(output_path + "results.csv", rows)


def run_force_biased_AJpp(
    fs: list[Fiber],
    image_size,
    output_path: str = "examples/outputs/",
    verbose: bool = True,
    is_periodic: bool = True,
    hardcore_ratio: float = 1.0,
    output_step_size: int = 100,
    max_steps: int = 1000,
    tol: float = 1e-3,
    dir_matrix = np.diag([1,1,1]),
    tau: float = 0.25,
    rho: float = 0.25,
    alpha_s: float = 1*np.pi/180, alpha_e: float = 2*np.pi/180,
    x_s: float = 0.25, x_e: float = 0.5
):
    """
    Run the force-biased packing by Altendorf & Jeulin in the refined version

    :param fs: list[Fiber]
        Fiber system to be packed.
    :param image_size: array-like of int, shape (3,)
        Size of the image domain in each spatial dimension (micrometers).
    :param output_path: str (default: ``"examples/outputs/"``)
        Directory path for intermediate ``.gad`` snapshots of the current
        fiber system and the final ``results.csv`` statistics file.
    :param verbose: bool (default: True)
        If ``True``, prints per-step statistics and writes intermediate
        ``.gad`` snapshots of the current fiber system
    :param is_periodic: bool (default: True)
        Whether to apply periodic boundary conditions.
    :param hardcore_ratio: float (default: 1.0)
        Fraction of each ball's radius that may not be penetrated.
    :param boundary_size: int (default: 0)
        Width of the boundary layer (voxels) added around the domain when
        *is_periodic* is ``False``.
    :param output_step_size: int (default: 100)
        Number of iteration steps between statistics snapshots and saving the current fiber system.
    :param max_steps: int (default: 1000)
        Maximum number of packing iterations.
    :param tol: float (default: 1e-3)
        Relative-change tolerance for the AJ++ and Contact++ stop criteria.
    :param dir_matrix: array-like, shape (3, 3), default ``np.diag([1, 1, 1])``
        Direction matrix used for orientation statistics in the verbose output.
    :param tau: float (default: 0.25)
        Step size of the repulsion force
    :param rho: float (default: 0.25)
        Step size of the recovery forces
    :param alpha_s: float (default: pi/180)
        Start angle for the angle-force cut-off smoothing window (radians).
    :param alpha_e: float (default: 2pi/180)
        End angle for the angle-force cut-off smoothing window (radians).
    :param x_s: float (default: 0.25)
        Start distance ratio for the spring-force cut-off smoothing window.
    :param x_e: float (default: 0.5)
        End distance ratio for the spring-force cut-off smoothing window.
    """
    rows = []

    max_radius = max(fiber.get_max_radius() for fiber in fs)

    grid = sh.SpatialHashing(image_size, 5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum =\
        calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
            tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
    )
    if optim_sum < 1:
        optim_sum = 1
    prev_optim_sum = optim_sum*optim_sum
    if verbose:
        rows.append(print_stats_row(fs, 0, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum, 0, image_size, dir_matrix, is_periodic))
    eps = np.finfo(float).eps

    for i in range(1, max_steps):
        if abs(optim_sum - prev_optim_sum)/optim_sum < tol and overlap < eps:
            break
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        prev_optim_sum = optim_sum
        total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
            = calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
            tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
        )
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, overlap, neighbor_dist,
                                        shortlist_distance_sum, optim_sum, abs(optim_sum - prev_optim_sum)/optim_sum,
                                        image_size, dir_matrix, is_periodic))
            write_gad(
                fs,
                output_path + "model.gad",
                image_size,
                is_periodic=is_periodic,
                )

    if verbose:
        rows.append(print_stats_row(fs, i, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum, abs(optim_sum - prev_optim_sum)/optim_sum,
                                    image_size, dir_matrix, is_periodic))
        write_gad(
            fs,
            output_path + "model.gad",
            image_size,
            is_periodic=is_periodic,
            )
        print_stats(output_path + "results.csv", rows)


def run_force_biased_AJppfast(
    fs: list[Fiber],
    image_size,
    output_path: str = "examples/outputs/",
    verbose: bool = True,
    is_periodic: bool = True,
    hardcore_ratio: float = 1.0,
    output_step_size: int = 100,
    max_steps: int = 1000,
    dir_matrix = np.diag([1,1,1]),
    tau: float = 0.25,
    rho: float = 0.25,
    alpha_s: float = 1*np.pi/180, alpha_e: float = 2*np.pi/180,
    x_s: float = 0.25, x_e: float = 0.5
):
    """
    Run AJ++ (the refined version of the Altendorf-Jeulin packing), stopping as soon as
    all overlaps are resolved (overlap < machine epsilon).

    Unlike ``run_force_biased_AJpp``, this variant ignores the relative change
    of the optimisation sum and stops purely when the system is overlap-free.

    :param fs: list[Fiber]
        Fiber system to be packed.
    :param image_size: array-like of int, shape (3,)
        Size of the image domain in each spatial dimension (micrometers).
    :param output_path: str (default: ``"examples/outputs/"``)
        Directory path for intermediate ``.gad`` snapshots of the current
        fiber system and the final ``results.csv`` statistics file.
    :param verbose: bool (default: True)
        If ``True``, prints per-step statistics and writes intermediate
        ``.gad`` snapshots of the current fiber system
    :param is_periodic: bool (default: True)
        Whether to apply periodic boundary conditions.
    :param hardcore_ratio: float (default: 1.0)
        Fraction of each ball's radius that may not be penetrated.
    :param boundary_size: int (default: 0)
        Width of the boundary layer (voxels) added around the domain when
        *is_periodic* is ``False``.
    :param output_step_size: int (default: 100)
        Number of iteration steps between statistics snapshots and saving the current fiber system.
    :param max_steps: int (default: 1000)
        Maximum number of packing iterations.
    :param dir_matrix: array-like, shape (3, 3), default ``np.diag([1, 1, 1])``
        Direction matrix used for orientation statistics in the verbose output.
    :param tau: float (default: 0.25)
        Step size of the repulsion force
    :param rho: float (default: 0.25)
        Step size of the recovery forces
    :param alpha_s: float (default: pi/180)
        Start angle for the angle-force cut-off smoothing window (radians).
    :param alpha_e: float (default: 2pi/180)
        End angle for the angle-force cut-off smoothing window (radians).
    :param x_s: float (default: 0.25)
        Start distance ratio for the spring-force cut-off smoothing window.
    :param x_e: float (default: 0.5)
        End distance ratio for the spring-force cut-off smoothing window.
    """
    rows = []

    max_radius = max(fiber.get_max_radius() for fiber in fs)

    grid = sh.SpatialHashing(image_size, 5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum =\
        calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
            tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
    )
    if optim_sum < 1:
        optim_sum = 1
    prev_optim_sum = optim_sum*optim_sum
    if verbose:
        rows.append(print_stats_row(fs, 0, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum, 0, image_size, dir_matrix, is_periodic))
    eps = np.finfo(float).eps

    for i in range(1, max_steps):
        if overlap < eps:
            break
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        prev_optim_sum = optim_sum
        total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
            = calculate_forces(
            grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
            tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
        )
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, overlap, neighbor_dist,
                                        shortlist_distance_sum, optim_sum, abs(optim_sum - prev_optim_sum)/optim_sum,
                                        image_size, dir_matrix, is_periodic))
            write_gad(
                fs,
                output_path + "model.gad",
                image_size,
                is_periodic=is_periodic,
                )

    if verbose:
        rows.append(print_stats_row(fs, i, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum, abs(optim_sum - prev_optim_sum)/optim_sum,
                                    image_size, dir_matrix, is_periodic))
        write_gad(
            fs,
            output_path + "model.gad",
            image_size,
            is_periodic=is_periodic,
            )
        print_stats(output_path + "results.csv", rows)


def run_force_biased_contactpp(
    fs: list[Fiber],
    image_size,
    output_path: str = "examples/outputs/",
    verbose: bool = True,
    is_periodic: bool = True,
    shortlist = [],
    hardcore_ratio: float = 1.0,
    contact_distance: float = 0.0,
    output_step_size: int = 100,
    steps_tension_reduction: int = 500,
    max_steps: int = 1000,
    tol: float = 1e-3,
    dir_matrix = np.diag([1,1,1]),
    tau: float = 0.25,
    rho: float = 0.25,
    alpha_s: float = 1*np.pi/180, alpha_e: float = 2*np.pi/180,
    x_s: float = 0.25, x_e: float = 0.5
):
    """
    Run the Contact++ force-biased packing with shortlist-based contact forces.

    Applies repulsion forces to resolve overlaps and spring/angle forces to
    attract shortlisted contact pairs.  After *steps_tension_reduction*
    iterations, contact edges that are either no longer in contact or under
    compression are pruned from the shortlist to allow convergence.

    :param fs: list[Fiber]
        Fiber system to be packed.
    :param image_size: array-like of int, shape (3,)
        Size of the image domain in each spatial dimension (micrometers).
    :param output_path: str (default: ``"examples/outputs/"``)
        Directory path for intermediate ``.gad`` snapshots of the current
        fiber system and the final ``results.csv`` statistics file.
    :param verbose: bool (default: True)
        If ``True``, prints per-step statistics and writes intermediate
        ``.gad`` snapshots of the current fiber system
    :param is_periodic: bool (default: True)
        Whether to apply periodic boundary conditions.
    :param shortlist: list (default: [])
        Pre-computed contact shortlist (ball-pair edges).
    :param hardcore_ratio: float (default: 1.0)
        Fraction of each ball's radius that may not be penetrated.
    :param contact_distance: float (default: 1.0)
        Maximum surface-to-surface distance for two balls to be considered
        in contact.
    :param output_step_size: int (default: 100)
        Number of iteration steps between statistics snapshots and saving the current fiber system.
    :param steps_tension_reduction: int (default: 500)
        Iteration step at which tense edges are removed from the shortlist,
        i.e., those that are still not in contact or whose balls are overlapping.
    :param max_steps: int (default: 1000)
        Maximum number of packing iterations.
    :param tol: float (default: 1e-3)
        Relative-change tolerance for the AJ++ and Contact++ stop criteria.
    :param dir_matrix: array-like, shape (3, 3), default ``np.diag([1, 1, 1])``
        Direction matrix used for orientation statistics in the verbose output.
    :param tau: float (default: 0.25)
        Step size of the repulsion force
    :param rho: float (default: 0.25)
        Step size of the recovery forces
    :param alpha_s: float (default: pi/180)
        Start angle for the angle-force cut-off smoothing window (radians).
    :param alpha_e: float (default: 2pi/180)
        End angle for the angle-force cut-off smoothing window (radians).
    :param x_s: float (default: 0.25)
        Start distance ratio for the spring-force cut-off smoothing window.
    :param x_e: float (default: 0.5)
        End distance ratio for the spring-force cut-off smoothing window.
    """
    rows = []

    max_radius = max(fiber.get_max_radius() for fiber in fs)

    grid = sh.SpatialHashing(image_size, 5 * max_radius)
    grid.add_fiber_system(fs, is_periodic=is_periodic)
    total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, optim_sum\
        = (calculate_forces(
        grid, fiber_system=fs, is_periodic=is_periodic, shortlist=shortlist, hardcore_ratio = hardcore_ratio,
        contact_distance=contact_distance, repulsion_factor=1.05,
        tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
    ))
    if optim_sum < 1:
        optim_sum = 1
    prev_optim_sum = optim_sum*optim_sum
    if verbose:
        rows.append(print_stats_row(fs, 0, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum, 0,
                                    image_size, dir_matrix, is_periodic))
    print("We run the force-biased algorithm:")
    eps = np.finfo(float).eps

    for i in range(1, max_steps):
        if abs(optim_sum - prev_optim_sum)/optim_sum < tol and overlap < eps:
            break
        if i == steps_tension_reduction:
            print("remove tense links")
            tense_links = list()
            for contact_edge in shortlist:
                ball = fs[contact_edge[0][0]].balls[contact_edge[0][1]]
                ball2 = fs[contact_edge[1][0]].balls[contact_edge[1][1]]
                is_in_contact, dist = test_in_contact(ball,
                    ball2,
                    image_size=grid.image_size,
                    is_periodic=is_periodic,
                    contact_distance=contact_distance,
                )
                if not is_in_contact or dist < 0:
                    tense_links.append(contact_edge)
            shortlist = [x for x in shortlist if x not in tense_links]
            print(len(shortlist))

        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 5 * max_radius)
        grid.add_fiber_system(fs, is_periodic)
        if i < steps_tension_reduction:
            prev_optim_sum = optim_sum
            total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum,\
                optim_sum = (calculate_forces(
                grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
                shortlist=shortlist, contact_distance=contact_distance, repulsion_factor=1.05,
                tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
            ))
        else:
            prev_optim_sum = optim_sum
            total_force_strength, max_force_strength, overlap, neighbor_dist, angle_diff, shortlist_distance_sum, \
                optim_sum = (calculate_forces(
                grid, fiber_system=fs, is_periodic=is_periodic, hardcore_ratio = hardcore_ratio,
                contact_distance=contact_distance, repulsion_factor=1.1,
                tau=tau, rho=rho, alpha_s=alpha_s, alpha_e=alpha_e, x_s=x_s, x_e=x_e
            ))
        if verbose and i % output_step_size == 0:
            rows.append(print_stats_row(fs, i, overlap, neighbor_dist,
                                        shortlist_distance_sum, optim_sum, abs(optim_sum - prev_optim_sum)/optim_sum,
                                        image_size, dir_matrix, is_periodic))
            write_gad(
                fs,
                output_path + "model.gad",
                image_size,
                is_periodic=is_periodic,
                )

    if verbose:
        rows.append(print_stats_row(fs, i, overlap, neighbor_dist,
                                    shortlist_distance_sum, optim_sum, abs(optim_sum - prev_optim_sum)/optim_sum,
                                    image_size, dir_matrix, is_periodic))
        write_gad(
            fs,
            output_path + "model.gad",
            image_size,
            is_periodic=is_periodic,
            )
        print_stats(output_path + "results.csv", rows)


def end_step_radius(fs: list[Fiber], overlap: float, max_overlap: float):
    """
    The end step where radii are reduced

    :param fs: list[list[Ball]]
        the fiber system to be packed
    :param overlap: float
        The currently maximal overlap in the fiber system
    :param max_overlap: float
        The maximal overlap that is permitted for the fiber system
    """
    if overlap > max_overlap:  # why not only do this for radii that are too large?
        for fiber in fs:
            for ball in fiber.balls:
                new_radius = ball.radius - ball.overlap
                if new_radius <= 0:
                    raise ValueError("Radius cannot be reduced sufficiently.")
                ball.radius = new_radius
                ball.overlap = 0


def end_step_repulsion(
    fs: list[Fiber], max_radius: float, overlap: float, image_size: tuple[int, int, int]
):
    """
    The end step where only the repulsion force is applied

    :param fs: list[list[Ball]]
        the fiber system to be packed
    :param max_radius: float
        The maximal radius in the fiber system
    :param overlap: float
        The currently maximal overlap in the fiber system
    :param image_size: tuple[int, int, int]
    """
    for fiber in fs:
        for ball in fiber.balls:
            ball.force = np.array([0, 0, 0])
            ball.overlap = 0
    grid = sh.SpatialHashing(image_size, 5 * max_radius)
    grid.add_fiber_system(fs)
    while overlap > 0:
        force_strength, overlap = calculate_forces_endstep(grid, fiber_system=fs)
        apply_forces(fs)
        grid = sh.SpatialHashing(image_size, 5 * max_radius)
        grid.add_fiber_system(fs)

