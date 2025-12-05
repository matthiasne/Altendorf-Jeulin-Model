import tifffile
import numpy as np
from Fiber import Ball
import SpatialHashing


def print_fiber_positions(fiber_system: list[list[Ball]],
                          max_fibers: int = 10, max_spheres: int = 10):
    """
    Print fibers as positions.

    Parameters
    ---------------------
    :param fiber_system: list[Fiber]
        A list of fibers, each represented as a list of balls
    :param max_fibers: int, optional
        maximal number of fibers to be printed
    :param max_spheres:  int, optional
        maximal number of spheres to be printed
    """
    for i, fiber in enumerate(fiber_system[:max_fibers]):
        coords = ' '.join(f"[{ball.coordinate[0]:.2f},"
                          f" {ball.coordinate[1]:.2f}, {ball.coordinate[2]:.2f}]"
                          for ball in fiber[:max_spheres])
        print("Fiber ", i, ":", coords)


def save_fibers_as_tif(fiber_system: list[list[Ball]],
                       shape: tuple[int, int, int] = (64, 64, 64),
                       path: str = "outputs/spheres.tif"):
    """
    Save fibers as tif-image

    Parameters
    ---------------------
    :param fiber_system: list[list[Ball]]
        A list of fibers, each represented as a list of 3D np.arrays
    :param shape: tuple(int, int, int) optional
        z,y,x coordinate of the image
    :param path: string optional
        The path where the tif image will be saved
    """
    volume = np.zeros(shape, dtype=np.uint8)
    for i, fiber in enumerate(fiber_system):
        for ball in fiber:
            # create mesh grid
            x = np.arange(shape[2])
            y = np.arange(shape[1])
            z = np.arange(shape[0])
            X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

            # set pixels inside sphere to 255 (white)
            x0, y0, z0 = ball.coordinate
            dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2)
            volume[dist <= ball.radius] = 255
    tifffile.imwrite(path, volume)


def plot_fibers_in_2D(fiber_system: list[list[Ball]],
                      path: str = "outputs/spheres.png"):
    """
    Plot fibers in a 2D image with 3D representations

    Parameters
    ---------------------
    :param fiber_system: list[list[Ball]
        A list of fibers, each represented as a list of Balls
    :param path: string optional
        The path where the image will be saved
    :return:
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = cm.get_cmap('viridis', len(fiber_system))

    # Create a sphere parametrically
    u = np.linspace(0, 2 * np.pi, 20)  # angle around z-axis
    v = np.linspace(0, np.pi, 20)  # angle from top to bottom

    for i, fiber in enumerate(fiber_system):
        color = cmap(i)
        for ball in fiber:
            x0, y0, z0 = ball.coordinate
            radius = ball.radius
            x = x0 + radius * np.outer(np.cos(u), np.sin(v))
            y = y0 + radius * np.outer(np.sin(u), np.sin(v))
            z = z0 + radius * np.outer(np.ones_like(u), np.cos(v))

            ax.plot_surface(x, y, z, color=color, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Fiber System')
    plt.savefig(path, dpi=300)


def print_grid(grid: SpatialHashing):
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
