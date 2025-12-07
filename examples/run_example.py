import Altendorf_Jeulin_Model.FiberModel as fm
import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.ForceBiased import run_force_biased


def main():
    # create a fiber system
    print("This is the Altendorf-Jeulin model")
    image_size = (200, 200, 200)
    N = 10
    radius = 4
    fs = fm.initialize_fiber_system(N, 35, radius, 1, image_size, 10, 100)

    # output fiber system
    print("We generated the following fibers")
    io.print_fiber_positions(fs, 10)
    io.plot_fibers_in_2D(fs, path = "examples/outputs/spheres.png")
    io.save_fibers_as_tif(fs, path = "examples/outputs/spheres++.png")

    print("We run the force-biased algorithm:")
    run_force_biased(fs, image_size)

    io.print_fiber_positions(fs, 10)
    io.plot_fibers_in_2D(fs, path="examples/outputs/spheres++.png")
    io.save_fibers_as_tif(fs, path="examples/outputs/spheres++.tif")


if __name__ == "__main__":
    main()
