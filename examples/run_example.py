import Altendorf_Jeulin_Model.FiberModel as fm
import Altendorf_Jeulin_Model.io_utils as io
from Altendorf_Jeulin_Model.ForceBiased import run_force_biased
from scipy.stats import poisson, uniform


def main():
    # create a fiber system
    print("This is the Altendorf-Jeulin model")
    image_size = (200, 200, 200)
    N = 100
    L = poisson(35)
    R = uniform(loc=4, scale=1)
    beta = 1
    fs = fm.initialize_fiber_system(N, L, R, beta, image_size, 10, 100)

    # output fiber system
    print("We generated the following fibers")
    io.print_fiber_positions(fs, 10)
    # io.plot_fibers_in_2D(fs, path="examples/outputs/spheres.png")
    # io.plot_fibers_in_2D_mod(fs, image_size, path="examples/outputs/spheres_mod.png")

    # io.save_fibers_as_tif(fs, path = "examples/outputs/spheres++.png")

    run_force_biased(fs, image_size, beta)

    io.print_fiber_positions(fs, 10)
    #io.plot_fibers_in_2D(fs, path="examples/outputs/spheres++.png")
    # io.plot_fibers_in_2D_mod(fs, image_size, path="examples/outputs/spheres++mod.png")

    # io.save_fibers_as_tif(fs, path="examples/outputs/spheres++.tif")


def test_AJ_setup():
    # create a fiber system
    image_size = (100, 100, 100)
    N = 100
    L = 50
    R = 5
    beta = 1
    fs = fm.initialize_fiber_system(N, L, R, beta, image_size, 10, 100, seed=1)
    run_force_biased(fs, image_size, beta, output_file="results_beta1_1.csv")
    fs = fm.initialize_fiber_system(N, L, R, beta, image_size, 10, 100, seed=2)
    run_force_biased(fs, image_size, beta, output_file="results_beta1_2.csv")
    fs = fm.initialize_fiber_system(N, L, R, beta, image_size, 10, 100, seed=3)
    run_force_biased(fs, image_size, beta, output_file="results_beta1_3.csv")

    beta = 0.1
    fs = fm.initialize_fiber_system(N, L, R, beta, image_size, 10, 100, seed=1)
    run_force_biased(fs, image_size, beta, output_file="results_beta0p1_1.csv")
    fs = fm.initialize_fiber_system(N, L, R, beta, image_size, 10, 100, seed=2)
    run_force_biased(fs, image_size, beta, output_file="results_beta0p1_2.csv")
    fs = fm.initialize_fiber_system(N, L, R, beta, image_size, 10, 100, seed=3)
    run_force_biased(fs, image_size, beta, output_file="results_beta0p1_3.csv")


if __name__ == "__main__":
    main()
