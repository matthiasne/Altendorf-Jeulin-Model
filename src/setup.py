import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["Altendorf_Jeulin_Model/CalculateForces.pyx", "Altendorf_Jeulin_Model/Fiber.pyx",
                             "Altendorf_Jeulin_Model/FiberModel.pyx", "Altendorf_Jeulin_Model/ForceBiased.pyx",
                             "Altendorf_Jeulin_Model/PoissonLines.pyx", "Altendorf_Jeulin_Model/ContactModel.pyx",
                             "utils/SpatialHashing.pyx", "utils/Statistics.pyx", "utils/utils.pyx",
                             "utils/rv_utils.pyx", "utils/Graph.pyx", "utils/io_utils.pyx"],
                            compiler_directives={'language_level': 3}),
    include_dirs = [np.get_include()],
)
