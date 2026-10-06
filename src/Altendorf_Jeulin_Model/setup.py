import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["CalculateForces.pyx", "Fiber.pyx", "FiberModel.pyx", "ForceBiased.pyx", "io_utils.pyx",
                             "SpatialHashing.pyx", "Statistics.pyx", "utils.pyx", "ContactModel.pyx", "rv_utils.pyx",
                             "Graph.pyx"],
                            compiler_directives={'language_level': 3}),
    include_dirs = [np.get_include()],
)
