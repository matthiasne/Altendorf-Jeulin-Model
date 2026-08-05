import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["CalculateForces.pyx", "Fiber.py", "FiberModel.pyx", "ForceBiased.py", "io_utils.py",
                             "SpatialHashing.py", "Statistics.py", "utils.pyx", "VectorizedCalculateForces.pyx", "VectorizedForceBiased.py"],
                            compiler_directives={'language_level': 3}),
    include_dirs = [np.get_include()],
)
