import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["CalculateForces.py", "Fiber.py", "FiberModel.py", "ForceBiased.py", "io_utils.py",
                             "SpatialHashing.py", "Statistics.py", "utils.py"]),
    include_dirs = [np.get_include()],
)
