import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("Altendorf_Jeulin_Model.CalculateForces",
              ["Altendorf_Jeulin_Model/CalculateForces.pyx"]),
    Extension("Altendorf_Jeulin_Model.Fiber",
              ["Altendorf_Jeulin_Model/Fiber.pyx"]),
    Extension("Altendorf_Jeulin_Model.FiberModel",
              ["Altendorf_Jeulin_Model/FiberModel.pyx"]),
    Extension("Altendorf_Jeulin_Model.ForceBiased",
              ["Altendorf_Jeulin_Model/ForceBiased.pyx"]),
    Extension("Altendorf_Jeulin_Model.PoissonLines",
              ["Altendorf_Jeulin_Model/PoissonLines.pyx"]),
    Extension("Altendorf_Jeulin_Model.ContactModel",
              ["Altendorf_Jeulin_Model/ContactModel.pyx"]),
    Extension("Altendorf_Jeulin_Model.utils.SpatialHashing",
              ["Altendorf_Jeulin_Model/utils/SpatialHashing.pyx"]),
    Extension("Altendorf_Jeulin_Model.utils.Statistics",
              ["Altendorf_Jeulin_Model/utils/Statistics.pyx"]),
    Extension("Altendorf_Jeulin_Model.utils.utils",
              ["Altendorf_Jeulin_Model/utils/utils.pyx"]),
    Extension("Altendorf_Jeulin_Model.utils.rv_utils",
              ["Altendorf_Jeulin_Model/utils/rv_utils.pyx"]),
    Extension("Altendorf_Jeulin_Model.utils.Graph",
              ["Altendorf_Jeulin_Model/utils/Graph.pyx"]),
    Extension("Altendorf_Jeulin_Model.utils.io_utils",
              ["Altendorf_Jeulin_Model/utils/io_utils.pyx"]),
]

setup(
    ext_modules=cythonize(extensions,
                          compiler_directives={'language_level': 3}),
    include_dirs=[np.get_include()],
)