# cython: language_level=3, infer_type=True, exception_check=False, cdivision=True
import cython

#cdef inline void vonmises_fisher(double kappa, double mu_x, double mu_y, double mu_z, rng random_state,
#                    double* dir_x, double* dir_y, double* dir_z) noexcept