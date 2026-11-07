cdef class Ball:
    """
    A sphere representing one segment of a fiber in the Altendorf–Jeulin model.

    All fields are public so they can be read and written from Python.

    :param coordinate: double[3]
        Cartesian center position (x, y, z)
    :param radius: double
        sphere radius
    :param fiber_label: int
        index of the fiber this ball belongs to; -1 if unassigned
    :param ball_label: int
        index of this ball within its fiber; -1 if unassigned
    :param force: double[3]
        force vector (gradient) acting on this ball during packing, initialized to (0, 0, 0)
    :param overlap: double
        maximal overlap with neighboring balls, initialized to 0
    :param angle: double
        target bending angle at this ball (radians), i.e. the angle between the incoming and outgoing segments;
        defaults to pi (straight)
    :param angle_diff: double
        difference between the realized bending angle and the target angle after packing, initialized to 0
    :param neighbor_dist: double
        distance to the next ball set to radius / 2 at construction
    :param optim_sum: double
        summand of the optimization objective, initialized to 0
    """
    cdef public double radius
    cdef public int fiber_label
    cdef public int ball_label
    cdef public double coordinate[3]
    cdef public double force[3]
    cdef public double overlap
    cdef public double angle
    cdef public double neighbor_dist
    cdef public double angle_diff
    cdef public double optim_sum

