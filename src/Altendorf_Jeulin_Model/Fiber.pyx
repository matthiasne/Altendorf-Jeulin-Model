# cython: language_level=3, infer_type=True, exception_check=False, cdivision=True
import numpy as np
import cython
from libc.stdint cimport int64_t
cimport numpy as np
np.import_array()

cdef class Ball:
    """
    Ball contains balls used in the fiber model

    :param coordinate: np.ndarray
        coordinate of the center of the ball
    :param radius: float
        radius of the ball
    :param fiber_label: int
        label of the fiber that the ball belongs to
    :param ball_label: int
        index of the ball within the fiber
    :param angle: float
        angle between the incident edges of the ball
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

    def __cinit__(self, np.ndarray[np.double_t, ndim=1] coordinate, double radius, int fiber_label, int ball_label, double angle = 0):
        coordinate = np.ascontiguousarray(coordinate, dtype=np.float64)
        self.coordinate[0] = coordinate[0]
        self.coordinate[1] = coordinate[1]
        self.coordinate[2] = coordinate[2]
        self.radius = radius
        self.fiber_label = fiber_label
        self.ball_label = ball_label
        self.force[0] = 0.0
        self.force[1] = 0.0
        self.force[2] = 0.0
        self.overlap = 0.0
        self.angle = angle
        self.neighbor_dist = radius / 2.0
        self.angle_diff = 0.0


class Fiber:
    """
    Fiber contains a list of the balls forming the fiber

    Attributes:
        balls: [Ball]
            list of the balls
    """

    def __init__(self, ball: Ball):
        self.balls = [ball]

    def get_number_of_balls(self):
        return len(self.balls)

    def get_min_radius(self):
        min_radius = np.min([ball.radius for ball in self.balls])
        return min_radius

    def get_max_radius(self):
        max_radius = np.max([ball.radius for ball in self.balls])
        return max_radius

    def get_mean_radius(self):
        mean_radius = np.mean([ball.radius for ball in self.balls])
        return mean_radius

    def get_length(self):
        length = 0
        for i in range(1, len(self.balls)):
            # if i == 0:
            #    length += self.balls[0].radius
            #    length += self.balls[-1].radius
            # else:
            coord = self.balls[i].coordinate
            prev_coord = self.balls[i - 1].coordinate
            length += np.linalg.norm(coord - prev_coord)
        return length

    def get_direction(self):
        return np.array(self.balls[-1].coordinate) - np.array(self.balls[0].coordinate)

    def add_ball(self, ball: Ball):
        self.balls.append(ball)
