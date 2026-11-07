import numpy as np
from Altendorf_Jeulin_Model.utils.utils cimport cdistance_ball, cartesian_to_spherical


cdef class Ball:
    """
    A ball representing one segment of a fiber.

    See Fiber.pxd for the full attribute list.

    :param coordinate: array-like of shape (3,)
        Cartesian center position (x, y, z)
    :param radius: float
        ball radius (must be >= 0)
    :param fiber_label: int, optional
        index of the owning fiber; default -1 (unassigned)
    :param ball_label: int, optional
        index within the fiber's ball list; default -1 (unassigned)
    :param angle: float, optional
        target bending angle in radians; default pi (straight segment)
    """

    def __init__(
        self,
        coordinate: np.ndarray,
        radius: float,
        fiber_label: int = -1,
        ball_label: int = -1,
        angle: float = np.pi,
    ):
        """
        Initialize all fields of the ball.

        Forces, overlap, angle_diff and optim_sum are always set to zero.
        neighbor_dist is set to radius / 2.

        :param coordinate: array-like of shape (3,)
            Cartesian center position
        :param radius: float
            ball radius
        :param fiber_label: int, optional
            owning fiber index; default -1
        :param ball_label: int, optional
            index within the fiber; default -1
        :param angle: float, optional
            target bending angle in radians; default pi
        """
        self.coordinate[0] = coordinate[0]
        self.coordinate[1] = coordinate[1]
        self.coordinate[2] = coordinate[2]
        self.radius = radius
        self.fiber_label = fiber_label
        self.ball_label = ball_label
        self.force[0] = 0
        self.force[1] = 0
        self.force[2] = 0
        self.overlap = 0
        self.angle = angle
        self.neighbor_dist = radius / 2.0
        self.angle_diff = 0
        self.optim_sum = 0


class Fiber:
    """
    A fiber modeled as an ordered chain of overlapping balls.

    The fiber is built incrementally: start with a single Ball passed to
    the constructor, then append further balls with add_ball().  The
    overall fiber direction is defined as the vector from the first ball's
    center to the last ball's center.

    :param balls: list[Ball]
        ordered list of balls forming the fiber, in the order they were added
    """

    def __init__(self, ball: Ball):
        """
        Create a fiber containing a single ball.

        :param ball: Ball
            the first (and initially only) ball of the fiber
        """
        self.balls = [ball]

    def get_number_of_balls(self):
        """
        Return the number of balls in the fiber.

        :return: int
            length of the balls list
        """
        return len(self.balls)

    def get_min_radius(self):
        """
        Return the minimum ball radius in the fiber.

        :return: float
            smallest radius across all balls
        """
        min_radius = np.min([ball.radius for ball in self.balls])
        return min_radius

    def get_max_radius(self):
        """
        Return the maximum ball radius in the fiber.

        :return: float
            largest radius across all balls
        """
        max_radius = np.max([ball.radius for ball in self.balls])
        return max_radius

    def get_mean_radius(self):
        """
        Return the mean ball radius in the fiber.

        :return: float
            arithmetic mean of all ball radii
        """
        mean_radius = np.mean([ball.radius for ball in self.balls])
        return mean_radius

    def get_length(self):
        """
        Return the arc length of the fiber.

        Computed as the sum of Euclidean distances between consecutive
        ball centers.  Returns 0 for a single-ball fiber.

        :return: float
            total arc length
        """
        length = 0
        for i in range(1, len(self.balls)):
            length += cdistance_ball(self.balls[i], self.balls[i - 1])
        return length

    def get_direction(self):
        """
        Return the end-to-end direction vector of the fiber.

        Defined as last_ball.coordinate - first_ball.coordinate.
        Not normalised — use utils.normalized() if a unit vector is needed.

        :return: np.ndarray, shape (3,)
            direction vector from first to last ball center
        :raises ValueError:
            if the fiber has fewer than 2 balls
        """
        cdef Ball first_ball = self.balls[0]
        cdef Ball last_ball = self.balls[len(self.balls) - 1]
        if len(self.balls) < 2:
            raise ValueError("Fiber must have at least 2 balls to compute a direction")

        return np.array([
            last_ball.coordinate[0] - first_ball.coordinate[0],
            last_ball.coordinate[1] - first_ball.coordinate[1],
            last_ball.coordinate[2] - first_ball.coordinate[2],
            ], dtype=np.float64)


    def add_ball(self, ball: Ball):
        """
        Append a ball to the end of the fiber.

        :param ball: Ball
            the ball to append
        """
        self.balls.append(ball)
