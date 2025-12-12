import numpy as np


class Ball:
    """
    Ball contains balls used in the fiber model

    Attributes:
        coordinate: np.ndarray
            coordinate of the center of the ball
        radius: float
            radius of the ball
        fiber_label: int
            label of the fiber that the ball belongs to
        ball_label: int
            index of the ball within the fiber
        angle: float
            angle between the incident edges of the ball
    """
    def __init__(self, coordinate: np.ndarray, radius: float,
                 fiber_label: int = -1, ball_label: int = -1, angle: float = np.pi):
        self.coordinate = coordinate
        self.radius = radius
        self.fiber_label = fiber_label
        self.ball_label = ball_label
        self.force: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.overlap = 0
        self.angle = angle
        self.neighbor_dist = radius/2.0


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
        min_radius = min([ball.radius for ball in self.balls])
        return min_radius

    def get_max_radius(self):
        max_radius = max([ball.radius for ball in self.balls])
        return max_radius

    def add_ball(self, ball: Ball):
        self.balls.append(ball)
