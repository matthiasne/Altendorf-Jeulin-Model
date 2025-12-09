import numpy as np


class Ball:
    def __init__(self, coordinate: np.ndarray, radius: float,
                 fiber_label: int = -1, ball_label: int = -1, angle: float = np.pi):
        self.coordinate = coordinate
        self.radius = radius
        self.fiber_label = fiber_label
        self.ball_label = ball_label
        self.force: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.overlap = 0
        self.angle = angle


class Fiber:
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
