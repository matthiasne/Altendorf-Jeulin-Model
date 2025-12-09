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
    def __init__(self, coordinates: list[np.ndarray], radii: list[float]):
        if not (isinstance(coordinates, list) and all(isinstance(v, (list, tuple)) and len(v) == 3 and
                                                      all(isinstance(x, (int, float)) for x in v) for v in
                                                      coordinates)):
            raise TypeError('Coordinates must be a list of 3D vectors')
        if not all(isinstance(r, (int, float)) and r > 0 for r in radii):
            raise TypeError('Radii must be a list of positive real numbers')
        if not len(coordinates) == len(radii):
            raise ValueError('Coordinates and radii must have the same length')
        self.coordinates = coordinates
        self.radii = radii

    def __init__(self, ball:Ball):
        self.balls = [ball]

    def get_number_of_balls(self):
        return len(self.balls)

    def get_min_radius(self):
        min_radius = min([ball.radius for ball in self.balls])
        return min_radius

    def get_max_radius(self):
        max_radius = max([ball.radius for ball in self.balls])
        return max_radius

    def add_ball(self, ball:Ball):
        self.balls.append(ball)
