import numpy as np


class Ball:
    def __init__(self, coordinate: np.ndarray, radius: float,
                 fiber_label: int = -1, ball_label: int = -1, angle:float = np.pi):
        self.coordinate = coordinate
        self.radius = radius
        self.fiber_label = fiber_label
        self.ball_label = ball_label
        self.force: np.ndarray = np.array([0.0,0.0,0.0])
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

    def __init__(self, coordinate: np.ndarray, radius: float):
        b = Ball(coordinate, radius)
        self.balls = [b]

    def get_number_of_balls(self):
        return len(self.coordinates)

    def get_min_radius(self):
        return min(self.radii)

    def get_max_radius(self):
        return max(self.radii)
