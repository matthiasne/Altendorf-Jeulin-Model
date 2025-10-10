class Fiber:
    def __init__(self, coordinates, radii):
        if not (isinstance(coordinates, list) and all(isinstance(v, (list, tuple)) and len(v) == 3 and
                                              all(isinstance(x, (int, float)) for x in v) for v in coordinates)):
            raise TypeError('Coordinates must be a list of 3D vectors')
        if not all(isinstance(r, (int, float)) and r > 0 for r in radii):
            raise TypeError('Radii must be a list of positive real numbers')
        if not len(coordinates) == len(radii):
            raise ValueError('Coordinates and radii must have the same length')
        self.coordinates = coordinates
        self.radii = radii

    def get_number_of_balls(self):
        return len(self.coordinates)

    def get_min_radius(self):
        return min(self.radii)

    def get_max_radius(self):
        return max(self.radii)
