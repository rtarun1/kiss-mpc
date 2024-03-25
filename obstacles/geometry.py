import numpy as np
from abc import ABC
import casadi as ca


class Geometry(ABC):
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def calculate_distance(self, state: np.ndarray) -> float:
        raise NotImplementedError

    def calculate_symbolic_distance(self, state: ca.MX) -> ca.MX:
        raise NotImplementedError


class Circle(Geometry):
    def __init__(self, x: float, y: float, z: float, radius: float):
        super().__init__(x, y, z)
        self.radius = radius

    def calculate_distance(self, state: np.ndarray) -> float:
        return np.linalg.norm(state[:2] - np.array([self.x, self.y])) - self.radius

    def calculate_symbolic_distance(self, state: ca.MX) -> ca.MX:
        return (
            ca.sqrt((state[0] - self.x) ** 2 + (state[1] - self.y) ** 2) - self.radius
        )


class Rectangle(Geometry):
    def __init__(self, x: float, y: float, z: float, width: float, height: float):
        super().__init__(x, y, z)
        self.width = width
        self.height = height
        self.bottom_left_bounds = self.x - self.width / 2, self.y - self.height / 2
        self.top_right_bounds = self.x + self.width / 2, self.y + self.height / 2

    def calculate_distance(self, state: np.ndarray) -> float:
        x_coordinate = abs(state[0] - self.x)
        y_coordinate = abs(state[1] - self.y)

        return np.sqrt(
            np.max([x_coordinate - self.width / 2, 0]) ** 2
            + [y_coordinate - self.height / 2, 0] ** 2
        )

    def calculate_symbolic_distance(self, state: ca.MX) -> ca.MX:
        x_coordinate = ca.fabs(state[0] - self.x)
        y_coordinate = ca.fabs(state[1] - self.y)
        return ca.sqrt(
            ca.mmax(ca.vertcat(x_coordinate - self.width / 2, 0)) ** 2
            + ca.mmax(ca.vertcat(y_coordinate - self.height / 2, 0)) ** 2
        )


class Ellipsoid(Geometry):
    def __init__(self, x: float, y: float, z: float, a: float, b: float):
        super().__init__(x, y, z)
        self.a = a
        self.b = b

    def from_circle(circle: Circle):
        return Ellipsoid(circle.x, circle.y, circle.z, circle.radius, circle.radius)

    def from_rectange(rectangle: Rectangle):
        # Return ellipsoid approximating the rectangle
        return Ellipsoid(
            rectangle.x,
            rectangle.y,
            rectangle.z,
            rectangle.width / 2,
            rectangle.height / 2,
        )

    def calculate_distance(self, state: np.ndarray) -> float:
        x_coordinate = abs(state[0] - self.x)
        y_coordinate = abs(state[1] - self.y)
        return np.sqrt((x_coordinate / self.a) ** 2 + (y_coordinate / self.b) ** 2)

    def calculate_symbolic_distance(self, state: ca.MX) -> ca.MX:
        x_coordinate = ca.fabs(state[0] - self.x)
        y_coordinate = ca.fabs(state[1] - self.y)
        return ca.sqrt((x_coordinate / self.a) ** 2 + (y_coordinate / self.b) ** 2)
