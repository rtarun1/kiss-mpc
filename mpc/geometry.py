from abc import ABC, abstractmethod

import casadi as ca
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.patches import Patch


class Geometry(ABC):
    def __init__(self):
        self.patch: Patch = None

    @abstractmethod
    def calculate_distance(self, distance_to: np.ndarray, state: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def calculate_symbolic_distance(
        self, distance_to: ca.MX, state: np.ndarray
    ) -> ca.MX:
        raise NotImplementedError

    @abstractmethod
    def update_patch(self, state: np.ndarray):
        raise NotImplementedError


class Circle(Geometry):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        self.patch: mpatches.Circle = mpatches.Circle(
            (0, 0), radius, fill=False, color="black", linestyle="--"
        )

    def calculate_distance(self, distance_to: np.ndarray, state: np.ndarray) -> float:
        return np.linalg.norm(distance_to[:2] - state[:2]) - self.radius

    def calculate_symbolic_distance(
        self, distance_to: ca.MX, state: np.ndarray
    ) -> ca.MX:
        return (
            ca.sqrt((distance_to[0] - state[0]) ** 2 + (distance_to[1] - state[1]) ** 2)
            - self.radius
        )

    def update_patch(self, state: np.ndarray):
        # Plot the circle
        self.patch.set_center((state[0], state[1]))


class Rectangle(Geometry):
    def __init__(self, width: float, height: float):
        super().__init__()
        self.width = width
        self.height = height
        self.patch: mpatches.Rectangle = mpatches.Rectangle(
            (0, 0),
            self.width,
            self.height,
            fill=False,
            color="black",
            linestyle="--",
        )

    def calculate_distance(self, distance_to: np.ndarray, state: np.ndarray) -> float:
        x_coordinate = abs(distance_to[0] - state[0])
        y_coordinate = abs(distance_to[1] - state[0])

        return np.sqrt(
            np.max([x_coordinate - self.width / 2, 0]) ** 2
            + np.max([y_coordinate - self.height / 2, 0]) ** 2
        )

    def calculate_symbolic_distance(
        self, distance_to: ca.MX, state: np.ndarray
    ) -> ca.MX:
        x_coordinate = ca.fabs(distance_to[0] - state[0])
        y_coordinate = ca.fabs(distance_to[1] - state[1])
        return ca.sqrt(
            ca.mmax(ca.vertcat(x_coordinate - self.width / 2, 0)) ** 2
            + ca.mmax(ca.vertcat(y_coordinate - self.height / 2, 0)) ** 2
        )

    def update_patch(self, state: np.ndarray):
        # Plot the rectangle
        self.patch.set_xy((state[0] - self.width / 2, state[1] - self.height / 2))


class Ellipsoid(Geometry):
    def __init__(self, a: float, b: float):
        super().__init__()
        self.a = a
        self.b = b
        self.patch: mpatches.Ellipse = mpatches.Ellipse(
            (0, 0), 2 * self.a, 2 * self.b, fill=False, color="black", linestyle="--"
        )

    def from_circle(circle: Circle):
        return Ellipsoid(circle.radius, circle.radius)

    def from_rectangle(rectangle: Rectangle):
        # Return ellipsoid approximating the rectangle
        return Ellipsoid(
            rectangle.width / 2,
            rectangle.height / 2,
        )

    def calculate_distance(self, distance_to: np.ndarray, state: np.ndarray) -> float:
        x_coordinate = abs(distance_to[0] - state[0])
        y_coordinate = abs(distance_to[1] - state[1])
        return np.sqrt((x_coordinate / self.a) ** 2 + (y_coordinate / self.b) ** 2)

    def calculate_symbolic_distance(
        self, distance_to: ca.MX, state: np.ndarray
    ) -> ca.MX:
        x_coordinate = ca.fabs(distance_to[0] - state[0])
        y_coordinate = ca.fabs(distance_to[1] - state[1])
        return ca.sqrt((x_coordinate / self.a) ** 2 + (y_coordinate / self.b) ** 2)

    def update_patch(self, state: np.ndarray):
        # Plot the ellipse
        self.patch.set_center((state[0], state[1]))
