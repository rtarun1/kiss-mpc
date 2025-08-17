from abc import ABC, abstractmethod
import casadi as ca
import numpy as np

class Geometry(ABC):
    @property
    @abstractmethod
    def location(self):
        raise NotImplementedError
    
    @location.setter
    @abstractmethod
    def location(self, value):
        raise NotImplementedError

    @abstractmethod
    def calculate_distance(self, distance_to, custom_self_location=None):
        raise NotImplementedError

    @abstractmethod
    def calculate_symbolic_distance(self, distance_to, custom_self_location=None):
        raise NotImplementedError


class Circle(Geometry):
    def __init__(self, center, radius):
        super().__init__()
        self.radius = radius
        self.center = np.array(center, dtype=np.float64)

    @property
    def location(self):
        return tuple(self.center)
    
    @location.setter
    def location(self, value):
        self.center += np.array(value) - self.center

    def calculate_distance(self, distance_to, custom_self_location=None):
        if custom_self_location is not None:
            center = np.array(custom_self_location)
        else:
            center = self.center
        return np.linalg.norm(np.array(distance_to[:2] - center) - self.radius)
    
    def calculate_symbolic_distance(self, distance_to, custom_self_location=None):
        if custom_self_location is not None:
            center = np.array(custom_self_location)
        else:
            center = self.center
        return (ca.sqrt((distance_to[0] - center[0]) ** 2 + (distance_to[1] - center[1]) ** 2) - self.radius)
   