from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np
import casadi as ca
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
