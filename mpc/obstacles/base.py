import numpy as np
from typing import Tuple, Optional, List, cast
from mpc.geometries import Geometry
import casadi as ca
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


class Obstacle(ABC):
    def __init__(
        self,
        id: int,
        geometry: Geometry,
        position: Tuple[float, float],
        orientation: float = 90,
    ):
        self.id = id
        self.geometry = geometry
        self.state = np.array([*position, orientation])

    @abstractmethod
    def calculate_matrix_distance(self, states_matrix: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        raise NotImplementedError

    def calculate_distance(self, state: np.ndarray):
        return self.geometry.calculate_distance(state, self.state)

    def calculate_symbolic_distance(self, symbolic_state: ca.MX):
        return self.geometry.calculate_symbolic_distance(symbolic_state, self.state)
