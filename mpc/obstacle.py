from abc import ABC, abstractmethod
from typing import Tuple, cast

import casadi as ca
import numpy as np

from mpc.geometry import Geometry


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


class StaticObstacle(Obstacle):
    def __init__(
        self,
        id: int,
        geometry: Geometry,
        position: Tuple[float, float],
        orientation: float = np.pi / 2,
    ):
        super().__init__(
            id=id, geometry=geometry, position=position, orientation=orientation
        )
        self.state = np.array([*position, orientation])

    def calculate_matrix_distance(self, states_matrix: np.ndarray):
        return np.stack(
            [
                self.geometry.calculate_distance(state, self.state)
                for state in states_matrix.T
            ]
        )

    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        return cast(
            ca.MX,
            ca.vertcat(
                *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:, time_step], self.state
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
            ),
        )
