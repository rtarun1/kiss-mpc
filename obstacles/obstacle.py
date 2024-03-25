import numpy as np
from typing import Tuple, Optional, List, cast
from obstacles.geometry import Geometry
import casadi as ca


class Obstacle:
    def __init__(
        self,
        id: int,
        geometry: Geometry,
        horizon: int = 50,
    ):
        self.id = id
        self.geometry = geometry
        self.states_matrix = np.tile(
            np.array([self.geometry.x, self.geometry.y, self.geometry.z]),
            (horizon + 1, 1),
        ).T

    def calculate_distance(self, states_matrix: np.ndarray):
        return np.stack(
            [self.geometry.calculate_distance(state) for state in states_matrix.T]
        )

    def calculate_symbolic_distance(self, symbolic_states_matrix: ca.MX):
        return cast(
            ca.MX,
            ca.vertcat(
                *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:, time_step]
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
            ),
        )
