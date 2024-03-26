import numpy as np
from typing import Tuple, cast
from mpc.geometries import Geometry
from .base import Obstacle
from mpc.agents.shadow import ShadowAgent
import casadi as ca


class StaticObstacle(Obstacle):
    def __init__(
        self,
        id: int,
        geometry: Geometry,
        position: Tuple[float, float],
    ):
        super().__init__(id=id, geometry=geometry, position=position)

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


class DynamicObstacle(Obstacle):
    def __init__(
        self,
        id: int,
        geometry: Geometry,
        position: Tuple[float, float],
        goal_position: Tuple[float, float],
        initial_linear_velocity: float = 0,
        initial_angular_velocity: float = 0,
        horizon: int = 50,
    ):
        super().__init__(id=id, geometry=geometry, position=position)
        self.shadow_agent = ShadowAgent(
            id=id,
            initial_position=tuple(self.state[:2]),
            initial_orientation=float(self.state[2]),
            goal_position=goal_position,
            goal_orientation=float(self.state[2]),
            initial_linear_velocity=initial_linear_velocity,
            initial_angular_velocity=initial_angular_velocity,
            horizon=horizon,
        )

    def calculate_matrix_distance(self, states_matrix: np.ndarray):
        return np.stack(
            [
                self.geometry.calculate_distance(
                    state, self.shadow_agent.states_matrix[:, index]
                )
                for index, state in enumerate(states_matrix.T)
            ]
        )

    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        return cast(
            ca.MX,
            ca.vertcat(
                *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:, time_step],
                        self.shadow_agent.states_matrix[:, time_step],
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
            ),
        )

    def step(self):
        self.shadow_agent.step()
        self.state = self.shadow_agent.state

    def reset(self):
        self.shadow_agent.reset()
        self.state = self.shadow_agent.initial_state
