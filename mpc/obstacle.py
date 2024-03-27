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
        initial_linear_velocity: float = 0,
        initial_angular_velocity: float = -np.pi,
    ):
        super().__init__(id=id, geometry=geometry, position=position)
        self.linear_velocity = initial_linear_velocity
        self.angular_velocity = initial_angular_velocity
        self.initial_state = self.state
        self.linear_velocity_bounds = (-4, 0)
        self.angular_velocity_bounds = (0, 0)

    def _perturb_velocity(self):
        # Modify velocity randomly within bounds
        self.linear_velocity = np.clip(
            self.linear_velocity + np.random.uniform(-0.1, 0.1),
            *self.linear_velocity_bounds,
        )
        self.angular_velocity = np.clip(
            self.angular_velocity + np.random.uniform(-0.1, 0.1),
            *self.angular_velocity_bounds,
        )

    def _predict_state(self, state: np.ndarray):
        self._perturb_velocity()
        return np.array(
            [
                state[0] + self.linear_velocity * np.cos(np.deg2rad(state[2])),
                state[1] + self.linear_velocity * np.sin(np.deg2rad(state[2])),
                state[2] + self.angular_velocity,
            ]
        )

    def _get_predicted_states_matrix(self, horizon: int):
        predicted_states = np.zeros((3, horizon))
        predicted_states[:, 0] = self.state

        for time_step in range(1, horizon):
            predicted_states[:, time_step] = self._predict_state(
                predicted_states[:, time_step - 1]
            )

        return predicted_states

    def calculate_matrix_distance(self, states_matrix: np.ndarray):
        obstacle_states_matrix = self._get_predicted_states_matrix(
            states_matrix.shape[1]
        )
        return np.stack(
            [
                self.geometry.calculate_distance(
                    state, obstacle_states_matrix[:, index]
                )
                for index, state in enumerate(states_matrix.T)
            ]
        )

    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        obstacle_states_matrix = self._get_predicted_states_matrix(
            symbolic_states_matrix.shape[1]
        )
        return cast(
            ca.MX,
            ca.vertcat(
                *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:, time_step],
                        obstacle_states_matrix[:, time_step],
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
            ),
        )

    def step(self):
        self.state = self._predict_state(self.state)

    def reset(self):
        self.state = self.initial_state
