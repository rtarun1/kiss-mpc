import casadi as ca
import numpy as np

from obstacle_handling.geometry import Circle
from obstacle_handling.obstacle import Obstacle

class DynamicObstacle(Obstacle):
    def __init__(self, id, position, orientation=np.deg2rad(90), linear_velocity=1.0, angular_velocity=0.0, horizon=6):
        super().__init__(id=id, geometry=Circle(center=position, radius=0.3))
        self.linear_velocity = linear_velocity
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.horizon = horizon
        self.states_matrix = self._get_predicted_states_matrix(horizon)

    @property
    def state(self):
        return np.array(self.geometry.location + (self.orientation, ))
    
    def _predict_state(self, state):
        dt = 0.1
        return np.array(
            [
                state[0] + self.linear_velocity * np.cos(np.deg2rad(state[2])) * dt,
                state[1] + self.linear_velocity * np.sin(np.deg2rad(state[2])) * dt,
                state[2] + self.angular_velocity * dt,
            ]
        )
    
    def _get_predicted_states_matrix(self, horizon):
        predicted_states = np.zeros((3, horizon))
        predicted_states[:, 0] = self.state

        for time_step in range(1, horizon):
            predicted_states[:, time_step] = self._predict_state(predicted_states[:, time_step - 1])

        return predicted_states
    
    def calculate_matrix_distance(self, states_matrix):
        return np.stack(
            [
                self.geometry.calculate_distance(state, self.states_matrix[:2, index])
                for state, index in enumerate(states_matrix.T)
            ]
        )
    
    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix):
        return ca.vertcat(
                *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:2, time_step],
                        self.states_matrix[:2, time_step],
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
            ),