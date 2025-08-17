import casadi as ca
import numpy as np
from abc import ABC, abstractmethod

class Obstacle(ABC):
    def __init__(self, id, geometry):
        self.id = id
        self.geometry = geometry

    @property
    def state(self):
        return np.array(self.geometry.location + (0,))

    @abstractmethod
    def calculate_matrix_distance(self, states_matrix):
        raise NotImplementedError

    @abstractmethod
    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix):
        raise NotImplementedError

    def calculate_distance(self, state):
        return self.geometry.calculate_distance(state)

    def calculate_symbolic_distance(self, symbolic_state):
        return self.geometry.calculate_symbolic_distance(symbolic_state)
    
class StaticObstacle(Obstacle):
    def __init__(self, id, geometry):
        super().__init__(id=id, geometry=geometry)

    def calculate_matrix_distance(self, states_matrix):
        return np.stack(
            [self.geometry.calculate_distance(state) for state in states_matrix]
        )
    
    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix):
        return ca.vertcat(
            *[
                    self.geometry.calculate_symbolic_distance(
                        symbolic_states_matrix[:2, time_step]
                    )
                    for time_step in range(symbolic_states_matrix.shape[1])
                ]
        )
    
