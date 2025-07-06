import casadi as ca
import numpy as np 
from typing import List, Tuple
from abc import ABC, abstractmethod
import time
import cv2

class Circle:
    def __init__(self, center: Tuple, radius: float):
        self.radius = radius
        self.center = np.array(center, dtype=np.float64)
        
    # query_point - is the ego across state matrix
    # obstacle_position - is used for dynamic obstacles
    def calculate_distance(
        self, query_point: Tuple, obstacle_position: Tuple = None
    ) -> float:
        if obstacle_position is not None:
            center = np.array(obstacle_position)
        else:
            center = self.center
        return np.linalg.norm(np.array(query_point[:2]) - center) - self.radius

    def calculate_symbolic_distance(
        self, query_point: ca.MX, obstacle_position: Tuple = None
    ) -> ca.MX:
        if obstacle_position is not None:
            center = np.array(obstacle_position)
        else:
            center = self.center
        return (
            ca.sqrt(
                (query_point[0] - center[0]) ** 2 + (query_point[1] - center[1]) ** 2
            )
            - self.radius
        )
        
class StaticObstacle:
    def __init__(self, id: int, circle: Circle):
        self.id = id
        self.circle = circle
        
    
    def calculate_matrix_distance(self, state_matrix:np.ndarray):
        return np.stack(
            [self.circle.calculate_distance(state) for state in state_matrix.T]
        )
        
    def calculate_symbolic_matrix_distance(self, symbolic_states_matrix: ca.MX):
        start_time = time.time()
        result = ca.vertcat(
            *[
                self.circle.calculate_symbolic_distance(
                    symbolic_states_matrix[:2, time_step]
                )
                for time_step in range(symbolic_states_matrix.shape[1])
            ]
        )
        duration = time.time() - start_time 
        print(f"Symbolic distance generation took: {duration:.6f} seconds")
        
        return result