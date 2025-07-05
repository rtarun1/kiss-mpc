import casadi as ca
import numpy as np 
from typing import List, Tuple
from abc import ABC, abstractmethod

import cv2

class Circle(ABC):
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