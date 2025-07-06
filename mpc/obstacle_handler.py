import casadi as ca
import numpy as np 
from obstacle_geometry import Circle, StaticObstacle
from mpc.model import Model
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from mpc.model import Model
    from mpc.obstacle_geometry import StaticObstacle


class ObstacleHandler:
    def __init__(
        self,
        static_obstacles: List[StaticObstacle],
        circle: Circle,
        model: Model
    ):  
        self.model = model
        self.circle = circle
        self.static_obstacles = static_obstacles
    def step(self):
        static_obstacles_dict = {
            obstacle.calculate_distance(self.model.state()): obstacle
            for obstacle in self.static_obstacles
        }
        filtered_static_obstacles = [
            static_obstacles_dict[distance]
            for distance in sorted(static_obstacles_dict.keys())
            if True
            # if distance <= self.agent.sensor_radius
        ]
        print("Number of Static Obstacles:", len(filtered_static_obstacles))
        
        self.model.step(
            static_obstacles=filtered_static_obstacles,
            state_override=True
        )
        