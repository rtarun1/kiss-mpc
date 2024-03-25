import numpy as np
from typing import Tuple, Optional, List

from obstacles.obstacle import Obstacle


class Agent:
    def __init__(
        self,
        id: int,
        initial_state: Tuple,
        goal_state: Tuple,
        horizon: int = 50,
        radius: float = 1.0,
        sensor_radius: float = 50.0,
        avoid_obstacles: bool = False,
    ):
        assert horizon > 0, "Horizon must be greater than 0"

        self.id = id
        self.radius = radius
        self.sensor_radius = sensor_radius

        self.avoid_obstacles = avoid_obstacles
        self.all_obstacles: List[Obstacle] = []

        self.initial_state = np.array(initial_state)
        self.goal_state = np.array(goal_state)

        self.horizon = horizon

        self.linear_velocity_bounds: Tuple[float, float] = (0, 12)
        self.angular_velocity_bounds: Tuple[float, float] = (-np.pi / 4, np.pi / 4)
        self.linear_acceleration_bounds: Tuple[float, float] = (-50, 50)
        self.angular_acceleration_bounds: Tuple[float, float] = (-50, 50)

        self.left_right_lane_bounds: Tuple[float, float] = (-1000.5, 1000.5)

        self.time_step = 0.041

        self.states_matrix = np.tile(self.initial_state, (self.horizon + 1, 1)).T
        self.controls_matrix = np.zeros((2, self.horizon))

        self.linear_velocity: float = 5
        self.angular_velocity: float = 0
        self.state: np.ndarray = self.states_matrix[:, 1]

    @property
    def visible_obstacles(self) -> List[Obstacle]:
        return (
            [
                obstacle
                for obstacle in self.all_obstacles
                if obstacle.geometry.calculate_distance(self.state)
                <= self.sensor_radius
            ]
            if self.avoid_obstacles
            else []
        )

    def update_state(self):
        self.state = self.states_matrix[:, 1]

    def update_velocities(self):
        self.linear_velocity += self.controls_matrix[0, 0] * self.time_step
        self.angular_velocity += self.controls_matrix[1, 0] * self.time_step
