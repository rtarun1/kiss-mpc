from .base import Agent
from mpc.obstacles.base import Obstacle
from typing import List, Optional, Tuple
import numpy as np


class EgoAgent(Agent):
    def __init__(
        self,
        id: int,
        initial_position: Tuple[float, float],
        initial_orientation: float,
        goal_position: Tuple[float, float],
        goal_orientation: float,
        planning_time_step: float = 0.041,
        initial_linear_velocity: float = 0,
        initial_angular_velocity: float = 0,
        horizon: int = 50,
        sensor_radius: float = 50.0,
        linear_velocity_bounds: Tuple[float, float] = (0, 12),
        angular_velocity_bounds: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        linear_acceleration_bounds: Tuple[float, float] = (-50, 50),
        angular_acceleration_bounds: Tuple[float, float] = (-50, 50),
        left_right_lane_bounds: Tuple[float, float] = (-1000.5, 1000.5),
    ):
        super().__init__(
            id=id,
            initial_position=initial_position,
            initial_orientation=initial_orientation,
            goal_position=goal_position,
            goal_orientation=goal_orientation,
            planning_time_step=planning_time_step,
            initial_linear_velocity=initial_linear_velocity,
            initial_angular_velocity=initial_angular_velocity,
            horizon=horizon,
            sensor_radius=sensor_radius,
            avoid_obstacles=True,
            linear_velocity_bounds=linear_velocity_bounds,
            angular_velocity_bounds=angular_velocity_bounds,
            linear_acceleration_bounds=linear_acceleration_bounds,
            angular_acceleration_bounds=angular_acceleration_bounds,
            left_right_lane_bounds=left_right_lane_bounds,
        )

    def step(self, obstacles: Optional[List[Obstacle]] = None):
        self.states_matrix, self.controls_matrix = self.planner.solve(
            current_state=self.state,
            current_linear_velocity=self.linear_velocity,
            current_angular_velocity=self.angular_velocity,
            goal_state=self.goal_state,
            states_matrix=self.states_matrix,
            controls_matrix=self.controls_matrix,
            left_right_lane_bounds=self.left_right_lane_bounds,
            linear_velocity_bounds=self.linear_velocity_bounds,
            angular_velocity_bounds=self.angular_velocity_bounds,
            linear_acceleration_bounds=self.linear_acceleration_bounds,
            angular_acceleration_bounds=self.angular_acceleration_bounds,
            inflation_radius=(2 * self.geometry.radius + 0.5),
            obstacles=obstacles if self.avoid_obstacles else None,
        )
        self.linear_velocity += self.controls_matrix[0, 0] * self.time_step
        self.angular_velocity += self.controls_matrix[1, 0] * self.time_step
