from .base import Agent
from typing import Tuple


class ShadowAgent(Agent):
    def __init__(
        self,
        id: int,
        initial_position: Tuple[float, float],
        initial_orientation: float,
        goal_position: Tuple[float, float],
        goal_orientation: float,
        initial_linear_velocity: float = 0,
        initial_angular_velocity: float = 0,
        horizon: int = 50,
    ):
        super().__init__(
            id=id,
            initial_position=initial_position,
            initial_orientation=initial_orientation,
            goal_position=goal_position,
            goal_orientation=goal_orientation,
            initial_linear_velocity=initial_linear_velocity,
            initial_angular_velocity=initial_angular_velocity,
            horizon=horizon,
            sensor_radius=0,
            avoid_obstacles=False,
        )

    def step(self):
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
        )
        self.linear_velocity += self.controls_matrix[0, 0] * self.time_step
        self.angular_velocity += self.controls_matrix[1, 0] * self.time_step
