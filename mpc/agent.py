from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from mpc.geometry import Circle
from mpc.obstacle import Obstacle
from mpc.planner import MotionPlanner


class Agent(ABC):
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
        geometry: Circle = Circle(1),
        avoid_obstacles: bool = True,
        linear_velocity_bounds: Tuple[float, float] = (0, 12),
        angular_velocity_bounds: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        linear_acceleration_bounds: Tuple[float, float] = (-50, 50),
        angular_acceleration_bounds: Tuple[float, float] = (-np.pi, np.pi),
        left_right_lane_bounds: Tuple[float, float] = (-1000.5, 1000.5),
    ):
        assert horizon > 0, "Horizon must be greater than 0"

        self.id = id
        self.geometry = geometry

        self.avoid_obstacles = avoid_obstacles

        self.initial_state = np.array([*initial_position, initial_orientation])
        self.goal_state = np.array([*goal_position, goal_orientation])

        self.horizon = horizon

        self.time_step = planning_time_step

        self.linear_velocity_bounds: Tuple[float, float] = linear_velocity_bounds
        self.angular_velocity_bounds: Tuple[float, float] = angular_velocity_bounds
        self.linear_acceleration_bounds: Tuple[float, float] = (
            linear_acceleration_bounds
        )
        self.angular_acceleration_bounds: Tuple[float, float] = (
            angular_acceleration_bounds
        )

        self.left_right_lane_bounds: Tuple[float, float] = left_right_lane_bounds

        self.initial_linear_velocity: float = initial_linear_velocity
        self.initial_angular_velocity: float = initial_angular_velocity

        self.linear_velocity: float = self.initial_linear_velocity
        self.angular_velocity: float = self.initial_angular_velocity

        self.states_matrix = np.tile(self.initial_state, (self.horizon + 1, 1)).T
        self.controls_matrix = np.zeros((2, self.horizon))

        self.planner = MotionPlanner(time_step=self.time_step, horizon=self.horizon)

    @property
    def state(self):
        return self.states_matrix[:, 1]

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @property
    def at_goal(self):
        return self.geometry.calculate_distance(self.state, self.goal_state) - 1 <= 0

    def reset(self, matrices_only: bool = False):
        self.states_matrix = np.tile(self.initial_state, (self.horizon + 1, 1)).T
        self.controls_matrix = np.zeros((2, self.horizon))
        if not matrices_only:
            self.linear_velocity = self.initial_linear_velocity
            self.angular_velocity = self.initial_angular_velocity


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
        geometry: Circle = Circle(1),
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
            geometry=geometry,
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
