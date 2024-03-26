import numpy as np
from typing import Tuple
from mpc.planners import MotionPlanner
from mpc.geometries import Circle

from abc import ABC, abstractmethod


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
        sensor_radius: float = 50.0,
        avoid_obstacles: bool = True,
        linear_velocity_bounds: Tuple[float, float] = (0, 12),
        angular_velocity_bounds: Tuple[float, float] = (-np.pi / 4, np.pi / 4),
        linear_acceleration_bounds: Tuple[float, float] = (-50, 50),
        angular_acceleration_bounds: Tuple[float, float] = (-50, 50),
        left_right_lane_bounds: Tuple[float, float] = (-1000.5, 1000.5),
    ):
        assert horizon > 0, "Horizon must be greater than 0"

        self.id = id
        self.geometry = geometry
        self.sensor_radius = sensor_radius

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
        return self.geometry.calculate_distance(self.state, self.goal_state) <= 0

    def reset(self):
        self.states_matrix = np.tile(self.initial_state, (self.horizon + 1, 1)).T
        self.controls_matrix = np.zeros((2, self.horizon))
        self.linear_velocity = self.initial_linear_velocity
        self.angular_velocity = self.initial_angular_velocity
