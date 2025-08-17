from abc import ABC, abstractmethod
import numpy as np
import time

from mpc.optimizer import MotionPlanner
from obstacle_handling.geometry import Circle

class Agent(ABC):
    def __init__(
        self,
        id,
        radius,
        initial_position,
        initial_orientation,
        planning_time_step,
        horizon,
        sensor_radius,
        avoid_obstacles,
        initial_linear_velocity,
        initial_angular_velocity,
        linear_velocity_bounds, 
        angular_velocity_bounds,
        state_bounds,
        goal_position=None,
        goal_orientation=None,
        waypoints=None,
        use_warm_start=False,        
    ):
        assert horizon > 0
        self.waypoints = waypoints
        self.waypoint_index = 0
        
        self.id = id
        self.sensor_radius = sensor_radius
        self.avoid_obstacles = avoid_obstacles
        self.geometry = Circle(center=initial_position, radius=radius)

        self.initial_state = np.array([*initial_position, initial_orientation])
        self.goal_state = (
            np.array([*goal_position, goal_orientation])
            if goal_position
            else self.initial_state
        )

        self.horizon = horizon

        self.time_step = planning_time_step

        self.linear_velocity_bounds = linear_velocity_bounds
        self.angular_velocity_bounds = angular_velocity_bounds
        self.state_bounds = state_bounds
        
        self.initial_linear_velocity = initial_linear_velocity
        self.initial_angular_velocity = initial_angular_velocity

        self.linear_velocity = self.initial_linear_velocity
        self.angular_velocity = self.initial_angular_velocity
       
        self.states_matrix = np.tile(self.initial_state, (self.horizon + 1, 1)).T
        self.controls_matrix = np.zeros((2, self.horizon))
        
        self.planner = MotionPlanner(time_step=self.time_step, horizon=self.horizon)
        
        self.use_warm_start = use_warm_start
        self.goal_radius = 0.5
        
    def update_goal(self, goal):
        self.goal_state = goal if (goal is not None) else self.initial_state

    @property
    def state(self):
        return self.states_matrix[:, 1]
    
    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    @property
    def at_goal(self):
        return self.geometry.calculate_distance(self.goal_state) - self.goal_radius <= 0

    def reset(self, matrices_only=False, to_initial_state=True):
        self.states_matrix = np.tile(
            (self.initial_state if to_initial_state else self.state),
            (self.horizon + 1, 1),
        ).T
        self.controls_matrix = np.zeros((2, self.horizon))
        if not matrices_only:
            self.linear_velocity = self.initial_linear_velocity
            self.angular_velocity = self.initial_angular_velocity

class EgoAgent(Agent):
    def __init__(
        self,
        id,
        radius,
        initial_position,
        initial_orientation,
        planning_time_step=0.041,
        horizon=50,
        sensor_radius=5,
        initial_linear_velocity=0,
        initial_angular_velocity=0,
        linear_velocity_bounds=(-0.2, 0.5), 
        angular_velocity_bounds=(-0.5, 0.5),
        state_bounds=(-20, 20),
        goal_position=None,
        goal_orientation=None,
        waypoints=None,
        use_warm_start=False,        
    ):
        
        super().__init__(
        id=id,
        radius=radius,
        initial_position=initial_position,
        initial_orientation=initial_orientation,
        planning_time_step=planning_time_step,
        horizon=horizon,
        sensor_radius=sensor_radius,
        avoid_obstacles=True,
        initial_linear_velocity=initial_linear_velocity,
        initial_angular_velocity=initial_angular_velocity,
        linear_velocity_bounds=linear_velocity_bounds, 
        angular_velocity_bounds=angular_velocity_bounds,
        state_bounds=state_bounds,
        goal_position=goal_position,
        use_warm_start=False,)

    def step(
        self,
        static_obstacles=[],
        dynamic_obstacles=[],
        state_override=False
    ):
        # if not self.use_warm_start:
        #     self.reset(matrices_only=True, to_initial_state=False)
               
        self.states_matrix, self.controls_matrix = self.planner.solve(
            current_state=self.state if not state_override else self.initial_state,
            current_linear_velocity=self.linear_velocity,
            current_angular_velocity=self.angular_velocity,
            goal_state=self.goal_state,
            states_matrix=self.states_matrix, 
            controls_matrix=self.controls_matrix,
            state_bounds=self.state_bounds,
            linear_velocity_bounds=self.linear_velocity_bounds,
            angular_velocity_bounds=self.angular_velocity_bounds,
            inflation_radius=(self.geometry.radius + 0.1),
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
        )       
        self.geometry.location = self.state[:2] if not state_override else self.initial_state[:2]
        self.linear_velocity = self.controls_matrix[0, 0]
        self.angular_velocity = self.controls_matrix[1, 0]