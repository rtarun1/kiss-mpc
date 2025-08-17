import numpy as np
import time
from typing import TYPE_CHECKING, List, Tuple

import numpy as np


class ROSEnvironment:
    def __init__(self, agent, static_obstacles, dynamic_obstacles, waypoints):
        self.agent = agent
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles

        for obstacle in self.dynamic_obstacles:
            assert obstacle.horizon == agent.horizon

        self.waypoints = waypoints
        self.waypoint_index = 0
        self.agent.update_goal(self.current_waypoint)

        self.rollout_times = []

    @property
    def current_waypoint(self):
        return (
            self.waypoints[self.waypoint_index]
            if self.waypoint_index < len(self.waypoints)
            else None
        )

    @property
    def final_goal_reached(self):
        return self.waypoint_index == len(self.waypoints) - 1 and self.agent.at_goal

    @property
    def obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles
    
    def step(self):
        if self.waypoint_index == len(self.waypoints) - 1:
            print("heading to final goal")
            self.agent.goal_radius = 0.5

        else:
            self.agent.goal_radius = 0.5

        t1 = time.perf_counter()
        static_obstacles_dict = {
            obstacle.calculate_distance(self.agent.state): obstacle
            for obstacle in self.static_obstacles
        }
        filtered_static_obstacles = [
            static_obstacles_dict[distance]
            for distance in sorted(static_obstacles_dict.keys())
            if distance <= self.agent.sensor_radius
        ]
        dynamic_obstacles_dict = {
            obstacle.calculate_distance(self.agent.state): obstacle
            for obstacle in self.dynamic_obstacles
        }
        filtered_dynamic_obstacles = [
            dynamic_obstacles_dict[distance]
            for distance in sorted(dynamic_obstacles_dict.keys())
            if distance <= self.agent.sensor_radius
        ]

        print("Number of Dyn Obstacles:", len(filtered_dynamic_obstacles))

        self.agent.step(static_obstacles=filtered_static_obstacles, dynamic_obstacles=filtered_dynamic_obstacles)

        t2 = time.perf_counter
        print("Rollout Time:", t2 - t1)

        print("Current Waypoint", self.current_waypoint)
        print("Waypoints", self.waypoints)

        if self.agent.at_goal and not self.final_goal_reached:
            print("Reached waypoint", self.waypoint_index + 1)
            self.waypoint_index += 1
            self.agent.update_goal(self.current_waypoint)
    
    def reset(self):
        self.agent.reset()
        self.waypoint_index = 0
        self.agent.update_goal(self.current_waypoint)