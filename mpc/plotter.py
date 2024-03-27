from typing import List

import numpy as np
from matplotlib import pyplot as plt

from mpc.agent import EgoAgent
from mpc.obstacle import DynamicObstacle, StaticObstacle


class Plotter:
    def __init__(
        self,
        agent: EgoAgent,
        static_obstacles: List[StaticObstacle],
        dynamic_obstacles: List[DynamicObstacle],
    ):
        self.agent = agent
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles

        # Create the plot
        _, axes = plt.subplots()
        axes: plt.Axes

        axes.add_patch(self.agent.geometry.patch)
        agent.geometry.update_patch(self.agent.state)

        # add agent id to plot
        self.agent_id = axes.text(
            self.agent.state[0],
            self.agent.state[1],
            f"Agent {self.agent.id}",
            fontsize=12,
            color="black",
        )

        for obstacle in self.obstacles:
            axes.add_patch(obstacle.geometry.patch)
            obstacle.geometry.update_patch(obstacle.state)

        self.static_obstacle_ids = [
            axes.text(
                obstacle.state[0],
                obstacle.state[1],
                f"Obstacle {obstacle.id}",
                fontsize=12,
                color="black",
            )
            for obstacle in self.static_obstacles
        ]

        self.dynamic_obstacle_ids = [
            axes.text(
                obstacle.state[0],
                obstacle.state[1],
                f"Obstacle {obstacle.id}",
                fontsize=12,
                color="black",
            )
            for obstacle in self.dynamic_obstacles
        ]

        goal_plot = axes.plot(
            self.agent.goal_state[0], self.agent.goal_state[1], marker="x", color="r"
        )[0]

        self.states_plot = axes.plot(
            self.agent.states_matrix[0, 1:],
            self.agent.states_matrix[1, 1:],
            marker=".",
            color="blue",
            # s=1.5,
        )[0]

        # plot velocity arrows for dynamic obstacles
        self.dynamic_obstacle_plots = [
            axes.arrow(
                obstacle.state[0],
                obstacle.state[1],
                obstacle.linear_velocity * np.cos(obstacle.state[2]),
                obstacle.linear_velocity * np.sin(obstacle.state[2]),
                head_width=0.2,
                head_length=0.2,
                fc="g",
                ec="g",
            )
            for obstacle in self.dynamic_obstacles
        ]

        # self.obstacle_plots = [
        #     axes.plot(
        #         obstacle.shadow_agent.states_matrix[0, 1:],
        #         obstacle.shadow_agent.states_matrix[1, 1:],
        #         marker=".",
        #         color="green",
        #         # s=1.5,
        #     )[0]
        #     for obstacle in self.dynamic_obstacles
        # ]

        goal_plot.set_data(self.agent.goal_state[0], self.agent.goal_state[1])

        self.recenter_plot()

    @property
    def obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def recenter_plot(self):
        # Center plot to agent
        axes = plt.gca()
        print(axes.get_children())
        axes.set_xlim(self.agent.state[0] - 10, self.agent.state[0] + 10)
        axes.set_ylim(self.agent.state[1] - 10, self.agent.state[1] + 10)

    def update_plot(self):
        # Plot using matplotlib
        plt.pause(0.01)

        self.recenter_plot()

        self.agent.geometry.update_patch(self.agent.state)

        self.agent_id.set_position((self.agent.state[0], self.agent.state[1]))

        for index, obstacle in enumerate(self.dynamic_obstacles):
            self.dynamic_obstacle_ids[index].set_position(
                (obstacle.state[0], obstacle.state[1])
            )
            obstacle.geometry.update_patch(obstacle.state)

        self.states_plot.set_data(
            self.agent.states_matrix[0, 1:], self.agent.states_matrix[1, 1:]
        )

        for obstacle_plot, obstacle in zip(
            self.dynamic_obstacle_plots, self.dynamic_obstacles
        ):
            obstacle_plot.set_data(
                x=obstacle.state[0],
                y=obstacle.state[1],
                dx=obstacle.linear_velocity * np.cos(obstacle.state[2]),
                dy=obstacle.linear_velocity * np.sin(obstacle.state[2]),
            )

    def close(self):
        plt.pause(2)
        plt.close()
