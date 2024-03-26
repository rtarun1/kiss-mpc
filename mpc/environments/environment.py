from typing import List

from matplotlib import pyplot as plt
from mpc.agents.ego import EgoAgent
from mpc.obstacles.obstacle import DynamicObstacle, StaticObstacle


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

        self.obstacle_plots = [
            axes.plot(
                obstacle.shadow_agent.states_matrix[0, 1:],
                obstacle.shadow_agent.states_matrix[1, 1:],
                marker=".",
                color="green",
            )[0]
            for obstacle in self.dynamic_obstacles
        ]

        goal_plot.set_data(self.agent.goal_state[0], self.agent.goal_state[1])

        self.recenter_plot()

    @property
    def obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def recenter_plot(self):
        # Center plot to agent
        axes = plt.gca()
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

        for obstacle_plot, obstacle in zip(self.obstacle_plots, self.dynamic_obstacles):
            obstacle_plot.set_data(
                obstacle.shadow_agent.states_matrix[0, 1:],
                obstacle.shadow_agent.states_matrix[1, 1:],
            )

    def close(self):
        plt.pause(2)
        plt.close()


class Environment:
    def __init__(
        self,
        agent: EgoAgent,
        static_obstacles: List[StaticObstacle],
        dynamic_obstacles: List[DynamicObstacle],
        plot=True,
    ):
        self.agent = agent
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.plot = plot

    @property
    def obstacles(self):
        return self.static_obstacles + self.dynamic_obstacles

    def step(self):
        self.agent.step(
            obstacles=[
                obstacle
                for obstacle in self.obstacles
                if obstacle.calculate_distance(self.agent.state)
                <= self.agent.sensor_radius
            ]
        )

        for obstacle in self.dynamic_obstacles:
            obstacle.step()

    def reset(self):
        self.agent.reset()

        for obstacle in self.dynamic_obstacles:
            obstacle.reset()

    def loop(self, max_timesteps: int = 10000):
        if self.plot:
            plotter = Plotter(
                agent=self.agent,
                static_obstacles=self.static_obstacles,
                dynamic_obstacles=self.dynamic_obstacles,
            )

        while (not self.agent.at_goal) and max_timesteps > 0:
            self.step()

            if self.plot:
                plotter.update_plot()

            max_timesteps -= 1
            print(self.agent.state)

        if self.plot:
            plotter.close()
