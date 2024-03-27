from typing import List

from mpc.agent import EgoAgent
from mpc.obstacle import DynamicObstacle, StaticObstacle
from mpc.plotter import Plotter


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
