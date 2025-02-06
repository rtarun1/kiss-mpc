import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class Geometry(ABC):
    @abstractmethod
    def create_patch(self) -> mpatches.Patch:
        raise NotImplementedError

    @abstractmethod
    def update_patch(self, patch: mpatches.Patch):
        raise NotImplementedError


class Polygon(Geometry):
    def __init__(self, vertices: List[Tuple]):
        super().__init__()
        self.vertices = np.array(vertices)

    def create_patch(self) -> mpatches.Polygon:
        return mpatches.Polygon(self.vertices, fill=False, color="black")

    def update_patch(self, patch: mpatches.Polygon):
        patch.set_xy(self.vertices)

    def from_rectangle(height: float, width: float, location: Tuple) -> "Polygon":
        return Polygon(
            [
                (location[0] - width / 2, location[1] - height / 2),
                (location[0] + width / 2, location[1] - height / 2),
                (location[0] + width / 2, location[1] + height / 2),
                (location[0] - width / 2, location[1] + height / 2),
            ]
        )


class Circle(Geometry):
    def __init__(self, center: Tuple[float, float], radius: float):
        super().__init__()
        self.radius = radius
        self.center = np.array(center, dtype=np.float64)

    def create_patch(self, color="black") -> mpatches.Circle:
        return mpatches.Circle(
            self.center, self.radius, fill=False, color=color, linestyle="--"
        )

    def update_patch(self, patch: mpatches.Circle):
        # Plot the circle
        patch.set_center(self.center)


@dataclass
class AgentEntity:
    geometry_patch: mpatches.Circle
    states_plot: Line2D


@dataclass
class AgentPlotData:
    id: int
    radius: float
    state: Tuple[float, float]
    future_states: np.ndarray


@dataclass
class ObstaclePlotData:
    id: int
    geometry: Geometry


class Plotter:
    def __init__(
        self,
        ego_agent_id: int,
        agents: List[AgentPlotData],
        obstacles: List[ObstaclePlotData],
        goals: List[Tuple[float, float]],
        video_path: Optional[Path] = None,
    ):
        self.video_path = video_path

        if self.video_path:
            os.makedirs(self.video_path, exist_ok=True)

        self.num_frames = 0

        self.PLOT_SIZE_DELTA = 10

        # Create the plot
        _, axes = plt.subplots()
        axes: plt.Axes

        axes.set_aspect("equal")

        self.ego_agent_id = ego_agent_id
        self.agents = {
            agent.id: AgentEntity(
                geometry_patch=Circle(
                    center=agent.state, radius=agent.radius
                ).create_patch(color="red" if agent.id == ego_agent_id else "black"),
                states_plot=axes.plot(
                    agent.future_states[0, 1:],
                    agent.future_states[1, 1:],
                    marker=".",
                    color="red" if agent.id == ego_agent_id else "grey",
                )[0],
            )
            for agent in agents
        }
        self.obstacles = {
            obstacle.id: axes.add_patch(obstacle.geometry.create_patch())
            for obstacle in obstacles
        }
        self.goals = goals
        self.goal_plots = [
            axes.plot(goal[0], goal[1], marker="x", color="green")[0] for goal in goals
        ]

    def reset_plot_limits(self):
        # Remove limits
        axes = plt.gca()
        axes.autoscale()

    def freeze_plot(self):
        plt.show()

    def recenter_plot(self, ego_agent_state: Tuple[float, float]):
        # Center plot to agent
        axes = plt.gca()
        axes.set_aspect("equal")
        axes.set_xlim(
            ego_agent_state[0] - self.PLOT_SIZE_DELTA,
            ego_agent_state[0] + self.PLOT_SIZE_DELTA,
        )
        axes.set_ylim(
            ego_agent_state[1] - self.PLOT_SIZE_DELTA,
            ego_agent_state[1] + self.PLOT_SIZE_DELTA,
        )

    def save_frame(self):
        # Save frame to video
        plt.gcf().savefig(self.video_path / f"frame_{(self.num_frames + 1):04d}.png")

    def update_plot(self, agent_updates: List[AgentPlotData]):
        # Plot using matplotlib
        plt.pause(0.01)

        for agent_update in agent_updates:
            agent = self.agents[agent_update.id]
            agent.geometry_patch.set_center(np.array(agent_update.state))
            agent.states_plot.set_data(
                agent_update.future_states[0, 1:], agent_update.future_states[1, 1:]
            )

            if agent_update.id == self.ego_agent_id:
                self.recenter_plot(agent_update.state)

        if self.video_path:
            self.save_frame()

        self.num_frames += 1

    def collapse_frames_to_video(self):
        frame_pattern = f"{self.video_path}/frame_%04d.png"
        video_path = f"{self.video_path}/video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-framerate",
                "20",
                "-i",
                frame_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
        )
        # Delete frames
        for file in os.listdir(self.video_path):
            if file.endswith(".png"):
                os.remove(os.path.join(self.video_path, file))

    def close(self):
        plt.pause(2)
        plt.close()
