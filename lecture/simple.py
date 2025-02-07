# %%
# %matplotlib ipympl
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, cast

import casadi as ca
import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# %%
# @title Geometry


class Geometry(ABC):
    @property
    @abstractmethod
    def location(self) -> Tuple:
        raise NotImplementedError

    @location.setter
    @abstractmethod
    def location(self, value: Tuple) -> None:
        raise NotImplementedError

    @abstractmethod
    def calculate_distance(
        self, distance_to: Tuple, custom_self_location: Tuple = None
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def calculate_symbolic_distance(
        self, distance_to: ca.MX, custom_self_location: Tuple = None
    ) -> ca.MX:
        raise NotImplementedError

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

    @property
    def location(self) -> Tuple:
        return tuple(np.mean(self.vertices, axis=0))

    @location.setter
    def location(self, value: Tuple) -> None:
        self.vertices += value - self.location

    def calculate_distance(
        self, distance_to: Tuple, custom_self_location: Tuple = None
    ) -> float:
        # Calculate the distance from the point (distance_to) to the polygon
        if custom_self_location is not None:
            a = self.vertices + custom_self_location - self.location
        else:
            a = self.vertices
        b = np.roll(a, -1, axis=0)
        edge = b - a
        v = np.array(distance_to[:2]) - a
        pq = (
            v
            - edge
            * np.clip(np.sum(v * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1)[
                :, None
            ]
        )
        distance = np.min(np.sum(pq**2, axis=1))

        v2 = distance_to[:2] - b
        val3 = np.roll(edge, 1, axis=1) * v
        val3 = val3[:, 1] - val3[:, 0]
        condition = np.stack([v[:, 1] >= 0, v2[:, 1] < 0, val3 > 0])
        not_condition = np.stack([v[:, 1] < 0, v2[:, 1] >= 0, val3 < 0])
        condition = np.all(np.all(condition, axis=0))
        not_condition = np.all(np.all(not_condition, axis=0))
        s = -1 if condition or not_condition else 1
        return np.sqrt(distance) * s

    def calculate_symbolic_distance(
        self, distance_to: ca.MX, custom_self_location: Tuple = None
    ) -> ca.MX:
        # Calculate the distance from the point (distance_to) to the polygon
        if custom_self_location is not None:
            a = self.vertices + custom_self_location - self.location
        else:
            a = self.vertices
        b = np.roll(a, -1, axis=0)
        edge = b - a
        v = ca.repmat(distance_to[:2].T, a.shape[0], 1) - a
        pq = v - edge * ca.fmin(ca.fmax(ca.sum2(v * edge) / ca.sum2(edge * edge), 0), 1)
        distance = ca.mmin(ca.sum2(pq**2))

        v2 = ca.repmat(distance_to[:2].T, b.shape[0], 1) - b
        val3 = np.roll(edge, 1, axis=1) * v
        val3 = val3[:, 1] - val3[:, 0]
        condition = ca.horzcat(v[:, 1] >= 0, v2[:, 1] < 0, val3 > 0)
        not_condition = ca.horzcat(v[:, 1] < 0, v2[:, 1] >= 0, val3 < 0)
        condition = ca.sum1(ca.sum2(condition))
        not_condition = ca.sum1(ca.sum2(not_condition))
        return ca.if_else(
            ca.eq(ca.sum1(ca.vertcat(condition, not_condition)), 1),
            ca.sqrt(distance) * -1,
            ca.sqrt(distance) * 1,
        )

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
    def __init__(self, center: Tuple, radius: float):
        super().__init__()
        self.radius = radius
        self.center = np.array(center, dtype=np.float64)

    @property
    def location(self) -> Tuple:
        return tuple(self.center)

    @location.setter
    def location(self, value: Tuple) -> None:
        self.center += np.array(value) - self.center

    def calculate_distance(
        self, distance_to: Tuple, custom_self_location: Tuple = None
    ) -> float:
        if custom_self_location is not None:
            center = np.array(custom_self_location)
        else:
            center = self.center
        return np.linalg.norm(np.array(distance_to[:2]) - center) - self.radius

    def calculate_symbolic_distance(
        self, distance_to: ca.MX, custom_self_location: Tuple = None
    ) -> ca.MX:
        if custom_self_location is not None:
            center = np.array(custom_self_location)
        else:
            center = self.center
        return (
            ca.sqrt(
                (distance_to[0] - center[0]) ** 2 + (distance_to[1] - center[1]) ** 2
            )
            - self.radius
        )

    def create_patch(self, color="black") -> mpatches.Circle:
        return mpatches.Circle(
            self.location, self.radius, fill=False, color=color, linestyle="--"
        )

    def update_patch(self, patch: mpatches.Circle):
        patch.set_center(self.location)


# %%
# @title Plotter


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
        video_path: Optional[str] = None,
    ):
        self.video_path = video_path

        if self.video_path:
            self.video_path = Path(self.video_path)
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
        # Add patches
        for agent in self.agents.values():
            axes.add_patch(agent.geometry_patch)
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


# %%
# @title Utilities
def MX_horzcat(*args: ca.MX) -> ca.MX:
    return ca.horzcat(*args)


def SX_horzcat(*args: ca.SX) -> ca.SX:
    return ca.horzcat(*args)


def DM_horzcat(*args: ca.DM) -> ca.DM:
    return ca.horzcat(*args)


def MX_vertcat(*args: ca.MX) -> ca.MX:
    return ca.vertcat(*args)


def SX_vertcat(*args: ca.SX) -> ca.SX:
    return ca.vertcat(*args)


def DM_vertcat(*args: ca.DM) -> ca.DM:
    return ca.vertcat(*args)


def create_symbolic_scalar(name: str, size: Optional[Tuple[int, int]] = None) -> ca.SX:
    return ca.SX.sym(name, size) if size else ca.SX.sym(name)


def create_symbolic_matrix(name: str, size: Optional[Tuple[int, int]] = None) -> ca.MX:
    return ca.MX.sym(name, size) if size else ca.MX.sym(name)


# %% [markdown]
# ## Model Predictive Control Equations
# $$
# \begin{aligned}
# & \underset{X}{\text{min}}& &f(X; P) = \sum_{t=1}^{N} \omega_x (x_t - x_g)^2 + \omega_y (y_t - y_g)^2 + \omega_\theta (\theta_t - \theta_g)^2 \\
# & \underset{U}{\text{min}}& &f(U) = \sum_{t=0}^{N-1} \omega_a a_t^2 + \omega_x \alpha_t^2 \\
# & \text{subject to :}& & x_0 - x_I = 0; \quad \text{and} \quad y_0 - y_I = 0; \quad \text{and} \quad \theta _0 - \theta _I = 0\\
# & & & \forall t \in \{1, \dots, N\}, \quad x_t - (x_{t-1} + (v_I + \sum_{k=1}^{t} a_{k-1}T) \cos(\theta_{t-1}) T) = 0\\
# & & & \forall t \in \{1, \dots, N\}, \quad y_t - (y_{t-1} + (v_I + \sum_{k=1}^{t} a_{k-1}T) \sin(\theta_{t-1}) T) = 0\\
# & & & \forall t \in \{1, \dots, N\}, \quad \theta _{t} - (\theta _{t-1} + (\omega _I + \sum_{k=1}^{t} \alpha _{k-1}T) T) = 0\\
# & & & \forall i \in \{0, \dots, O\}, \forall t \in \{1, \dots, N\}, \quad \text{dist}(x_t, o_i) \geq I\\
# & & & v_L \leq v_i + \sum_{k=1}^{t} a_{k-1}T \leq v_U\\
# & & & \omega _L \leq \omega _i + \sum_{k=1}^{t} \alpha _{k-1}T \leq \omega _U\\
# & & & \forall t \in \{0, \dots, N+1\}, \quad l_L \leq x_t \leq l_U\\
# & & & \forall t \in \{0, \dots, N\}, \quad u_L \leq a_t \leq u_U \quad \text{and} \quad \alpha _L \leq \alpha _t \leq \alpha _U\\
# & \text{where :}& & X = \{x_0, \dots, x_N, \quad y_0, \dots, y_N, \quad \theta _0, \dots, \theta_N\} \\
# & & & U = \{a_0, \dots, a_{N-1}, \quad \alpha_0, \dots, \alpha_{N-1}\}\\
# & & & P = \{x_I, y_I, \theta _I, x_G, y_G, \theta _G\}
# \end{aligned}
# $$


# %%
@dataclass
class PlanningData:
    time_step: float
    horizon: int

    def __post_init__(self):
        # States
        self.symbolic_states = ca.vertcat(
            create_symbolic_scalar("x"),
            create_symbolic_scalar("y"),
            create_symbolic_scalar("theta"),
        )
        self.num_states = self.symbolic_states.numel()

        # Controls
        self.symbolic_controls = ca.vertcat(
            create_symbolic_scalar("a"),
            create_symbolic_scalar("alpha"),
        )
        self.num_controls = self.symbolic_controls.numel()

        # Weight matrix for goal cost
        self.goal_weight_x = ca.DM(100)
        self.goal_weight_y = ca.DM(100)
        self.goal_weight_theta = ca.DM(0)
        # self.weight_matrix = ca.DM(ca.diagcat(100, 100, 0))

        self.linear_acceleration_weight = ca.DM(50)
        self.angular_acceleration_weight = ca.DM(30)

        # Matrix of states over the prediction horizon
        # (contains an extra column for the initial state)
        self.symbolic_states_matrix = ca.SX.sym(
            "X", (self.num_states, self.horizon + 1)
        )

        # Matrix of controls over the prediction horizon
        self.symbolic_controls_matrix = ca.SX.sym(
            "U", (self.num_controls, self.horizon)
        )

        # Initial state and Goal state vector
        self.symbolic_terminal_states_vector = ca.SX.sym(
            "P", (self.num_states + self.num_states, 1)
        )

        # Optimization variables
        self.symbolic_optimization_variables = ca.vertcat(
            self.symbolic_states_matrix.reshape((-1, 1)),
            self.symbolic_controls_matrix.reshape((-1, 1)),
        )

        print(self.symbolic_optimization_variables)


# %% [markdown]
# ## Objective Functions

# %% [markdown]
# ### Goal Reaching Objective Function

# %% [markdown]
# $$
# \begin{aligned}
# & \underset{X}{\text{min}}& &f(X; P) = \sum_{t=1}^{N} \omega_x (x_t - x_g)^2 + \omega_y (y_t - y_g)^2 + \omega_\theta (\theta_t - \theta_g)^2
# \end{aligned}
# $$


# %%
def goal_reaching_objective_function(planner_data: PlanningData) -> ca.SX:
    return sum(
        planner_data.goal_weight_x
        * (
            planner_data.symbolic_states_matrix[0, t]
            - planner_data.symbolic_terminal_states_vector[3]
        )
        ** 2
        + planner_data.goal_weight_y
        * (
            planner_data.symbolic_states_matrix[1, t]
            - planner_data.symbolic_terminal_states_vector[4]
        )
        ** 2
        + planner_data.goal_weight_theta
        * (
            planner_data.symbolic_states_matrix[2, t]
            - planner_data.symbolic_terminal_states_vector[5]
        )
        ** 2
        for t in range(1, planner_data.horizon + 1)
    )


# %% [markdown]
# ### Damping Objective Function

# %% [markdown]
# $$
# \begin{aligned}
# & \underset{U}{\text{min}}& &f(U) = \sum_{t=0}^{N-1} \omega_a a_t^2 + \omega_x \alpha_t^2 \\
# \end{aligned}
# $$


# %%
def damping_objective_function(planner_data: PlanningData) -> ca.MX:
    return sum(
        planner_data.linear_acceleration_weight
        * planner_data.symbolic_controls_matrix[0, t] ** 2
        + planner_data.angular_acceleration_weight
        * planner_data.symbolic_controls_matrix[1, t] ** 2
        for t in range(planner_data.horizon)
    )


# %% [markdown]
# ## Constraints

# %% [markdown]
# ### State Constraints

# %% [markdown]
# $$
# \begin{aligned}
# x_0 - x_I = 0; \quad \text{and} \quad y_0 - y_I = 0; \quad \text{and} \quad \theta _0 - \theta _I = 0\\
# \forall t \in \{1, \dots, N\}, \quad x_t - (x_{t-1} + (v_I + \sum_{k=1}^{t} a_{k-1}T) \cos(\theta_{t-1}) T) = 0\\
# \forall t \in \{1, \dots, N\}, \quad y_t - (y_{t-1} + (v_I + \sum_{k=1}^{t} a_{k-1}T) \sin(\theta_{t-1}) T) = 0\\
# \forall t \in \{1, \dots, N\}, \quad \theta _{t} - (\theta _{t-1} + (\omega _I + \sum_{k=1}^{t} \alpha _{k-1}T) T) = 0
# \end{aligned}
# $$


# %%
from itertools import chain


def state_constraints(
    planning_data: PlanningData,
    current_linear_velocity: float,
    current_angular_velocity: float,
) -> ca.SX:
    timestep_linear_velocity = current_linear_velocity
    timestep_angular_velocity = current_angular_velocity
    return ca.vertcat(
        *[
            planning_data.symbolic_states_matrix[:, 0]
            - planning_data.symbolic_terminal_states_vector[: planning_data.num_states],
        ]
        + list(
            chain.from_iterable(
                [
                    planning_data.symbolic_states_matrix[0, t + 1]
                    - (
                        planning_data.symbolic_states_matrix[0, t]
                        + (
                            (
                                timestep_linear_velocity := timestep_linear_velocity
                                + planning_data.symbolic_controls_matrix[0, t]
                                * planning_data.time_step
                            )
                            * ca.cos(planning_data.symbolic_states_matrix[2, t])
                            * planning_data.time_step
                        )
                    ),
                    planning_data.symbolic_states_matrix[1, t + 1]
                    - (
                        planning_data.symbolic_states_matrix[1, t]
                        + (
                            (
                                timestep_linear_velocity
                                + planning_data.symbolic_controls_matrix[0, t]
                                * planning_data.time_step
                            )
                            * ca.sin(planning_data.symbolic_states_matrix[2, t])
                            * planning_data.time_step
                        )
                    ),
                    planning_data.symbolic_states_matrix[2, t + 1]
                    - (
                        planning_data.symbolic_states_matrix[2, t]
                        + (
                            (
                                timestep_angular_velocity := timestep_angular_velocity
                                + planning_data.symbolic_controls_matrix[1, t]
                                * planning_data.time_step
                            )
                            * planning_data.time_step
                        )
                    ),
                ]
                for t in range(planning_data.horizon)
            )
        )
    )


def state_constraints_bounds(planning_data: PlanningData) -> Tuple[ca.DM, ca.DM]:
    zeros = ca.vertcat(
        *[
            ca.DM.zeros((planning_data.num_states, 1)),
        ]
        + [ca.DM(0) for _ in range(planning_data.horizon * planning_data.num_states)]
    )
    return (zeros, zeros)


# %% [markdown]
# ### Collision Constraints

# %% [markdown]
# $$
# \begin{aligned}
# \forall i \in \{0, \dots, O\}, \forall t \in \{1, \dots, N\}, \quad \text{dist}(x_t, o_i) \geq I\\
# \end{aligned}
# $$


# %%
def collision_constraints(
    planning_data: PlanningData, obstacles: List[Geometry]
) -> ca.MX:
    return ca.horzcat(
        *[
            ca.vertcat(
                *[
                    obstacle.calculate_symbolic_distance(
                        planning_data.symbolic_states_matrix[:2, time_step]
                    )
                    for time_step in range(
                        planning_data.symbolic_states_matrix.shape[1] - 1
                    )
                ]
            )
            for obstacle in obstacles
        ]
    )


def collision_constraints_bounds(
    planning_data: PlanningData, inflation_radius: float, num_obstacles: int
) -> Tuple[ca.DM, ca.DM]:
    constraints_lower_bound = ca.vertcat(
        *[ca.DM(inflation_radius) for _ in range(planning_data.horizon * num_obstacles)]
    )
    constraints_upper_bound = ca.vertcat(
        *[ca.DM(ca.inf) for _ in range(planning_data.horizon * num_obstacles)]
    )
    return constraints_lower_bound, constraints_upper_bound


# %% [markdown]
# ### Velocity Constraints

# %% [markdown]
# $$
# \begin{aligned}
# v_L \leq v_i + \sum_{k=1}^{t} a_{k-1}T \leq v_U
# \end{aligned}
# $$


# %%
def linear_velocity_constraints(
    planning_data: PlanningData, current_linear_velocity: float
) -> ca.SX:
    timestep_linear_velocity = current_linear_velocity
    return ca.vertcat(
        *[
            (
                timestep_linear_velocity := timestep_linear_velocity
                + planning_data.symbolic_controls_matrix[0, t] * planning_data.time_step
            )
            for t in range(planning_data.horizon)
        ]
    )


def linear_velocity_constraints_bounds(
    planning_data: PlanningData, linear_velocity_bounds: Tuple[float, float]
) -> Tuple[ca.DM, ca.DM]:
    constraints_lower_bound = ca.vertcat(
        *[ca.DM(linear_velocity_bounds[0]) for _ in range(planning_data.horizon)]
    )
    constraints_upper_bound = ca.vertcat(
        *[ca.DM(linear_velocity_bounds[1]) for _ in range(planning_data.horizon)]
    )

    return constraints_lower_bound, constraints_upper_bound


# %% [markdown]
# $$
# \begin{aligned}
# \omega _L \leq \omega _i + \sum_{k=1}^{t} \alpha _{k-1}T \leq \omega _U
# \end{aligned}
# $$


# %%
def angular_velocity_constraints(
    planning_data: PlanningData, current_angular_velocity: float
) -> ca.SX:
    timestep_angular_velocity = current_angular_velocity
    return ca.vertcat(
        *[
            (
                timestep_angular_velocity := timestep_angular_velocity
                + planning_data.symbolic_controls_matrix[1, t] * planning_data.time_step
            )
            for t in range(planning_data.horizon)
        ]
    )


def angular_velocity_constraints_bounds(
    planning_data: PlanningData, angular_velocity_bounds: Tuple[float, float]
) -> Tuple[ca.DM, ca.DM]:
    constraints_lower_bound = ca.vertcat(
        *[ca.DM(angular_velocity_bounds[0]) for _ in range(planning_data.horizon)]
    )
    constraints_upper_bound = ca.vertcat(
        *[ca.DM(angular_velocity_bounds[1]) for _ in range(planning_data.horizon)]
    )

    return constraints_lower_bound, constraints_upper_bound


# %% [markdown]
# ### Lane Constraints

# %% [markdown]
# $$
# \begin{aligned}
# \forall t \in \{0, \dots, N+1\}, \quad l_L \leq x_t \leq l_U\\
# \end{aligned}
# $$


# %%
def lane_constraints_bounds(
    planning_data: PlanningData, left_right_lane_bounds: Tuple[float, float]
) -> Tuple[ca.DM, ca.DM]:
    # lane_lower_bounds = ca.vertcat(
    #     *[left_right_lane_bounds[0] for _ in range(planning_data.horizon + 1)]
    #     + [-ca.inf for _ in range(planning_data.horizon + 1)]
    #     + [-ca.inf for _ in range(planning_data.horizon + 1)]
    # )
    # lane_upper_bounds = ca.vertcat(
    #     *[left_right_lane_bounds[1] for _ in range(planning_data.horizon + 1)]
    #     + [ca.inf for _ in range(planning_data.horizon + 1)]
    #     + [ca.inf for _ in range(planning_data.horizon + 1)]
    # )

    lane_lower_bounds = ca.repmat(
        DM_vertcat(
            left_right_lane_bounds[0],
            -ca.inf,
            -ca.inf,
        ),
        (1, planning_data.horizon + 1),
    )
    lane_upper_bounds = ca.repmat(
        DM_vertcat(
            left_right_lane_bounds[1],
            ca.inf,
            ca.inf,
        ),
        (1, planning_data.horizon + 1),
    )

    return lane_lower_bounds, lane_upper_bounds


# %% [markdown]
# ### Acceleration Constraints

# %% [markdown]
# $$
# \begin{aligned}
# \forall t \in \{0, \dots, N\}, \quad u_L \leq a_t \leq u_U \quad \text{and} \quad \alpha _L \leq \alpha _t \leq \alpha _U
# \end{aligned}
# $$


# %%
def acceleration_constraints_bounds(
    planning_data: PlanningData,
    linear_acceleration_bounds: Tuple[float, float],
    angular_acceleration_bounds: Tuple[float, float],
) -> Tuple[ca.DM, ca.DM]:
    # control_lower_bounds = ca.vertcat(
    #     *[
    #         ca.DM(linear_acceleration_bounds[0])
    #         for _ in range(planning_data.horizon)
    #     ] + [
    #         ca.DM(angular_acceleration_bounds[0])
    #         for _ in range(planning_data.horizon)
    #     ]
    # )
    # control_upper_bounds = ca.vertcat(
    #     *[
    #         ca.DM(linear_acceleration_bounds[1])
    #         for _ in range(planning_data.horizon)
    #     ] + [
    #         ca.DM(angular_acceleration_bounds[1])
    #         for _ in range(planning_data.horizon)
    #     ]
    # )
    control_lower_bounds = ca.repmat(
        DM_vertcat(
            linear_acceleration_bounds[0],
            angular_acceleration_bounds[0],
        ),
        (1, planning_data.horizon),
    )
    control_upper_bounds = ca.repmat(
        DM_vertcat(
            linear_acceleration_bounds[1],
            angular_acceleration_bounds[1],
        ),
        (1, planning_data.horizon),
    )

    return control_lower_bounds, control_upper_bounds


# %% [markdown]
# ## Putting It Together


# %%
@dataclass
class AgentData:
    id: int
    initial_position: Tuple[float, float]
    initial_orientation: float
    radius: float = 1
    planning_time_step: float = 0.8
    initial_linear_velocity: float = 0
    initial_angular_velocity: float = 0
    planning_horizon: int = 10
    sensor_radius: float = 3
    avoid_obstacles: bool = True
    linear_velocity_bounds: Tuple[float, float] = (0, 0.3)
    angular_velocity_bounds: Tuple[float, float] = (-0.5, 0.5)
    linear_acceleration_bounds: Tuple[float, float] = (-0.5, 0.5)
    angular_acceleration_bounds: Tuple[float, float] = (-1, 1)
    left_right_lane_bounds: Tuple[float, float] = (-10, 10)
    inflation_radius: float = 0.3
    goal_position: Tuple[float, float] = None
    goal_orientation: float = None

    def __post_init__(self):
        self.geometry = Circle(self.initial_position, self.radius)
        self.linear_velocity = self.initial_linear_velocity
        self.angular_velocity = self.initial_angular_velocity
        self.initial_state = np.array(
            [*self.initial_position, self.initial_orientation]
        )
        self.goal_state = (
            np.array([*self.goal_position, self.goal_orientation])
            if self.goal_position
            else self.initial_state
        )
        self.goal_radius = 0.5
        self.states_matrix = np.tile(
            self.initial_state, (self.planning_horizon + 1, 1)
        ).T
        self.controls_matrix = np.zeros((2, self.planning_horizon))
        self.planning_data = PlanningData(
            self.planning_time_step, self.planning_horizon
        )

    def to_plot_data(self) -> AgentPlotData:
        return AgentPlotData(
            id=self.id,
            radius=self.radius,
            state=(self.state[0], self.state[1]),
            future_states=self.states_matrix,
        )

    @property
    def state(self):
        return self.states_matrix[:, 1]

    @property
    def at_goal(self):
        return self.geometry.calculate_distance(self.goal_state) - self.goal_radius <= 0

    def reset(self, matrices_only: bool = False, to_initial_state: bool = True):
        self.states_matrix = np.tile(
            (self.initial_state if to_initial_state else self.state),
            (self.planning_horizon + 1, 1),
        ).T
        self.controls_matrix = np.zeros((2, self.planning_horizon))
        if not matrices_only:
            self.linear_velocity = self.initial_linear_velocity
            self.angular_velocity = self.initial_angular_velocity

    def update_goal(self, goal: np.ndarray):
        self.goal_state = goal if (goal is not None) else self.initial_state

    def update_state(self, states_matrix: np.ndarray, controls_matrix: np.ndarray):
        self.states_matrix, self.controls_matrix = states_matrix, controls_matrix
        self.geometry.location = self.state[:2]
        self.linear_velocity += self.controls_matrix[0, 0] * self.planning_time_step
        self.angular_velocity += self.controls_matrix[1, 0] * self.planning_time_step


# %%
def plan(
    agent_data: AgentData,
    obstacles: List[Geometry],
):
    planning_data = agent_data.planning_data

    # Problem Definition
    optimization_variables = ca.vertcat(
        planning_data.symbolic_states_matrix.reshape((-1, 1)),
        planning_data.symbolic_controls_matrix.reshape((-1, 1)),
    )

    objective_functions = goal_reaching_objective_function(
        planning_data
    ) + damping_objective_function(planning_data)

    constraints = ca.vertcat(
        state_constraints(
            planning_data,
            agent_data.linear_velocity,
            agent_data.angular_velocity,
        ).reshape((-1, 1)),
        linear_velocity_constraints(planning_data, agent_data.linear_velocity).reshape(
            (-1, 1)
        ),
        angular_velocity_constraints(
            planning_data, agent_data.angular_velocity
        ).reshape((-1, 1)),
        collision_constraints(planning_data, obstacles).reshape((-1, 1)),
    )

    non_linear_program = {
        "x": optimization_variables,
        "f": objective_functions,
        "g": constraints,
        "p": planning_data.symbolic_terminal_states_vector,
    }

    # Options
    solver_options = {
        "ipopt": {
            "max_iter": 2000,
            "print_level": 0,
            "acceptable_tol": 1e-8,
            "acceptable_obj_change_tol": 1e-6,
        },
        "print_time": 0,
    }

    solver = ca.nlpsol("solver", "ipopt", non_linear_program, solver_options)

    # Constraints Bounds
    constraints_bounds = [
        state_constraints_bounds(planning_data),
        linear_velocity_constraints_bounds(
            planning_data, agent_data.linear_velocity_bounds
        ),
        angular_velocity_constraints_bounds(
            planning_data, agent_data.angular_velocity_bounds
        ),
        collision_constraints_bounds(
            planning_data,
            inflation_radius=agent_data.radius + agent_data.inflation_radius,
            num_obstacles=len(obstacles),
        ),
    ]

    # Optimization Variable Bounds
    optimization_variable_bounds = [
        lane_constraints_bounds(planning_data, agent_data.left_right_lane_bounds),
        acceleration_constraints_bounds(
            planning_data,
            linear_acceleration_bounds=agent_data.linear_acceleration_bounds,
            angular_acceleration_bounds=agent_data.angular_acceleration_bounds,
        ),
    ]

    # Solution
    solution = solver(
        x0=ca.vertcat(
            ca.reshape(
                ca.DM(agent_data.states_matrix),
                (planning_data.num_states * (planning_data.horizon + 1), 1),
            ),
            ca.reshape(
                ca.DM(agent_data.controls_matrix),
                (planning_data.num_controls * planning_data.horizon, 1),
            ),
        ),
        lbx=ca.vertcat(
            *[
                lower_bound.reshape((-1, 1))
                for lower_bound, _ in optimization_variable_bounds
            ]
        ),
        ubx=ca.vertcat(
            *[
                upper_bound.reshape((-1, 1))
                for _, upper_bound in optimization_variable_bounds
            ]
        ),
        lbg=ca.vertcat(
            *[lower_bound.reshape((-1, 1)) for lower_bound, _ in constraints_bounds]
        ),
        ubg=ca.vertcat(
            *[upper_bound.reshape((-1, 1)) for _, upper_bound in constraints_bounds]
        ),
        p=ca.vertcat(ca.DM(agent_data.state), ca.DM(agent_data.goal_state)),
    )

    updated_states_matrix = ca.reshape(
        solution["x"][: planning_data.num_states * (planning_data.horizon + 1)],
        (planning_data.num_states, planning_data.horizon + 1),
    )
    updated_controls_matrix = ca.reshape(
        solution["x"][planning_data.num_states * (planning_data.horizon + 1) :],
        (planning_data.num_controls, planning_data.horizon),
    )

    return (
        np.array(updated_states_matrix.full()),
        np.array(updated_controls_matrix.full()),
    )


# %%
agent = AgentData(
    id=1,
    initial_position=(-16, -16),
    initial_orientation=np.deg2rad(90),
)

obstacle_agent = AgentData(
    id=2,
    initial_position=(-5, -2),
    initial_orientation=np.deg2rad(-90),
    goal_position=(-5, -10),
    goal_orientation=np.deg2rad(-90),
)

walls = [
    Polygon.from_rectangle(height=1, width=39, location=(0, -20)),
    Polygon.from_rectangle(height=1, width=39, location=(0, 20)),
    Polygon.from_rectangle(height=39, width=1, location=(-20, 0)),
    Polygon.from_rectangle(height=39, width=1, location=(20, 0)),
    Polygon.from_rectangle(height=9, width=1, location=(-12, -15)),
    Polygon.from_rectangle(height=1, width=13, location=(-13, -5)),
    Polygon.from_rectangle(height=25, width=1, location=(1, -7)),
]

circles = [
    Circle(center=(1, 7), radius=1),
    Circle(center=(1, 14), radius=1),
    Circle(center=(1, 18), radius=1),
]

obstacles: List[Geometry] = [*walls, *circles]

waypoints = [
    (-2, -2, np.deg2rad(90)),
    # (-2, 10, np.deg2rad(90)),
    # (10, 5, np.deg2rad(90)),
]

video_path = "videos/"

plotter = Plotter(
    ego_agent_id=1,
    agents=[agent.to_plot_data(), obstacle_agent.to_plot_data()],
    obstacles=[
        ObstaclePlotData(id=index + 1, geometry=obstacle)
        for index, obstacle in enumerate(obstacles)
    ],
    goals=[(waypoint[0], waypoint[1]) for waypoint in waypoints],
    # video_path=video_path,
)

# %%
final_goal_reached = False
max_timesteps = 10000
waypoint_index = 0
agent.update_goal(waypoints[waypoint_index])

while max_timesteps > 0:
    print(f"Time step: {10000 - max_timesteps}")
    ego_obstacles = [
        obstacle
        for obstacle in obstacles
        if obstacle.calculate_distance(agent.state) <= agent.sensor_radius
    ] + [obstacle_agent.geometry]
    ego_matrices = plan(agent, obstacles=ego_obstacles)
    agent.update_state(*ego_matrices)

    obstacle_matrices = plan(obstacle_agent, obstacles=[])
    obstacle_agent.update_state(*obstacle_matrices)

    plotter.update_plot(
        agent_updates=[agent.to_plot_data(), obstacle_agent.to_plot_data()]
    )

    if agent.at_goal:
        waypoint_index += 1
        if waypoint_index >= len(waypoints):
            break
        agent.update_goal(waypoints[waypoint_index])
    max_timesteps -= 1

plotter.close()
# plotter.collapse_frames_to_video()

# %%
# #@title View Video
# from IPython.display import HTML
# from base64 import b64encode
# mp4 = open(f'{video_path}/video.mp4','rb').read()
# data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
# HTML("""
# <video width=400 controls>
#       <source src="%s" type="video/mp4">
# </video>
# """ % data_url)
