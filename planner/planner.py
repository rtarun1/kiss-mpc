import casadi as ca
from agents.agent import Agent
from typing import cast, Callable, Any, List, Tuple, Optional
import numpy as np
import copy


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


def create_symbolic_scalar(
    name: str, size: Optional[Tuple[int, int] | int] = None
) -> ca.SX:
    return ca.SX.sym(name, size) if size else ca.SX.sym(name)


def create_symbolic_matrix(name: str, size: Optional[Tuple[int, int]] = None) -> ca.MX:
    return ca.MX.sym(name, size) if size else ca.MX.sym(name)


class MotionPlanner:
    def __init__(self, agent: Agent):
        self.agent = agent

        # States
        self.symbolic_states = SX_vertcat(
            position_x := create_symbolic_scalar("x"),
            position_y := create_symbolic_scalar("y"),
            heading := create_symbolic_scalar("theta"),
        )
        self.num_states = cast(int, self.symbolic_states.numel())

        # Controls
        self.symbolic_controls = SX_vertcat(
            linear_acceleration := create_symbolic_scalar("a"),
            angular_acceleration := create_symbolic_scalar("alpha"),
        )
        self.num_controls = cast(int, self.symbolic_controls.numel())

        # Weight matrix for state error
        self.weight_matrix = ca.DM(ca.diagcat(Q_x := 100, Q_y := 100, Q_theta := 500))

        # Obstacle cost weight
        self.obstacle_cost_weight = 100

        # Matrix of states over the prediction horizon
        # (contains an extra column for the initial state)
        self.symbolic_states_matrix = create_symbolic_matrix(
            "X", (self.num_states, self.agent.horizon + 1)
        )

        # Matrix of controls over the prediction horizon
        self.symbolic_controls_matrix = create_symbolic_matrix(
            "U", (self.num_controls, self.agent.horizon)
        )

        # Initial state and Goal state vector
        self.symbolic_terminal_states_vector = create_symbolic_matrix(
            "P", (self.num_states + self.num_states, 1)
        )

        # Optimization variables
        self.symbolic_optimization_variables = MX_vertcat(
            self.symbolic_states_matrix.reshape((-1, 1)),
            self.symbolic_controls_matrix.reshape((-1, 1)),
        )

    @property
    def symbolic_goal_cost(self) -> ca.MX:
        error = cast(
            ca.MX,
            self.symbolic_states_matrix[:, 1:-1]
            - self.symbolic_terminal_states_vector[self.num_states :],
        )
        cost = cast(ca.MX, cast(ca.MX, (error.T @ self.weight_matrix)).T * error)
        return cast(
            ca.MX,
            ca.sum2(ca.sum1(cost)),
        )

    @property
    def symbolic_angular_acceleration_cost(self) -> ca.SX:
        squared_angular_acceleration = cast(
            ca.MX, self.symbolic_controls_matrix[1, :] ** 2
        )
        return cast(
            ca.MX,
            ca.sum2(ca.sum1(squared_angular_acceleration)),
        )

    @property
    def symbolic_costs(self) -> ca.MX:
        return self.symbolic_goal_cost + self.symbolic_angular_acceleration_cost

    # @property
    # def lane_cost(self, lane_bounds_x: ca.SX):
    #     cost = 0
    #     for timestep in range(self.agent.horizon+1):
    #         state = self.symbolic_states_matrix[0, timestep]
    #         cost += (state - lane_bounds_x)**2
    #     return cost

    @property
    def lane_bounds(self) -> Tuple[ca.DM, ca.DM]:
        lane_lower_bounds = cast(
            ca.DM,
            ca.repmat(
                DM_vertcat(
                    self.agent.left_right_lane_bounds[0] + self.agent.radius,
                    -ca.inf,
                    -ca.inf,
                ),
                (1, self.agent.horizon + 1),
            ),
        )
        lane_upper_bounds = cast(
            ca.DM,
            ca.repmat(
                DM_vertcat(
                    self.agent.left_right_lane_bounds[1] - self.agent.radius,
                    ca.inf,
                    ca.inf,
                ),
                (1, self.agent.horizon + 1),
            ),
        )

        return lane_lower_bounds, lane_upper_bounds

    @property
    def control_bounds(self) -> Tuple[ca.DM, ca.DM]:
        control_lower_bounds = cast(
            ca.DM,
            ca.repmat(
                DM_vertcat(
                    self.agent.linear_acceleration_bounds[0],
                    self.agent.angular_acceleration_bounds[0],
                ),
                (1, self.agent.horizon),
            ),
        )
        control_upper_bounds = cast(
            ca.DM,
            ca.repmat(
                DM_vertcat(
                    self.agent.linear_acceleration_bounds[1],
                    self.agent.angular_acceleration_bounds[1],
                ),
                (1, self.agent.horizon),
            ),
        )

        return control_lower_bounds, control_upper_bounds

    @property
    def optimization_variable_bounds(self) -> Tuple[ca.DM, ca.DM]:
        lower_lane_bounds, upper_lane_bounds = self.lane_bounds
        lower_control_bounds, upper_control_bounds = self.control_bounds

        optimization_variable_lower_bounds = DM_vertcat(
            lower_lane_bounds.reshape((-1, 1)), lower_control_bounds.reshape((-1, 1))
        )
        optimization_variable_upper_bounds = DM_vertcat(
            upper_lane_bounds.reshape((-1, 1)), upper_control_bounds.reshape((-1, 1))
        )

        return optimization_variable_lower_bounds, optimization_variable_upper_bounds

    @property
    def state_constraints_bounds(self) -> Tuple[ca.DM, ca.DM]:
        return (
            cast(ca.DM, ca.DM.zeros((3, self.agent.horizon + 1))),
            cast(ca.DM, ca.DM.zeros((3, self.agent.horizon + 1))),
        )

    @property
    def symbolic_states_constraints(self) -> ca.MX:
        current_velocities = cast(
            ca.MX,
            self.symbolic_controls_matrix * ca.DM(self.agent.time_step)
            + DM_horzcat(
                DM_vertcat(
                    ca.DM(self.agent.linear_velocity),
                    ca.DM(self.agent.angular_velocity),
                ),
                ca.MX.zeros((2, self.agent.horizon - 1)),
            ),
        )
        # current_velocities[:, 0] = current_velocities[:, 0] + DM_vertcat(
        #     ca.DM(self.agent.linear_velocity),
        #     ca.DM(self.agent.angular_velocity),
        # )
        current_velocities = cast(
            ca.MX,
            ca.cumsum(
                current_velocities,
                1,
            ),
        )
        next_states_bounds = MX_vertcat(
            self.symbolic_states_matrix[0, :-1]
            + (
                current_velocities[0, :]
                * ca.cos(self.symbolic_states_matrix[2, :-1])
                * ca.DM(self.agent.time_step)
            ),
            self.symbolic_states_matrix[1, :-1]
            + (
                current_velocities[0, :]
                * ca.sin(self.symbolic_states_matrix[2, :-1])
                * ca.DM(self.agent.time_step)
            ),
            self.symbolic_states_matrix[2, :-1]
            + (current_velocities[1, :] * ca.DM(self.agent.time_step)),
        )
        return MX_horzcat(
            self.symbolic_states_matrix[:, 0]
            - self.symbolic_terminal_states_vector[: self.num_states],
            self.symbolic_states_matrix[:, 1:] - next_states_bounds,
        )

    @property
    def symbolic_obstacle_constraints(self) -> ca.MX:
        return MX_horzcat(
            *[
                obstacle.calculate_symbolic_distance(
                    symbolic_states_matrix=self.symbolic_states_matrix[:, 1:]
                )
                for obstacle in self.agent.visible_obstacles
            ]
        )

    @property
    def obstacle_constraints_bounds(self) -> Tuple[ca.DM, ca.DM]:
        constraints_lower_bound = cast(
            ca.DM,
            ca.repmat(
                ca.DM(2 * self.agent.radius + 0.5),
                (1, self.agent.horizon * len(self.agent.visible_obstacles)),
            ),
        )
        constraints_upper_bound = cast(
            ca.DM,
            ca.repmat(
                ca.DM(ca.inf),
                (1, self.agent.horizon * len(self.agent.visible_obstacles)),
            ),
        )
        return constraints_lower_bound, constraints_upper_bound

    @property
    def symbolic_linear_velocity_constraints(self) -> ca.MX:
        current_velocities = cast(
            ca.MX,
            self.symbolic_controls_matrix[0, :] * ca.DM(self.agent.time_step)
            + ca.horzcat(
                ca.DM(self.agent.linear_velocity),
                ca.MX.zeros((1, self.agent.horizon - 1)),
            ),
        )
        return cast(
            ca.MX,
            ca.cumsum(
                current_velocities,
                1,
            ),
        )

    @property
    def linear_velocity_constraints_bounds(self) -> Tuple[ca.DM, ca.DM]:
        constraints_lower_bound = cast(
            ca.DM,
            ca.repmat(
                ca.DM(self.agent.linear_velocity_bounds[0]), (1, self.agent.horizon)
            ),
        )
        constraints_upper_bound = cast(
            ca.DM,
            ca.repmat(
                ca.DM(self.agent.linear_velocity_bounds[1]), (1, self.agent.horizon)
            ),
        )

        return constraints_lower_bound, constraints_upper_bound

    @property
    def symbolic_angular_velocity_constraints(self) -> ca.MX:
        current_velocities = cast(
            ca.MX,
            self.symbolic_controls_matrix[1, :] * ca.DM(self.agent.time_step)
            + ca.horzcat(
                ca.DM(self.agent.angular_velocity),
                ca.MX.zeros((1, self.agent.horizon - 1)),
            ),
        )
        return cast(
            ca.MX,
            ca.cumsum(
                current_velocities,
                1,
            ),
        )

    @property
    def angular_velocity_constraints_bounds(self) -> Tuple[ca.DM, ca.DM]:
        constraints_lower_bound = cast(
            ca.DM,
            ca.repmat(
                ca.DM(self.agent.angular_velocity_bounds[0]), (1, self.agent.horizon)
            ),
        )
        constraints_upper_bound = cast(
            ca.DM,
            ca.repmat(
                ca.DM(self.agent.angular_velocity_bounds[1]), (1, self.agent.horizon)
            ),
        )

        return constraints_lower_bound, constraints_upper_bound

    @property
    def symbolic_constraints(self) -> ca.MX:
        return MX_vertcat(
            self.symbolic_states_constraints.reshape((-1, 1)),
            self.symbolic_linear_velocity_constraints.reshape((-1, 1)),
            self.symbolic_angular_velocity_constraints.reshape((-1, 1)),
            self.symbolic_obstacle_constraints.reshape((-1, 1)),
        )

    @property
    def constraints_bounds(self) -> Tuple[ca.DM, ca.DM]:
        (
            state_constraints_lower_bound,
            state_constraints_upper_bound,
        ) = self.state_constraints_bounds

        (
            linear_velocity_constraints_lower_bound,
            linear_velocity_constraints_upper_bound,
        ) = self.linear_velocity_constraints_bounds

        (
            angular_velocity_constraints_lower_bound,
            angular_velocity_constraints_upper_bound,
        ) = self.angular_velocity_constraints_bounds

        (
            obstacle_constraints_lower_bound,
            obstacle_constraints_upper_bound,
        ) = self.obstacle_constraints_bounds

        constraints_lower_bound = DM_vertcat(
            state_constraints_lower_bound.reshape((-1, 1)),
            linear_velocity_constraints_lower_bound.reshape((-1, 1)),
            angular_velocity_constraints_lower_bound.reshape((-1, 1)),
            obstacle_constraints_lower_bound.reshape((-1, 1)),
        )
        constraints_upper_bound = DM_vertcat(
            state_constraints_upper_bound.reshape((-1, 1)),
            linear_velocity_constraints_upper_bound.reshape((-1, 1)),
            angular_velocity_constraints_upper_bound.reshape((-1, 1)),
            obstacle_constraints_upper_bound.reshape((-1, 1)),
        )

        return constraints_lower_bound, constraints_upper_bound

    def solve(self):
        (
            constraints_lower_bounds,
            constraints_upper_bounds,
        ) = self.constraints_bounds

        (
            optimization_variable_lower_bounds,
            optimization_variable_upper_bounds,
        ) = self.optimization_variable_bounds

        non_linear_program = {
            "x": self.symbolic_optimization_variables,
            "f": self.symbolic_costs,
            "g": self.symbolic_constraints,
            "p": self.symbolic_terminal_states_vector,
        }

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

        solution = solver(
            x0=DM_vertcat(
                ca.reshape(
                    ca.DM(self.agent.states_matrix),
                    (self.num_states * (self.agent.horizon + 1), 1),
                ),
                ca.reshape(
                    ca.DM(self.agent.controls_matrix),
                    (self.num_controls * self.agent.horizon, 1),
                ),
            ),
            lbx=optimization_variable_lower_bounds,
            ubx=optimization_variable_upper_bounds,
            lbg=constraints_lower_bounds,
            ubg=constraints_upper_bounds,
            p=DM_vertcat(ca.DM(self.agent.state), ca.DM(self.agent.goal_state)),
        )

        updated_states_matrix = cast(
            ca.DM,
            ca.reshape(
                solution["x"][: self.num_states * (self.agent.horizon + 1)],
                (self.num_states, self.agent.horizon + 1),
            ),
        )
        updated_controls_matrix = cast(
            ca.DM,
            ca.reshape(
                solution["x"][self.num_states * (self.agent.horizon + 1) :],
                (self.num_controls, self.agent.horizon),
            ),
        )
        self.agent.states_matrix = np.array(updated_states_matrix.full())
        self.agent.controls_matrix = np.array(updated_controls_matrix.full())
