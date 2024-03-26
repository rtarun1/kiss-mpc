from mpc.environments import Environment
from mpc.agents.ego import EgoAgent
from mpc.obstacles.obstacle import StaticObstacle, DynamicObstacle
from mpc.geometries import Circle, Rectangle, Ellipsoid

agent1 = EgoAgent(
    id=1,
    initial_position=(3, -4),
    initial_orientation=90,
    goal_position=(5, 50),
    goal_orientation=90,
    horizon=20,
)

static_obstacle = StaticObstacle(
    id=2,
    geometry=Rectangle(height=3, width=10),
    position=(5, 20),
)

dynamic_obstacle = DynamicObstacle(
    id=3,
    geometry=Circle(radius=1),
    position=(-6, 5),
    goal_position=(30, 40),
    horizon=20,
)

environment = Environment(
    agent=agent1,
    static_obstacles=[static_obstacle],
    dynamic_obstacles=[],
)
environment.loop()

# from agents.agent import Agent
# from planners.motionplanner import MotionPlanner
# import os
# import numpy as np
# import time
# import casadi as ca
# import matplotlib.pyplot as plt
# import copy
# import utils as utils
# from obstacles.obstacle import Obstacle
# from geometries.geometry import Circle, Rectangle

# np.set_printoptions(suppress=True)


# if __name__ == "__main__":
#     rec_video = False

#     exp_name = "Dynamic Obstacle Avoidance"
#     repo_path = os.path.abspath(os.path.dirname(__file__))
#     results_path = repo_path + "/results"
#     exp_path = results_path + "/" + exp_name
#     plt_dir = exp_path + "/tmp/"
#     os.makedirs(exp_path + "/tmp/", exist_ok=True)

#     timeout = 100

#     agent_v_ub = 12
#     agent_v_lb = 0

#     y_lane = np.arange(-1000, 1000)
#     x1_l_lane = 1.5 * np.ones(y_lane.shape)
#     x1_r_lane = 4.5 * np.ones(y_lane.shape)
#     x2_l_lane = -1.5 * np.ones(y_lane.shape)
#     x3_l_lane = -4.5 * np.ones(y_lane.shape)

#     draw_list = []

#     y_l_lim = -10
#     y_u_lim = 40

#     # Fixed target state
#     x_target = 5  # Replace with actual x-coordinate of the target
#     y_target = 50  # Replace with actual y-coordinate of the target
#     theta_target = np.deg2rad(90)  # Replace with actual orientation in radians

#     agent1 = Agent(
#         1, (3, -4, np.deg2rad(90)), (x_target, y_target, theta_target), horizon=30
#     )
#     agent1.linear_velocity_bounds = (0, 12)
#     agent1.linear_velocity = 5
#     draw_list.append(agent1)

#     obs = Obstacle(2, Rectangle(3, 18, 0, 3, 4), horizon=30)
#     # obs = Obstacle(2, Circle(3, 18, 0, 3), horizon=30)

#     # Initialize first obstacle
#     # obs = Agent(2, [3, 18, np.deg2rad(90)], [3, 18 + 30, np.deg2rad(90)], 30)
#     # obs = Agent(2, (-6, 5, np.deg2rad(90)), (30, 40, np.deg2rad(90)), horizon=30)
#     # obs.linear_velocity_bounds = (0, 8)
#     # obs.linear_velocity = 6
#     draw_list.append(obs)

#     # # Initialize new obstacle moving from goal position to agent's starting position
#     # obs2 = Agent(3, (7, 20, np.deg2rad(90)), (3, 20, np.deg2rad(90)), horizon=30)
#     # obs2.linear_velocity_bounds = (0, 8)
#     # obs2.linear_velocity = 6
#     # draw_list.append(obs2)

#     # # Initialize new obstacle moving from goal position to agent's starting position
#     # obs3 = Agent(4, (9, 5, np.deg2rad(90)), (5, 18 + 20, np.deg2rad(90)), horizon=30)
#     # obs3.linear_velocity_bounds = (0, 8)
#     # obs3.linear_velocity = 6
#     # draw_list.append(obs3)

#     # # Initialize new obstacle moving from goal position to agent's starting position
#     # obs4 = Agent(5, (9, 15, np.deg2rad(90)), (5 + 7, 15, np.deg2rad(90)), horizon=30)
#     # obs4.linear_velocity_bounds = (0, 8)
#     # obs4.linear_velocity = 6
#     # draw_list.append(obs4)

#     # # Initialize new obstacle moving from goal position to agent's starting position
#     # obs5 = Agent(6, (-5, 20, np.deg2rad(90)), (9, 50, np.deg2rad(90)), horizon=30)
#     # obs5.linear_velocity_bounds = (0, 7)
#     # obs5.linear_velocity = 2
#     # draw_list.append(obs5)

#     agent1.all_obstacles.append(obs)
#     # agent1.all_obstacles.append(obs2)
#     # agent1.all_obstacles.append(obs3)
#     # agent1.all_obstacles.append(obs4)
#     # agent1.all_obstacles.append(obs5)

#     agent1.avoid_obstacles = True

#     controller1 = MotionPlanner(agent1)
#     # obscontroller = MotionPlanner(obs)
#     # obscontroller2 = MotionPlanner(obs2)
#     # obscontroller3 = MotionPlanner(obs3)
#     # obscontroller4 = MotionPlanner(obs4)
#     # obscontroller5 = MotionPlanner(obs5)

#     # controller1.predict_controls()
#     controller1.solve()
#     # obscontroller.solve()
#     # obscontroller2.solve()
#     # # obscontroller3.solve()
#     # # obscontroller4.solve()
#     # obscontroller5.solve()

#     if rec_video:
#         plt_sv_dir = plt_dir
#         p = 0

#     # print(agent1.state, agent1.goal_state)

#     while (ca.norm_2(agent1.state - agent1.goal_state) >= 1) and timeout > 0:
#         timeout -= agent1.time_step

#         t1 = time.time()

#         # Predict controls for all agents and obstacles
#         controller1.solve()
#         # obscontroller.solve()
#         # obscontroller2.solve()
#         # obscontroller3.solve()
#         # obscontroller4.solve()
#         # obscontroller5.solve()

#         print(time.time() - t1)
#         print("#################")

#         agent1.update_velocities()
#         agent1.update_state()

#         # obs.update_velocities()
#         # obs.update_state()

#         # obs2.update_velocities()
#         # obs2.update_state()

#         # obs3.update_velocities()
#         # obs3.update_state()

#         # obs4.update_velocities()
#         # obs4.update_state()

#         # obs5.update_velocities()
#         # obs5.update_state()

#         utils.draw(draw_list)
#         # Commented out lane plotting
#         # plt.plot(x1_r_lane, y_lane, 'k', linewidth=1)
#         # plt.plot(x1_l_lane, y_lane, 'k', linewidth=1)
#         # plt.plot(x2_l_lane, y_lane, 'k', linewidth=1)
#         # plt.plot(x3_l_lane, y_lane, 'k', linewidth=1)

#         # Plot size
#         plt.xlim([-40, 40])
#         plt.ylim([y_l_lim, y_u_lim])

#         if rec_video:
#             plt.savefig(plt_sv_dir + str(p) + ".png", dpi=500, bbox_inches="tight")
#             p += 1
#             plt.clf()
#         else:
#             plt.pause(1e-10)
#             plt.clf()

#         y_l_lim = agent1.state[1] - 10
#         y_u_lim = agent1.state[1] + 40

#     plt.close()
