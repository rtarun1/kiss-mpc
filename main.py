import numpy as np

from mpc.agent import EgoAgent
from mpc.environment import LocalEnvironment
from mpc.geometry import Circle, Ellipsoid, Rectangle
from mpc.obstacle import DynamicObstacle, StaticObstacle

agent1 = EgoAgent(
    id=1,
    initial_position=(8, -20),
    initial_orientation=np.pi / 2,
    goal_position=(8, 15),
    goal_orientation=np.pi / 2,
    horizon=30,
    # angular_velocity_bounds=(-np.pi * 2, np.pi * 2),
    # left_right_lane_bounds=(-np.inf, np.inf),
    use_warm_start=True,
)

static_obstacle_rectangle = StaticObstacle(
    id=1,
    geometry=Rectangle(height=2, width=10),
    position=(-4, 10),
)
static_obstacle_rectangle_2 = StaticObstacle(
    id=2,
    geometry=Rectangle(height=2, width=10),
    position=(8, 10),
)

static_obstacle_ellipse = StaticObstacle(
    id=2,
    geometry=Ellipsoid.from_rectangle(Rectangle(height=2, width=8)),
    position=(8, 10),
)

static_obstacle_circle = StaticObstacle(
    id=3,
    geometry=Circle(radius=1),
    position=(6, 10),
)
static_obstacle_circle_2 = StaticObstacle(
    id=3,
    geometry=Circle(radius=1),
    position=(8, 10),
)
static_obstacle_circle_3 = StaticObstacle(
    id=3,
    geometry=Circle(radius=1),
    position=(10, 10),
)

dynamic_obstacle = DynamicObstacle(
    id=4,
    geometry=Circle(radius=1),
    position=(3, 20),
)

environment = LocalEnvironment(
    agent=agent1,
    static_obstacles=[
        # static_obstacle_rectangle_2,
        # static_obstacle_ellipse,
        static_obstacle_circle,
        static_obstacle_circle_2,
        static_obstacle_circle_3,
    ],
    dynamic_obstacles=[],
    # save_video=True,
)
environment.loop()
