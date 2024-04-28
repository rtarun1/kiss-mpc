import numpy as np

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import DynamicObstacle
from mpc.environment import LocalEnvironment
from mpc.geometry import Circle, Ellipsoid, Rectangle
from mpc.obstacle import StaticObstacle

agent = EgoAgent(
    id=1,
    initial_position=(8, -20),
    initial_orientation=np.pi / 2,
    horizon=30,
    use_warm_start=True,
)

static_obstacle_rectangle = StaticObstacle(
    id=1,
    geometry=Rectangle(height=2, width=8),
    position=(-4, 10),
)
static_obstacle_rectangle_2 = StaticObstacle(
    id=2,
    geometry=Rectangle(height=2, width=8),
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
    position=(13, 12.5),
    orientation=np.deg2rad(-90),
    goal_position=(3, -12.5),
    goal_orientation=np.deg2rad(90),
    horizon=30,
)

environment = LocalEnvironment(
    agent=agent,
    static_obstacles=[
        # static_obstacle_rectangle,
        # static_obstacle_rectangle_2,
        # static_obstacle_ellipse,
        # static_obstacle_circle,
        # static_obstacle_circle_2,
        # static_obstacle_circle_3,
    ],
    dynamic_obstacles=[
        dynamic_obstacle,
    ],
    waypoints=[((8, 15), np.deg2rad(90))],
    save_video=True,
)
environment.loop()
