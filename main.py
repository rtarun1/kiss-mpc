import numpy as np

from mpc.agent import EgoAgent
from mpc.environment import LocalEnvironment
from mpc.geometry import Circle, Ellipsoid, Rectangle
from mpc.obstacle import DynamicObstacle, StaticObstacle

agent1 = EgoAgent(
    id=1,
    initial_position=(5, -4),
    initial_orientation=np.pi / 2,
    goal_position=(2.5, 12.5),
    goal_orientation=np.pi / 2,
    horizon=20,
)

static_obstacle_rectangle = StaticObstacle(
    id=1,
    geometry=Rectangle(height=2, width=10),
    position=(-4, 10),
)
static_obstacle_rectangle_2 = StaticObstacle(
    id=2,
    geometry=Rectangle(height=2, width=10),
    position=(10, 10),
)

static_obstacle_ellipse = StaticObstacle(
    id=2,
    geometry=Ellipsoid.from_rectangle(Rectangle(height=3, width=10)),
    position=(5, 22),
)

static_obstacle_circle = StaticObstacle(
    id=3,
    geometry=Circle(radius=3),
    position=(5, 20),
)

dynamic_obstacle = DynamicObstacle(
    id=4,
    geometry=Circle(radius=1),
    position=(3, 20),
)

environment = LocalEnvironment(
    agent=agent1,
    static_obstacles=[static_obstacle_rectangle, static_obstacle_rectangle_2],
    dynamic_obstacles=[],
    # save_video=True,
)
environment.loop()
