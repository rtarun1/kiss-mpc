import numpy as np

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import SimulatedDynamicObstacle
from mpc.environment import LocalEnvironment
from mpc.geometry import Circle
from mpc.obstacle import StaticObstacle

agent = EgoAgent(
    id=1,
    radius=1,
    initial_position=(-16, -16),
    initial_orientation=np.deg2rad(90),
    horizon=10,
    use_warm_start=True,
    planning_time_step=0.8,
    linear_velocity_bounds=(0, 0.3),
    angular_velocity_bounds=(-0.5, 0.5),
    linear_acceleration_bounds=(-0.5, 0.5),
    angular_acceleration_bounds=(-1, 1),
    sensor_radius=5,
)

walls = [
    # Polygon.from_rectangle(height=1, width=39, location=(0, -20)),
    # Polygon.from_rectangle(height=1, width=39, location=(0, 20)),
    # Polygon.from_rectangle(height=39, width=1, location=(-20, 0)),
    # Polygon.from_rectangle(height=39, width=1, location=(20, 0)),
    # Polygon.from_rectangle(height=9, width=1, location=(-12, -15)),
    # Polygon.from_rectangle(height=1, width=13, location=(-13, -5)),
    # Polygon.from_rectangle(height=25, width=1, location=(1, -7)),
]

# circles = [
#     Circle(center=(1, 7), radius=1),
#     Circle(center=(1, 14), radius=1),
#     Circle(center=(1, 18), radius=1),
# ]

# Add 500 static obstacles representing walls
circles = []
circles += [Circle(center=(-20, j), radius=0.1) for j in np.arange(-20, 20, 40 / 500)]
circles += [Circle(center=(1, j), radius=0.1) for j in np.arange(-20, 5, 25 / 100)]

polygon_obstacles = [
    StaticObstacle(id=i, geometry=polygon) for i, polygon in enumerate(walls)
]

obstacle_geometries = circles + polygon_obstacles
static_obstacles = [
    StaticObstacle(id=i, geometry=geometry)
    for i, geometry in enumerate(obstacle_geometries)
]

dynamic_obstacle = SimulatedDynamicObstacle(
    id=1,
    position=(-4, -2),
    orientation=np.deg2rad(-90),
    goal_position=(-4, -10),
    goal_orientation=np.deg2rad(-90),
    horizon=10,
)

dynamic_obstacles = [
    dynamic_obstacle,
]

# All static obstacles must have same radius
assert all(
    isinstance(obstacle.geometry, Circle)
    and obstacle.geometry.radius == obstacle_geometries[0].radius
    for obstacle in obstacle_geometries
), "All static obstacles must have the same radius"

# All dynamic obstacles must have same radius
assert all(
    isinstance(obstacle.geometry, Circle)
    and obstacle.geometry.radius == dynamic_obstacles[0].geometry.radius
    for obstacle in dynamic_obstacles
), "All dynamic obstacles must have the same radius"


environment = LocalEnvironment(
    agent=agent,
    static_obstacles=static_obstacles,
    dynamic_obstacles=dynamic_obstacles,
    waypoints=[
        (-2, -2, np.deg2rad(90)),
        (-2, 10, np.deg2rad(90)),
        (10, 5, np.deg2rad(90)),
    ],
    # save_video=True,
)
# environment.view_environment()
environment.loop()
