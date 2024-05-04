#!/usr/bin/env python3
from typing import List, cast

import numpy as np
import rospy
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Point32, PoseWithCovariance, Twist
from nav_msgs.msg import Odometry, Path
from people_msgs.msg import People, Person
from tf.transformations import euler_from_quaternion

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import DynamicObstacle
from mpc.environment import ROSEnvironment
from mpc.geometry import Circle, Polygon
from mpc.obstacle import StaticObstacle


class ROSInterface:
    """
    ROSInterface class to interface with ROS
    Creates a node and subscribes to people messages for obstacles, and publishes commands on the /cmd_vel topic
    Also subscribes to waypoint pose messages for the next goal
    """

    def __init__(self):
        self.environment = ROSEnvironment(
            agent=EgoAgent(
                id=1,
                radius=0.3,
                initial_position=(0, 0),
                initial_orientation=np.deg2rad(90),
                horizon=20,
                use_warm_start=True,
                planning_time_step=0.2,
                linear_velocity_bounds=(-0.26, 0.26),
                angular_velocity_bounds=(-1.82, 1.82),
                linear_acceleration_bounds=(-0.1, 0.1),
                angular_acceleration_bounds=(-0.1, 0.1),
                sensor_radius=20,
            ),
            static_obstacles=[],
            dynamic_obstacles=[],
            waypoints=[],
            plot=False,
        )

        rospy.init_node("ros_mpc_interface")

        # rospy.Subscriber("/people", People, self.people_callback)
        rospy.Subscriber(
            "/move_base/GlobalPlanner/plan",
            Path,
            self.waypoint_callback,
        )
        # rospy.Subscriber(
        #     "/costmap_converter/costmap_obstacles",
        #     ObstacleArrayMsg,
        #     self.obstacle_callback,
        # )
        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)

        self.velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # polygons = [
        #     [
        #         (-2.875, 0.025000037625432014),
        #         (-2.575000047683716, -0.42499998211860657),
        #         (-2.7249999046325684, 0.3750000298023224),
        #         # (-2.875, 0.025000037625432014),
        #     ],
        #     [
        #         (-2.674999952316284, 0.3750000298023224),
        #         (-2.2750000953674316, 1.125),
        #         # (-2.674999952316284, 0.3750000298023224),
        #     ],
        #     [
        #         (-2.625, -0.3749999701976776),
        #         (-2.2249999046325684, -1.125),
        #         # (-2.625, -0.3749999701976776),
        #     ],
        #     [
        #         (-2.2249999046325684, 1.125),
        #         (-1.875, 1.7750000953674316),
        #         # (-2.2249999046325684, 1.125),
        #     ],
        #     [
        #         (-2.174999952316284, -1.225000023841858),
        #         (-1.774999976158142, -1.875),
        #         (-2.174999952316284, -1.125),
        #         # (-2.174999952316284, -1.225000023841858),
        #     ],
        #     [
        #         (-1.875, 1.8250000476837158),
        #         (-1.225000023841858, 2.174999952316284),
        #         (-1.6749999523162842, 2.125),
        #         # (-1.875, 1.8250000476837158),
        #     ],
        #     [
        #         (-1.774999976158142, -1.9249999523162842),
        #         (-1.1749999523162842, -2.375),
        #         (-1.3249999284744263, -1.9249999523162842),
        #         # (-1.774999976158142, -1.9249999523162842),
        #     ],
        #     [
        #         (-1.1749999523162842, -2.325000047683716),
        #         (-1.0749999284744263, -2.5250000953674316),
        #         (-0.5249999761581421, -2.5250000953674316),
        #         # (-1.1749999523162842, -2.325000047683716),
        #     ],
        #     [
        #         (-1.1749999523162842, -1.125),
        #         (-0.9749999642372131, -1.1749999523162842),
        #         (-0.9249999523162842, -0.9249999523162842),
        #         (-1.1749999523162842, -0.9249999523162842),
        #         # (-1.1749999523162842, -1.125),
        #     ],
        #     [
        #         (-1.1749999523162842, 0.025000037625432014),
        #         (-0.9249999523162842, -0.07499996572732925),
        #         (-0.9249999523162842, 0.2250000387430191),
        #         # (-1.1749999523162842, 0.025000037625432014),
        #     ],
        #     [
        #         (-1.1749999523162842, 1.125),
        #         (-0.9749999642372131, 1.0250000953674316),
        #         (-0.824999988079071, 1.1750000715255737),
        #         (-1.024999976158142, 1.3250000476837158),
        #         # (-1.1749999523162842, 1.125),
        #     ],
        #     [
        #         (-0.875, 0.12500004470348358),
        #         # (-0.875, 0.12500004470348358),
        #     ],
        #     [
        #         (-0.47499996423721313, -2.5250000953674316),
        #         (0.2250000387430191, -2.4749999046325684),
        #         # (-0.47499996423721313, -2.5250000953674316),
        #     ],
        #     [
        #         (-0.07499996572732925, -1.125),
        #         (0.17500004172325134, -1.1749999523162842),
        #         (0.17500004172325134, -0.9249999523162842),
        #         (0.025000037625432014, -0.875),
        #         # (-0.07499996572732925, -1.125),
        #     ],
        #     [
        #         (-0.07499996572732925, 0.025000037625432014),
        #         (0.17500004172325134, -0.07499996572732925),
        #         (0.2250000387430191, 0.12500004470348358),
        #         (0.025000037625432014, 0.2250000387430191),
        #         # (-0.07499996572732925, 0.025000037625432014),
        #     ],
        #     [
        #         (-0.07499996572732925, 1.0750000476837158),
        #         (0.17500004172325134, 0.9750000238418579),
        #         (0.27500003576278687, 1.1750000715255737),
        #         (0.025000037625432014, 1.2750000953674316),
        #         # (-0.07499996572732925, 1.0750000476837158),
        #     ],
        #     [
        #         (0.2250000387430191, 1.225000023841858),
        #         # (0.2250000387430191, 1.225000023841858),
        #     ],
        #     [
        #         (0.27500003576278687, -2.5250000953674316),
        #         (0.6250000596046448, -2.4749999046325684),
        #         # (0.27500003576278687, -2.5250000953674316),
        #     ],
        # ]
        static_obstacle_circle = StaticObstacle(
            id=3,
            geometry=Circle(center=(-2, 1), radius=0.2),
        )
        self.static_obstacle_list = []
        self.static_obstacle_list.append(static_obstacle_circle)
        # self.polygon_obstacles = [
        #     StaticObstacle(id=i, geometry=Polygon(vertices=vertices))
        #     for i, vertices in enumerate(polygons)
        # ]

        rospy.spin()

    def run(self):
        pass
        # rate = rospy.Rate(1)

        # # self.environment.static_obstacles = self.polygon_obstacles
        # # self.environment.plotter.update_static_obstacles(self.polygon_obstacles)

        # while not rospy.is_shutdown():
        #     self.environment.step()
        #     # print(self.environment.agent.goal_state, self.environment.agent.state)
        #     # print(
        #     #     "Velocity",
        #     #     self.environment.agent.linear_velocity,
        #     #     self.environment.agent.angular_velocity,
        #     # )

        #     # Publish the control command
        #     control_command = Twist()
        #     control_command.linear.x = self.environment.agent.linear_velocity
        #     control_command.angular.z = self.environment.agent.angular_velocity
        #     print(control_command.linear.x, control_command.angular.z)

        #     self.velocity_publisher.publish(control_command)

        #     rate.sleep()

    def odom_callback(self, message: Odometry):
        # Update the agent's state with the current position and orientation
        self.environment.agent.initial_state = np.array(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                euler_from_quaternion(
                    [
                        message.pose.pose.orientation.x,
                        message.pose.pose.orientation.y,
                        message.pose.pose.orientation.z,
                        message.pose.pose.orientation.w,
                    ]
                )[2],
            ]
        )
        self.environment.static_obstacles = self.static_obstacle_list
        self.environment.agent.reset(matrices_only=True)
        self.environment.step()
        print(self.environment.agent.goal_state, self.environment.agent.state)
        print(
            "Velocity",
            self.environment.agent.linear_velocity,
            self.environment.agent.angular_velocity,
        )

        # Publish the control command
        control_command = Twist()
        control_command.linear.x = self.environment.agent.linear_velocity
        control_command.angular.z = self.environment.agent.angular_velocity
        print(control_command.linear.x, control_command.angular.z)

        self.velocity_publisher.publish(control_command)

    def obstacle_callback(self, message: ObstacleArrayMsg):
        static_obstacle_list = []

        for obstacle in message.obstacles:
            obstacle: ObstacleMsg
            # Create a static obstacle for each polygon
            if len(obstacle.polygon.points[:-1]) > 1:
                points = [
                    (point.x, point.y)
                    for point in cast(List[Point32], obstacle.polygon.points[:-1])
                ]
            else:
                continue
            static_obstacle_list.append(
                StaticObstacle(
                    id=obstacle.id,
                    geometry=Polygon(
                        # id=obstacle.id,
                        vertices=points,
                    ),
                )
            )

        self.environment.static_obstacles = static_obstacle_list
        # self.environment.plotter.update_static_obstacles(static_obstacle_list)

    def people_callback(self, message: People):
        # Create a dynamic obstacle for each person
        dynamic_obstacle_list = []

        for person in message.people:
            person: Person
            dynamic_obstacle_list.append(
                DynamicObstacle(
                    id=person.name,
                    position=(person.position.x, person.position.y),
                    orientation=np.arctan2(person.velocity.y, person.velocity.x),
                    linear_velocity=(person.velocity.x**2 + person.velocity.y**2),
                    angular_velocity=0,
                    horizon=20,
                )
            )

        self.environment.dynamic_obstacles = dynamic_obstacle_list

    def waypoint_callback(self, message: Path):
        # Update the agent's goal with the waypoint position
        if message.header.seq == 0:
            print("Updating waypoints")
            waypoints = [
                (
                    pose.pose.position.x,
                    pose.pose.position.y,
                    euler_from_quaternion(
                        [
                            pose.pose.orientation.x,
                            pose.pose.orientation.y,
                            pose.pose.orientation.z,
                            pose.pose.orientation.w,
                        ]
                    )[2],
                )
                for pose in message.poses[::20]
            ]
        waypoints = []
        # orientation_euler = euler_from_quaternion((0, 0, message.poses[-1].pose.orientation, 0))
        waypoints.append(
            (
                message.poses[-1].pose.position.x,
                message.poses[-1].pose.position.y,
                euler_from_quaternion(
                    [
                        message.poses[-1].pose.orientation.x,
                        message.poses[-1].pose.orientation.y,
                        message.poses[-1].pose.orientation.z,
                        message.poses[-1].pose.orientation.w,
                    ]
                )[2],
            )
        )
        print("Waypoint", waypoints)
        self.environment.waypoints = np.array(waypoints)
        self.environment.waypoint_index = 0
        self.environment.agent.update_goal(self.environment.current_waypoint)
        # self.environment.plotter.update_goal(self.environment.waypoints)


if __name__ == "__main__":
    ros_interface = ROSInterface()
    ros_interface.run()
