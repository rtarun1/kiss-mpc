import numpy as np
import rospy
from geometry_msgs.msg import Pose, Twist
from people_msgs.msg import People, Person

from mpc.agent import EgoAgent
from mpc.dynamic_obstacle import DynamicObstacle
from mpc.environment import ROSEnvironment
from mpc.geometry import Circle, Rectangle


class ROSInterface:
    """
    ROSInterface class to interface with ROS
    Creates a node and subscribes to people messages for obstacles, and publishes commands on the /cmd_vel topic
    Also subscribes to waypoint pose messages for the next goal
    """

    def __init__(self):
        self.environment = ROSEnvironment(
            agent=EgoAgent(
                id=0,
                initial_position=(0, 0),
                initial_orientation=0,
                goal_position=(0, 0),
                goal_orientation=0,
                planning_time_step=0.041,
                initial_linear_velocity=0,
                initial_angular_velocity=0,
                horizon=30,
                geometry=Circle.from_rectangle(Rectangle(width=0.5, height=0.5)),
                linear_velocity_bounds=(0, 12),
                angular_velocity_bounds=(-np.pi / 4, np.pi / 4),
                linear_acceleration_bounds=(-50, 50),
                angular_acceleration_bounds=(-np.pi, np.pi),
                left_right_lane_bounds=(-1000.5, 1000.5),
            ),
            static_obstacles=[],
            dynamic_obstacles=[],
        )

        rospy.init_node("ros_mpc_interface")

        rospy.Subscriber("/people", People, self.people_callback)
        rospy.Subscriber("/waypoint", Pose, self.waypoint_callback)
        rospy.Subscriber("/odom", Pose, self.odom_callback)

        self.velocity_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.spin()

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.environment.step()

            # Publish the control command
            control_command = Twist()
            control_command.linear.x = self.environment.agent.linear_velocity
            control_command.angular.z = self.environment.agent.angular_velocity
            self.velocity_publisher.publish(control_command)

            rate.sleep()

    def odom_callback(self, message: Pose):
        # Update the agent's state with the current position and orientation
        self.environment.agent.initial_state = np.array(
            [message.position.x, message.position.y, message.orientation.z]
        )
        self.environment.agent.reset(matrices_only=True)

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
                    geometry=Rectangle(width=0.5, height=0.5),
                )
            )

        self.environment.dynamic_obstacles = dynamic_obstacle_list

    def waypoint_callback(self, message: Pose):
        # Update the agent's goal with the waypoint position
        self.environment.agent.goal_state = np.array(
            [message.position.x, message.position.y, message.orientation.z]
        )


if __name__ == "__main__":
    ros_interface = ROSInterface()
    ros_interface.run()
