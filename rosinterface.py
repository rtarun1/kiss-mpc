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
                id=1,
                radius=1,
                initial_position=(0, 0),
                initial_orientation=np.deg2rad(90),
                horizon=20,
                use_warm_start=True,
            ),
            static_obstacles=[],
            dynamic_obstacles=[],
            waypoints=[],
        )

        rospy.init_node("ros_mpc_interface")

        rospy.Subscriber("/people", People, self.people_callback)
        rospy.Subscriber("/waypoint", Pose, self.waypoint_callback)
        rospy.Subscriber("/odom", Pose, self.odom_callback)
        # rospy.Subcriber("/obstacles")

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
                    horizon=20,
                )
            )

        self.environment.dynamic_obstacles = dynamic_obstacle_list

    def waypoint_callback(self, message: Pose):
        # Update the agent's goal with the waypoint position
        self.environment.waypoints = np.array(
            [message.position.x, message.position.y, message.orientation.z]
        )
        self.environment.waypoint_index = 0


if __name__ == "__main__":
    ros_interface = ROSInterface()
    ros_interface.run()
