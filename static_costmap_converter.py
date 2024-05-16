#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from std_msgs.msg import Float32MultiArray


class StaticCostmapConveter:
    def __init__(self) -> None:

        rospy.init_node("static_costmap_converter")
        rospy.Subscriber(
            "/costmap_converter/costmap_obstacles",
            ObstacleArrayMsg,
            self.centre_callback,
        )
        self.centre_publisher = rospy.Publisher(
            "/ros_interface",
            Float32MultiArray,
            self.compute_origin_callback,
        )
        self.obstacle_vertices = []
        rospy.spin()

    def centre_callback(self, message: ObstacleArrayMsg):
        """
        Here goes the code to accept polygon vertices and append them
        to a list only if a new vertex is received.
        """

        for obstacle in message.obstacles:
            single_obstacle = []
            if obstacle.polygon.points not in self.obstacle_vertices:
                single_obstacle.append(
                    [(point.x, point.y) for point in obstacle.polygon.points]
                )
            self.obstacle_vertices.append(single_obstacle)
        print(self.obstacle_vertices)

    def compute_origin_callback():
        """
        Here goes all the code to convert polygon vertices to centres.
        """
        pass


if __name__ == "__main__":
    static_map_interface = StaticCostmapConveter()
