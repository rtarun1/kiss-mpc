#!/usr/bin/env python3

import message_filters
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from std_msgs.msg import Float32MultiArray

from tf2_geometry_msgs import do_transform_pose_stamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from scipy.spatial.transform import (
    Rotation as R,  
)
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from mpc.model import Model

def euler_from_quaternion(quat, degree = False):
    return R.from_quat(quat).as_euler('xyz')

class ROS2Interface(Node):
    def __init__(self):
        super().__init__('ros_mpc_interface')
        
        self.model = Model(
            id = 1,
            initial_position=(0, 0),
            initial_orientation=np.deg2rad(90),
            horizon=7,
            use_warm_start=True,
            planning_time_step=0.8,
            linear_velocity_bounds=(0, 0.3),
            angular_velocity_bounds=(-0.3, 0.3),
            waypoints=[],
        )
        
        self.counter = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.waypoints=[]
        
        self.create_subscription(Float32MultiArray, '/waypoint', self.waypoint_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        self.velocity_publisher = self.create_publisher(Twist, '/husky_velocity_controller/cmd_vel_unstamped', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/future_states', 10)
        self.timer = self.create_timer(0.01, self.run)
    def run(self):
        if not self.waypoints:
            return
        # print("run function is running")
        self.model.step()
        self.future_states_pub()
        
        control_command = Twist()
        control_command.linear.x = self.model.linear_velocity
        control_command.angular.z = self.model.angular_velocity
        self.velocity_publisher.publish(control_command)
        
    def future_states_pub(self):
        marker_array = MarkerArray()
        future_states = self.model.states_matrix
        for i, state in enumerate(future_states.T):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position.x = float(state[0])
            marker.pose.position.y = float(state[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)

        self.marker_publisher.publish(marker_array)
        
    def odom_callback(self, message: Odometry):
        pose = message.pose.pose
        self.model.initial_state = np.array(
            [
                pose.position.x,
                pose.position.y,
                euler_from_quaternion(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                )[2],
            ]
        )
        self.model.reset(matrices_only=True)
        
    def waypoint_callback(self, message: Float32MultiArray):

        data = message.data
        
        # Check if we have valid data (even number of elements for x,y pairs)
        if len(data) == 0 or len(data) % 2 != 0:
            print(f"Invalid waypoint data: {len(data)} elements (should be even)")
            return
        
        # Extract x,y pairs from the data
        waypoints = []
        for i in range(0, len(data), 2):
            x = float(data[i])
            y = float(data[i + 1])
            # Add orientation (theta) as 0 for now, or you could calculate it based on direction
            theta = 0.0  # You might want to calculate this based on the direction between waypoints
            waypoints.append((x, y, theta))
        
        # Check if waypoints have changed significantly
        try:
            if self.waypoints:
                # Compare the last waypoint
                last_old = np.array(self.waypoints[-1][:2])  # Only x,y for comparison
                last_new = np.array(waypoints[-1][:2])
                diff = np.linalg.norm(last_old - last_new)
            else:
                diff = float('inf')  # Force update if no previous waypoints
        except Exception:
            diff = float('inf')
            
        if not self.waypoints or abs(diff) > 0.01:  # Threshold for waypoint change
            print(f"Updating waypoints: received {len(waypoints)} waypoints")
            
            # Calculate orientation for each waypoint based on direction to next waypoint
            for i in range(len(waypoints) - 1):
                x1, y1, _ = waypoints[i]
                x2, y2, _ = waypoints[i + 1]
                theta = np.arctan2(y2 - y1, x2 - x1)
                waypoints[i] = (x1, y1, theta)
            
            # For the last waypoint, use the orientation from the previous waypoint
            if len(waypoints) > 1:
                waypoints[-1] = (waypoints[-1][0], waypoints[-1][1], waypoints[-2][2])
            
            self.waypoints = waypoints
            self.model.waypoints = np.array(waypoints)
            self.model.waypoint_index = 0
            self.model.update_goal(self.model.current_waypoint())
            
            print(f"Updated waypoints: {waypoints}")

def main(args=None):
    rclpy.init(args=args)
    ros_interface = ROS2Interface()
    # ros_interface.run()
    rclpy.spin(ros_interface)
    ros_interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()