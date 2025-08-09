#!/usr/bin/env python3
import os
import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from sensor_msgs_py import point_cloud2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from ultralytics import YOLO
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import message_filters
import ros2_numpy
from sklearn.cluster import DBSCAN
import cv2
import time
import rosbag2_py
from rclpy.serialization import deserialize_message

# QoS Profiles
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

tf_static_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)
class BagReader(Node):
    def __init__(self):
        super().__init__("ros_bag_reader")

        self.image_pub = self.create_publisher(
            Image, '/camera1/camera1/color/image_raw', qos_profile
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, '/camera1/camera1/color/camera_info', qos_profile
        )
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/livox/lidar', qos_profile
        )

        self.tf_pub = self.create_publisher(
            TFMessage, '/tf', 10
        )
        self.tf_static_pub = self.create_publisher(
            TFMessage, '/tf_static', tf_static_qos
        )

        self.reader = rosbag2_py.SequentialReader()
        bag_path = os.path.expanduser('/home/container_user/wheelchair2/src/records/2025-08-05_22-54-16/rosbag/rosbag_0.db3')
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.reader.open(storage_options, converter_options)

        self.topic = {
            '/camera1/camera1/color/image_raw': (Image, self.image_pub),
            '/camera1/camera1/color/camera_info': (CameraInfo, self.camera_info_pub),
            '/livox/lidar' : (PointCloud2, self.point_cloud_pub),
            '/tf': (TFMessage, self.tf_pub),
            '/tf_static': (TFMessage, self.tf_static_pub)
        }

        self.create_timer(0.1, self.ros_play)

        self.played = False


    def ros_play(self):
        if self.played:
            return
        self.played = True
        self.get_logger().info('Started playing bag')
        wall_start_time = time.time()
        bag_start_time = None

        while self.reader.has_next():
            try:
                (topic, data, timestamp) = self.reader.read_next()
                if bag_start_time is None:
                    bag_start_time = timestamp
                intended_time = wall_start_time + (timestamp - bag_start_time) / 1e9
                now = time.time()
                sleep_time = intended_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if topic in self.topic:
                    message_type, publisher = self.topic[topic]
                    msg = deserialize_message(data, message_type)
                    publisher.publish(msg)

            except Exception as e:
                self.get_logger().error(f'Error reading bag: {e}')
                break

class DetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector")
       
        self.bridge = CvBridge()
        self.model = YOLO('yolo11n-seg.pt')
        self.model.to('cuda')
        self.conf_threshold = 0.5
        self.target_class = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_intrinsics = None
        self.dbscan_eps = 0.5
        self.dbscan_min_samples = 10

        self.frame_count = 0

        #subs
        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            "/camera1/camera1/color/image_raw",   
        )

        self.lidar_sub = message_filters.Subscriber(
            self,
            PointCloud2,
            "/livox/lidar",
        )

        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=10,
            slop=0.1,
        ) 
        self.time_sync.registerCallback(self.synchronized_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera1/camera1/color/camera_info',
            self.camera_info_callback,
            1
        )

        self.seg_pc_pub = self.create_publisher(
            PointCloud2,
            '/humans/pointcloud',
            10
        )
        self.seg_image_pub = self.create_publisher(
            Image,
            "/segmentation/image",
            10
        )
    def camera_info_callback(self, msg):
        self.camera_intrinsics = np.array(msg.k).reshape((3, 3))
        self.get_logger().info("Camera intrinsics received.")
        self.destroy_subscription(self.camera_info_sub)

    def synchronized_callback(self, image_msg, pc_msg):

        if self.camera_intrinsics is None:
            self.get_logger().warn("Waiting for camera intrinsics...")
            return
        
        # 1. Hardcode your provided extrinsic values [x, y, z, qx, qy, qz, qw]
        # This is T_lidar_camera (transforms from camera to lidar)
        T_lidar_camera_arr = [
            0.08592069025315134, 0, -0.10986527445745775,
            0.5041754569158885, 0.510006761582378, 0.495582854424067, 0.5040033761446712
        ]
        translation_m = ros2_numpy.geometry.transformations.translation_matrix(T_lidar_camera_arr[0:3])
        rotation_m = ros2_numpy.geometry.transformations.quaternion_matrix(T_lidar_camera_arr[3:7])
        T_lidar_camera = np.dot(translation_m, rotation_m)
        transform_matrix = np.linalg.inv(T_lidar_camera)
        # print(transform_matrix)

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        results = self.model.track(
            source = cv_image,
            conf = self.conf_threshold,
            classes = [self.target_class],
            persist=True,
        )
        
        if results[0].masks is None or len(results[0].masks) == 0:
            self.get_logger().info("No person detected in the frame.")
            return
         
        self.get_logger().info(f"Detected {len(results[0].boxes)} person(s). Publishing filtered point cloud.")

        segmentation_image = results[0].plot(conf=False, labels=True, boxes=True, masks=True)
        segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_image, encoding="bgr8")
        segmentation_msg.header = image_msg.header 
        self.seg_image_pub.publish(segmentation_msg)

        combined_mask = np.zeros((image_msg.height, image_msg.width), dtype=np.uint8)

        cloud_array = ros2_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
        points = ros2_numpy.point_cloud2.get_xyz_points(cloud_array, remove_nans=False)

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_camera_frame = (transform_matrix @ points_homogeneous.T).T[:, :3]

        in_front_mask = points_camera_frame[:, 2] > 0
        points_for_projection = points_camera_frame[in_front_mask]
        original_points_for_publishing = points[in_front_mask]
        
        if len(points_for_projection) == 0:
            return

        projected_points = (self.camera_intrinsics @ points_for_projection.T).T
        pixel_coords = (projected_points[:, :2] / projected_points[:, 2:3]).astype(int)

        on_image_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_msg.width) & \
                        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_msg.height)
        
        pixel_coords_on_image = pixel_coords[on_image_mask]
        original_points_on_image = original_points_for_publishing[on_image_mask]
        
        if len(pixel_coords_on_image) == 0:
            return
        
        masks = results[0].masks.data
        track_ids = results[0].boxes.id.int().cpu().numpy()
        
        # Use a dictionary to map track_id to its points. This is robust.
        human_points_map = {tid: [] for tid in track_ids}

        for i, track_id in enumerate(track_ids):
            small_mask = masks[i].cpu().numpy().astype(np.uint8)
            resized_mask = cv2.resize(small_mask, (image_msg.width, image_msg.height), interpolation=cv2.INTER_NEAREST)
            
            mask_values = resized_mask[pixel_coords_on_image[:, 1], pixel_coords_on_image[:, 0]]
            person_points_mask = (mask_values == 1)
            
            person_points_3d = original_points_on_image[person_points_mask]
            
            if len(person_points_3d) > 0:
                human_points_map[track_id].extend(person_points_3d.tolist())

        all_human_points_list = []
        for track_id, points in human_points_map.items():
            all_human_points_list.extend(points)

        if len(all_human_points_list) > 0:
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            final_points = np.vstack(all_human_points_list)
            output_array = np.zeros(len(final_points), dtype=dtype)
            output_array['x'] = final_points[:, 0]
            output_array['y'] = final_points[:, 1]
            output_array['z'] = final_points[:, 2]

            final_pc_msg = ros2_numpy.point_cloud2.array_to_pointcloud2(
                output_array, 
                stamp=pc_msg.header.stamp, 
                frame_id=pc_msg.header.frame_id
            )
            self.seg_pc_pub.publish(final_pc_msg)
        
        self.frame_count += 1

def main(args=None):
    RUN_WITH_BAG = False

    rclpy.init(args=args)

    humandetector = DetectorNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(humandetector)

    bag_reader = None
    if RUN_WITH_BAG:
        print("RUNNING IN BAG PLAYBACK MODE")
        bag_reader = BagReader()
        executor.add_node(bag_reader)
    else:
        print("RUNNING IN LIVE MODE (DETECTOR ONLY)")

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if bag_reader is not None:
            bag_reader.destroy_node()
        humandetector.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
