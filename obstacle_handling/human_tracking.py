#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import message_filters
import ros2_numpy
from sklearn.cluster import DBSCAN
import cv2

def create_point_cloud_xyz_intensity(header, points_xyz, intensity):

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    
    # Create the PointCloud2 message
    points = np.hstack((points_xyz, intensity))
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.is_dense = True  # Assumes no NaN or Inf values
    msg.is_bigendian = False
    msg.fields = fields
    
    # Set the point_step (size of a single point in bytes)
    msg.point_step = 16  # 3 fields * 4 bytes/float
    
    # Set the row_step (total size of a row in bytes)
    msg.row_step = msg.point_step * msg.width
    
    # Convert the numpy array to a byte array and assign it to the data field
    msg.data = np.asarray(points, dtype=np.float32).tobytes()
    
    return msg


class HumanDetectorNode(Node):
    def __init__(self):
        super().__init__("yolov8_human_detector")
       
        self.bridge = CvBridge()
        self.model = YOLO('yolo11n-seg.pt')
        self.model.to('cuda')
        self.conf_threshold = 0.5
        self.target_class = 0
        self.camera_frame = 'camera1_link'
        self.lidar_frame = 'livox_frame'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_intrinsics = None
        
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
        
        # transform = self.tf_buffer.lookup_transform(
        #     self.camera_frame, self.lidar_frame, rclpy.time.Time()
        # )
        
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

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        
        results = self.model.predict(
            source = cv_image,
            conf = self.conf_threshold,
            classes = [self.target_class],
            verbose=False,
        )
        
        if results[0].masks is None or len(results[0].masks) == 0:
            self.get_logger().info("No person detected in the frame.")
            return
        
        self.get_logger().info(f"Detected {len(results[0].boxes)} person(s). Publishing filtered point cloud.")

        combined_mask = np.zeros((image_msg.height, image_msg.width), dtype=np.uint8)
        target_shape = (image_msg.width, image_msg.height)
        for mask_data in results[0].masks.data:
            small_mask = mask_data.cpu().numpy().astype(np.uint8)
            resized_mask = cv2.resize(small_mask, target_shape, interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, resized_mask * 255)

        segmentation_image = results[0].plot(conf=False, labels=False, boxes=False, masks=True)
        segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_image, encoding="bgr8")
        segmentation_msg.header = image_msg.header 
        self.seg_image_pub.publish(segmentation_msg)

        pc_dict = ros2_numpy.point_cloud2.point_cloud2_to_array(pc_msg)
        points = pc_dict['xyz']
        intensity = pc_dict['intensity']

        # t = transform.transform.translation
        # q = transform.transform.rotation
        # translation_matrix = ros2_numpy.geometry.transformations.translation_matrix([t.x, t.y, t.z])
        # rotation_matrix = ros2_numpy.geometry.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        # transform_matrix = np.dot(translation_matrix, rotation_matrix) # by this we get the transformation between camera and lidar

        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_camera_frame = (transform_matrix @ points_homogeneous.T).T[:, :3]

        in_front_mask = points_camera_frame[:, 2] > 0
        points_for_projection = points_camera_frame[in_front_mask]
        original_points_for_publishing = points[in_front_mask]
        original_intensity_for_publishing = intensity[in_front_mask]


        if len(points_for_projection) == 0:
            return

        projected_points = (self.camera_intrinsics @ points_for_projection.T).T
        pixel_coords = (projected_points[:, :2] / projected_points[:, 2:3]).astype(int)

        on_image_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_msg.width) & \
                        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_msg.height)
        
        pixel_coords_on_image = pixel_coords[on_image_mask]
        original_points_on_image = original_points_for_publishing[on_image_mask]
        original_intensity_on_image = original_intensity_for_publishing[on_image_mask]


        if len(pixel_coords_on_image) == 0:
            return
        
        mask_values = combined_mask[pixel_coords_on_image[:, 1], pixel_coords_on_image[:, 0]]
        human_points_mask = (mask_values == 255)

        final_points = original_points_on_image[human_points_mask]
        final_intensity = original_intensity_on_image[human_points_mask]


        if len(final_points) > 0:
            header = pc_msg.header
            filtered_pc_msg = create_point_cloud_xyz_intensity(header, final_points, final_intensity)

            self.seg_pc_pub.publish(filtered_pc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
