#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import message_filters
import ros2_numpy
from sklearn.cluster import DBSCAN

class HumanDetectorNode(Node):
    def __init__(self):
        super().__init__("yolov8_human_detector")
       
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n-seg.pt')
        self.model.to('cuda')
        self.conf_threshold = 0.4
        self.target_class = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        #subs
        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            "/camera/camera/color/image_raw",   
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
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            1
        )

        self.det_image_pub = self.create_publisher(
            Image,
            "/detection/image",
            10
        )
        self.seg_image_pub = self.create_publisher(
            Image,
            "/segmentation/image",
            10
        )

    def image_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        results = self.model.predict(
            source = cv_image,
            conf = self.conf_threshold,
            classes = [self.target_class],
            verbose=False,
        )
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            num_persons = len(results[0].boxes)
            self.get_logger().info(f"Detected {num_persons} person(s).")

            detection_image = results[0].plot(conf=True, labels=True, boxes=True, masks=False)
            detection_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding="bgr8")
            detection_msg.header = msg.header
            self.det_image_pub.publish(detection_msg)

            if results[0].masks is not None:
                    segmentation_image = results[0].plot(conf=False, labels=False, boxes=False, masks=True)
                    segmentation_msg = self.bridge.cv2_to_imgmsg(segmentation_image, encoding="bgr8")
                    segmentation_msg.header = msg.header 
                    self.seg_image_pub.publish(segmentation_msg)
        else:
            self.get_logger().info("No person detected in the frame.")

def main(args=None):
    rclpy.init(args=args)
    node = HumanDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()
