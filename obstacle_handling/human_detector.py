#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch

class HumanDetectorNode(Node):
    def __init__(self):
        super().__init__("yolov8_human_detector")
        self.subscription = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10
        )
        self.br = CvBridge()
        self.model = YOLO('yolov8n.pt')  

    def image_callback(self, msg):
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        results = self.model.predict(source=cv_image, conf=0.4, classes=[self.target_class], verbose=False)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            self.get_logger().info(f"Detected {len(boxes)} person(s) in the frame.")
        else:
            self.get_logger().info("No person detected.")

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
