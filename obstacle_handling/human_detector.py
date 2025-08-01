#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class HumanDetectorNode(Node):
    def __init__(self):
        super().__init__("yolov8_human_detector")

        self.declare_parameter('model', 'yolov8x-seg.pt')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('threshold', 0.4)
        
        model_name = self.get_parameter('model').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('threshold').get_parameter_value().double_value
        
        self.bridge = CvBridge()
        self.model = YOLO(model_name)
        self.model.to(device)
        self.target_class = 0


        self.subscription = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            10
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
