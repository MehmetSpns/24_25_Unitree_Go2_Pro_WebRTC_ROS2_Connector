# camera_obstacle_detector_node.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraObstacleDetector(Node):
    def __init__(self):
        super().__init__('camera_obstacle_detector')

        self.bridge = CvBridge()
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)
        self.get_logger().info('Camera Obstacle Detector Node Started')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Simple threshold to detect large close objects (white means "obstacle")
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacle_detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # Threshold area for obstacle detection
                obstacle_detected = True
                break

        cmd = Twist()
        if obstacle_detected:
            self.get_logger().info('Obstacle detected! Stopping.')
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = 0.3  # Move forward

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CameraObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
