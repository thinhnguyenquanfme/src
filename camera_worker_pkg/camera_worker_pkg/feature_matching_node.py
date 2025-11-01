import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class OrbMatching(Node):
    def __init__(self):
        super().__init__('ORB_matching_node')
        self.bridge = CvBridge()
        self.undistorted_img_sub = self.create_subscription(Image, '/camera/undistorted_image', self.undistorted_img_cb, 1)

    def undistorted_img_cb(self, msg):
        try:
            undistort_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            
        except:
            self.get_logger().error('Undistorted image error!')

def main(args=None):
    rclpy.init(args=args)
    my_node = OrbMatching()
    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()