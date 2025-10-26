#!/usr/bin/env python3

import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class EdgeDetect(Node):
    def __init__(self):
        super().__init__('edge_detect')

        # Params
        self.declare_parameter('frame_id', 'edge_image')
        self.declare_parameter('pixel_format', 'mono8')
        self.declare_parameter('canny_thres1', int(100))
        self.declare_parameter('canny_thres2', int(200))
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.pixel_format = self.get_parameter('pixel_format').get_parameter_value().string_value
        self.canny_thres1 = self.get_parameter('canny_thres1').get_parameter_value().integer_value
        self.canny_thres2 = self.get_parameter('canny_thres2').get_parameter_value().integer_value
        

        self.subscriber_ = self.create_subscription(Image, '/camera/undistorted_image', self.img_callback, 1)
        self.publishers_ = self.create_publisher(Image, '/camera/edge_image', 1)
        self.bridge = CvBridge()

    def img_callback(self, msg):
        try:
            self.canny_thres1 = self.get_parameter('canny_thres1').get_parameter_value().integer_value
            self.canny_thres2 = self.get_parameter('canny_thres2').get_parameter_value().integer_value

            undistort_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.pixel_format)
            edge_img = cv.Canny(undistort_img, self.canny_thres1, self.canny_thres2)
            # self.get_logger().info(f'{self.canny_thres1}')

            rosimg = self.bridge.cv2_to_imgmsg(edge_img, encoding=self.pixel_format)
            rosimg.header.stamp = self.get_clock().now().to_msg()
            rosimg.header.frame_id = self.frame_id

            self.get_logger().info('Canny edge convert success.')

            self.publishers_.publish(rosimg)
        except:
            self.get_logger().error('Canny edge image error!')

def main(args=None):
    rclpy.init(args=args)
    my_node = EdgeDetect()
    
    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()