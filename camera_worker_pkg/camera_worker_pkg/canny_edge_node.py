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
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('pixel_format', 'mono8')
        self.declare_parameter('canny_thres1', float(100))
        self.declare_parameter('canny_thres2', float(200))
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.pixel_format = self.get_parameter('pixel_format').get_parameter_value().string_value
        self.canny_thres1 = self.get_parameter('canny_thres1').get_parameter_value().double_value
        self.canny_thres2 = self.get_parameter('canny_thres2').get_parameter_value().double_value
        

        self.undistorted_img_sub = self.create_subscription(Image, '/camera/undistorted_image', self.img_callback, 1)
        self.edge_img_pub = self.create_publisher(Image, '/camera/canny_node/edge_image', 1)
        self.dx_main_pub = self.create_publisher(Image, '/camera/canny_node/dx_main', 1)
        self.dy_main_pub = self.create_publisher(Image, '/camera/canny_node/dy_main', 1)
        self.bridge = CvBridge()

    def img_callback(self, msg):
        try:
            self.canny_thres1 = self.get_parameter('canny_thres1').get_parameter_value().double_value
            self.canny_thres2 = self.get_parameter('canny_thres2').get_parameter_value().double_value

            undistort_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.pixel_format)
            edge_img = cv.Canny(undistort_img, self.canny_thres1, self.canny_thres2)

            dx_main = cv.Sobel(undistort_img, cv.CV_32F, 1, 0, ksize=3)
            dy_main = cv.Sobel(undistort_img, cv.CV_32F, 0, 1, ksize=3)

            rosimg = self.bridge.cv2_to_imgmsg(edge_img, encoding=self.pixel_format)
            rosimg.header.stamp = self.get_clock().now().to_msg()
            rosimg.header.frame_id = self.frame_id

            ros_dx_main = self.bridge.cv2_to_imgmsg(dx_main, encoding='passthrough')
            ros_dx_main.header.stamp = self.get_clock().now().to_msg()
            ros_dx_main.header.frame_id = self.frame_id
            
            ros_dy_main = self.bridge.cv2_to_imgmsg(dy_main, encoding='passthrough')
            ros_dy_main.header.stamp = self.get_clock().now().to_msg()
            ros_dy_main.header.frame_id = self.frame_id

            self.get_logger().info('Canny edge convert success.')
            self.edge_img_pub.publish(rosimg)
            self.dx_main_pub.publish(ros_dx_main)
            self.dy_main_pub.publish(ros_dy_main)
        except Exception as e:
            self.get_logger().error(f"Canny error: {e!r}")

def main(args=None):
    rclpy.init(args=args)
    my_node = EdgeDetect()
    
    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()