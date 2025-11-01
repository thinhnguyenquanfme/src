import cv2 as cv
import numpy as np
import glob

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraUndistort(Node):
    def __init__(self):
        super().__init__('frame_undistort')

        # Params
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('pixel_format', 'mono8')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.pixel_format = self.get_parameter('pixel_format').get_parameter_value().string_value

        self.data = np.load('/home/thinh/ros2_ws/src/camera_worker_pkg/data_source/Basler_calib_data/basler_camera_params.npz')
        self.mtx = self.data['camera_matrix'] 
        self.dist = self.data['dist_coeffs']
        self.rvecs = self.data['rvecs']
        self.tvecs = self.data['tvecs']

        self.bridge = CvBridge()
        self.subscriber_ = self.create_subscription(Image, '/camera/raw_snapshot', self.img_callback, 1)
        self.publisher_ = self.create_publisher(Image, '/camera/undistorted_image', 1)

    def img_callback(self, msg):
        
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.pixel_format)
        print(frame)
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))

        dst = cv.undistort(frame, self.mtx, self.dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # Convert into ROS image
        rosimg = self.bridge.cv2_to_imgmsg(dst, encoding=self.pixel_format)
        rosimg.header.stamp = self.get_clock().now().to_msg()
        rosimg.header.frame_id = self.frame_id
        self.get_logger().info('Undistort image success.')
        self.publisher_.publish(rosimg)

def main(args=None):
    rclpy.init(args=args)
    my_node = CameraUndistort()
    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()