#!/usr/bin/env python3

'''
This is a service node to grab frame
'''

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from example_interfaces.srv import Trigger

from pypylon import pylon
from pypylon import genicam


class BaslerSnapshotServer(Node):
    def __init__(self):
        super().__init__('basler_snapshot_server')
        # Params
        self.declare_parameter('frame_id', 'camera_optical_frame')
        self.declare_parameter('pixel_format', 'MONO8')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.pixel_format = self.get_parameter('pixel_format').get_parameter_value().string_value

        # Publisher
        self.pub_snapshot = self.create_publisher(Image, '/camera/raw_snapshot', 1)
        self.bridge = CvBridge()

        # Open camera
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.get_logger().info('Camera opened. Ready for grab.')

        except:
            self.get_logger().error('can not open camera!')

        # Conver to opencv mono8
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Create service
        self.srv = self.create_service(Trigger, '/camera/grab_one', self.handle_grab_one)

    def handle_grab_one(self, req, resp):
        if self.camera is None:
            resp.success = False
            resp.message = 'Camera not avaiable'
            return resp
        try:
            r = self.camera.GrabOne(100)
            if r is None or not r.GrabSucceeded():
                msg = 'time out' if r is None else f'{r.ErrorCode} {r.ErrorDescription}'
                if r is not None: r.Release()
                resp.success = False
                resp.message = f'grab failed: {msg}'
                return resp
            
            img = self.converter.Convert(r).GetArray()
            r.Release()

            # Convert into ROS image
            rosimg = self.bridge.cv2_to_imgmsg(img, encoding='mono8')
            rosimg.header.stamp = self.get_clock().now().to_msg()
            rosimg.header.frame_id = self.frame_id

            # Publish 1 image
            self.pub_snapshot.publish(rosimg)
            resp.success = True
            resp.message = 'snapshot published on /camera/grab_one'

            self.get_logger().info('Image grab success.')

            return resp

        except genicam.GenericException as e:
            resp.success = False
            resp.message = f'exception {e}'

            return resp
        
    def destroy_node(self):
        try:
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    my_node = BaslerSnapshotServer()

    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

