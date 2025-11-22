import numpy as np
import cv2 as cv

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from robot_interfaces.srv import PoseListSrv
from robot_interfaces.msg import PoseListMsg
from geometry_msgs.msg import Transform, Vector3, PoseStamped, Twist
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from std_srvs.srv import SetBool

class ConvertCoordinate(Node):
    def __init__(self):
        super().__init__('coordinate_convert')

        # =============== Publisher define ===============

        # =============== Subscriber define ===============
        self.object_center_sub = self.create_subscription(PoseStamped, '/geometry/camera_coord/object_center', self.convert_to_robot_axis, 1)

    def convert_to_robot_axis(self, msg):
        

def main(args=None):
    rclpy.init(args=args)
    my_node = ConvertCoordinate()
    rclpy.spin(my_node)
    rclpy.shutdown