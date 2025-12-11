import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math

class PointSim(Node):
    def __init__(self):
        super().__init__('point_sim')

        self.pub = self.create_publisher(Marker, 'robot_point', 10)
        self.t = 0.0
        self.timer = self.create_timer(0.02, self.update)

    def update(self):
        self.t += 0.02

        x = 1.0 * math.cos(self.t)
        y = 1.0 * math.sin(self.t)
        z = 0.0

        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        self.pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = PointSim()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
