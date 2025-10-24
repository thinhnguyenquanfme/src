#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 

class DrawCircleNode(Node):
    def __init__(self):
        super().__init__('turtle_draw_circle')
        self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.create_timer(1.0, self.send_velocity_cmd)
        

    def send_velocity_cmd(self):
        msg = Twist()
        msg.linear.x = 2.0
        msg.linear.y = 2.0
        msg.angular.z = 1.0
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    my_node = DrawCircleNode()

    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()