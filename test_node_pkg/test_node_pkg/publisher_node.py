#!/usr/bin/env python3
import cv2 as cv
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32

class MyNode(Node):
    def __init__(self):
        super().__init__("Publisher_node")
        self.create_timer(1.0, self.send_mess)
        self.publisher_  = self.create_publisher(UInt32, 'topic', 10)

    def send_mess(self):
        msg = UInt32()
        now = self.get_clock().now()
        stamp = now.to_msg()
        msg.data = stamp.sec
        self.publisher_.publish(msg)
        self.get_logger().info(f"{msg.data}")

        


def main(args=None):
    rclpy.init(args=args)
    my_node = MyNode()

    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()