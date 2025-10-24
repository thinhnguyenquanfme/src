#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt32


class SubscriberNode(Node):
    def __init__(self):
        super().__init__("Subscriber_node")
        self.subscriber_ = self.create_subscription(UInt32, "topic", self.subs_callback, 10)


    def subs_callback(self, msg):
        self.get_logger().info(f"I heard: {msg.data}")


def main(args = None):
    rclpy.init(args=args)
    subscriber = SubscriberNode()

    rclpy.spin(subscriber)
    rclpy.shutdown()

if __name__ == "__main__":
    main()