#!/usr/bin/env python3
"""
Publish dummy object centers to /geometry/camera_coord/object_center at a
configurable frequency and position. Enable/disable via SetBool service or
parameter updates, so the GUI can toggle it.
"""

from typing import List

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped


class ObjectSpawn(Node):
    def __init__(self):
        super().__init__("object_spawn_node")

        # Parameters (can be changed at runtime)
        self.declare_parameter("frequency_hz", 1.0)         # publish rate
        self.declare_parameter("pos_x", 450.0)              # default fixed position
        self.declare_parameter("pos_y", 100.0)
        self.declare_parameter("randomize", False)          # if true, choose random pos each publish
        self.declare_parameter("x_min", 400.0)
        self.declare_parameter("x_max", 500.0)
        self.declare_parameter("y_min", 80.0)
        self.declare_parameter("y_max", 120.0)
        self.declare_parameter("frame_id", "geometry")
        self.declare_parameter("enabled", False)

        self.enabled = self.get_parameter("enabled").get_parameter_value().bool_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.pos_x = self.get_parameter("pos_x").get_parameter_value().double_value
        self.pos_y = self.get_parameter("pos_y").get_parameter_value().double_value
        self.frequency_hz = self.get_parameter("frequency_hz").get_parameter_value().double_value
        self.randomize = self.get_parameter("randomize").get_parameter_value().bool_value
        self.x_min = self.get_parameter("x_min").get_parameter_value().double_value
        self.x_max = self.get_parameter("x_max").get_parameter_value().double_value
        self.y_min = self.get_parameter("y_min").get_parameter_value().double_value
        self.y_max = self.get_parameter("y_max").get_parameter_value().double_value

        # Publisher
        self.obj_pub = self.create_publisher(PoseStamped, "/geometry/camera_coord/object_center", 10)

        # Service to enable/disable publishing (for GUI)
        self.enable_srv = self.create_service(SetBool, "/system_sim/object_spawn/enable", self.handle_enable)

        # Parameter change callback
        self.add_on_set_parameters_callback(self.on_param_set)

        # Timer for publishing
        self.timer = None
        self._reset_timer()

        self.get_logger().info("ObjectSpawn node ready.")

    # ---------------- Timer helpers ----------------
    def _reset_timer(self):
        # Cancel previous timer if exists
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

        period_sec = 1.0 / max(self.frequency_hz, 1e-6)
        self.timer = self.create_timer(period_sec, self._publish_once)
        self.get_logger().info(f"Publishing every {period_sec:.3f}s (freq={self.frequency_hz:.3f} Hz)")

    # ---------------- Publish ----------------
    def _publish_once(self):
        if not self.enabled:
            return

        # Chọn vị trí: cố định hoặc ngẫu nhiên trong khoảng
        if self.randomize:
            import random
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
        else:
            x = float(self.pos_x)
            y = float(self.pos_y)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = x
        msg.pose.position.y = y

        # self.get_logger().info(str(msg))
        self.obj_pub.publish(msg)
        self.get_logger().info('published point')

    # ---------------- Service ----------------
    def handle_enable(self, req, resp):
        self.enabled = bool(req.data)
        resp.success = True
        resp.message = "Object spawn enabled" if self.enabled else "Object spawn disabled"
        self.get_logger().info(resp.message)
        return resp

    # ---------------- Parameter callback ----------------
    def on_param_set(self, params: List[Parameter]):
        for p in params:
            if p.name == "enabled":
                self.enabled = p.value
            elif p.name == "frame_id":
                self.frame_id = p.value
            elif p.name == "pos_x":
                self.pos_x = float(p.value)
            elif p.name == "pos_y":
                self.pos_y = float(p.value)
            elif p.name == "randomize":
                self.randomize = bool(p.value)
            elif p.name == "x_min":
                self.x_min = float(p.value)
            elif p.name == "x_max":
                self.x_max = float(p.value)
            elif p.name == "y_min":
                self.y_min = float(p.value)
            elif p.name == "y_max":
                self.y_max = float(p.value)
            elif p.name == "pos_z":
                self.pos_z = float(p.value)
            elif p.name == "frequency_hz":
                if p.value <= 0:
                    msg = "frequency_hz must be > 0"
                    self.get_logger().error(msg)
                    return SetParametersResult(successful=False, reason=msg)
                self.frequency_hz = float(p.value)
                self._reset_timer()

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSpawn()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
