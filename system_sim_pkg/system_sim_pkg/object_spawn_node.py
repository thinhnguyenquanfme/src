#!/usr/bin/env python3
import random
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool
from robot_interfaces.msg import PoseStampedConveyor


class ObjectSpawn(Node):
    def __init__(self):
        super().__init__("object_spawn_node")

        # ===== PARAMETERS =====
        self.declare_parameter("simulate_mode", 0)   # 0=fixed, 1=fixed_random_y, 2=burst_mode
        self.declare_parameter("frequency_hz", 0.25)

        self.declare_parameter("pos_x", 450.0)
        self.declare_parameter("pos_y", 100.0)

        self.declare_parameter("y_min", 50.0)
        self.declare_parameter("y_max", 170.0)

        self.declare_parameter("frame_id", "geometry")
        self.declare_parameter("enabled", False)

        # ===== LOAD PARAMETERS =====
        self.enabled = self.get_parameter("enabled").value
        self.simulate_mode = self.get_parameter("simulate_mode").value
        self.frequency_hz = self.get_parameter("frequency_hz").value

        self.pos_x = self.get_parameter("pos_x").value
        self.pos_y = self.get_parameter("pos_y").value

        self.y_min = self.get_parameter("y_min").value
        self.y_max = self.get_parameter("y_max").value

        self.frame_id = self.get_parameter("frame_id").value

        # ID tăng dần
        self._next_id = 1

        # ===== PUB & SERVICE =====
        self.obj_pub = self.create_publisher(PoseStampedConveyor, "/geometry/camera_coord/object_center", 10)
        self.enable_srv = self.create_service(SetBool, "/system_sim/object_spawn/enable", self.handle_enable)

        self.add_on_set_parameters_callback(self.on_param_set)

        # ===== TIMER =====
        self.timer = None
        self._reset_timer()

        self.get_logger().info("ObjectSpawn ready.")

    # =========================================================
    # TIMER HANDLING
    # =========================================================
    def _reset_timer(self):
        if self.timer:
            self.timer.cancel()

        if self.simulate_mode in (0, 1):
            # fixed frequency
            period = 1.0 / max(self.frequency_hz, 1e-6)
            self.timer = self.create_timer(period, self._publish_once)
            self.get_logger().info(f"[Mode {self.simulate_mode}] Fixed freq = {self.frequency_hz} Hz")

        elif self.simulate_mode == 2:
            # burst mode → timer with dynamic period
            self.get_logger().info("[Mode 2] Burst mode (avg = 4 sec)")
            self._schedule_next_burst()

    def _schedule_next_burst(self):
        """Generate next random interval with mean 4 seconds."""
        dt = random.expovariate(1.0 / 4.0)   # mean = 4s
        if self.timer:
            self.timer.cancel()
        self.timer = self.create_timer(dt, self._burst_callback)

    def _burst_callback(self):
        self._publish_once()
        self._schedule_next_burst()

    # =========================================================
    # PUBLISH ONE OBJECT
    # =========================================================
    def _publish_once(self):
        if not self.enabled:
            return

        msg = PoseStampedConveyor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # =============== Mode logic ===============

        # MODE 0 — fixed point
        if self.simulate_mode == 0:
            x = self.pos_x
            y = self.pos_y

        # MODE 1 — fixed freq, random Y
        elif self.simulate_mode == 1:
            x = self.pos_x
            y = random.uniform(self.y_min, self.y_max)

        # MODE 2 — burst mode (random interval)
        elif self.simulate_mode == 2:
            x = self.pos_x
            y = random.uniform(self.y_min, self.y_max)

        # ID để node khác nhận dạng
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(self._next_id)
        msg.conv_pose = 0.0

        self._next_id += 1

        self.obj_pub.publish(msg)
        self.get_logger().info(
            f"Spawned object id={msg.pose.position.z:.0f} at x={x:.1f}, y={y:.1f}, conv_pose={msg.conv_pose:.1f}"
        )

    # =========================================================
    # SERVICE
    # =========================================================
    def handle_enable(self, req, resp):
        self.enabled = bool(req.data)
        resp.success = True
        resp.message = "Enabled" if self.enabled else "Disabled"
        self.get_logger().info(resp.message)
        return resp

    # =========================================================
    # PARAMETER UPDATE CALLBACK
    # =========================================================
    def on_param_set(self, params):
        for p in params:
            if p.name == "enabled":
                self.enabled = bool(p.value)
            elif p.name == "simulate_mode":
                self.simulate_mode = int(p.value)
                self._reset_timer()
            elif p.name == "frequency_hz":
                self.frequency_hz = float(p.value)
                self._reset_timer()
            elif p.name == "pos_x":
                self.pos_x = float(p.value)
            elif p.name == "pos_y":
                self.pos_y = float(p.value)
            elif p.name == "y_min":
                self.y_min = float(p.value)
            elif p.name == "y_max":
                self.y_max = float(p.value)
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
