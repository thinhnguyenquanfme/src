#!/usr/bin/env python3
"""
Simulate robot pose feedback by playing back PoseListMsg waypoints over time.
Subscribes to /geometry/trajectory_data, interpolates positions by current ROS
time, and publishes a PoseStamped on /simulation/robot_pose at a fixed rate.
Can be toggled via parameter or SetBool service while waiting for real PLC.

Bổ sung:
- Ghi lại (t, x, y, z) trong suốt quá trình chạy mỗi trajectory
- Khi trajectory kết thúc -> tính vận tốc (vx, vy, vz) bằng sai phân
- Xuất ra 1 file CSV / trajectory trong thư mục cấu hình
"""

from bisect import bisect_right
from dataclasses import dataclass
from typing import List
import os
import csv

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
from robot_interfaces.msg import PoseListMsg


@dataclass
class Trajectory:
    times: List[float]
    xs: List[float]
    ys: List[float]
    zs: List[float]


class RobotStateSim(Node):
    def __init__(self):
        super().__init__("robot_state_sim_node")

        # Parameters
        self.declare_parameter("update_hz", 50.0)
        self.declare_parameter("frame_id", "geometry")
        self.declare_parameter("enabled", False)

        # Tham số cho logging CSV
        self.declare_parameter("enable_csv_log", True)
        self.declare_parameter("csv_output_dir", "/home/thinh/velocity_logs")

        self.update_hz = float(self.get_parameter("update_hz").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.enabled = bool(self.get_parameter("enabled").value)

        self.enable_csv = bool(self.get_parameter("enable_csv_log").value)
        self.csv_dir = str(self.get_parameter("csv_output_dir").value)
        os.makedirs(self.csv_dir, exist_ok=True)

        # Trajectory buffer (FIFO)
        self.queue: List[Trajectory] = []

        # Logging cho trajectory hiện tại
        # current_log: list of (t, x, y, z)
        self.current_log: List[tuple[float, float, float, float]] = []
        self.logging_active: bool = False
        self.traj_index: int = 0  # đếm số trajectory đã ghi file

        # Pub/Sub
        self.pose_pub = self.create_publisher(PoseStamped, "/simulation/robot_pose", 10)
        self.traj_sub = self.create_subscription(PoseListMsg, "/geometry/trajectory_data", self.traj_cb, 10)

        # Enable/disable service for GUI
        self.enable_srv = self.create_service(SetBool, "/system_sim/robot_state_sim/enable", self.handle_enable)

        # Parameter callback
        self.add_on_set_parameters_callback(self.on_param_set)

        # Timer
        self.timer = self.create_timer(1.0 / max(self.update_hz, 1e-3), self.publish_pose)

        self.get_logger().info(
            "RobotStateSim ready: listening to /geometry/trajectory_data and publishing /simulation/robot_pose"
        )

    # ---------------- Trajectory handling ----------------
    def traj_cb(self, msg: PoseListMsg):
        tfs = msg.trajectory.transforms
        stamps = list(msg.stamp)
        if not tfs:
            self.get_logger().warn("Received empty trajectory, ignoring")
            return

        # Ensure we have timestamps; if missing, make relative from now
        if not stamps or len(stamps) != len(tfs):
            now_s = self.get_clock().now().nanoseconds * 1e-9
            dt = 0.1
            stamps = [now_s + i * dt for i in range(len(tfs))]

        # Ensure monotonic order
        paired = sorted(zip(stamps, tfs), key=lambda x: x[0])
        times = [float(p[0]) for p in paired]
        xs = [float(p[1].translation.x) for p in paired]
        ys = [float(p[1].translation.y) for p in paired]
        zs = [float(p[1].translation.z) for p in paired]

        self.queue.append(Trajectory(times=times, xs=xs, ys=ys, zs=zs))
        self.get_logger().info(f"Queued trajectory with {len(times)} waypoints (queue size={len(self.queue)})")

    # ---------------- Publishing ----------------
    def publish_pose(self):
        if not self.enabled:
            return
        if not self.queue:
            # không có trajectory nào
            return

        now_s = self.get_clock().now().nanoseconds * 1e-9
        traj = self.queue[0]

        # ===== Trajectory đã kết thúc? =====
        if now_s > traj.times[-1]:
            # Publish final point once
            x_last = traj.xs[-1]
            y_last = traj.ys[-1]
            z_last = traj.zs[-1]

            # Log thêm 1 điểm cuối
            if self.enable_csv and self.logging_active:
                self.current_log.append((traj.times[-1], x_last, y_last, z_last))
                self._write_csv_for_current_traj()
                self.current_log = []
                self.logging_active = False

            self.pose_pub.publish(self._make_pose(traj.times[-1], x_last, y_last, z_last))
            self.queue.pop(0)
            return

        # ===== Trajectory đang chạy =====
        # Bắt đầu logging nếu chưa bật
        if self.enable_csv and not self.logging_active:
            self.current_log = []
            self.logging_active = True

        # Find segment
        idx = bisect_right(traj.times, now_s) - 1
        if idx < 0:
            idx = 0
        if idx >= len(traj.times) - 1:
            x = traj.xs[-1]
            y = traj.ys[-1]
            z = traj.zs[-1]
        else:
            t0 = traj.times[idx]
            t1 = traj.times[idx + 1]
            w = 0.0 if t1 == t0 else (now_s - t0) / (t1 - t0)
            x = traj.xs[idx] + w * (traj.xs[idx + 1] - traj.xs[idx])
            y = traj.ys[idx] + w * (traj.ys[idx + 1] - traj.ys[idx])
            z = traj.zs[idx] + w * (traj.zs[idx + 1] - traj.zs[idx])

        # Log point (t, x, y, z)
        if self.enable_csv and self.logging_active:
            self.current_log.append((now_s, x, y, z))

        msg = self._make_pose(now_s, x, y, z)
        self.pose_pub.publish(msg)

    def _make_pose(self, stamp_s: float, x: float, y: float, z: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp.sec = int(stamp_s)
        msg.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1e9)
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        return msg

    # ---------------- CSV Writing ----------------
    def _write_csv_for_current_traj(self):
        if not self.current_log:
            return

        times, xs, ys, zs = zip(*self.current_log)

        # Tính vận tốc bằng sai phân
        vxs: List[float] = []
        vys: List[float] = []
        vzs: List[float] = []

        for i in range(len(times)):
            if i == 0:
                # điểm đầu: cho vận tốc = 0 hoặc bằng điểm kế tiếp
                vxs.append(0.0)
                vys.append(0.0)
                vzs.append(0.0)
            else:
                dt = times[i] - times[i - 1]
                if dt <= 0:
                    vxs.append(vxs[-1])
                    vys.append(vys[-1])
                    vzs.append(vzs[-1])
                else:
                    vxs.append((xs[i] - xs[i - 1]) / dt)
                    vys.append((ys[i] - ys[i - 1]) / dt)
                    vzs.append((zs[i] - zs[i - 1]) / dt)

        self.traj_index += 1
        filename = f"trajectory_{self.traj_index:03d}.csv"
        filepath = os.path.join(self.csv_dir, filename)

        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "x", "y", "z", "vx", "vy", "vz"])
                for t, x, y, z, vx, vy, vz in zip(times, xs, ys, zs, vxs, vys, vzs):
                    writer.writerow([f"{t:.6f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                                     f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}"])
            self.get_logger().info(f"CSV trajectory saved: {filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV: {e}")

    # ---------------- Services & params ----------------
    def handle_enable(self, req, resp):
        self.enabled = bool(req.data)
        resp.success = True
        resp.message = "Robot state sim enabled" if self.enabled else "Robot state sim disabled"
        self.get_logger().info(resp.message)
        return resp

    def on_param_set(self, params: List[Parameter]):
        for p in params:
            if p.name == "enabled":
                self.enabled = bool(p.value)
            elif p.name == "frame_id":
                self.frame_id = str(p.value)
            elif p.name == "update_hz":
                if p.value <= 0:
                    return SetParametersResult(successful=False, reason="update_hz must be > 0")
                self.update_hz = float(p.value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.update_hz, self.publish_pose)
            elif p.name == "enable_csv_log":
                self.enable_csv = bool(p.value)
            elif p.name == "csv_output_dir":
                self.csv_dir = str(p.value)
                os.makedirs(self.csv_dir, exist_ok=True)

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateSim()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
