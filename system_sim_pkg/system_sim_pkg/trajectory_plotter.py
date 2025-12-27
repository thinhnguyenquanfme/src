#!/usr/bin/env python3
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from robot_interfaces.msg import PoseListMsg, PoseStampedConveyor


class TrajectoryPlotter(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_plotter")

        # Parameters
        self.conveyor_speed_x = float(self.declare_parameter("conveyor_speed_x", -20.0).value)
        self.working_space_upper = float(self.declare_parameter("working_space_upper", 10.0).value)
        self.view_scale = float(self.declare_parameter("view_scale", 1.5).value)
        self.circle_radius = float(self.declare_parameter("circle_radius", 25.0).value)
        self.show_traj_line = True
        self.guide_z = 30.0

        # Subscriptions
        self.sub = self.create_subscription(PoseStamped, "/simulation/robot_pose", self._on_pose, 10)
        self.obj_sub = self.create_subscription(PoseStampedConveyor, "/geometry/camera_coord/object_center", self._on_object, 10)
        self.traj_sub = self.create_subscription(PoseListMsg, "/geometry/trajectory_data", self._on_traj, 10)

        # Timer
        self.timer = self.create_timer(0.05, self._on_timer)

        # Buffers
        self.points: List[Tuple[float, float, float, float]] = []
        self.t0: float | None = None
        self.objects = []  # {"t0": float, "p0": (x,y,z), "id": float}
        self.current_job_id: float | None = None
        self.seen_object_ids = set()
        self.chosen_object_ids = set()
        self.completed_object_ids = set()

        # Figure setup
        plt.ion()
        self.fig = plt.figure("Trajectory Plotter", figsize=(14, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.invert_zaxis()
        self.ax.set_box_aspect((1, 1, 0.2))

        self.scatter = self.ax.scatter([], [], [], color="red")
        self.traj_line, = self.ax.plot([], [], [], color="blue", linewidth=1, alpha=0.7)
        # Guide lines
        self.line_y50, = self.ax.plot([0.0, 500.0], [50.0, 50.0], [self.guide_z, self.guide_z], color="gray", linestyle="--", linewidth=1)
        self.line_y170, = self.ax.plot([0.0, 500.0], [170.0, 170.0], [self.guide_z, self.guide_z], color="gray", linestyle="--", linewidth=1)
        # Objects and circle
        self.obj_scatter_active = self.ax.scatter([], [], [], color="orange", marker="x", label="Active obj")
        self.obj_scatter_other = self.ax.scatter([], [], [], color="green", marker="x", label="Other obj")
        self.obj_scatter_done = self.ax.scatter([], [], [], color="red", marker="x", label="Done obj")
        self.circle_line, = self.ax.plot([], [], [], color="purple", linestyle="--", linewidth=1, label="Circle")

        self.text = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)

    def _on_pose(self, msg: PoseStamped) -> None:
        stamp = msg.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.t0 is None:
            self.t0 = t

        p = msg.pose.position
        self.points.append((t, p.x, p.y, p.z))
        if len(self.points) > 5000:
            self.points = self.points[-5000:]

    def _on_object(self, msg: PoseStampedConveyor) -> None:
        stamp = msg.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        p = msg.pose.position
        obj_id = float(p.z)
        self.objects.append({"t0": t, "p0": (p.x, p.y, self.guide_z), "id": obj_id})
        self.seen_object_ids.add(obj_id)

    def _on_traj(self, msg: PoseListMsg) -> None:
        tfs = msg.trajectory.transforms
        if not tfs:
            return
        tfs_msg = msg.trajectory
        tid = float(tfs_msg.time_from_start.sec) + float(tfs_msg.time_from_start.nanosec) * 1e-9
        # Mark previous job as done if switching
        if self.current_job_id is not None and abs(self.current_job_id - tid) > 1e-6:
            self.completed_object_ids.add(self.current_job_id)
        self.current_job_id = tid
        self.chosen_object_ids.add(tid)

    def _on_timer(self) -> None:
        plt.pause(0.001)
        if not self.points:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        t_rel = now - self.t0
        current = self.points[-1]

        # Robot point
        self.scatter.remove()
        self.scatter = self.ax.scatter([current[1]], [current[2]], [current[3]], color="red", s=40)

        # Trajectory line
        if self.show_traj_line:
            xs = [p[1] for p in self.points]
            ys = [p[2] for p in self.points]
            zs = [p[3] for p in self.points]
        else:
            xs, ys, zs = [], [], []
        self.traj_line.set_data(xs, ys)
        self.traj_line.set_3d_properties(zs)

        # Objects
        # ================== Objects ==================
        alive_objects = []
        active_positions = []   # list of (pos, id)
        other_positions = []    # list of pos
        done_positions = []     # list of pos

        for obj in self.objects:
            dt = now - obj["t0"]
            x0, y0, z0 = obj["p0"]
            x_curr = x0 + self.conveyor_speed_x * dt

            # remove if out of workspace
            if x_curr < self.working_space_upper:
                continue

            pos = (x_curr, y0, z0)
            oid = obj["id"]

            if oid in self.completed_object_ids:
                done_positions.append(pos)
            elif (self.current_job_id is not None) and (abs(oid - self.current_job_id) < 1e-6):
                active_positions.append((pos, oid))
            else:
                other_positions.append(pos)

            alive_objects.append(obj)

        self.objects = alive_objects

        # Draw scatters
        self.obj_scatter_active.remove()
        self.obj_scatter_other.remove()
        self.obj_scatter_done.remove()

        if active_positions:
            ax, ay, az = zip(*[p for p, _ in active_positions])
        else:
            ax, ay, az = [], [], []

        if other_positions:
            ox, oy, oz = zip(*other_positions)
        else:
            ox, oy, oz = [], [], []

        if done_positions:
            dx, dy, dz = zip(*done_positions)
        else:
            dx, dy, dz = [], [], []

        self.obj_scatter_active = self.ax.scatter(ax, ay, az, color="orange", marker="x", label="Active obj")
        self.obj_scatter_other = self.ax.scatter(ox, oy, oz, color="green", marker="x", label="Other obj")
        self.obj_scatter_done = self.ax.scatter(dx, dy, dz, color="red", marker="x", label="Done obj")

        # Display IDs for active objects
        for txt in getattr(self, "obj_texts", []):
            txt.remove()
        self.obj_texts = []
        for pos, oid in active_positions:
            x, y, z = pos
            label = f"ID:{int(oid)}"
            self.obj_texts.append(self.ax.text(x, y, z, label, color="orange", fontsize=8))

        # Circle following active object center
        circle_x: List[float] | np.ndarray = []
        circle_y: List[float] | np.ndarray = []
        circle_z: List[float] | np.ndarray = []

        if active_positions:
            active_center, active_id = active_positions[0]
            cx, cy, cz = active_center
            theta = np.linspace(0.0, 2.0 * math.pi, 100)
            circle_x = cx + self.circle_radius * np.cos(theta)
            circle_y = cy + self.circle_radius * np.sin(theta)
            circle_z = np.full_like(circle_x, cz)

        self.circle_line.set_data(circle_x, circle_y)
        self.circle_line.set_3d_properties(circle_z)

        # Axis limits
        # Limits based on robot + objects + guide lines (circle not stretching view)
        all_x = [p[1] for p in self.points] + list(ax) + list(ox) + list(dx) + [0.0, 500.0, 0.0, 500.0]
        all_y = [p[2] for p in self.points] + list(ay) + list(oy) + list(dy) + [50.0, 50.0, 170.0, 170.0]
        all_z = [p[3] for p in self.points] + list(az) + list(oz) + list(dz) + [self.guide_z] * 4
        if all_x and all_y and all_z:
            margin = 0.1
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            z_min, z_max = 0.0, 100.0

            x_mid = (x_min + x_max) / 2.0
            y_mid = (y_min + y_max) / 2.0
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 100.0)
            half = (max_range / 2.0 + margin) * max(self.view_scale, 1.0)

            self.ax.set_xlim(x_mid - half, x_mid + half)
            self.ax.set_ylim(y_mid - half, y_mid + half)
            self.ax.set_zlim(z_max, z_min)
            self.ax.set_box_aspect((1, 1, 0.2))
            # self.fig.set_size_inches(14, 5, forward=True)

        seen_cnt = len(self.seen_object_ids)
        chosen_cnt = len(self.chosen_object_ids)
        self.text.set_text(
            f"t_rel = {t_rel:.2f}s\n"
            f"XYZ = ({current[1]:.3f}, {current[2]:.3f}, {current[3]:.3f})\n"
            f"Input Objects: {seen_cnt} | Done: {chosen_cnt}"
        )
        self.fig.canvas.draw_idle()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryPlotter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.close("all")


if __name__ == "__main__":
    main()
