import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from robot_interfaces.msg import PoseListMsg


class TrajectoryPlotter(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_plotter")

        self.sub = self.create_subscription(
            PoseListMsg, "/geometry/trajectory_data", self._on_traj, 10
        )
        self.timer = self.create_timer(0.05, self._on_timer)

        # Lưu (t, x, y, z) từ message gần nhất
        self.points: List[Tuple[float, float, float, float]] = []
        self.t0: float = 0.0

        # Chuẩn bị figure 3D
        plt.ion()
        self.fig = plt.figure("Trajectory Plotter")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.scatter = self.ax.scatter([], [], [], color="red")
        self.text = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)

    def _on_traj(self, msg: PoseListMsg) -> None:
        tfs = msg.trajectory.transforms
        stamps = msg.stamp
        if not tfs:
            self.get_logger().warn("Received trajectory with 0 transforms")
            return

        # Lưu điểm và thời gian (giả sử stamp cùng chiều dài với transforms)
        self.points = []
        for i, tf in enumerate(tfs):
            if i < len(stamps):
                t = float(stamps[i])
            else:
                # nếu thiếu stamp, dùng index như giây để không crash
                t = float(i)
            self.points.append((t, tf.translation.x, tf.translation.y, tf.translation.z))

        # Thời gian gốc để tính t_relative
        self.t0 = self.points[0][0]

        # Cập nhật giới hạn trục để nhìn hết dữ liệu
        xs = [p[1] for p in self.points]
        ys = [p[2] for p in self.points]
        zs = [p[3] for p in self.points]
        margin = 0.1
        self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        self.ax.set_zlim(min(zs) - margin, max(zs) + margin)

        self.get_logger().info(f"Loaded trajectory with {len(self.points)} points")

    def _on_timer(self) -> None:
        plt.pause(0.001)  # cho matplotlib xử lý event loop
        if not self.points:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        t_rel = now - self.t0

        # Chọn điểm có stamp <= now gần nhất, nếu vượt thì giữ điểm cuối
        current = self.points[-1]
        for p in self.points:
            if p[0] <= now:
                current = p
            else:
                break

        # Cập nhật scatter
        self.scatter.remove()
        self.scatter = self.ax.scatter([current[1]], [current[2]], [current[3]], color="red", s=40)
        self.text.set_text(f"t_rel = {t_rel:.2f}s\nXYZ = ({current[1]:.3f}, {current[2]:.3f}, {current[3]:.3f})")
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
