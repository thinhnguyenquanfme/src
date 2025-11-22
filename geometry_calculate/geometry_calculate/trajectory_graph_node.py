#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from robot_interfaces.msg import PoseListMsg
from std_msgs.msg import Float64MultiArray

import numpy as np

import csv
import os
from datetime import datetime

class TrajectoryGraph(Node):
    def __init__(self):
        super().__init__('trajectory_graph')
        self.create_subscription(PoseListMsg, '/geometry/trajectory_data', self.cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/geometry/trajectory_points', 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, '/geometry/velocity_profile', 10)


        # Create folder only once
        self.log_dir = "/home/thinh/trajectory_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def cb(self, msg: PoseListMsg):
        # Extract transforms
        tfs = msg.trajectory.transforms
        n = len(tfs)
        if n == 0:
            return

        # Convert transforms to Nx3 numpy array for 3D plot
        pts = np.array([[tf.translation.x,
                         tf.translation.y,
                         tf.translation.z]
                        for tf in tfs],
                       dtype=float)

        # Publish 3D points for GUI
        arr = Float64MultiArray()
        arr.data = pts.flatten().tolist()
        self.pub.publish(arr)

        # ====== Build time & velocity for velocity graph ======
        # Time stamps from msg.stamp (list of float)
        if msg.stamp:
            times = np.array(msg.stamp, dtype=float)
        else:
            # fallback if no stamp provided
            times = np.arange(n, dtype=float)

        # Make first time = 0 (relative time)
        t0 = times[0]
        rel_times = times - t0

        # Velocities from msg.trajectory.velocities (Twist list)
        vel_list = msg.trajectory.velocities
        if len(vel_list) == n:
            speeds = np.array([tw.linear.x for tw in vel_list], dtype=float)
        else:
            # if mismatch, just fill with zeros
            speeds = np.zeros(n, dtype=float)

        # Pack as [t0, v0, t1, v1, ...]
        tv = np.vstack([rel_times, speeds]).T  # shape (N, 2)

        vel_msg = Float64MultiArray()
        vel_msg.data = tv.flatten().tolist()
        self.vel_pub.publish(vel_msg)

        # ====== Optionally: save full CSV with time, xyz, vel ======
        self.save_csv(rel_times, pts, speeds)



    def save_csv(self, times, pts, speeds=None):
        # Create unique filename per trajectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"traj_{timestamp}.csv"
        full_path = os.path.join(self.log_dir, filename)

        with open(full_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            if speeds is None:
                writer.writerow(["idx", "time_s", "X", "Y", "Z"])
            else:
                writer.writerow(["idx", "time_s", "X", "Y", "Z", "speed"])

            # Rows
            for idx, (t, p) in enumerate(zip(times, pts)):
                x, y, z = p
                if speeds is None:
                    writer.writerow([idx, t, x, y, z])
                else:
                    writer.writerow([idx, t, x, y, z, speeds[idx]])

        self.get_logger().info(f"Saved CSV: {full_path}")



def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGraph()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
