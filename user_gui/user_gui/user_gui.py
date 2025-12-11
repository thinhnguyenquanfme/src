import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from example_interfaces.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from robot_interfaces.msg import PoseListMsg
from std_srvs.srv import SetBool
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped

import sys
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QRubberBand, QVBoxLayout
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QPointF, QTimer, QRect, QRectF, QPoint
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic
from interactivegraphicsview import InteractiveGraphicsView 
import pyqtgraph as pg

from pathlib import Path
import os
import cv2 as cv
import numpy as np
import datetime


import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class RosQtBridge(QObject):
    # defind signal to cross threads
    topic_msg = pyqtSignal(str)
    trigger_done = pyqtSignal(bool, str)
    distorted_image_received = pyqtSignal(QPixmap)
    edge_image_received = pyqtSignal(QPixmap)
    seg_image_received = pyqtSignal(QPixmap)
    trajectory_plot = pyqtSignal(object)
    trajectory_3d = pyqtSignal(object)
    velocity_plot = pyqtSignal(object)
    sim_robot_pose = pyqtSignal(object)   # np.array([x,y,z])
    sim_object_pose = pyqtSignal(object)


class RosUserGui(Node):
    def __init__(self, bridge: RosQtBridge):
        super().__init__('user_gui_node')
        self.bridge_cv = CvBridge()
        self.bridge = bridge

        # Image params
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('pixel_format', 'mono8')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.pixel_format = self.get_parameter('pixel_format').get_parameter_value().string_value

        # Def grab image client
        self.grab_one_cli = self.create_client(Trigger, '/camera/grab_one')

        # ============= Define Subscriber =============
        # Def undistorted subscriber
        self.undistorted_sub = self.create_subscription(Image, '/camera/undistorted_image', self.undistorted_frame_cb, 1)
        # Def segment image result subscriber
        self.seg_img_sub = self.create_subscription(Image, '/camera/segment_node/segmented_image', self.seg_image_cb, 1)
        # Def trajectory tranform data
        self.trajectory_data_sub = self.create_subscription(PoseListMsg, '/geometry/trajectory_data', self.trajectory_data_cb, 1)
        # Def 3d point trajectory
        self.trajectory_3d_sub = self.create_subscription(Float64MultiArray, '/geometry/trajectory_points', self.trajectory_3d_cb, 1)
        # Def velocity profile
        self.velocity_sub = self.create_subscription(Float64MultiArray, '/geometry/velocity_profile', self.velocity_cb, 1)
        # Simulated data
        self.sim_robot_pose_sub = self.create_subscription(PoseStamped, '/simulation/robot_pose', self.sim_robot_pose_cb, 10)
        self.sim_object_pose_sub = self.create_subscription(PoseStamped, '/geometry/camera_coord/object_center', self.sim_object_pose_cb, 10)

        # ============= Define client =============
        self.trajectory_planning_start_cli = self.create_client(SetBool, '/geometry/set_trajectory_start')
        self.sim_spawn_cli = self.create_client(SetBool, '/system_sim/object_spawn/enable')
        self.sim_robot_state_cli = self.create_client(SetBool, '/system_sim/robot_state_sim/enable')

    def call_grab_one_trigger(self):
        if not self.grab_one_cli.wait_for_service(timeout_sec=0.5):
            self.bridge.trigger_done.emit(False, "Service unavaiable")
            return
        future = self.grab_one_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_grab_one_trigger_done)

    def _on_grab_one_trigger_done(self, fut):
        try:
            res = fut.result()
            self.bridge.trigger_done.emit(bool(res.success), str(res.message))
        except Exception as e:
            self.bridge.trigger_done.emit(False, f"Error {e}")

    def undistorted_frame_cb(self, msg):
        try: 
            # Convert ROSmsg to cv
            cv_image = self.bridge_cv.imgmsg_to_cv2(msg, desired_encoding=self.pixel_format)
            h, w = cv_image.shape
            bytes_per_line = w
            # Draw an border
            pt1 = (0, 0)
            pt2 = (w - 1, h - 1)
            border_color = (0, 0, 255) #BGR
            border_thickness = 2
            img_w_border = cv.rectangle(cv_image, pt1, pt2, border_color, border_thickness)

            # Convert cv to QImage
            qimg = QImage(img_w_border.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8).copy()

            # Convet QImage to QPixmap
            pixmap = QPixmap.fromImage(qimg)

            # Emit signals
            self.bridge.distorted_image_received.emit(pixmap)
            
            # Save image for training
            # now = datetime.datetime.now()
            # timestamp = now.strftime("%Y%m%d_%H%M%S")
            # filename = f"speaker_{timestamp}.jpg"
            # full_path = os.path.join('/home/thinh/yolov11/speaker_training_image', filename)
            # success = cv.imwrite(full_path, cv_image)
            
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')

    def seg_image_cb(self, msg):
        try: 
            # Convert ROSmsg to cv
            cv_image = self.bridge_cv.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w, channel = cv_image.shape
            bytes_per_line = w * channel          

            # Convert cv to QImage
            qimg = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()

            # Convet QImage to QPixmap
            pixmap = QPixmap.fromImage(qimg)

            # Emit signals
            self.bridge.seg_image_received.emit(pixmap)

        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
    
    def trajectory_data_cb(self, msg):
        # Extract x,y from incoming transforms
        xs = []
        ys = []
        for tf in msg.trajectory.transforms:   # iterate directly
            xs.append(tf.translation.x)
            ys.append(tf.translation.y)

        if not xs:  # nothing to plot
            return

        xy = np.vstack([xs, ys])  # shape (2, N)
        self.bridge.trajectory_plot.emit(xy)   # send to Qt thread


    def call_start_task(self, flag:bool):
        req = SetBool.Request()
        req.data = flag
        future = self.trajectory_planning_start_cli.call_async(req)
        future.add_done_callback(self._callback_start_response)

    def _callback_start_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Start trajectory planning success: {response.message}")
            else:
                self.get_logger().warn(f"Start trajectory planning failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def trajectory_3d_cb(self, msg: Float64MultiArray):
        # self.get_logger().info(str(msg))
        data = np.array(msg.data, dtype=float)
        if data.size == 0:
            return
        pts = data.reshape(-1, 3)   # shape (N, 3)
        self.bridge.trajectory_3d.emit(pts)

    def velocity_cb(self, msg: Float64MultiArray):
        data = np.array(msg.data, dtype=float)
        if data.size == 0:
            return

        pts = data.reshape(-1, 2)  # each row: [time, velocity]
        self.bridge.velocity_plot.emit(pts)

    def sim_robot_pose_cb(self, msg: PoseStamped):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self.bridge.sim_robot_pose.emit(p)

    def sim_object_pose_cb(self, msg: PoseStamped):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self.bridge.sim_object_pose.emit(p)

    def call_set_sim_mode(self, flag: bool):
        # enable/disable both sim services
        services = [
            (self.sim_spawn_cli, "object_spawn"),
            (self.sim_robot_state_cli, "robot_state_sim"),
        ]
        for cli, name in services:
            if not cli.wait_for_service(timeout_sec=0.5):
                self.get_logger().warn(f"Service {name} unavailable")
                continue
            req = SetBool.Request()
            req.data = flag
            fut = cli.call_async(req)
            fut.add_done_callback(lambda f, n=name: self._on_sim_service_done(f, n, flag))

    def _on_sim_service_done(self, future, name: str, flag: bool):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"{name} set to {flag}")
            else:
                self.get_logger().warn(f"{name} failed: {res.message}")
        except Exception as e:
            self.get_logger().error(f"Service {name} call error: {e}")

class MainWindow(QMainWindow):
    def __init__(self, node: RosUserGui, bridge: RosQtBridge):
        super().__init__()
        self.setWindowTitle('Gluedispensor Robot GUI')

        # ============= Create ros node data =============
        self.node = node
        self.bridge = bridge

        # Get path to .py file
        script_path = Path(__file__).resolve()
        # Get path to folder contain .py file
        script_dir = script_path.parent
        ui_path = script_dir / 'user_gui.ui'
        uic.loadUi(str(ui_path), self)

        # ============= Pressed timer =============
        self.videoTimer = QTimer(self)
        self.videoTimer.setInterval(100)
        self.videoTimer.timeout.connect(self.start_grabbing_video)

        # Graphics view state
        self._first_undistorted_frame = True
        self._first_edge_frame = True
        self._first_ght_frame = True

        # ============= Connect function signal =============
        # Snap one button
        self.snapButton.clicked.connect(self.node.call_grab_one_trigger)
        # Video button
        self.videoButton.setCheckable(True) 
        self.videoButton.toggled.connect(
            lambda checked: self.videoTimer.start() if checked else self.videoTimer.stop()
            )
        # START button
        self.startButton.setCheckable(True) 
        self.startButton.toggled.connect(self.on_start_button_toggled)
        # Sim Mode button
        self.simModeButton.setCheckable(True)
        self.simModeButton.toggled.connect(self.on_sim_mode_toggled)

        # Replace the placeholder QWidget (named trajectoryPlot) with a pyqtgraph PlotWidget

        # -------------- 3D trajectory --------------
        self.init_3d_plot()
        # -------------- Velocity plot --------------
        self.init_velocity_plot()
        # -------------- Simulation plot --------------
        self.init_simulation_plot()

        # Buffers for simulation plot
        self.sim_robot_pts = []
        self.sim_object_pts = []

        # ============= Connect bridge signals to GUI slots =============
        self.bridge.trigger_done.connect(self.on_trigger_done, Qt.ConnectionType.QueuedConnection) 
        self.bridge.distorted_image_received.connect(self.update_undistorted_image, Qt.ConnectionType.QueuedConnection)
        self.bridge.seg_image_received.connect(self.update_seg_image, Qt.ConnectionType.QueuedConnection)
        self.bridge.trajectory_3d.connect(self.update_trajectory_3d, Qt.ConnectionType.QueuedConnection)
        self.bridge.velocity_plot.connect(self.update_velocity_plot, Qt.ConnectionType.QueuedConnection)
        self.bridge.sim_robot_pose.connect(self.on_sim_robot_pose, Qt.ConnectionType.QueuedConnection)
        self.bridge.sim_object_pose.connect(self.on_sim_object_pose, Qt.ConnectionType.QueuedConnection)

        # Simulation plot timer (100ms)
        self.simPlotTimer = QTimer(self)
        self.simPlotTimer.setInterval(100)
        self.simPlotTimer.timeout.connect(self.update_simulation_plot)
        self.simPlotTimer.start()

    # ++++++++++++++++ Helper ++++++++++++++++
    def qimage_mono8_to_cv_image(self, qimage):
        # Neu format goc khong phai la Grayscale8, chuyen doi no
        if qimage.format() != QImage.Format.Format_Grayscale8:
            qimage = qimage.convertToFormat(QImage.Format.Format_Grayscale8)

        height = qimage.height()
        width = qimage.width()
        bytes_per_line = qimage.bytesPerLine() 
        ptr = qimage.bits()
        ptr.setsize(qimage.sizeInBytes())
        arr_1d = np.frombuffer(ptr, dtype=np.uint8).copy()
        
        if bytes_per_line == width:
            cv_image = arr_1d.reshape(height, width)
        else:
            arr_padded = arr_1d.reshape(height, bytes_per_line)
            cv_image = arr_padded[:, :width].copy() 
        return cv_image

    def draw_coordinate_frame(self, origin=(0.0, 0.0, 0.0), length=50.0):
        """
        Draw a simple 3D coordinate frame (X-red, Y-green, Z-blue)
        inside ax3d, starting at 'origin' with given 'length'.
        """
        ox, oy, oz = origin

        # X axis (red)
        self.ax3d.plot(
            [ox, ox + length],
            [oy, oy],
            [oz, oz],
            color='r',
            linewidth=2,
        )

        # Y axis (green)
        self.ax3d.plot(
            [ox, ox],
            [oy, oy + length],
            [oz, oz],
            color='g',
            linewidth=2,
        )

        # Z axis (blue)
        self.ax3d.plot(
            [ox, ox],
            [oy, oy],
            [oz, oz + length],
            color='b',
            linewidth=2,
        )

        # Optional axis labels near the tip
        self.ax3d.text(ox + length, oy, oz, 'X', color='r')
        self.ax3d.text(ox, oy + length, oz, 'Y', color='g')
        self.ax3d.text(ox, oy, oz + length, 'Z', color='b')

    # ++++++++++++++++ Normal Function ++++++++++++++++
    def on_trigger_done(self, ok: bool, msg: str):
        text = f"{'OK' if ok else 'FAIL'}: {msg}"
        if hasattr(self, "statusLabel"):
            self.statusLabel.setText(text)
        else:
            self.setWindowTitle(text)

    def update_undistorted_image(self, pixmap: QPixmap):
        self.undistortedFrame.setPixmap(pixmap, fit_first=self._first_undistorted_frame)
        undistorted_qimg = pixmap.toImage()
        self.undistorted_cv_img = self.qimage_mono8_to_cv_image(undistorted_qimg)
        self._first_undistorted_frame = False

    def start_grabbing_video(self):
        self.node.call_grab_one_trigger()

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def update_seg_image(self, pixmap: QPixmap):
        self.segImageFrame.setPixmap(pixmap, fit_first=self._first_ght_frame)
        self._first_ght_frame = False

    def on_start_button_toggled(self, checked: bool):
        # Send to ROS service
        self.node.call_start_task(checked)

        # # Optionally start/stop the video timer
        # if checked:
        #     self.videoTimer.start()
        # else:
        #     self.videoTimer.stop()

    def on_sim_mode_toggled(self, checked: bool):
        self.node.call_set_sim_mode(checked)
        self.simModeButton.setText("Simulation ON" if checked else "Simulation Mode")

    def init_3d_plot(self):
        """Embed a Matplotlib 3D canvas into the placeholder QWidget 'trajectoryPlot'."""
        # Create a Figure and a Canvas
        self.fig = Figure( tight_layout=False)
        self.canvas = FigureCanvas(self.fig)

        # Add 3D axes
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("Robot trajectory")

        # Optional
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)


        # Put canvas + toolbar into the placeholder QWidget in .ui
        # trajectoryPlot is a QWidget created in Qt Designer
        layout = self.trajectoryPlot.layout()
        if layout is None:
            layout = QVBoxLayout(self.trajectoryPlot)
            self.trajectoryPlot.setLayout(layout)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def update_trajectory_3d(self, pts):
        if pts.size == 0:
            return

        self.ax3d.cla()

        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]

        # === Equal scale logic (như bạn đã làm) ===
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        z_min, z_max = zs.min(), zs.max()

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6)
        half = max_range / 2

        # self.ax3d.set_xlim(x_mid - half, x_mid + half)
        # self.ax3d.set_ylim(y_mid - half, y_mid + half)
        # self.ax3d.set_zlim(z_mid - half, z_mid + half)

        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("Robot Trajectory")

        # If you want Z pointing down:
        self.ax3d.invert_zaxis()

        self.ax3d.set_box_aspect([1, 1, 1])

        # 🔹 Draw coordinate axes (choose origin & length)
        self.draw_coordinate_frame(origin=(0.0, 0.0, 0.0), length=50.0)

        # Plot trajectory
        self.ax3d.plot(xs, ys, zs, linewidth=2)
        self.ax3d.scatter(xs[0], ys[0], zs[0], color='green', s=40, label='Start')
        self.ax3d.scatter(xs[-1], ys[-1], zs[-1], color='red', s=40, label='End')
        self.ax3d.legend()

        # Force immediate repaint in GUI thread
        self.canvas.draw()


    def init_velocity_plot(self):
        """Embed a Matplotlib 2D canvas into the placeholder QWidget 'velocityGraph'."""
        self.fig_vel = Figure(tight_layout=True)
        self.canvas_vel = FigureCanvas(self.fig_vel)

        self.ax_vel = self.fig_vel.add_subplot(111)
        self.ax_vel.set_xlabel("Time (s)")
        self.ax_vel.set_ylabel("Velocity")
        self.ax_vel.set_title("Velocity profile")
        self.ax_vel.grid(True)

        self.toolbar_vel = NavigationToolbar(self.canvas_vel, self)

        layout = self.velocityGraph.layout()
        if layout is None:
            layout = QVBoxLayout(self.velocityGraph)
            self.velocityGraph.setLayout(layout)

        layout.addWidget(self.toolbar_vel)
        layout.addWidget(self.canvas_vel)

    def update_velocity_plot(self, pts):
        """
        pts: Nx2 numpy array
        pts[:,0] = time (s), starting at 0
        pts[:,1] = velocity
        """
        if pts.size == 0:
            return

        t = pts[:, 0]
        v = pts[:, 1]

        self.ax_vel.cla()
        self.ax_vel.set_xlabel("Time (s)")
        self.ax_vel.set_ylabel("Velocity")
        self.ax_vel.set_title("Velocity profile")
        self.ax_vel.grid(True)

        self.ax_vel.plot(t, v, linewidth=2)
        self.canvas_vel.draw()

    # ===== Simulation plot (robot + object) =====
    def init_simulation_plot(self):
        self.fig_sim = Figure(tight_layout=True)
        self.canvas_sim = FigureCanvas(self.fig_sim)
        self.ax_sim = self.fig_sim.add_subplot(111, projection="3d")
        self.ax_sim.set_xlabel("X")
        self.ax_sim.set_ylabel("Y")
        self.ax_sim.set_zlabel("Z")
        self.ax_sim.set_title("Simulation 3D")
        self.ax_sim.grid(True)
        self.ax_sim.invert_zaxis()  # Z points downward
        self.ax_sim.set_box_aspect([1, 1, 1])

        layout = self.simulationPlot.layout()
        if layout is None:
            layout = QVBoxLayout(self.simulationPlot)
            self.simulationPlot.setLayout(layout)
        layout.addWidget(self.canvas_sim)

    def on_sim_robot_pose(self, p):
        self.sim_robot_pts.append(p)
        if len(self.sim_robot_pts) > 2000:
            self.sim_robot_pts = self.sim_robot_pts[-2000:]

    def on_sim_object_pose(self, p):
        self.sim_object_pts.append(p)
        if len(self.sim_object_pts) > 2000:
            self.sim_object_pts = self.sim_object_pts[-2000:]

    def update_simulation_plot(self):
        self.ax_sim.cla()
        self.ax_sim.set_xlabel("X")
        self.ax_sim.set_ylabel("Y")
        self.ax_sim.set_zlabel("Z")
        self.ax_sim.set_title("Simulation 3D")
        self.ax_sim.grid(True)
        self.ax_sim.invert_zaxis()  # Z points downward
        self.ax_sim.set_box_aspect([1, 1, 1])

        # Collect points for equal aspect limits
        pts_for_bounds = []

        # Static colored rectangle on XY plane (Z=180)
        rect_pts = np.array([
            [0.0, 50.0, 180.0],
            [500.0, 50.0, 180.0],
            [500.0, 150.0, 180.0],
            [0.0, 150.0, 180.0],
            [0.0, 50.0, 180.0],
        ])
        rect_face = Poly3DCollection([rect_pts], facecolors="lightgray", alpha=0.3, edgecolors="gray", linewidths=1.5)
        rect_face.set_label("Target area")
        self.ax_sim.add_collection3d(rect_face)
        # Outline for legend reliability
        self.ax_sim.plot(rect_pts[:, 0], rect_pts[:, 1], rect_pts[:, 2], color="gray", linestyle=":", linewidth=1.5)
        pts_for_bounds.append(rect_pts)

        if self.sim_robot_pts:
            rp = np.vstack(self.sim_robot_pts)
            self.ax_sim.plot(rp[:, 0], rp[:, 1], rp[:, 2], color="blue", linewidth=2, label="Robot path")
            self.ax_sim.scatter(rp[-1, 0], rp[-1, 1], rp[-1, 2], color="blue", s=40, marker="o", label="Robot")
            pts_for_bounds.append(rp)

        if self.sim_object_pts:
            op = np.vstack(self.sim_object_pts)
            self.ax_sim.plot(op[:, 0], op[:, 1], op[:, 2], color="orange", linewidth=1.5, linestyle="--", label="Object path")
            self.ax_sim.scatter(op[-1, 0], op[-1, 1], op[-1, 2], color="orange", s=40, marker="x", label="Object")
            pts_for_bounds.append(op)

        # Enforce cubic aspect by expanding axes to the largest range
        if pts_for_bounds:
            all_pts = np.vstack(pts_for_bounds)
            mins = all_pts.min(axis=0)
            maxs = all_pts.max(axis=0)
            centers = (mins + maxs) / 2.0
            max_range = np.max(maxs - mins)
            if max_range < 1e-6:
                max_range = 1.0  # avoid zero-size box
            half = max_range / 2.0
            x0, y0, z0 = centers
            self.ax_sim.set_xlim(x0 - half, x0 + half)
            self.ax_sim.set_ylim(y0 - half, y0 + half)
            self.ax_sim.set_zlim(z0 - half, z0 + half)
            self.ax_sim.set_box_aspect([1, 1, 1])
            self.ax_sim.invert_zaxis()  # Z points downward

        handles, labels = self.ax_sim.get_legend_handles_labels()
        if handles:
            self.ax_sim.legend(handles, labels)

        # Force repaint even when user is not interacting
        self.canvas_sim.draw()



def main():
    # ROS
    rclpy.init(args=sys.argv)

    # Qt app
    app = QApplication(sys.argv)

    # Bridge + node
    bridge = RosQtBridge()
    ros_node = RosUserGui(bridge)

    # executor + spin thread
    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)

    def spin():
        executor.spin()

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()

    # GUI
    win = MainWindow(ros_node, bridge)
    win.show()

    # shutdown when GUI quit
    def on_about_to_quit():
        executor.shutdown()
        executor.remove_node(ros_node)
        ros_node.destroy_node()
        rclpy.shutdown()

    app.aboutToQuit.connect(on_about_to_quit)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
