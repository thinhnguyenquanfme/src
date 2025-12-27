import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from example_interfaces.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
from robot_interfaces.msg import PoseStampedConveyor, PlcCommand, PlcReadWord

import sys
import threading
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic
from interactivegraphicsview import InteractiveGraphicsView 

from pathlib import Path
import os
import cv2 as cv
import numpy as np
import datetime

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


CONV_REGISTER = 'D200'


class RosQtBridge(QObject):
    # defind signal to cross threads
    topic_msg = pyqtSignal(str)
    trigger_done = pyqtSignal(bool, str)
    distorted_image_received = pyqtSignal(QPixmap)
    seg_image_received = pyqtSignal(QPixmap)
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
        # Simulated data
        self.sim_robot_pose_sub = self.create_subscription(PoseStamped, '/simulation/robot_pose', self.sim_robot_pose_cb, 10)
        self.sim_object_pose_sub = self.create_subscription(PoseStampedConveyor, '/geometry/camera_coord/object_center', self.sim_object_pose_cb, 10)
        self.plc_read_word_sub = self.create_subscription(PlcReadWord, '/plc/plc_read_word', self._plc_read_word_cb, 10)

        # ============= Define client =============
        self.trajectory_planning_start_cli = self.create_client(SetBool, '/geometry/set_trajectory_start')
        self.sim_spawn_cli = self.create_client(SetBool, '/system_sim/object_spawn/enable')
        self.sim_robot_state_cli = self.create_client(SetBool, '/system_sim/robot_state_sim/enable')
        self.plc_cmd_pub = self.create_publisher(PlcCommand, '/plc/plc_cmd', 10)
        self.custom_object_pub = self.create_publisher(PoseStampedConveyor, '/geometry/camera_coord/object_center', 10)

        self.pending_custom_point = None

    def words_to_int32(self, lo16, hi16):
        raw = (int(hi16) << 16) | int(lo16)
        if raw & 0x80000000:
            raw -= 0x100000000
        return raw

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

    def sim_robot_pose_cb(self, msg: PoseStamped):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self.bridge.sim_robot_pose.emit(p)

    def sim_object_pose_cb(self, msg: PoseStampedConveyor):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self.bridge.sim_object_pose.emit(p)

    def send_custom_point(self, x: float, y: float):
        stamp = self.get_clock().now().to_msg()
        self.pending_custom_point = (x, y, stamp)

        cmd = PlcCommand()
        cmd.op = 'READ_WORD'
        cmd.device = CONV_REGISTER[0]
        cmd.start_addr = CONV_REGISTER[1:]
        cmd.read_count = 2
        self.plc_cmd_pub.publish(cmd)

    def _plc_read_word_cb(self, msg: PlcReadWord):
        if self.pending_custom_point is None:
            return
        if len(msg.data) < 2:
            self.get_logger().warn(f"Not enough PLC data, expected >=2 words, got {len(msg.data)}")
            self.pending_custom_point = None
            return
        conv_raw = self.words_to_int32(msg.data[0], msg.data[1])
        conv_pose_value = float(conv_raw) / 100.0

        x, y, stamp = self.pending_custom_point
        self.pending_custom_point = None

        out = PoseStampedConveyor()
        out.header.stamp = stamp
        out.header.frame_id = 'geometry'
        out.pose.position.x = float(x)
        out.pose.position.y = float(y)
        out.pose.position.z = 0.0
        out.conv_pose = conv_pose_value
        self.custom_object_pub.publish(out)
        self.get_logger().info(
            f'Published custom object center: {out.pose.position.x:.3f}, '
            f'{out.pose.position.y:.3f}, {out.header.stamp}, conv_pose={conv_pose_value:.3f}'
        )

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
        # Custom point send button
        self.customPointSend.clicked.connect(self.on_custom_point_send)

        # -------------- Simulation plot --------------
        self.init_simulation_plot()

        # Buffers for simulation plot
        self.sim_robot_pts = []
        self.sim_object_pts = []

        # ============= Connect bridge signals to GUI slots =============
        self.bridge.trigger_done.connect(self.on_trigger_done, Qt.ConnectionType.QueuedConnection) 
        self.bridge.distorted_image_received.connect(self.update_undistorted_image, Qt.ConnectionType.QueuedConnection)
        self.bridge.seg_image_received.connect(self.update_seg_image, Qt.ConnectionType.QueuedConnection)
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

    def on_custom_point_send(self):
        try:
            x_text = self.customPointX.text().strip()
            y_text = self.customPointY.text().strip()
            x = float(x_text) if x_text else 0.0
            y = float(y_text) if y_text else 0.0
        except ValueError:
            self.setWindowTitle("Invalid custom point input")
            self.node.get_logger().warn("Custom point input is not a valid number")
            return

        self.node.send_custom_point(x, y)
        self.setWindowTitle(f"Custom point sent: {x:.3f}, {y:.3f}")

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
        self.ax_sim.add_collection3d(rect_face)
        # Outline for legend reliability
        self.ax_sim.plot(rect_pts[:, 0], rect_pts[:, 1], rect_pts[:, 2], color="gray", linestyle=":", linewidth=1.5, label="Target area")
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
