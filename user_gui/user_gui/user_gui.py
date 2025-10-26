import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from example_interfaces.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import sys
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QRubberBand, QVBoxLayout
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QPointF, QTimer, QRect, QRectF, QPoint
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic
from interactivegraphicsview import InteractiveGraphicsView 

from pathlib import Path
import os
import cv2 as cv
import numpy as np


class RosQtBridge(QObject):
    # defind signal to cross threads
    topic_msg = pyqtSignal(str)
    trigger_done = pyqtSignal(bool, str)
    distorted_image_received = pyqtSignal(QPixmap)
    edge_image_received = pyqtSignal(QPixmap)


class RosUserGui(Node):
    def __init__(self, bridge: RosQtBridge):
        super().__init__('user_gui_node')
        self.bridge_cv = CvBridge()
        self.bridge = bridge

        # Image params
        self.declare_parameter('frame_id', 'user_gui_node')
        self.declare_parameter('pixel_format', 'mono8')
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.pixel_format = self.get_parameter('pixel_format').get_parameter_value().string_value

        # Def grab image client
        self.grab_one_cli = self.create_client(Trigger, '/camera/grab_one')
        # Def undistorted subscriber
        self.undistorted_sub = self.create_subscription(Image, '/camera/undistorted_image', self.undistorted_frame_cb, 1)
        self.edge_sub = self.create_subscription(Image, '/camera/edge_image', self.edge_frame_cb, 1)

        # Create client to call /edge_detect/set_parameters
        self.param_cli = self.create_client(SetParameters, '/edge_detect/set_parameters')
        while not self.param_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /edge_detect/set_parameters ...")
        
        # Create publisher to publish edge ROI
        self.roi_edge_pub = self.create_publisher(Image, '/camera/roi_edge', 1)

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

        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')

    def edge_frame_cb(self, msg):
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
            self.bridge.edge_image_received.emit(pixmap)

        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')

    def update_cannyThres1(self, value: int):
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='canny_thres1', value=int(value)).to_parameter_msg()
        ]
        fut = self.param_cli.call_async(req)
        fut.add_done_callback(self._on_param_updated)

    def update_cannyThres2(self, value: int):
        req = SetParameters.Request()
        req.parameters = [
            Parameter(name='canny_thres2', value=int(value)).to_parameter_msg()
        ]
        fut = self.param_cli.call_async(req)
        fut.add_done_callback(self._on_param_updated)

    def _on_param_updated(self, fut):
        try:
            result = fut.result()  # rcl_interfaces.srv.SetParameters_Response
            if not result.results:
                self.get_logger().warn("No results returned by set_parameters")
                return
            r = result.results[0]  # rcl_interfaces.msg.SetParametersResult
            if r.successful:
                self.get_logger().info("✅ Parameter updated successfully.")
            else:
                self.get_logger().warn(f"⚠️ Parameter update failed: {r.reason}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def publish_roi_edge(self, cv_img):
        rosimg = self.bridge_cv.cv2_to_imgmsg(cv_img)
        self.roi_edge_pub.publish(rosimg)
            

class MainWindow(QMainWindow):
    def __init__(self, node: RosUserGui, bridge: RosQtBridge):
        super().__init__()
        self.setWindowTitle('Gluedispensor Robot GUI')

        # ---- Create ros node data ----
        self.node = node
        self.bridge = bridge

        # Get path to .py file
        script_path = Path(__file__).resolve()
        # Get path to folder contain .py file
        script_dir = script_path.parent
        ui_path = script_dir / 'user_gui.ui'
        uic.loadUi(str(ui_path), self)

        # ---- Pressed timer ----
        self.videoTimer = QTimer(self)
        self.videoTimer.setInterval(100)
        self.videoTimer.timeout.connect(self.start_grabbing_video)

        # Graphics view state
        self._first_undistorted_frame = True
        self._first_edge_frame = True

        # ---- Connect function signal ----
        # Snap one button
        self.snapButton.clicked.connect(self.node.call_grab_one_trigger)
        # Video button
        self.videoButton.setCheckable(True) 
        self.videoButton.toggled.connect(
            lambda checked: self.videoTimer.start() if checked else self.videoTimer.stop()
            )
        # Crop ROI button
        self.roiCropButton.clicked.connect(self.roi_crop_trigger)
        # Canny thresold slider
        self.cannyThres1.valueChanged.connect(self.on_update_cannyThres1)
        self.cannyThres2.valueChanged.connect(self.on_update_cannyThres2)

        # ---- Connect bridge signals to GUI slots ----
        self.bridge.trigger_done.connect(self.on_trigger_done) 
        self.bridge.distorted_image_received.connect(self.update_undistorted_image)
        self.bridge.edge_image_received.connect(self.update_edge_image)

        # ---- Extract ROI frame for positioning ----

    # ---- Helper ----
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

    # ---- Normal Function ----
    def on_trigger_done(self, ok: bool, msg: str):
        text = f"{'OK' if ok else 'FAIL'}: {msg}"
        if hasattr(self, "statusLabel"):
            self.statusLabel.setText(text)
        else:
            self.setWindowTitle(text)

    def update_undistorted_image(self, pixmap: QPixmap):
        self.undistortedFrame.setPixmap(pixmap, fit_first=self._first_undistorted_frame)
        self._first_undistorted_frame = False
    
    def update_edge_image(self, pixmap: QPixmap):
        self.edgeFrame.setPixmap(pixmap, fit_first=self._first_edge_frame)    
        self._first_edge_frame = False

    def start_grabbing_video(self):
        self.node.call_grab_one_trigger()

    def roi_crop_trigger(self):
        roi_qimg = self.undistortedFrame.get_roi_qimage()
        if roi_qimg is None:
            if hasattr(self, "statusLabel"):
                self.statusLabel.setText("No ROI selected")
            return

        roi_pix = QPixmap.fromImage(roi_qimg)
        # show on the other view; fit only on first use if you like
        self.roiView.setPixmap(roi_pix, fit_first=True)
        cv_img = self.qimage_mono8_to_cv_image(roi_qimg)
        self.node.publish_roi_edge(cv_img)

    def on_update_cannyThres1(self, value: int):
        self.node.update_cannyThres1(value)
        self.cannyThres1Val.setText(str(value))

    def on_update_cannyThres2(self, value: int):
        self.node.update_cannyThres2(value)
        self.cannyThres2Val.setText(str(value))

    def resizeEvent(self, event):
        super().resizeEvent(event)
    

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