import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class GeneralizedHough(Node):
    def __init__(self):
        super().__init__('generalized_hough_node')
        self.bridge = CvBridge()

        # Create message_filter subscriber
        self.edge_img_sub= self.create_subscription(Image, '/camera/undistorted_image', self.edge_img_cb, 1)
        self.roi_edge_sub = self.create_subscription(Image, '/camera/roi_edge', self.roi_edge_cb, 1)

        # Create result publisher
        self.ght_result_pub = self.create_publisher(Image, '/camera/ght_result', 1)

        # self.edge_cv_image = None
        # self.roi_edge_cv_image = None

    def roi_edge_cb(self, msg):
        try:
            # Convert ROS Image -> NumPy, accept 8UC1, mono8, etc.
            templ = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if templ.ndim == 3:
                templ = cv.cvtColor(templ, cv.COLOR_BGR2GRAY)
            if templ.dtype != np.uint8:
                templ = templ.astype(np.uint8)
            self.roi_edge_cv_image = templ
        except Exception as e:
            self.get_logger().warning(f'Failed to set template: {e!r}')
    
    def edge_img_cb(self, msg):
        try:
            self.edge_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            if self.roi_edge_cv_image is not None:        
                self.ght_process(self.edge_cv_image, self.roi_edge_cv_image)
            else:
                self.get_logger().info('Edge-base matching templete do no define yet!')
        except:
            self.get_logger().error('GHT data input error or no sample image!')
    
    def ght_process(self, img_gray, templ_gray):
        try:
            # --- Sanity: ensure single-channel uint8 ---
            if img_gray.ndim == 3:
                img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
            if templ_gray.ndim == 3:
                templ_gray = cv.cvtColor(templ_gray, cv.COLOR_BGR2GRAY)
            if img_gray.dtype != np.uint8:   img_gray = img_gray.astype(np.uint8)
            if templ_gray.dtype != np.uint8: templ_gray = templ_gray.astype(np.uint8)

            h, w = templ_gray.shape[:2]
            if h < 5 or w < 5:
                raise ValueError(f"Template too small: {w}x{h}")
            if cv.countNonZero(templ_gray) == 0:
                raise ValueError("Template has no nonzero pixels (check edges).")

            # --- Create & configure GUIL (rotation + scale control) ---
            guil = cv.createGeneralizedHoughGuil()
            guil.setMinDist(10)
            guil.setLevels(360)          # angle quantization (1 deg bins if 0..360)
            guil.setDp(3)                # accumulator downscale
            guil.setMaxBufferSize(1000)

            # Rotation search window
            guil.setMinAngle(0)      # <-- adjust to your expected range
            guil.setMaxAngle( 30)
            guil.setAngleStep(1)       # finer -> slower
            guil.setAngleThresh(1500)    # gradient correlation threshold

            # Scale search window (tighten for speed & stability)
            guil.setMinScale(0.9)
            guil.setMaxScale(1.10)
            guil.setScaleStep(0.01)
            guil.setScaleThresh(50)

            # Position peak threshold (suppress weak detections)
            guil.setPosThresh(10)
            
            # Let GHT compute edges internally
            guil.setCannyLowThresh(100)
            guil.setCannyHighThresh(200)
            templ_edges = templ_gray
            scene_edges = img_gray

            # It’s valid to pass an edge map as the template
            guil.setTemplate(templ_edges)

            self.get_logger().info("GHT(Guil) detection start…")
            pos_guil, votes_guil = guil.detect(scene_edges)  # Nx1x4 or Nx4: [x, y, scale, angle]

            # --- Draw & publish ---
            out_bgr = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

            if pos_guil is not None and len(pos_guil) > 0:
                P = np.asarray(pos_guil, dtype=np.float32).reshape(-1, 4)
                for (cx, cy, scale, angle_deg) in P:
                    self.draw_rotated_box(out_bgr, cx, cy, w, h, scale, angle_deg, (0, 255, 0), 2)

            # Save debug image (optional)
            cv.imwrite('/home/thinh/ros2_ws/src/camera_worker_pkg/camera_worker_pkg/result_guil.png', out_bgr)

            rosimg = self.bridge.cv2_to_imgmsg(out_bgr, encoding='bgr8')
            self.ght_result_pub.publish(rosimg)
            self.get_logger().info("GHT(Guil) success.")
        except Exception as e:
            self.get_logger().error(f"GHT(Guil) error: {e!r}")

            

        
        except cv.error as e_cv:
            self.get_logger().error('--- LỖI OPENCV ---')
            self.get_logger().error(f'Hàm (Function): {e_cv.func}')
            self.get_logger().error(f'Mã lỗi (Code): {e_cv.code}')
            self.get_logger().error(f'Thông điệp (Message): {e_cv.err}')
            self.get_logger().error(f'File: {e_cv.filename}')
            self.get_logger().error(f'Dòng: {e_cv.line}')
            self.get_logger().error('--------------------')
        except:
            self.get_logger().error('GHT process error!')

    def draw_rotated_box(self, img, cx, cy, w, h, scale, angle_deg, color, thickness):
        # RotatedRect in OpenCV Python is represented by ((cx,cy),(w,h),angle_deg)
        rr = ((float(cx), float(cy)), (float(w*scale), float(h*scale)), float(angle_deg))
        box = cv.boxPoints(rr)  # 4x2 float
        box = np.int32(box)
        cv.polylines(img, [box], isClosed=True, color=color, thickness=thickness, lineType=cv.LINE_AA)

def main(args=None):
    rclpy.init(args=args)
    my_node = GeneralizedHough()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
