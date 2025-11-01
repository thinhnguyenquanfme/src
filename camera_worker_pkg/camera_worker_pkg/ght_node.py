import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class GeneralizedHough(Node):
    def __init__(self):
        super().__init__('generalized_hough_node')
        self.bridge = CvBridge()

        # Template var
        self.edge_templ_cv_img = None
        self.dx_templ_cv_img = None
        self.dy_templ_cv_img = None
        self.template_ready = False

        # Params
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('pixel_format', 'mono8') # Pixel format for process purpose
        self.declare_parameter('display_pixel_format', 'bgr8') # Pixel format for display purpose
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.pixel_format = self.get_parameter('pixel_format').get_parameter_value().string_value
        self.display_pixel_format = self.get_parameter('display_pixel_format').get_parameter_value().string_value

        # Template Subscriber
        self.dx_templ_sub = self.create_subscription(Image, '/camera/user_gui/dx_templ', self.dx_templ_cb, 1)
        self.dy_templ_sub = self.create_subscription(Image, '/camera/user_gui/dy_templ', self.dy_templ_cb, 1)
        self.roi_edge_sub = self.create_subscription(Image, '/camera/user_gui/roi_edge', self.roi_edge_cb, 1)

        # Main subscriber
        self.edge_main_sub = Subscriber(self, Image, '/camera/canny_node/edge_image')
        self.undistorted_sub = Subscriber(self, Image, '/camera/undistorted_image')
        self.dx_main_sub = Subscriber(self, Image, '/camera/canny_node/dx_main')
        self.dy_main_sub = Subscriber(self, Image, '/camera/canny_node/dy_main')

        # 'slop=0.5' nghĩa là các tin nhắn phải đến trong vòng 0.1 giây của nhau
        self.ts = ApproximateTimeSynchronizer(
            [self.edge_main_sub, self.undistorted_sub, self.dx_main_sub, self.dy_main_sub],
            queue_size=10, 
            slop=0.5 
        )
        self.ts.registerCallback(self.synchronized_data_cb)
        

    
        # Create result publisher
        self.ght_result_pub = self.create_publisher(Image, '/camera/ght/result', 1)
        # Create edge image publisher
        self.ght_edge_img_pub = self.create_publisher(Image, '/camera/ght/edge_img', 1)
        self.ght_edge_templ_pub = self.create_publisher(Image, '/camera/ght/edge_templ', 1)

        self.get_logger().info('GHT node started.')

        # --- Create & configure GUIL (rotation + scale control) ---
        self.guil = cv.createGeneralizedHoughGuil()
        self.guil.setMinDist(20)
        self.guil.setLevels(360)          # angle quantization (1 deg bins if 0..360)
        self.guil.setDp(5)                # accumulator downscale
        self.guil.setMaxBufferSize(1000)

        # Rotation search window
        self.guil.setMinAngle(0)      # <-- adjust to your expected range
        self.guil.setMaxAngle(90)
        self.guil.setAngleStep(1)       # finer -> sl)ower
        self.guil.setAngleThresh(2)    # gradient correlation threshold

        # Scale search window (tighten for speed & stability)
        self.guil.setMinScale(0.95)
        self.guil.setMaxScale(1.05)
        self.guil.setScaleStep(0.01)
        self.guil.setScaleThresh(1)

        # Position peak threshold (suppress weak detections)
        self.guil.setPosThresh(100)

    def check_template_ready(self):
        if self.template_ready:
            return True

        # Kiểm tra xem cả 3 biến template đã được gán chưa
        data_to_check = (
            self.edge_templ_cv_img,
            self.dx_templ_cv_img,
            self.dy_templ_cv_img
        )
        
        if not any(v is None for v in data_to_check):
            try:
                # --- GỌI SETTEMPLATE Ở ĐÂY (CHỈ 1 LẦN) ---
                h, w = self.edge_templ_cv_img.shape[:2]
                if h < 5 or w < 5: raise ValueError("Template quá nhỏ")
                if cv.countNonZero(self.edge_templ_cv_img) == 0: raise ValueError("Template không có cạnh")
                
                center = (w // 2, h // 2)
                self.guil.setTemplate(self.edge_templ_cv_img, self.dx_templ_cv_img, self.dy_templ_cv_img, center)
                
                # Lưu lại w, h để vẽ
                self.template_w = w
                self.template_h = h
                
                self.template_ready = True
                self.get_logger().info('GHT TEMPLATE ĐÃ SẴN SÀNG VÀ ĐƯỢC GÁN.')
                return True
            except Exception as e:
                self.get_logger().error(f'GÁN TEMPLATE THẤT BẠI: {e!r}')
                # Đặt lại để thử lại (nếu cần)
                self.edge_templ_cv_img = None 
                self.dx_templ_cv_img = None
                self.dy_templ_cv_img = None
                return False
        return False

    def roi_edge_cb(self, msg):
        try:
            self.edge_templ_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.pixel_format)
            self.check_template_ready()
            self.get_logger().info('GHT template created.')
        except Exception as e:
            self.get_logger().warning(f'Failed to set template: {e!r}')

    # def undistorted_img_cb(self, msg):
    #     try:
    #         self.undistorted_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.pixel_format)
    #         self.get_logger().info('GHT undistorted image grabbed.')
    #     except:
    #         self.get_logger().warning(f'Failed to get undistorted image: {e!r}')

    # def dx_main_cb(self, msg):
    #     try:
    #         self.dx_main_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    #         self.get_logger().info('GHT main dx created.')
    #     except:
    #         self.get_logger().warning(f'Failed to get dx_main image: {e!r}')
    # def dy_main_cb(self, msg):
    #     try:
    #         self.dy_main_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    #     except:
    #         self.get_logger().warning(f'Failed to get dy_main image: {e!r}')
    def dx_templ_cb(self, msg):
        try:
            self.dx_templ_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.check_template_ready()
            self.get_logger().info('GHT templ dx created.')
        except:
            self.get_logger().warning(f'Failed to get dx_templ image: {e!r}')
    def dy_templ_cb(self, msg):
        try:
            self.dy_templ_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.check_template_ready()
        except:
            self.get_logger().warning(f'Failed to get dy_templ image: {e!r}')


    
    # def edge_img_cb(self, msg):
    #     try:
    #         self.edge_main_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.pixel_format)
    #         if self.check_ght_source():        
    #             self.ght_process()
    #         else:
    #             self.get_logger().info('Edge-base matching templete do no define yet!')
    #     except Exception as e:
    #         self.get_logger().error(f'GHT data input error or no sample image!: {e!r}')

    def synchronized_data_cb(self, edge_main_msg, undistorted_msg, dx_main_msg, dy_main_msg):
        # 1. Kiểm tra xem template đã sẵn sàng chưa
        if not self.template_ready:
            self.get_logger().info('Đã nhận dữ liệu chính, nhưng template chưa sẵn sàng. Đang đợi...')
            return

        # 2. Template đã sẵn sàng, dữ liệu chính đã đồng bộ -> XỬ LÝ
        try:
            # Chuyển đổi tất cả 4 tin nhắn
            edge_main_cv_img = self.bridge.imgmsg_to_cv2(edge_main_msg, desired_encoding=self.pixel_format)
            undistorted_cv_img = self.bridge.imgmsg_to_cv2(undistorted_msg, desired_encoding=self.pixel_format)
            dx_main_cv_img = self.bridge.imgmsg_to_cv2(dx_main_msg, desired_encoding='passthrough')
            dy_main_cv_img = self.bridge.imgmsg_to_cv2(dy_main_msg, desired_encoding='passthrough')
            
            # 3. Gọi ght_process với TẤT CẢ 7 ảnh (4 từ đây, 3 từ self)
            self.ght_process(
                edge_main_cv_img, 
                undistorted_cv_img, 
                dx_main_cv_img, 
                dy_main_cv_img
            )
        except Exception as e:
            self.get_logger().error(f'Lỗi xử lý callback đồng bộ: {e!r}')
    
    def ght_process(self, edge_main_img, undistorted_img, dx_main_img, dy_main_img):
        try:
            self.get_logger().info("GHT(Guil) detection start…")
            pos_guil, votes_guil = self.guil.detect(edge_main_img, dx_main_img, dy_main_img)

            # --- Draw & publish ---
            out_bgr = cv.cvtColor(undistorted_img, cv.COLOR_GRAY2BGR)

            if pos_guil is not None and votes_guil is not None:
                # normalize shapes
                P = np.asarray(pos_guil,  dtype=np.float32).reshape(-1, 4)   # [N,4]
                V = np.asarray(votes_guil, dtype=np.float32).reshape(-1)     # [N]

                # match lengths and handle empty
                N = min(len(P), len(V))
                if N == 0:
                    self.get_logger().warn("GHT: no detections")
                else:
                    P = P[:N]; V = V[:N]

                    # choose the single most-voted detection
                    best_idx = int(np.argmax(V))
                    cx, cy, scale, angle_deg = map(float, P[best_idx])
                    best_vote = float(V[best_idx])

                    # draw best only
                    self.draw_rotated_box(out_bgr, cx, cy, self.template_w, self.template_h, scale, angle_deg, (255, 0, 0), 3)
                    # Dòng 1: In Vote
                    cv.putText(out_bgr, f"V: {best_vote:.0f}", (int(cx), int(cy) - 20),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv.LINE_AA)

                    # Dòng 2: In Scale và Angle ngay bên dưới
                    text_sa = f"S: {scale:.2f} A: {angle_deg:.1f}"
                    cv.putText(out_bgr, text_sa, (int(cx), int(cy) - 6),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv.LINE_AA)
                    
                    try:
                        # 1. Lấy tâm của ảnh mẫu (template)
                        center_x = self.template_w / 2.0
                        center_y = self.template_h / 2.0

                        # 2. Lấy ma trận biến đổi (Rotation + Scale) từ tâm của mẫu
                        M = cv.getRotationMatrix2D((center_x, center_y), float(angle_deg), float(scale))

                        # 3. Thêm ma trận tịnh tiến (Translation)
                        # Dịch chuyển tâm ảnh mẫu (center_x, center_y) 
                        # đến vị trí (cx, cy) đã tìm thấy
                        M[0, 2] += (float(cx) - center_x)
                        M[1, 2] += (float(cy) - center_y)

                        # 4. Kích thước ảnh đầu ra (phải bằng out_bgr)
                        dsize = (out_bgr.shape[1], out_bgr.shape[0]) # (width, height)

                        # 5. Biến đổi (Warp) ảnh cạnh của mẫu
                        # Kết quả là một ảnh xám (mono8)
                        warped_edges_mask = cv.warpAffine(self.edge_templ_cv_img, M, dsize)

                        # 6. Dùng ảnh đã biến đổi làm mặt nạ (mask) để vẽ màu xanh lá
                        # Chỉ vẽ ở những nơi mặt nạ > 0 (là các cạnh)
                        out_bgr[warped_edges_mask > 0] = (0, 255, 0) # Màu Xanh lá (B, G, R)

                    except Exception as e_draw:
                        self.get_logger().warn(f"Không thể vẽ chồng cạnh: {e_draw!r}")
                    # --- KẾT THÚC CODE MỚI ---
                    # --- Lặp qua TẤT CẢ các phát hiện ---
                    # for i in range(N):
                    #     # Lấy thông tin của phát hiện thứ i
                    #     cx, cy, scale, angle_deg = map(float, P[i])
                    #     vote_count = float(V[i])

                    #     # Vẽ phát hiện thứ i
                    #     self.draw_rotated_box(out_bgr, cx, cy, self.template_w, self.template_h, scale, angle_deg, (0, 255, 0), 1) # Giảm độ dày 1 chút
                    #     cv.putText(out_bgr, f"{vote_count:.0f}", (int(cx), int(cy)-6),
                    #             cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA) # Giảm kích thước text
            else:
                self.get_logger().warn("GHT: pos or votes is None")

            # Save debug image (optional)
            cv.imwrite('/home/thinh/ros2_ws/src/camera_worker_pkg/camera_worker_pkg/result_guil.png', out_bgr)

            rosimg = self.bridge.cv2_to_imgmsg(out_bgr, encoding='bgr8')
            self.ght_result_pub.publish(rosimg)
            self.get_logger().info("GHT(Guil) success.")
        except cv.error as e_cv:
            self.get_logger().error('--- LỖI OPENCV ---')
            self.get_logger().error(f'Hàm (Function): {e_cv.func}')
            self.get_logger().error(f'Mã lỗi (Code): {e_cv.code}')
            self.get_logger().error(f'Thông điệp (Message): {e_cv.err}')
            self.get_logger().error(f'File: {e_cv.filename}')
            self.get_logger().error(f'Dòng: {e_cv.line}')
            self.get_logger().error('--------------------')

        except Exception as e:
            self.get_logger().error(f"GHT(Guil) error: {e!r}")


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
