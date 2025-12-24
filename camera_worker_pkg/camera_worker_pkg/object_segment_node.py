from ultralytics import YOLO

import numpy as np
import cv2 as cv

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from robot_interfaces.msg import PoseStampedConveyor

model = YOLO('/home/thinh/ros2_ws/src/camera_worker_pkg/data_source/best.pt')

CONV_REGISTER = 'D200'  # lowwer register of conv current feed

class ObjectSegmentYolo(Node):
    def __init__(self):
        super().__init__('object_segment_node')

        # Params
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('pixel_format', 'mono8')
        self.declare_parameter('display_pixel_format', 'bgr8')
        self.declare_parameter('conf_threshold', 0.1)
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.pixel_format = self.get_parameter('pixel_format').get_parameter_value().string_value
        self.display_pixel_format = self.get_parameter('display_pixel_format').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value

        self.bridge = CvBridge()

        # Subscriber
        self.undistorted_img_sub = self.create_subscription(Image, '/camera/undistorted_image', self.undistorted_img_cb, 1)

        # Publisher
        self.segmented_img_pub = self.create_publisher(Image, '/camera/segment_node/segmented_image', 1)
        self.segment_mask_pub = self.create_publisher(Image, '/camera/segment_node/segment_mask', 1)
        self.circle_segmented_pub = self.create_publisher(Image, '/camera/segment_node/circle_segmented', 1)
        self.object_center_pub = self.create_publisher(PoseStampedConveyor, '/geometry/camera_coord/object_center', 10)


    def undistorted_img_cb(self, msg):
        undst_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.pixel_format)
        color_img_for_yolo = cv.cvtColor(undst_cv_img, cv.COLOR_GRAY2BGR)
        image_to_draw = color_img_for_yolo.copy() 
        img_h, img_w = undst_cv_img.shape[:2]
        roi_top = img_h // 3
        roi_bottom = (img_h * 2) // 3
        # ROI is the middle third between two horizontal lines
        cv.line(image_to_draw, (0, roi_top), (img_w - 1, roi_top), (0, 255, 255), 2)
        cv.line(image_to_draw, (0, roi_bottom), (img_w - 1, roi_bottom), (0, 255, 255), 2)

        # Draw center of image
        # img_h, img_w = undst_cv_img.shape[:2]
        # cx, cy = img_w // 2, img_h // 2           
        # radius = min(img_w, img_h) // 4                
        # cv.circle(image_to_draw, (cx, cy), 180, (0, 255, 255), 3)
        # cv.circle(image_to_draw, (cx, cy), 10, (0, 255, 255), 5)

        results = model.predict(show=False, source=color_img_for_yolo, verbose=False, conf=self.conf_threshold)
        result = results[0]
        published = False

        # Iterate all detections (works for both detect-only and seg models)
        for i, box in enumerate(result.boxes):
            confidence = float(box.conf[0])
            if confidence < self.conf_threshold:
                self.get_logger().debug(f'Skip det conf={confidence:.2f} below {self.conf_threshold}')
                continue

            class_id = int(box.cls[0])
            class_name = model.names.get(class_id, str(class_id))

            coords_xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords_xyxy
            box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            box_cx, box_cy = box_center
            segment_fully_in_roi = False

            # Publish box center
            # publish only if the full segment lies inside the ROI

            # Draw bounding box and label
            cv.rectangle(image_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(image_to_draw, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.circle(image_to_draw, box_center, radius=4, color=(255, 0, 0), thickness=2)

            # If the model has segmentation masks, use them to draw overlay
            binary_mask_resized = None
            if result.masks:
                mask = result.masks[i]
                polygon_points = mask.xy[0].astype(int)
                H, W = result.orig_shape
                binary_mask_tensor = mask.data[0]
                binary_mask_np = binary_mask_tensor.cpu().numpy().astype(np.uint8)
                binary_mask_resized = cv.resize(binary_mask_np, (W, H), interpolation=cv.INTER_NEAREST)
                ys, xs = np.where(binary_mask_resized == 1)
                if ys.size > 0:
                    min_y = int(ys.min())
                    max_y = int(ys.max())
                    segment_fully_in_roi = (min_y >= roi_top) and (max_y <= (roi_bottom - 1))
                else:
                    segment_fully_in_roi = False

                gray = binary_mask_resized * 255
                bgr_mask_resized = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
                best_circle = self.robust_best_circle(binary_mask_resized, max_iter=500, tau=3.0)
                if best_circle is not None:
                    cx, cy, r = best_circle
                    cv.circle(bgr_mask_resized, (int(cx), int(cy)), int(r), (0, 255, 255), 3)
                    cv.circle(bgr_mask_resized, (int(cx), int(cy)), 2, (0, 0, 255), 3)
                mask_rosimg = self.bridge.cv2_to_imgmsg(bgr_mask_resized, encoding=self.display_pixel_format)
                self.segment_mask_pub.publish(mask_rosimg)

                cv.polylines(image_to_draw, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)
                color_overlay = np.zeros_like(image_to_draw, dtype=np.uint8)
                color_overlay[binary_mask_resized == 1] = (0, 0, 255)
                image_to_draw = cv.addWeighted(image_to_draw, 1, color_overlay, 0.5, 0)
            else:
                # Fallback to bbox when no mask is available
                segment_fully_in_roi = (y1 >= roi_top) and (y2 <= (roi_bottom - 1))

            if segment_fully_in_roi:
                box_center_msg = PoseStamped()
                box_center_msg.header.stamp = self.get_clock().now().to_msg()
                box_center_msg.header.frame_id = 'geometry'
                box_center_msg.pose.position.x = float(box_cx)
                box_center_msg.pose.position.y = float(box_cy)
                self.object_center_pub.publish(box_center_msg)
                self.get_logger().info(
                    f'Published object center: {box_center_msg.pose.position.x:.3f}, '
                    f'{box_center_msg.pose.position.y:.3f}, {box_center_msg.header.stamp}'
                )

            rosimg = self.bridge.cv2_to_imgmsg(image_to_draw, encoding=self.display_pixel_format)
            self.segmented_img_pub.publish(rosimg)
            published = True
        # GUI waits on this topic; publish a frame even when nothing is detected
        if not published:
            fallback_img = self.bridge.cv2_to_imgmsg(image_to_draw, encoding=self.display_pixel_format)
            blank_mask = np.zeros_like(image_to_draw)
            fallback_mask = self.bridge.cv2_to_imgmsg(blank_mask, encoding=self.display_pixel_format)
            self.segmented_img_pub.publish(fallback_img)
            self.segment_mask_pub.publish(fallback_mask)
            self.get_logger().info('No detection; published fallback frame')

    def detect_hough_circle(self, circle_mask, src_bgr):
        if circle_mask.dtype != np.uint8:
            circle_mask = circle_mask.astype(np.uint8)

        gray = circle_mask * 255 
        blur_gray = cv.medianBlur(gray, 5)
        # blur_gray = gray
        rows = blur_gray.shape[0]

        circles = cv.HoughCircles(blur_gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=50, maxRadius=500)
        bgr_mask = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (int(i[0]), int(i[1]))
                radius = int(i[2])
                cv.circle(bgr_mask, center, 1, (0, 100, 100), 3)     # tâm
                cv.circle(bgr_mask, center, radius, (255, 0, 255), 3) # đường tròn
        
        rosimg = self.bridge.cv2_to_imgmsg(bgr_mask, encoding=self.display_pixel_format)
        self.circle_segmented_pub.publish(rosimg)
            
        mask_rosimg = self.bridge.cv2_to_imgmsg(gray, encoding=self.pixel_format)
        self.segment_mask_pub.publish(mask_rosimg)

    def circle_from_3pts(self, p1, p2, p3):
        # return cx, cy, r; raise if collinear
        (x1,y1),(x2,y2),(x3,y3) = p1,p2,p3
        A = np.array([[x2-x1, y2-y1],
                    [x3-x1, y3-y1]], dtype=np.float64)
        B = np.array([((x2**2 - x1**2) + (y2**2 - y1**2))/2.0,
                    ((x3**2 - x1**2) + (y3**2 - y1**2))/2.0], dtype=np.float64)
        if abs(np.linalg.det(A)) < 1e-8:
            raise ValueError("collinear")
        cx, cy = np.linalg.solve(A, B)
        r = np.hypot(x1-cx, y1-cy)
        return cx, cy, r

    def fit_circle_lstsq(self, pts):
        # algebraic LS: x^2+y^2 + a x + b y + c = 0
        x = pts[:,0]; y = pts[:,1]
        A = np.c_[x, y, np.ones_like(x)]
        b = -(x**2 + y**2)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a,b_,c = sol
        cx = -a/2.0; cy = -b_/2.0
        r = np.sqrt(cx*cx + cy*cy - c)
        return float(cx), float(cy), float(r)

    def circle_mask(self, h, w, cx, cy, r):
        yy, xx = np.ogrid[:h, :w]
        cm = ((xx - cx)**2 + (yy - cy)**2) <= (r*r)
        return cm.astype(np.uint8)

    def iou_circle_vs_mask(self, mask_bin, cx, cy, r):
        h, w = mask_bin.shape[:2]
        cm = self.circle_mask(h, w, cx, cy, r)
        A = (mask_bin > 0).astype(np.uint8)
        inter = np.sum((A & cm) > 0)
        union = np.sum((A | cm) > 0)
        return inter / max(1, union)

    def robust_best_circle(self, mask_bin, max_iter=2000, tau=2.0, subsample_step=3):
        # 1) preprocess
        A = (mask_bin > 0).astype(np.uint8)*255
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        A = cv.morphologyEx(A, cv.MORPH_OPEN, kernel, iterations=1)

        # 2) contour
        cnts, _ = cv.findContours(A, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not cnts: return None
        cnt = max(cnts, key=cv.contourArea).reshape(-1,2)
        pts = cnt[::max(1,subsample_step)].astype(np.float64)
        n = len(pts)
        if n < 10: return None

        # 3) RANSAC
        best = None
        best_inliers = []
        rng = np.random.default_rng(0)
        for _ in range(max_iter):
            try:
                i,j,k = rng.choice(n, size=3, replace=False)
                cx,cy,r = self.circle_from_3pts(pts[i], pts[j], pts[k])
            except Exception:
                continue
            rad = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
            resid = np.abs(rad - r)
            inliers = resid < tau
            score = np.count_nonzero(inliers)
            if best is None or score > len(best_inliers):
                best = (cx,cy,r)
                best_inliers = inliers

        if best is None: return None

        # 4) refine with LS on inliers
        inlier_pts = pts[best_inliers]
        cx,cy,r = self.fit_circle_lstsq(inlier_pts)

        # 5) optional: choose by IoU if you also try several tau or models
        # iou = iou_circle_vs_mask(A//255, cx, cy, r)

        return cx, cy, r
    
    
def main(args=None):
    rclpy.init(args=args)
    my_node = ObjectSegmentYolo()
    rclpy.spin(my_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
