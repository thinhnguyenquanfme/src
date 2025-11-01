import cv2
from ultralytics import YOLO

model = YOLO('/home/thinh/ros2_ws/src/camera_worker_pkg/data_source/best.pt')
# cap = cv2.VideoCapture(0)  # webcam

# while True:
    # ret, frame = cap.read()
    # if not ret: break
results = model.predict(show=True, source='0')
    # annotated = results[0].plot()
    # cv2.imshow('YOLOv8 Detection', annotated)
    # if cv2.waitKey(1) == 27: break


# cap.release()
# cv2.destroyAllWindows()
# for box in results[0].boxes:
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
#     roi = frame[y1:y2, x1:x2]
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 100, 200)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(frame, [max(contours, key=cv2.contourArea)], -1, (0,255,0), 2)
