import math
from ultralytics import YOLO
import cv2



model = YOLO('yolo11n.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open the camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, show=False, device=0, classes=[0], conf=0.80)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes.xyxy
    names = results[0].names
    # cls = results[0].cls

    if len(boxes) >= 2:
        x1a, y1a, x2a, y2a = boxes[0]
        x1b, y1b, x2b, y2b = boxes[1]
        center_a = ((x1a + x2a) / 2, (y1a + y2a) / 2)
        center_b = ((x1b + x2b) / 2, (y1b + y2b) / 2)

        pixel_distance = math.dist(center_a, center_b)

        cv2.line(annotated_frame, (int(center_a[0]), int(center_a[1])), (int(center_b[0]), int(center_b[1])), (0, 255, 0), 2)
        text_position = (int((center_a[0] + center_b[0])/2), int((center_a[1] + center_b[1])/2))
        cv2.putText(annotated_frame, f"{pixel_distance:.1f}px", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Yolo Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()