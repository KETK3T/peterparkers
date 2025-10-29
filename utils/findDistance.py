import math

from numpy.distutils.system_info import x11_info
from ultralytics import YOLO
import cv2



model = YOLO('models/yolo11n.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open the camera")

DIST_THRESHOLD = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, show=False, device=0, conf=0.80)
    annotated_frame = results[0].plot()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    n = len(boxes)

    for i in range(n):
        for j in range(i + 1, n):
            boxA = boxes[i]
            boxB = boxes[j]

            x1A,y1A,x2A,y2A = boxA
            x1B,y1B,x2B,y2B = boxB

            h_overlap = max(0, min(x2A,x2B) - max(x1A,x1B))
            v_overlap = max(0, min(y2A,y2B) - max(y1A,y1B))

            h_gap = max(0, max(x1B - x2A, x1A - x2B))
            h_gap = max(0, max(y1B - y2A, y1A - y2B))

            if v_overlap > 0 and 0 < h_overlap < DIST_THRESHOLD:
                left = boxA if x2A < x1B else boxB
                right = boxB if x2A < x1B else boxA

                x_left = int(left[2])
                x_right = int(right[2])
                y_top = int(max(left[1] , right[1]))
                y_bottom = int(min(left[3], right[3]))


                cv2.rectangle(annotated_frame, (x_left,y_top), (x_right, y_bottom),
                              (0,0,255), thickness=-1)

    cv2.imshow("Yolo Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()