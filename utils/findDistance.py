import math

from numpy.distutils.system_info import x11_info
from ultralytics import YOLO
import cv2
import numpy as np







model = YOLO('models/yolo11m.pt')

cap = cv2.VideoCapture("tests/IMG_9663.mp4")
if not cap.isOpened():
    raise RuntimeError("Could not open the camera")

MIN_DIST_THRESHOLD = 300
MAX_DIST_THRESHOLD = 600
while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, show=False, device=0, conf=0.45, classes=[2,3,5,7])
    annotated_frame = results[0].plot()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    boxes = boxes[np.argsort((boxes[:, 0] + boxes[:, 2]) / 2)]
    # for i, box in enumerate(boxes):
    #     x1,y1,x2,y2 = map(int, box)
    #     print(f"Box {i}:  (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
    #
    #     corners  = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    #     for point in corners:
    #         cv2.circle(annotated_frame, point, 3, (255,0,0), -1)

        # label = f"(x1: {x1}, y1: {y1})"
        # cv2.putText(annotated_frame, label, (x1,y1 - 8),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        #
        # label = f"(x2: {x2},y1: {y1})"
        # cv2.putText(annotated_frame, label, (x2, y1 - 8),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # label = f"(x2: {x2}, y2: {y2})"
        # cv2.putText(annotated_frame, label, (x2, y2 - 8),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # label = f"(x1:{x1},y2:{y2})"
        # cv2.putText(annotated_frame, label, (x1, y2 - 8),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    n = len(boxes)

    for i in range(n):
        for j in range(i + 1, n):
            boxA = boxes[i]
            boxB = boxes[j]

            x1A,y1A,x2A,y2A = map(int,boxA)
            x1B,y1B,x2B,y2B = map(int,boxB)


            length = math.dist((x1A,y2A), (x1B,y2B))
            length2 = math.dist((x2A,y2A), (x2B,y2B))

            mid_x_A = int((x1A + x1B)/ 2)
            mid_x_B = int((x2A + x2B)/ 2)
            mid_y_A = int((y2A + y2B)/ 2)
            mid_y_B = int((y2A + y2B)/ 2)

            label = f"first: {length: .1f}px"
            label2 = f"second: {length2: .1f}px"
            if length > MIN_DIST_THRESHOLD and length < MAX_DIST_THRESHOLD and length2 > MIN_DIST_THRESHOLD and length2 < MAX_DIST_THRESHOLD:
                A1 = np.array([x1A, y2A])
                A2 = np.array([x1B, y2B])
                B1 = np.array([x2A, y2A])
                B2 = np.array([x2B, y2B])
                # cv2.line(annotated_frame, tuple(A1), tuple(A2), (0,0,255), 1)
                # cv2.line(annotated_frame, tuple(B1), tuple(B2), (0, 0, 255), 1)
                line1 = np.array([A1,A2])
                line2 = np.array([B1,B2])
                cv2.polylines(annotated_frame, [line1], False, (0,0,255),5)
                cv2.polylines(annotated_frame, [line2], False, (0,0,255),5)

                poly = np.vstack((line1, line2[::-1]))
                cv2.fillPoly(annotated_frame, [poly], (0,255,0))
                # cv2.putText(annotated_frame, label, (mid_x_A, mid_y_A),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,255), 2)
                # cv2.putText(annotated_frame, label2, (mid_x_B, mid_y_B),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # for t in np.linspace(0,1,50):
                #     p1 = (1 - t) * A1 + t * B1
                #     P2 = (1 - t) * B2 + t * A2
                #     cv2.line(annotated_frame, tuple(p1.astype(int)), tuple(P2.astype(int)),(0,100,255), 2)
                # # cv2.rectangle(annotated_frame,(x1A,y2A),(x2B,y2B), (0,25,255), 2)



    cv2.imshow("Yolo Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()