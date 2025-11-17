import cv2
import numpy as np


cap = cv2.VideoCapture("tests/IMG_9663.mp4")
src = np.float32([[639.5,719] , [0,719], [1279,719], [682,538]])
dst = np.float32([[600,719] , [0,719], [1000,719], [300,538]])


H = cv2.getPerspectiveTransform(src,dst)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h,w = frame.shape[:2]

    bird = cv2.warpPerspective(frame, H, (w,h))

    cv2.imshow("Original", frame)
    cv2.imshow("Bird view", bird)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
