import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(4)
cap3 = cv2.VideoCapture(8)

while True:
    ret, frame = cap.read()
    ret1, frame1 = cap2.read()
    if ret:
        cv2.imshow("Jetson Camera 1", frame)
    if ret1:
        cv2.imshow("Jetson Camera 2", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap2.release()
cv2.destroyAllWindows()