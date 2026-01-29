import cv2

#cap = cv2.VideoCapture("/dev/video0")
#cap2 = cv2.VideoCapture("/dev/video4")
cap3 = cv2.VideoCapture("/dev/video8")
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    #ret, frame = cap.read()
    #ret1, frame1 = cap2.read()
    ret2, frame2 = cap3.read()
    #if ret:
     #   cv2.imshow("Jetson Camera 1", frame)
    #if ret1:
     #   cv2.imshow("Jetson Camera 2", frame1)
    if ret2:
        cv2.imshow("Jetson Camera 3", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
#cap2.release()
cap3.release()
cv2.destroyAllWindows()