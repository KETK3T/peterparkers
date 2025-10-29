from ultralytics import YOLO
import cv2

model = YOLO('models/yolo11n.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open the camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model.predict(source=frame, show=False, device=0, classes=[0], conf=0.80)

    annotated_frame = results[0].plot()
    cv2.imshow("Yolo Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()