from ultralytics import YOLO
import cv2
from multiprocessing import Process



model = YOLO('models/yolo11n.pt')
# def pipeline(dev_path):
#     return(
#         f"v4l2src device=/dev/video4 ! "
#         "image/jpeg, width=640, height=480, framerate=30/1 ! "
#         "jpegdec ! videoconvert ! autovideosink"
#     )
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(4)
if not cap.isOpened() or not cap2.isOpened():
    raise RuntimeError("Error opening a camera")
camera = 0
def run_model_on_cams(cam,camera):
    camera += 1
    while True:
        ret, frame = cam.read()

        if not ret:
            break

        results  = model.predict(source=frame, show=False,device=0, conf=0.75)

        annotated_frame = results[0].plot()
        cv2.imshow(f"Yolo Live {camera}", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyWindow(f"Yolo Live {camera}")
# def run_cam(idx):
#
#     cap = cv2.VideoCapture(pipeline(1), cv2.CAP_GSTREAMER)
#     # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     # cap.set(cv2.CAP_PROP_FPS, 15)
#
#     if not cap.isOpened():
#         raise RuntimeError("Could not open the camera")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#
#         results = model.predict(source=frame, show=False, device=0, classes=[0], conf=0.80)
#
#         annotated_frame = results[0].plot()
#         cv2.imshow("Yolo Live", annotated_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows(f"Camera {idx}")

# cam_idxs = [0,1]

# cam_idxs = ["/dev/video0", "/dev/video4"]
cam_idxs = [cap,cap2]
processes = []

for idx in cam_idxs:
    process = Process(target=run_model_on_cams, args=(idx,camera))
    process.start()
    processes.append(process)

for process in processes:
    process.join()

cv2.destroyAllWindows()