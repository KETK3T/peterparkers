import cv2
from ultralytics import YOLO
import multiprocessing as mp
import time

class proc(mp.Process):
    def __init__(self,cam_id, name=None, model="models/yolo11n.pt"):
        super().__init__(name=name or f"Camera-{cam_id}")
        self.cam_id = cam_id
        self.model = model
        self.stop_event = mp.Event()
        self.frame_queue = mp.Queue(maxsize=1)
    def run(self):
        print(f"[{self.name}] Starting camera {self.cam_id}")
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)

        if not cap.isOpened():
            print(f"[{self.name}] Failed to open camera {self.cam_id}")
            return

        model_instance = YOLO(self.model)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[{self.name}] Failed to get a frame")
                break

            results = model_instance.predict(source=frame, show=False,device=0, conf=0.75)
            annotated_frame = results[0].plot()

            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(annotated_frame)

        cap.release()
        print(f"[{self.name}] Camera stopped")

    def stop(self):
        self.stop_event.set()
    def get_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None



if __name__ == "__main__":
    mp.set_start_method("spawn")

    cam1 = proc(0,model="models/yolo11n.pt",name="Front Cam")
    cam2 = proc(4,model="models/yolo11n.pt",name="Left Cam")
    cam3 = proc(8,model="models/yolo11n.pt",name="Right Cam")
    cam1.start()
    cam2.start()
    cam3.start()

    try:
        while True:
            f1 = cam1.get_frame()
            f2 = cam2.get_frame()
            f3 = cam3.get_frame()

            if f1 is not None:
                cv2.imshow("Front Cam", f1)
            if f2 is not None:
                cv2.imshow("Left Cam", f2)
            if f3 is not None:
                cv2.imshow("Right Cam", f3)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass

    print("Stopping all camera processes")
    cam1.stop()
    cam3.stop()
    cam2.stop()
    cam1.join()
    cam2.join()
    cam3.join()
    cv2.destroyAllWindows()
    print("Closed all processes")
