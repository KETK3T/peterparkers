import cv2
from ultralytics import YOLO
import multiprocessing as mp
import numpy as np
import math 
import time

class proc(mp.Process):
    def __init__(self,cam_id, name=None, model="models/yolo11n-seg.pt"):
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

            H = frame.shape[0]
            crp = int(H*0.25)
            frame = frame[crp:H,:]

            results = model_instance.predict(source=frame, show=False,device=0, conf=0.25)

            annotated_frame = results[0].plot()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            boxes = boxes[np.argsort((boxes[:, 0] + boxes[:, 2]) / 2)]
            for i, box in enumerate(boxes):
                x1,y1,x2,y2 = map(int,box)
                print(f"Box {i}:  (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

            n = len(boxes)

            for i in range(n - 1):
                x1a, y1a, x2a, y2a = map(int, boxes[i])
                x1b, y1b, x2b, y2b = map(int, boxes[i + 1])
                len2 = math.dist((x1a, y2a), (x1b, y2b))
                len1 = math.dist((x2a, y2a), (x2b, y2b))

                mid_x_a = int((x1a + x1b) / 2)
                mid_x_b = int((x2a + x2b) / 2)
                mid_y_a = int((y2a + y2b) / 2)
                mid_y_b = int((y2a + y2b) / 2)
                label = f"first: {len1: .1f}px"
                label2 = f"second: {len2: .1f}px"

                a1 = np.array([x1a, y2a])
                a2 = np.array([x1b, y2b])
                b1 = np.array([x2a, y2a])
                b2 = np.array([x2b, y2b])

                line1 = np.array([a1, a2])
                line2 = np.array([b1, b2])
                cv2.line(annotated_frame, (x1a, y2a), (x1b, y2b), (0, 0, 255), 2)
                cv2.line(annotated_frame, (x2a, y2a), (x2b, y2b), (0, 0, 255), 2)
                cv2.putText(annotated_frame, label, (mid_x_a, mid_y_a),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.putText(annotated_frame, label2, (mid_x_b, mid_y_b),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

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
