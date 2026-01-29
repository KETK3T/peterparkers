import atexit
import cv2
from ultralytics import YOLO
from flask import Flask, Response
from flask_cors import CORS
import multiprocessing as mp
import numpy as np
import math 
import time


app = Flask(__name__)
CORS(app)



class proc(mp.Process):
    def __init__(self,cam_id, name=None, model="models/best.pt"):
        super().__init__(name=name or f"Camera-{cam_id}")
        self.cam_id = cam_id
        self.model = model
        self.stop_event = mp.Event()
        self.frame_queue = mp.Queue(maxsize=1)
    def run(self):
        print(f"[{self.name}] Starting camera {self.cam_id}")
        cap = cv2.VideoCapture(self.cam_id,cv2.CAP_V4L2)
        # cap = cv2.VideoCapture("tests/F_tilted_6_cropped.mov")
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
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



def generate_stream(cam):
    try:
        while True:
            frame = cam.get_frame()

            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg',frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except GeneratorExit:
        print("Client disconnected from stream")
        return

@app.route('/video/front')
def video_front():
    return Response(
        generate_stream(cam1),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/video/left')
def video_left():
    return Response(
        generate_stream(cam2),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/video/right')
def video_right():
    return Response(
        generate_stream(cam3),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



cam1 = proc(0, model="models/best.pt", name="Front Cam")
cam2 = proc(2, model="models/best.pt", name="Left Cam")
cam3 = proc(6, model="models/best.pt", name="Right Cam")
cam1.start()
cam2.start()
cam3.start()

@app.route('/api/users')
def get_users():
    return {"users": ["Camera running properly!"]}

if __name__ == "__main__":

    app.run(host='localhost', port=8000, use_reloader=False)
    mp.set_start_method("spawn")

@atexit.register
def shutdown_cams():
    print("Shutting down all camera processes")

    cam1.stop()
    cam3.stop()
    cam2.stop()
    cam1.join()
    cam2.join()
    cam3.join()
    print("All cameras have stopped")
