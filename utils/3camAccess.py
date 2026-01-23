import cv2
import os
import time
from multiprocessing import Process, Queue, Lock
# from ultralytics import YOLO




class LastFrame:
    def __init__(self):
        self.lock = Lock()
        self.frame = None
        self.ok = True
        self.id = None
    def set(self, frame):
        with self.lock:
            self.frame = frame

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def capture(id,src, queue, w=640, h=480):


    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    print(f"frame size: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,w)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 15)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))

    if not cap.isOpened():
        print(f"Failed to open camera {id}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to return frames from camera{id}")
            break

        if not queue.full():
            queue.put(frame)

    cap.release()

if __name__ == "__main__":
    queues = [Queue(maxsize=1) for _ in range(3)]
    sources = [9, 4, 1]
    # model = YOLO("models/yolo11n.pt")
    processes = []
    for i, src in enumerate(sources):
        p = Process(target=capture, args=(i, src, queues[i]))
        p.start()
        processes.append(p)


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = []
    for i in range(3):
        fileName = f"cam{i}.mp4"

        if os.path.isfile(fileName):
            os.remove(fileName)
        frameSize = (640,480)
        if i == 2:
            frameSize = (1280,720)
        writer = cv2.VideoWriter(fileName,fourcc, 30, frameSize,True) #change resolution if needed
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer for cam{i}")

        writers.append(writer)
    try:
        while True:
            # frames = []
            #
            # for q in queues:
            #     if not q.empty():
            #         frames.append(q.get())
            #     else:
            #         frames.append(None)

            # for i, frame in enumerate(frames):
            #     if frame is not None:
            #         # results = model.predict(source=frames, show=False, device=0, conf=0.25)
            #         # annotated_frame = results[0].plot()
            #         cv2.imshow(f"cam{i}", frame)
            #
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # time.sleep(0.5)

            for i, q in enumerate(queues):
                if not q.empty():
                    frame = q.get()
                    if frame is not None and frame.size > 0:
                        cv2.imshow(f"cam{i}",frame)
                        writers[i].write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for p in processes:
            p.terminate()

        for w in writers:
            w.release()
        cv2.destroyAllWindows()
