import atexit
import cv2
from ultralytics import YOLO
from flask import Flask, Response
from flask_cors import CORS
from multiprocessing import Process, Queue, Lock, Event, Value
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time
from threading import Thread
import torch
import os
import re

_MAIN_PID = 0
from datetime import datetime
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.set_num_threads(2)
cv2.setNumThreads(2)

app = Flask(__name__)
CORS(app)
processes = []
stop_event = Event()

SOURCES = [0, 0, 0]

SHAPES = [
    (480, 640, 3),
    (480, 640, 3),
    (480, 640, 3)
]
CAPTURE_FPS = 15
INFERENCE_FPS = 5
IMGSZ = 160
CONF = 0.20
JPEG_QUALITY = 70
MODEL_PATH = "./models/best.engine"
MAX_BATCH_SIZE = 3
RECORD_FPS = 15
RECORD_PATH = "./recordings"
CACHE_CLEAR_INTERVAL = 10.0
RAW_SHM_OBJS = []
ANN_SHM_OBJS = []
RAW_BUFS = []
ANN_BUFS = []
RAW_LOCKS = []
ANN_LOCKS = []
RAW_READY = []
RAW_FRAME_ID = []
processes = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--record",
        "-R",
        action="store_true",
        help="Record Cameras instead of running streaming server"
    )
    return parser.parse_args()


def record(cam_idx, annotated_shm, annotated_locks, stop_event):
    print(f"Recording started for cam {cam_idx}")
    os.makedirs(RECORD_PATH, exist_ok=True)
    Rnow = datetime.now()
    H, W, _ = SHAPES[cam_idx]

    filename = os.path.join(RECORD_PATH, f"cam{cam_idx}_{Rnow.strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print(f"cam {cam_idx}: filename={filename}")
    print(f"cam {cam_idx}: resolution={W}X{H}, fps={RECORD_FPS}, fourcc={fourcc}")
    print(
        f"cam {cam_idx}: RECORD_PATH exists={os.path.exists(RECORD_PATH)}, writable={os.access(RECORD_PATH, os.W_OK)}")

    writer = cv2.VideoWriter(filename, fourcc, RECORD_FPS, (W, H), True)

    if not writer.isOpened():
        print(f"Failed to start writing for cam {cam_idx}")
        return

    frame_time = 1.0 / RECORD_FPS
    last_write = 0.0

    last_frame = np.zeros((H, W, 3), dtype=np.uint8)

    while not stop_event.is_set():

        now = time.time()
        if now - last_write < frame_time:
            time.sleep(0.01)
            continue

        try:
            with annotated_locks[cam_idx]:
                frame = annotated_shm[cam_idx].copy()

            if frame.sum() != 0:
                last_frame = frame

            writer.write(last_frame)
            last_write = now

        except Exception as e:
            print(f"Recording error cam {cam_idx} : {e}")
            time.sleep(0.05)

    writer.release()
    print(f"Recording saved: {filename}")


# MODEL_PATH = "best.engine"
# yolo export model=best.pt format=engine device=0 half=True imgsz=160
# yolo export model=best.pt format=engine device=0 half=True imgsz=160 workspace=1
# yolo export model=best.pt format=engine int8=True data=data.yaml

def sharedMemory_ndarray(shm: SharedMemory, shape):
    return np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)


def draw_results(frame: np.ndarray, result) -> np.ndarray:
    annotated = frame
    ih, iw = annotated.shape[:2]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    names = result.names
    ri_h, ri_w = result.orig_shape

    sx = iw / ri_w
    sy = ih / ri_h

    masks = result.masks
    if masks is not None:
        mask_data = masks.data.cpu().numpy()
        overlay = annotated.copy()

        for j, box in enumerate(boxes):
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            color = (0, 255, 0)

            m = mask_data[j]
            m_resized = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_LINEAR)

            binary = (m_resized > 0.5).astype(np.uint8)
            overlay[binary == 1] = color

        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, dst=annotated)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        color = (0, 255, 0)
        label = f"{names[cls]} {conf:.2f}"

        if (ri_h, ri_w) != (ih, iw):
            x1 = int(x1 * sx)
            y1 = int(y1 * sy)
            x2 = int(x2 * sx)
            y2 = int(y2 * sy)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated


def capture_worker(id, src, shape, raw_shared_mem_name, raw_lock, raw_ready, raw_frame_id, stop_event):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

    # cap = cv2.VideoCapture(src)
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)

    if id != 0:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    else:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[0])

    frame_counter = 0
    skip = 3 if id == 0 else 1
    if not cap.isOpened():
        print(f"Failed to open camera {id}")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera {id} opened with resolution {actual_w}x{actual_h} at {actual_fps} FPS")

    H, W, C = shape
    shm = SharedMemory(name=raw_shared_mem_name)
    buf = sharedMemory_ndarray(shm, shape)
    local_id = 0
    raw_ready.value = 0

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            print(f"Failed to return frames from camera {id}")
            break
        if id == 0:
            frame_counter += 1
            if frame_counter % skip != 0:
                continue
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        with raw_lock:
            np.copyto(buf, frame)

        local_id += 1
        raw_frame_id.value = local_id
        raw_ready.value = 1

    cap.release()
    shm.close()


def inference_loop(shapes, raw_shms, annotated_shms, raw_locks, annotated_locks,
                   raw_ready, raw_frame_id, model_path, stop_event):
    model = YOLO(model_path, task='segment')
    # model.to("cuda")

    if not model_path.endswith(".engine"):
        model.fuse()

    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    dummy_batch = [dummy,dummy,dummy]
    _ = model.predict(dummy_batch, imgsz=IMGSZ, conf=CONF, verbose=False, device='cuda',half=True)
    torch.cuda.empty_cache()

    last_seen = [0] * len(raw_shms)
    frame_time = 1.0 / INFERENCE_FPS
    last_time = 0.0
    last_cache_clear = 0.0

    while not stop_event.is_set():
        now = time.time()
        if now - last_time < frame_time:
            time.sleep(0.01)
            continue

        frames = []
        valid_indices = []
        raw_frames = []
        print([raw_frame_id[i].value for i in range(len(raw_shms))])
        for i in range(len(raw_shms)):
            fid = raw_frame_id[i].value
            if fid == 0 or fid == last_seen[i]:
                continue

            with raw_locks[i]:
                frame = raw_shms[i].copy()
            frames.append(frame)
            raw_frames.append(frame)
            valid_indices.append(i)

            if len(frames) >= MAX_BATCH_SIZE:
                break

        if not frames:
            time.sleep(0.002)
            continue

        try:
            while len(frames) < MAX_BATCH_SIZE:
                frames.append(np.zeros((IMGSZ,IMGSZ, 3), dtype=np.uint8))
                raw_frames.append(None)
                valid_indices.append(None)
            results = model.predict(frames, imgsz=IMGSZ, conf=CONF, verbose=False, device='cuda', half=True)

            for result, idx, raw_frame in zip(results, valid_indices, raw_frames):
                # print(f"yolo results from engine looking for not none: {result.masks}")
                if idx is None:
                    continue
                annotated = draw_results(raw_frame, result)

                H, W, _ = shapes[idx]
                if annotated.shape[:2] != (H, W):
                    annotated = cv2.resize(annotated, (W, H))
                with annotated_locks[idx]:
                    np.copyto(annotated_shms[idx], annotated)

                last_seen[idx] = raw_frame_id[idx].value

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Inference error: {e} ")
            torch.cuda.empty_cache()
            time.sleep(0.1)

        last_time = now

        if now - last_cache_clear > CACHE_CLEAR_INTERVAL:
            torch.cuda.empty_cache()
            last_cache_clear = now


def generate_stream(cam_index, annotated_shm, annotated_locks, stop_event):
    last_frame = None
    frame_time = 1.0 / 15
    last_send = 0
    while not stop_event.is_set():
        now = time.time()
        if now - last_send < frame_time:
            time.sleep(0.01)
            continue

        try:

            with annotated_locks[cam_index]:
                frame = annotated_shm[cam_index].copy()
                # print(f"In generate stream frame size is {frame.shape}")
            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            last_send = now
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)


@app.route('/video/front')
def video_front():
    return Response(
        generate_stream(0, ANN_BUFS, ANN_LOCKS, stop_event),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video/left')
def video_left():
    return Response(
        generate_stream(1, ANN_BUFS, ANN_LOCKS, stop_event),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/video/right')
def video_right():
    return Response(
        generate_stream(2, ANN_BUFS, ANN_LOCKS, stop_event),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@atexit.register
def shutdown_cams():
    if os.getpid() != _MAIN_PID:
        return
    stop_event.set()
    print("Shutting down all camera processes")
    for p in processes:
        try:
            p.join(timeout=3)
        except Exception:
            pass

    for shm in RAW_SHM_OBJS + ANN_SHM_OBJS:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
    print("All cameras have stopped and memory cleaned up")


if __name__ == "__main__":
    _MAIN_PID = os.getpid()
    os.system('v4l2-ctl --list-devices > camInfo.txt')
    with open("camInfo.txt") as f:
        while (True):
            usb_1 = 2.1
            usb_2 = 2.2
            line = f.readline()
            if not line:
                break

            if "Arducam_12MP" in line:
                line = f.readline()
                SOURCES[0] = int(re.search(r'\d+', line).group())
            elif "Arducam USB Camera" in line and usb_1 == float(re.search(r'\d+\.\d+', line).group()):
                line = f.readline()
                SOURCES[1] = int(re.search(r'\d+', line).group())
            elif "Arducam USB Camera" in line and usb_2 == float(re.search(r'\d+\.\d+', line).group()):
                line = f.readline()
                SOURCES[2] = int(re.search(r'\d+', line).group())
    f.close()

    args = parse_args()
    RECORD_MODE = args.record

    probe = cv2.VideoCapture(SOURCES[2])
    ok, first = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError(f"Failed to read from camera source {SOURCES[0]}")

    for shape in SHAPES:
        H, W, C = shape
        size = H * W * C
        raw = SharedMemory(create=True, size=size)
        ann = SharedMemory(create=True, size=size)
        RAW_SHM_OBJS.append(raw)
        ANN_SHM_OBJS.append(ann)
        RAW_BUFS.append(sharedMemory_ndarray(raw, shape))
        ANN_BUFS.append(sharedMemory_ndarray(ann, shape))

    for buf in ANN_BUFS:
        buf[:] = 0

    RAW_LOCKS = [Lock() for _ in SOURCES]
    ANN_LOCKS = [Lock() for _ in SOURCES]
    RAW_READY = [Value('i', 0) for _ in SOURCES]
    RAW_FRAME_ID = [Value('i', 0) for _ in SOURCES]

    for i, src in enumerate(SOURCES):
        p = Process(target=capture_worker, args=(i, src, SHAPES[i], RAW_SHM_OBJS[i].name,
                                                 RAW_LOCKS[i], RAW_READY[i],
                                                 RAW_FRAME_ID[i], stop_event), daemon=False)
        p.start()
        processes.append(p)

    t = Thread(target=inference_loop, args=(SHAPES, RAW_BUFS, ANN_BUFS,
                                            RAW_LOCKS, ANN_LOCKS,
                                            RAW_READY, RAW_FRAME_ID,
                                            MODEL_PATH, stop_event), daemon=True)
    t.start()

    if RECORD_MODE:
        print("Running in RECORD mode")

        print("Waiting for cameras to initialize")
        for i in range(len(SOURCES)):
            while RAW_FRAME_ID[i].value == 0:
                time.sleep(0.1)
            print(f"Camera {i} ready")

            while True:
                with ANN_LOCKS[i]:
                    ready = ANN_BUFS[i].any()
                if ready:
                    break
                time.sleep(0.1)
            print(f"Camera {i} inference ready")

        record_threads = []
        for i in range(len(SOURCES)):
            t = Thread(target=record,
                       args=(i, ANN_BUFS, ANN_LOCKS, stop_event),
                       daemon=False)
            t.start()
            record_threads.append(t)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping recording...")
            stop_event.set()
    else:
        print("Running in STREAM mode")
        app.run(host="localhost", port=8000, use_reloader=False, threaded=True)