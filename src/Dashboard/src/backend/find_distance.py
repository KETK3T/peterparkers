"""
Usage:
python find_distance.py                         Flask Api with no visible annotations
python find_distance.py -T, -t, --test          Display for testing and to confirm correct visuals
python find_distance.py -r,--record             Recording
python find_distance.py -T,-r                   display and record
pyhton find_distance.py -r -a                   Record with annotations

CPU core layout
core 0 - Main
core 1 - cam 0
core 2 - cam 1
core 3 - cam 2
core 4 - inference thread
core 5 - Display and record

PLEASE LOOK AT THE WARNINGS!!!
"""

import traceback
import atexit
import cv2
from ultralytics import YOLO
from flask import Flask, jsonify
from flask_cors import CORS
from multiprocessing import Process, Lock, Event, Value
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time
from threading import Thread
import torch
import os
import re
import threading
from datetime import datetime
import argparse
import signal

# ── Env / torch tuning ────────────────────────────────────────────────────────
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.set_num_threads(2)
cv2.setNumThreads(2)


def parse_args():
    p = argparse.ArgumentParser(description="Parking Finder")
    p.add_argument("-t","-T", "--test", action="store_true", help="Enable display")
    p.add_argument("-R","-r", "--record", action="store_true", help="Record")
    p.add_argument("-A", "-a", "--annotate", action="store_true", help="Records the annotated version (requires -r)")
    return p.parse_args()


CAP_W, CAP_H = 640, 480
DISP_W, DISP_H = 640, 480
CAP_SHAPE = (CAP_H, CAP_W, 3)
DISP_SHAPE = (DISP_H, DISP_W, 3)
SOURCES = [0, 0]
CAPTURE_FPS = 15
INFERENCEFPS = 15
IMGSZ = 160
CONF = 0.20
MODEL_PATH = "./models/yolo11n.engine"
MAX_BATCH = 3
CLASSES = [2, 3, 5, 7]
CAM_ORDER = ['Left', 'Right']
DISPLAY_ORDER = [1,0]
INF_IDX = [0, 1]
SPOT_CAMS = {'Left', 'Right'}
MIN_BOX_H = {'Left': 30, 'Right': 30}  # Ignore detections smaller than this
INTERSECT_ALLOWANCE = 0.10

ROIS = {
    'Left': [
        {'id': 'L1', 'poly': [(1, 261), (98, 219), (144, 222), (3, 303)]},
        {'id': 'L2', 'poly': [(1, 303), (144, 222), (192, 228), (62, 358), (0, 357)]},
        {'id': 'L3', 'poly': [(193, 225), (62, 359), (242, 359), (294, 225)]},
        {'id': 'L4', 'poly': [(293, 224), (243, 358), (449, 359), (379, 223)]},
        {'id': 'L5', 'poly': [(378, 222), (449, 359), (638, 357), (638, 338), (454, 222)]},
        {'id': 'L6', 'poly': [(453, 221), (637, 338), (638, 263), (529, 219)]},
    ],
    'Right': [
        {'id': 'R1', 'poly': [(0, 247), (80, 218), (166, 240), (111, 285), (100, 295), (0, 350)]},
        {'id': 'R2', 'poly': [(164, 238), (96, 298), (0, 347), (0, 355), (234, 357), (303, 258)]},
        {'id': 'R3', 'poly': [(302, 258), (234, 358), (445, 358), (414, 268)]},
        {'id': 'R4', 'poly': [(413, 265), (445, 358), (603, 359), (525, 272)]},
        {'id': 'R5', 'poly': [(525, 270), (604, 359), (638, 358), (638, 277)]},
    ],
}

SPOT_MASK: dict[str, list[np.ndarray]] = {}

for cam, spots in ROIS.items():
    SPOT_MASK[cam] = []
    for spot in spots:
        m = np.zeros((DISP_H, DISP_W), dtype=np.uint8)
        cv2.fillPoly(m, [np.array(spot['poly'], dtype=np.int32)], 1)
        SPOT_MASK[cam].append(m.astype(bool))

OCCUPIED_COLOR = (0, 0, 220)  # Red tint
EMPTY_COLOR = (0, 220, 80)  # Green tint
SPOT_ALPHA = 0.25  # Opacity

store_lock = threading.Lock()
frame_ready_event = Event()

spot_states: dict[str, list[str]] = {
    cam: ['Empty'] * len(spots) for cam, spots in ROIS.items()
}

RAW_SHM_OBJS: list[SharedMemory] = []
ANN_SHM_OBJS: list[SharedMemory] = []
RAW_BUFS: list[np.ndarray] = []
ANN_BUFS: list[np.ndarray] = []
ANN_LOCKS: list = []
RAW_LOCKS: list = []
RAW_FRAME_ID: list = []
PROCESSES: list = []

_MAIN_PID = 0
STOP_EVENT = Event()


def shm_ndarray(shm: SharedMemory, shape: tuple) -> np.ndarray:
    return np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)


app = Flask(__name__)
CORS(app)


@app.route('/detections', methods=['GET'])
def get_detections():
    with store_lock:
        data = {
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            'spots': {cam: list(states) for cam, states in spot_states.items()},
        }
    return jsonify(data)


def check_parking_spots(cam_name: str, disp_boxes: list):
    """
        Spot is occupied if bounding box overlaps 10% of its ROI
    """

    masks = SPOT_MASK.get(cam_name, [])
    min_h = MIN_BOX_H.get(cam_name, 0)
    states = []

    for mask in masks:
        occupied = False
        for box in disp_boxes:
            x1, y1, x2, y2 = box

            # skip small boxes
            if (y2 - y1) < min_h:
                continue

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(DISP_W, int(x2))
            y2 = min(DISP_H, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            roi = mask[y1:y2, x1:x2]

            if roi.sum() >= roi.size * INTERSECT_ALLOWANCE:
                occupied = True
                break

        states.append('Car' if occupied else 'Empty')

    with store_lock:
        spot_states[cam_name] = states


def annotate_frame(frame: np.ndarray, disp_boxes: list, scores: list, names: list, cam_name: str):
    """
        Draw annotation directly on frames, no copy less memory overhead
    """
    min_h = MIN_BOX_H.get(cam_name, 0)
    for box, score, name in zip(disp_boxes, scores, names):
        x1, y1, x2, y2 = box
        if (y2 - y1) < min_h:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = (0, 150, 150)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)


def draw_spot_overlays(frame: np.ndarray, cam_name: str) -> np.ndarray:
    """
        Blend ROI polygons into the frame
    """

    spots = ROIS.get(cam_name, [])
    states = spot_states.get(cam_name, ['Empty'] * len(spots))
    overlay = frame.copy()

    for spot, state in zip(spots, states):
        pts = np.array(spot['poly'], dtype=np.int32)
        color = OCCUPIED_COLOR if state == 'Car' else EMPTY_COLOR
        border = (255, 80, 80) if state == 'Car' else (80, 255, 130)

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], isClosed=True, color=border, thickness=2)

        cx = int(np.mean([p[0] for p in spot['poly']]))
        cy = int(np.mean([p[1] for p in spot['poly']]))
        label = f"{spot['id']}: {state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        cv2.rectangle(overlay,
                      (cx - tw // 2 - 3, cy - th - 6),
                      (cx + tw // 2 + 3, cy + 4),
                      (20, 20, 20), -1)

        cv2.putText(overlay, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, border, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, SPOT_ALPHA, frame, 1 - SPOT_ALPHA, 0)


def capture_worker(cam_id: int, src: int, raw_shm_name: str, raw_lock, raw_frame_id, stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)

    if cam_id == 0:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    else:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # only keep most recent frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    if not cap.isOpened():
        print(f"[CAM {cam_id}] Failed to open source {src}")
        return

    print(f'[cam {cam_id}] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
          f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} '
          f'@ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS')

    shm = SharedMemory(name=raw_shm_name)
    buf = shm_ndarray(shm, CAP_SHAPE)

    skip = 1
    counter = 0
    local_id = 0

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            print(f"[cam {cam_id}] read failed")
            break

        counter += 1
        if counter % skip != 0:
            continue

        if frame.shape[:2] != (CAP_H, CAP_W):
            frame = cv2.resize(frame, (CAP_W, CAP_H), interpolation=cv2.INTER_NEAREST)

        with raw_lock:
            np.copyto(buf, frame)

        local_id += 1
        raw_frame_id.value = local_id

        frame_ready_event.set()  # Wake inference thread immediately to work on the frame

    cap.release()
    shm.close()


def inference_loop(need_annotations: bool, stop_event):
    try:
        os.sched_setaffinity(0, {4})
    except Exception:
        pass

    model = YOLO(MODEL_PATH, task='detect')
    if not MODEL_PATH.endswith('.engine'):
        model.fuse()

    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model.predict([dummy] * MAX_BATCH, imgsz=IMGSZ, conf=CONF, verbose=False, device='cuda', half=True)
    print('[Inference] warm-up done')

    sx = DISP_W / IMGSZ
    sy = DISP_H / IMGSZ

    last_seen = [0] * 3
    frame_time = 1.0 / INFERENCEFPS
    last_time = 0.0

    while not stop_event.is_set():
        now = time.time()

        if now - last_time < frame_time:
            frame_ready_event.wait(timeout=0.05)
            frame_ready_event.clear()
            continue

        frames = []
        raw_full_list = []
        valid_indices = []

        for i in INF_IDX:
            fid = RAW_FRAME_ID[i].value

            if fid == 0 or fid == last_seen[i]:
                continue

            with RAW_LOCKS[i]:
                raw_full = RAW_BUFS[i].copy()

            raw_full_list.append(raw_full)

            frames.append(cv2.resize(raw_full, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST))
            valid_indices.append(i)

        if not frames:
            time.sleep(0.005)
            continue

        padded = list(frames)
        while len(padded) < MAX_BATCH:
            padded.append(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8))

        try:
            results = model.predict(padded, imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False, device='cuda',
                                    half=True)

            for result, i, raw_full in zip(results, valid_indices, raw_full_list):
                cam_name = CAM_ORDER[i]

                disp_boxes = [
                    [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                    for x1, y1, x2, y2 in result.boxes.xyxy.tolist()
                ]

                if cam_name in SPOT_CAMS:
                    check_parking_spots(cam_name, disp_boxes)

                if need_annotations:
                    scores = result.boxes.conf.tolist()
                    names = [model.names[int(c)] for c in result.boxes.cls]

                    # disp = cv2.resize(raw_full, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
                    disp = raw_full

                    annotate_frame(disp, disp_boxes, scores, names, cam_name)

                    if cam_name in SPOT_CAMS:
                        disp = draw_spot_overlays(disp, cam_name)

                    with ANN_LOCKS[i]:
                        np.copyto(ANN_BUFS[i], disp)

                last_seen[i] = RAW_FRAME_ID[i].value

        except Exception:
            traceback.print_exc()
            time.sleep(0.1)

        last_time = now


def display_loop(stop_event):
    """
        Displays what the cams see
    """

    try:
        os.sched_setaffinity(0, {5})
    except Exception:
        pass

    cv2.namedWindow('Parking Finder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Parking Finder', DISP_W * 3, DISP_H)

    frame_time = 1.0 / 30
    last_time = 0.0

    while not stop_event.is_set():
        now = time.time()
        if now - last_time < frame_time:
            time.sleep(0.005)
            continue

        panels = []
        for i in DISPLAY_ORDER:
            cam_name = CAM_ORDER[i]
            """
            if i == 0:
                with RAW_LOCKS[i]:
                    raw = RAW_BUFS[i].copy()
                panel = cv2.resize(raw, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
            """
            with ANN_LOCKS[i]:
                panel = ANN_BUFS[i].copy()
            
            cv2.putText(panel, cam_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 200), 2, cv2.LINE_AA)

            if cam_name in SPOT_CAMS:
                with store_lock:
                    states = spot_states.get(cam_name, [])
                occ = sum(1 for s in states if s == 'Car')
                cv2.putText(panel, f'{occ}/{len(states)} occupied', (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 230, 200), 1, cv2.LINE_AA)

            panels.append(panel)

        cv2.imshow('Parking Finder', np.hstack(panels))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
        last_time = now
    cv2.destroyAllWindows()


def record_loop(stop_event, use_anns=False):
    """
        Writes raw cam frames to the output file (Can be edited to write annotated frames)
    """
    try:
        os.sched_setaffinity(0, {5})
    except Exception:
        pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('recordings', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = []
    suffix = 'Annotated' if use_anns else 'Raw'
    for cam_name in CAM_ORDER:
        path = f"recordings/{timestamp}_{cam_name}.mp4"
        writers.append(cv2.VideoWriter(path, fourcc, CAPTURE_FPS, (CAP_W, CAP_H)))
        print(f"[Record] {cam_name} -> {path}")

    frame_time = 1.0 / CAPTURE_FPS
    last_time = 0.0

    if use_anns:
        print("\033[1;93mWarning: Recording annotated version\033[0m")
    while not stop_event.is_set():
        now = time.time()
        if now - last_time < frame_time:
            time.sleep(0.003)
            continue

        for i, writer in enumerate(writers):
            # change here to ann versions
            if use_anns:

                with ANN_LOCKS[i]:
                    writer.write(ANN_BUFS[i])
            else:
                with RAW_LOCKS[i]:
                    writer.write(RAW_BUFS[i])
        last_time = now

    for w in writers:
        w.release()
    print('[record] files saved')


@atexit.register
def shutdown():
    if os.getpid() != _MAIN_PID:
        return

    STOP_EVENT.set()
    print("Shutting down...")

    for p in PROCESSES:
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

    print('Done')


if __name__ == "__main__":
    args = parse_args()
    _MAIN_PID = os.getpid()
    if args.annotate and not args.record:
        print("\033[1;91mWarning: -a/-A/--annotate has no effect without -r/-R/--record\033[0m")
    need_annotation = args.test or args.record
    os.system('v4l2-ctl --list-devices > camInfo.txt')
    with open('camInfo.txt') as f:
        usb_1 = 2.1
        usb_2 = 2.2
        while True:
            line = f.readline()
            if not line:
                break

            if 'Arducam USB Camera' in line:
                port = float(re.search(r'\d+\.\d+', line).group())
                line = f.readline()
                idx = int(re.search(r'\d+', line).group())
                if port == usb_1:
                    SOURCES[0] = idx
                elif port == usb_2:
                    SOURCES[1] = idx

    if len(set(SOURCES)) != len(SOURCES):
        raise RuntimeError(f"Camera detection failed - possible duplicate sources: {SOURCES}")

    for src in SOURCES:
        probe = cv2.VideoCapture(src)
        ok, _ = probe.read()
        probe.release()
        if not ok:
            raise RuntimeError(f'Cannot read from camera source {src}')

    for _ in range(len(SOURCES)):
        raw = SharedMemory(create=True, size=CAP_H * CAP_W * 3)
        ann = SharedMemory(create=True, size=DISP_H * DISP_W * 3)
        RAW_SHM_OBJS.append(raw)
        ANN_SHM_OBJS.append(ann)
        RAW_BUFS.append(shm_ndarray(raw, CAP_SHAPE))
        ANN_BUFS.append(shm_ndarray(ann, DISP_SHAPE))

    # initialize ann buffs to 0 so we dont have garbage pixels in display
    for buf in ANN_BUFS:
        buf[:] = 0

    RAW_LOCKS = [Lock() for _ in range(len(SOURCES))]
    ANN_LOCKS = [Lock() for _ in range(len(SOURCES))]
    RAW_FRAME_ID = [Value('i', 0) for _ in range(len(SOURCES))]

    for i, src in enumerate(SOURCES):
        p = Process(
            target=capture_worker,
            args=(i, src, RAW_SHM_OBJS[i].name, RAW_LOCKS[i], RAW_FRAME_ID[i], STOP_EVENT),
            daemon=False,
        )
        p.start()
        try:
            os.sched_setaffinity(p.pid, {i + 1})
        except Exception:
            pass
        PROCESSES.append(p)

    inf_t = Thread(
        target=inference_loop,
        args=(need_annotation, STOP_EVENT),
        daemon=True
    )
    inf_t.start()

    print("Waiting for cameras...")
    for i in range(len(SOURCES)):
        while RAW_FRAME_ID[i].value == 0:
            time.sleep(0.1)
        print(f'Camera {i} ({CAM_ORDER[i]}) ready')

    Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True),
        daemon=True,
    ).start()
    print('Flask Api -> http://0.0.0.0/detections')

    if args.record:
        rec_t = Thread(target=record_loop, args=(STOP_EVENT, args.annotate), daemon=True)
        rec_t.start()

    if args.test:
        display_loop(STOP_EVENT)
    else:
        print("Headless mode. Ctrl + C to stop")
        try:
            while not STOP_EVENT.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            STOP_EVENT.set()