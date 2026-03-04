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
import matplotlib.pyplot as plt
from datetime import datetime

# ── Env / torch tuning ────────────────────────────────────────────────────────
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled   = True
torch.set_num_threads(2)
cv2.setNumThreads(2)

_MAIN_PID  = 0
stop_event = Event()

# ── Camera / shape config ─────────────────────────────────────────────────────
SOURCES = [0, 0, 0]   # [Front(0), Left(1), Right(2)] — filled by v4l2 detection

SHAPES = [
    (480, 640, 3),   # Front  – index 0
    (480, 640, 3),   # Left   – index 1
    (480, 640, 3),   # Right  – index 2
]

# Cameras that receive YOLO inference (Front is excluded)
INFER_INDICES = [1, 2]   # Left, Right

CAPTURE_FPS          = 15
INFERENCE_FPS        = 5
IMGSZ                = 160
CONF                 = 0.20
MODEL_PATH           = "./models/yolo11n.engine"
MAX_BATCH_SIZE       = 3    # engine was exported with batch=3
CACHE_CLEAR_INTERVAL = 10.0

# ── Display config ────────────────────────────────────────────────────────────
DISPLAY_W = 640
DISPLAY_H = 360

CAM_ORDER = ["Front", "Left", "Right"]
CAM_INDEX = {"Front": 0, "Left": 1, "Right": 2}
MIN_BOX_H = {"Left": 60, "Right": 60}

PARKING_SPOTS = {
    "Left": [
        {"id": "L1", "poly": [(1,   261), (98,  219), (144, 222), (3,   303)]},
        {"id": "L2", "poly": [(1,   303), (144, 222), (192, 228), (62,  358), (0, 357)]},
        {"id": "L3", "poly": [(193, 225), (62,  359), (242, 359), (294, 225)]},
        {"id": "L4", "poly": [(293, 224), (243, 358), (449, 359), (379, 223)]},
        {"id": "L5", "poly": [(378, 222), (449, 359), (638, 357), (638, 338), (454, 222)]},
        {"id": "L6", "poly": [(453, 221), (637, 338), (638, 263), (529, 219)]},
    ],
    "Right": [
        {"id": "R1", "poly": [(0,   247), (80,  218), (166, 240), (111, 285), (100, 295), (0, 350)]},
        {"id": "R2", "poly": [(164, 238), (96,  298), (0,   347), (0,   355), (234, 357), (303, 258)]},
        {"id": "R3", "poly": [(302, 258), (234, 358), (445, 358), (414, 268)]},
        {"id": "R4", "poly": [(413, 265), (445, 358), (603, 359), (525, 272)]},
        {"id": "R5", "poly": [(525, 270), (604, 359), (638, 358), (638, 277)]},
    ],
}
SPOT_CAMS = set(PARKING_SPOTS.keys())
CLASSES    = [2, 3, 5, 7]
SPOT_OCCUPIED_COLOR = (0,   0,   220)
SPOT_EMPTY_COLOR    = (0,   200,  80)
SPOT_ALPHA          = 0.25

# ── Shared memory globals (populated in __main__) ─────────────────────────────
RAW_SHM_OBJS = []
ANN_SHM_OBJS = []
RAW_BUFS     = []
ANN_BUFS     = []
RAW_LOCKS    = []
ANN_LOCKS    = []
RAW_READY    = []
RAW_FRAME_ID = []
processes    = []

# ── State shared between inference thread and display loop ────────────────────
import threading
store_lock  = threading.Lock()
spot_states = {cam: ["Empty"] * len(spots) for cam, spots in PARKING_SPOTS.items()}

# ── Flask detections API ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/detections', methods=['GET'])
def get_detections():
    with store_lock:
        data = {
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "spots": {cam: list(states) for cam, states in spot_states.items()}
        }
    return jsonify(data)

Thread(target=lambda: app.run(port=5000, debug=False, use_reloader=False), daemon=True).start()


# ── Shared-memory helpers ─────────────────────────────────────────────────────
def sharedMemory_ndarray(shm: SharedMemory, shape):
    return np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)


# ── Capture worker ────────────────────────────────────────────────────────────
def capture_worker(id, src, shape, raw_shm_name, raw_lock, raw_ready, raw_frame_id, stop_event):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ['OPENBLAS_NUM_THREADS']   = '2'
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)

    if id == 0:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)

    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[0])

    if not cap.isOpened():
        print(f"Failed to open camera {id}")
        return

    print(f"Camera {id}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS)} FPS")

    H, W, _       = shape
    shm           = SharedMemory(name=raw_shm_name)
    buf           = sharedMemory_ndarray(shm, shape)
    frame_counter = 0
    skip          = 3 if id == 0 else 1
    local_id      = 0
    raw_ready.value = 0

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            print(f"Camera {id}: failed to read frame")
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
        raw_ready.value    = 1

    cap.release()
    shm.close()


# ── Drawing helpers ───────────────────────────────────────────────────────────
def box_overlaps_spot(x1, y1, x2, y2, poly, iou_threshold=0.10):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    spot_mask = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.uint8)
    cv2.fillPoly(spot_mask, [np.array(poly, dtype=np.int32)], 1)
    box_mask  = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.uint8)
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 1, -1)
    intersection = np.logical_and(spot_mask, box_mask).sum()
    if intersection == 0:
        return False
    union = np.logical_or(spot_mask, box_mask).sum()
    return (intersection / union) >= iou_threshold if union > 0 else False


def check_parking_spots(cam_id, boxes):
    spots  = PARKING_SPOTS.get(cam_id, [])
    states = []
    for spot in spots:
        occupied = False
        for box in boxes:
            x1, y1, x2, y2 = box
            if cam_id in MIN_BOX_H and (y2 - y1) < MIN_BOX_H[cam_id]:
                continue
            if box_overlaps_spot(x1, y1, x2, y2, spot["poly"]):
                occupied = True
                break
        states.append("Car" if occupied else "Empty")
    with store_lock:
        spot_states[cam_id] = states


def draw_spot_overlays(frame, cam_id):
    spots   = PARKING_SPOTS.get(cam_id, [])
    states  = spot_states.get(cam_id, ["Empty"] * len(spots))
    overlay = frame.copy()
    for spot, state in zip(spots, states):
        pts       = np.array(spot["poly"], dtype=np.int32)
        color     = SPOT_OCCUPIED_COLOR if state == "Car" else SPOT_EMPTY_COLOR
        color_rgb = (color[2], color[1], color[0])
        cv2.fillPoly(overlay, [pts], color_rgb)
        border = (255, 80, 80) if state == "Car" else (80, 255, 130)
        cv2.polylines(overlay, [pts], isClosed=True, color=border, thickness=2)
        cx = int(np.mean([p[0] for p in spot["poly"]]))
        cy = int(np.mean([p[1] for p in spot["poly"]]))
        label = f"{spot['id']}: {state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (cx - tw//2 - 3, cy - th - 6),
                      (cx + tw//2 + 3, cy + 4), (20, 20, 20), -1)
        cv2.putText(overlay, label, (cx - tw//2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, border, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, SPOT_ALPHA, frame, 1 - SPOT_ALPHA, 0)


def annotate_frame(frame, boxes, scores, names, cam_id):
    """Draw ALL detections across the full frame — no ROI filtering."""
    out   = frame.copy()
    count = 0
    for box, score, name in zip(boxes, scores, names):
        x1, y1, x2, y2 = box
        if cam_id in MIN_BOX_H and (y2 - y1) < MIN_BOX_H[cam_id]:
            continue
        count += 1
        x1, y1, x2, y2 = map(int, box)
        np.random.seed(hash(name) % 1000)
        color     = tuple(int(c) for c in np.random.randint(80, 255, 3))
        color_bgr = (color[2], color[1], color[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
    return out, count


# ── Inference thread (Left + Right only) ─────────────────────────────────────
def inference_loop(shapes, raw_shms, annotated_shms, raw_locks, annotated_locks,
                   raw_ready, raw_frame_id, model_path, stop_event):
    model = YOLO(model_path, task='detect')
    if not model_path.endswith(".engine"):
        model.fuse()

    # Warm-up
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model.predict([dummy, dummy, dummy], imgsz=IMGSZ, conf=CONF,
                  verbose=False, device='cuda', half=True)
    torch.cuda.empty_cache()

    last_seen        = [0] * len(raw_shms)
    frame_time       = 1.0 / INFERENCE_FPS
    last_time        = 0.0
    last_cache_clear = 0.0

    while not stop_event.is_set():
        now = time.time()
        if now - last_time < frame_time:
            time.sleep(0.01)
            continue

        frames        = []
        raw_frames    = []
        valid_indices = []

        for i in INFER_INDICES:
            fid = raw_frame_id[i].value
            if fid == 0 or fid == last_seen[i]:
                continue
            with raw_locks[i]:
                frame = raw_shms[i].copy()
            frames.append(frame)
            raw_frames.append(frame)
            valid_indices.append(i)

        if not frames:
            time.sleep(0.002)
            continue

        try:
            # Pad to MAX_BATCH_SIZE=3
            padded_frames = list(frames)
            padded_raw    = list(raw_frames)
            padded_idx    = list(valid_indices)
            while len(padded_frames) < MAX_BATCH_SIZE:
                padded_frames.append(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8))
                padded_raw.append(None)
                padded_idx.append(None)

            results = model.predict(padded_frames, imgsz=IMGSZ, conf=CONF, classes=CLASSES,
                                    verbose=False, device='cuda', half=True)

            for result, idx, raw_frame in zip(results, padded_idx, padded_raw):
                if idx is None:
                    continue

                cam_id   = CAM_ORDER[idx]
                H, W, _  = shapes[idx]

                # Annotate at display resolution
                display  = cv2.resize(raw_frame, (DISPLAY_W, DISPLAY_H))
                disp_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                boxes  = result.boxes.xyxy.tolist()
                scores = result.boxes.conf.tolist()
                names  = [model.names[int(c)] for c in result.boxes.cls]

                if cam_id in SPOT_CAMS:
                    check_parking_spots(cam_id, boxes)

                ann, _ = annotate_frame(disp_rgb, boxes, scores, names, cam_id)

                if cam_id in SPOT_CAMS:
                    ann = draw_spot_overlays(ann, cam_id)

                # Store back as BGR at original shape
                ann_bgr  = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
                ann_full = cv2.resize(ann_bgr, (W, H))
                with annotated_locks[idx]:
                    np.copyto(annotated_shms[idx], ann_full)

                last_seen[idx] = raw_frame_id[idx].value

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Inference error: {e}")
            torch.cuda.empty_cache()
            time.sleep(0.1)

        last_time = now

        if now - last_cache_clear > CACHE_CLEAR_INTERVAL:
            torch.cuda.empty_cache()
            last_cache_clear = now


# ── Display helper ────────────────────────────────────────────────────────────
def read_display_frame(idx):
    if idx == 0:
        # Front: raw, no annotation
        with RAW_LOCKS[idx]:
            frame = RAW_BUFS[idx].copy()
    else:
        with ANN_LOCKS[idx]:
            frame = ANN_BUFS[idx].copy()
    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── Matplotlib layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(19, 7), facecolor="#0d1117")
fig.canvas.manager.set_window_title("YOLO Vision Monitor")

cam_axes = {}
for i, cam_id in enumerate(CAM_ORDER):
    ax = fig.add_axes([i * 0.333, 0.08, 0.328, 0.90])
    ax.set_title(cam_id, color="#00e6b4", fontsize=13, fontweight="bold", pad=6)
    ax.axis("off")
    cam_axes[cam_id] = ax

blank    = np.zeros((DISPLAY_H, DISPLAY_W, 3), np.uint8)
img_objs = {cam: cam_axes[cam].imshow(blank) for cam in CAM_ORDER}

for cam_id in CAM_ORDER:
    ax = cam_axes[cam_id]
    if cam_id in SPOT_CAMS:
        for spot in PARKING_SPOTS[cam_id]:
            poly = spot["poly"]
            xs   = [p[0] for p in poly] + [poly[0][0]]
            ys   = [p[1] for p in poly] + [poly[0][1]]
            ax.fill(xs, ys, color="#00ffb4", alpha=0.08, zorder=5)
            ax.plot(xs, ys, color="#00ffb4", linewidth=1.2, zorder=6)
            ax.text(int(np.mean([p[0] for p in poly])),
                    int(np.mean([p[1] for p in poly])),
                    spot["id"], color="#00ffb4", fontsize=7,
                    ha="center", va="center", zorder=7)

count_texts = {
    cam: cam_axes[cam].text(
        8, 30, "", color="#00e6b4", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117", alpha=0.7), zorder=10
    )
    for cam in CAM_ORDER
}

status_ax  = fig.add_axes([0.0, 0.0, 1.0, 0.07])
status_ax.set_facecolor("#090d12")
status_ax.axis("off")
state_text = status_ax.text(
    0.5, 0.5, "● LIVE", transform=status_ax.transAxes,
    color="#00eb78", fontsize=11, fontweight="bold", va="center", ha="center"
)

paused = False

def on_key(event):
    global paused
    if event.key == " ":
        paused = not paused
        state_text.set_text("❚❚ PAUSED" if paused else "● LIVE")
        state_text.set_color("#00b4ff" if paused else "#00eb78")
    elif event.key == "q":
        stop_event.set()
        plt.close("all")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)


def update(_=None):
    if not plt.fignum_exists(fig.number):
        return
    if not paused:
        for cam_id in CAM_ORDER:
            idx   = CAM_INDEX[cam_id]
            frame = read_display_frame(idx)
            img_objs[cam_id].set_data(frame)
            if cam_id == "Front":
                count_texts[cam_id].set_text("Raw (no inference)")
            else:
                with store_lock:
                    states   = spot_states.get(cam_id, [])
                occupied = sum(1 for s in states if s == "Car")
                count_texts[cam_id].set_text(f"{occupied}/{len(states)} spots occupied")
    fig.canvas.draw_idle()
    fig.canvas.manager.window.after(33, update)


# ── Cleanup ───────────────────────────────────────────────────────────────────
@atexit.register
def shutdown_cams():
    if os.getpid() != _MAIN_PID:
        return
    stop_event.set()
    print("Shutting down camera processes…")
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
    print("All cameras stopped and memory cleaned up.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _MAIN_PID = os.getpid()

    os.system('v4l2-ctl --list-devices > camInfo.txt')
    with open("camInfo.txt") as f:
        while True:
            usb_1 = 2.1
            usb_2 = 2.2
            line  = f.readline()
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

    probe = cv2.VideoCapture(SOURCES[2])
    ok, _ = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError(f"Failed to read from camera source {SOURCES[2]}")

    for shape in SHAPES:
        H, W, C = shape
        size    = H * W * C
        raw     = SharedMemory(create=True, size=size)
        ann     = SharedMemory(create=True, size=size)
        RAW_SHM_OBJS.append(raw)
        ANN_SHM_OBJS.append(ann)
        RAW_BUFS.append(sharedMemory_ndarray(raw, shape))
        ANN_BUFS.append(sharedMemory_ndarray(ann, shape))

    for buf in ANN_BUFS:
        buf[:] = 0

    RAW_LOCKS    = [Lock()        for _ in SOURCES]
    ANN_LOCKS    = [Lock()        for _ in SOURCES]
    RAW_READY    = [Value('i', 0) for _ in SOURCES]
    RAW_FRAME_ID = [Value('i', 0) for _ in SOURCES]

    for i, src in enumerate(SOURCES):
        p = Process(
            target=capture_worker,
            args=(i, src, SHAPES[i], RAW_SHM_OBJS[i].name,
                  RAW_LOCKS[i], RAW_READY[i], RAW_FRAME_ID[i], stop_event),
            daemon=False
        )
        p.start()
        processes.append(p)

    Thread(
        target=inference_loop,
        args=(SHAPES, RAW_BUFS, ANN_BUFS,
              RAW_LOCKS, ANN_LOCKS,
              RAW_READY, RAW_FRAME_ID,
              MODEL_PATH, stop_event),
        daemon=True
    ).start()

    print("Waiting for cameras to initialise…")
    for i in range(len(SOURCES)):
        while RAW_FRAME_ID[i].value == 0:
            time.sleep(0.1)
        print(f"  Camera {i} ready")

    plt.ion()
    plt.show()
    update()
    plt.ioff()
    plt.show()