import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify
from datetime import datetime
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

@app.route('/detections', methods=['GET'])
def get_detections():
    with store_lock:
        data = {
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "spots": {
                cam: list(states) for cam, states in spot_states.items()
            }
        }
    return jsonify(data)

threading.Thread(target=lambda: app.run(port=5000, debug=False, use_reloader=False), daemon=True).start()

VIDS = {
    "Front": cv2.VideoCapture("./recordings/cam0_20260219_090730.mp4"),
    "Left":  cv2.VideoCapture("./recordings/cam2_20260219_090730.mp4"),
    "Right": cv2.VideoCapture("./recordings/cam1_20260219_090730.mp4"),
}

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

ROI_POLYGONS = {
    "Front": [
        [(365, 204), (518, 359), (639, 357), (638, 212), (444, 177)],
        [(0,   355), (1,   215), (184, 177), (241, 204), (70,  358)],
    ],
}

SPOT_CAMS  = set(PARKING_SPOTS.keys())
CLASSES    = [2, 3, 5, 7]
MODEL      = YOLO("./models/yolo11n.engine")
CAM_ORDER  = ["Left", "Front", "Right"]
DISPLAY_W  = 640
DISPLAY_H  = 360
CONF       = 0.3
MIN_BOX_H  = {"Left": 60, "Right": 60}

# ── State ─────────────────────────────────────────────────────────────────────
paused     = False
spot_states = {cam: ["Empty"] * len(spots) for cam, spots in PARKING_SPOTS.items()}
store_lock  = threading.Lock()

SPOT_OCCUPIED_COLOR = (0,   0,   220)
SPOT_EMPTY_COLOR    = (0,   200, 80)
SPOT_ALPHA          = 0.25


def box_overlaps_spot(x1, y1, x2, y2, poly, iou_threshold=0.10):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    spot_mask = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.uint8)
    cv2.fillPoly(spot_mask, [np.array(poly, dtype=np.int32)], 1)

    box_mask = np.zeros((DISPLAY_H, DISPLAY_W), dtype=np.uint8)
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 1, -1)

    intersection = np.logical_and(spot_mask, box_mask).sum()
    if intersection == 0:
        return False

    union = np.logical_or(spot_mask, box_mask).sum()
    return (intersection / union) >= iou_threshold if union > 0 else False


def check_parking_spots(cam_id, boxes):
    spots  = PARKING_SPOTS.get(cam_id)
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
    spots   = PARKING_SPOTS.get(cam_id)
    states  = spot_states.get(cam_id, ["Empty"] * len(spots))
    overlay = frame.copy()

    for spot, state in zip(spots, states):
        pts       = np.array(spot["poly"], dtype=np.int32)
        color     = SPOT_OCCUPIED_COLOR if state == "Car" else SPOT_EMPTY_COLOR
        color_rgb = (color[2], color[1], color[0])

        cv2.fillPoly(overlay, [pts], color_rgb)
        border = (255, 80, 80) if state == "Car" else (80, 255, 130)
        cv2.polylines(overlay, [pts], isClosed=True, color=border, thickness=2)

        cx, cy = int(np.mean([p[0] for p in spot["poly"]])), int(np.mean([p[1] for p in spot["poly"]]))
        label  = f"{spot['id']}: {state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (cx - tw//2 - 3, cy - th - 6), (cx + tw//2 + 3, cy + 4), (20, 20, 20), -1)
        cv2.putText(overlay, label, (cx - tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, border, 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, SPOT_ALPHA, frame, 1 - SPOT_ALPHA, 0)

def annotate_frame(frame, boxes, scores, names, cam_id):
    out      = frame.copy()
    count    = 0
    polygons = ROI_POLYGONS.get(cam_id, [])
    if polygons and not isinstance(polygons[0], list):
        polygons = [polygons]

    for box, score, name in zip(boxes, scores, names):
        x1, y1, x2, y2 = box
        if cam_id in MIN_BOX_H and (y2 - y1) < MIN_BOX_H[cam_id]:
            continue

        # ROI filter
        if cam_id in SPOT_CAMS:
            check_pt = ((x1 + x2) / 2, y2)
            all_polys = [s["poly"] for s in PARKING_SPOTS[cam_id]]
            if not any(cv2.pointPolygonTest(np.array(p, np.int32), (float(check_pt[0]), float(check_pt[1])), False) >= 0 for p in all_polys):
                continue
        elif polygons:
            check_pt = ((x1 + x2) / 2, y1)
            if not any(cv2.pointPolygonTest(np.array(p, np.int32), (float(check_pt[0]), float(check_pt[1])), False) >= 0 for p in polygons):
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
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)

    return out, count

def get_frames():
    frames = {}
    for cam_id, cap in VIDS.items():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frames[cam_id] = cv2.cvtColor(cv2.resize(frame, (DISPLAY_W, DISPLAY_H)), cv2.COLOR_BGR2RGB) if ret \
                         else np.zeros((DISPLAY_H, DISPLAY_W, 3), np.uint8)
    return frames

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
            xs = [p[0] for p in poly] + [poly[0][0]]
            ys = [p[1] for p in poly] + [poly[0][1]]
            ax.fill(xs, ys, color="#00ffb4", alpha=0.08, zorder=5)
            ax.plot(xs, ys, color="#00ffb4", linewidth=1.2, zorder=6)
            ax.text(int(np.mean([p[0] for p in poly])), int(np.mean([p[1] for p in poly])),
                    spot["id"], color="#00ffb4", fontsize=7, ha="center", va="center", zorder=7)
    else:
        for poly in ROI_POLYGONS.get(cam_id, []):
            xs = [p[0] for p in poly] + [poly[0][0]]
            ys = [p[1] for p in poly] + [poly[0][1]]
            ax.fill(xs, ys, color="#00ffb4", alpha=0.10, zorder=5)
            ax.plot(xs, ys, color="#00ffb4", linewidth=1.5, zorder=6)

count_texts = {
    cam: cam_axes[cam].text(8, 30, "", color="#00e6b4", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117", alpha=0.7), zorder=10)
    for cam in CAM_ORDER
}

status_ax = fig.add_axes([0.0, 0.0, 1.0, 0.07])
status_ax.set_facecolor("#090d12")
status_ax.axis("off")
state_text = status_ax.text(
    0.5, 0.5, "● LIVE", transform=status_ax.transAxes,
    color="#00eb78", fontsize=11, fontweight="bold", va="center", ha="center"
)

def on_key(event):
    global paused
    if event.key == " ":
        paused = not paused
        state_text.set_text("❚❚ PAUSED" if paused else "● LIVE")
        state_text.set_color("#00b4ff" if paused else "#00eb78")
    elif event.key == "q":
        plt.close("all")
        for cap in VIDS.values():
            cap.release()
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)


def update(_=None):
    if not plt.fignum_exists(fig.number):
        return

    if not paused:
        raw = get_frames()
        for cam_id in CAM_ORDER:
            frame   = raw[cam_id]
            results = MODEL.predict(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), imgsz=160,
                                    classes=CLASSES, conf=CONF, verbose=False)
            result  = results[0]
            boxes   = result.boxes.xyxy.tolist()
            scores  = result.boxes.conf.tolist()
            names   = [MODEL.names[int(c)] for c in result.boxes.cls]

            if cam_id in SPOT_CAMS:
                check_parking_spots(cam_id, boxes)

            ann, cnt = annotate_frame(frame, boxes, scores, names, cam_id)

            if cam_id in SPOT_CAMS:
                ann = draw_spot_overlays(ann, cam_id)

            img_objs[cam_id].set_data(ann)
            count_texts[cam_id].set_text(f"ROI: {cnt} vehicles")

    fig.canvas.draw_idle()
    fig.canvas.manager.window.after(30, update)

plt.ion()
plt.show()
update()
plt.ioff()
plt.show()