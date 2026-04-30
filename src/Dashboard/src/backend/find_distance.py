## @file find_distance.py
# @breif Smart parking solution designed to help drivers efficiently locate available parking spaces at busy locations, such as university campus.
# 
# Captures frames from two cameras (left and right), runs batched YOLO inference to detect vehicle, and determines parking spot occupancy by checking bounding box
# ovelap against pre-defined ROI polygons. Results are streamed via a Flask REST API, with optional visualization and recording functionality.
# 
# @section usage Usage
# @code
# python find_distance.py                         # Headless mode: Flask API only
# python find_distance.py -T, -t, --test          # Display window for visual testing
# python find_distance.py -r,--record             # Record raw frames to disk
# python find_distance.py -T,-r                   # Display and record simultaneously
# pyhton find_distance.py -r -a                   # Record with detection annotations
# @endcode
# 
# @section cores CPU Core Affinity Layout
# Machine: Jetson Orin Nano
# CPU core layout
# core 0 - Main / Flask API
# core 1 - camera 0 capture process
# core 2 - camera 1 capture process
# core 3 - Auto Calibration thread
# core 4 - Yolo inference thread
# core 5 - Display / Record thread
# 
# @section roi_editor ROI Editor (requires -T / --test)
# 
#  | Key        | Action |
#  |------------|--------|
#  | I          | Toggle editor on/off |
#  | Left click | Add point to current polygon |
#  | Right click| Undo last point |
#  | Enter      | Finish polygon and print coordinates to terminal |
#  | Backspace  | Clear all points for current polygon |
#  | Tab        | Cycle active camera (Left / Right) |
#  | Q          | Quit |
#
# @warning Always read the printed warnings
#
# @note TensorRt export command:
# @code
# yolo export model=yolo11n.pt format=engine half=true device=0 imgsz=128 batch=3
# @endcode

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
import subprocess

# ── Env / torch tuning ────────────────────────────────────────────────────────
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.set_num_threads(1)
cv2.setNumThreads(1)

## @brief Lock Jetson hardware clocks at maximum frequency for consistent inference latency.
#
#  Stores the current clock state to JETSON_CLOCKS_CONF before locking so it can
#  be restored on shutdown. Should be called at startup (enable=True) and in the
#  atexit handler (enable=False). Prints a coloured warning so the state change
#  is visible in the terminal.
#
#  @param enable  True to lock clocks at maximum; False to restore original state.
def set_jetson_clocks(enable: bool):
    try:
        if enable:
            subprocess.run(['sudo', 'jetson_clocks', '--store', JETSON_CLOCKS_CONF], check=True)
            subprocess.run(['sudo', 'jetson_clocks'], check=True)
            print("\033[1;91mWarning: [SYSTEM] JETSON CLOCKS LOCKED\033[0m")
        else:
            subprocess.run(['sudo', 'jetson_clocks', '--restore', JETSON_CLOCKS_CONF], check=True)
            print("\033[1;91mWarning: [SYSTEM] JETSON CLOCKS RESTORED\033[0m")
    except Exception as e:
        print(f"\033[1;91mWarning: [SYSTEM] JETSON CLOCKS FAILED: {e}\033[0m")

## @brief Parse command-line arguments.
#
#  Supports three flags that can be combined freely:
#  - -t / -T / --test    Open the display window for visual testing.
#  - -r / -R / --record  Record raw (or annotated) frames to disk.
#  - -a / -A / --annotate  Burn annotations into the recording (requires -r).
#
#  @return argparse.Namespace with attributes: test (bool), record (bool), annotate (bool).
def parse_args():
    p = argparse.ArgumentParser(description="Parking Finder")
    p.add_argument("-t", "-T", "--test", action="store_true", help="Enable display")
    p.add_argument("-R", "-r", "--record", action="store_true", help="Record")
    p.add_argument("-A", "-a", "--annotate", action="store_true", help="Records the annotated version (requires -r)")
    return p.parse_args()

## @var JETSON_CLOCKS_CONF
# @brief Temporary file path used to store the original jetson clock state before locking.
JETSON_CLOCKS_CONF = '/tmp/jetson_clocks_backup.conf'

## @var CAP_W
# @brief capture frame width in pixels.
CAP_W = 640
## @var CAP_H
#  @brief Capture frame height in pixels.
CAP_H = 480
## @var DISP_W
#  @brief Display frame width in pixels.
DISP_W = 640
## @var DISP_H
#  @brief Display frame height in pixels.
DISP_H = 480
## @var CAP_SHAPE
#  @brief NumPy shape tuple for a raw capture frame (H, W, 3).
CAP_SHAPE = (CAP_H, CAP_W, 3)
## @var DISP_SHAPE
#  @brief NumPy shape tuple for a display frame (H, W, 3).
DISP_SHAPE = (DISP_H, DISP_W, 3)
## @var SOURCES
#  @brief Video sources for each camera. Can be V4L2 device indices (int) or file paths (str).
#  @note Indices are auto-detected at startup by reading v4l2-ctl output.
# SOURCES = [0, 0] #Left, Right
# SOURCES = ['./Left1.mp4', './Right1.mp4']
SOURCES = ['./recordings/left_side_cam.mp4', './recordings/right_side_ccam.mp4']
## @var CAPTURE_FPS
#  @brief Target capture framerate for live cameras and recording output.
CAPTURE_FPS = 30
## @var INFERENCEFPS
#  @brief Maximum inference framerate. Caps how often the YOLO model is called.
INFERENCEFPS = 15
## @var IMGSZ
#  @brief Input image size (square) fed to the YOLO model in pixels.
IMGSZ = 128
## @var CONF
#  @brief YOLO detection confidence threshold. Detections below this are discarded
CONF = 0.25
## @var MODEL_PATH
#  @brief Path to the YOLO model file. Supports .pt (PyTorch) and .engine (TensorRT).
MODEL_PATH = "./models/yolo11n.engine"
# MODEL_PATH = "./yolo11n.pt"
# MODEL_PATH = "./models/yolo26x.engine"
## @var MAX_BATCH
#  @brief Maximum batch size passed to the YOLO model per inference call.
#  @note Padding frames are added when fewer cameras are active.
MAX_BATCH = 3
## @var CLASSES
#  @brief COCO class IDs to detect. 2=car, 3=motorcycle, 5=bus, 7=truck.
CLASSES = [2, 3, 5, 7]
## @var CAM_ORDER
#  @brief Ordered list of camera name strings. Index matches SOURCES and shared memory lists.
CAM_ORDER = ['Left', 'Right']
## @var DISPLAY_ORDER
#  @brief Order in which camera panels are stitched horizontally in the display window
DISPLAY_ORDER = [0, 1]
## @var INF_IDX
#  @brief Camera indices that are sent to the inference pipeline.
INF_IDX = [0, 1]
## @var SPOT_CAMS
#  @brief Set of camera names that have parking spot ROIs defined and should be checked for occupancy.
SPOT_CAMS = {'Left', 'Right'}
## @var MIN_BOX_H
#  @brief Per camera minimum bounding box height in pixels. Boxes shorter than this are ignored
#         to filter out distant vehicles.
MIN_BOX_H = {'Left': 30, 'Right': 30}
## @var INTERSECT_ALLOWANCE
#  @brief Minimum fraction of a spot's mask area that a bounding box must overlap
#         to mark the spot as occupied. (0.15 = 15%)
INTERSECT_ALLOWANCE = 0.15
## @var AUTO_CALIBRATE_INTERVAL
#  @brief Seconds between automatic ROI recalibration passes. Set to 0 to disable.
AUTO_CALIBRATE_INTERVAL = 0
## @var ROIS
#  @brief Per camera list of parking spot definitions.
#  @note Populate these manually or let the calibration system generate them at runtime.
ROIS = {
    'Left': [
    ],
    'Right': [
    ],
}
## @var SPOT_MASK
#  @brief Per camera list of boolean NumPy arrays (DISP_H x DISP_W) pre-rasterised from ROIS.
#         True pixels belong to that spot's polygon. Used for fast overlap checks.
SPOT_MASK: dict[str, list[np.ndarray]] = {}

## @var CALIB_PARAMS
#  @brief Per camera calibration parameters used by generate_rois() and auto_caliberate_rois().
#
#  Keys per camera:
#  - vanishing_point (tuple): (x, y) pixel of the perspective vanishing point.
#  - near_y (int): Y pixel of the near (bottom) edge of the parking area.
#  - far_y (int): Y pixel of the far (top) edge of the parking area.
#  - left_x / right_x (int): Horizontal bounds of the parking area.
#  - n_spots (int): Expected number of parking spots per row.
#  - n_rows (int): Number of rows of spots.
#  - fisheye_distortion (float): lens distortion correction coefficient.
#  - perspective_strength (float): 0.0 = no perspective warp, 1.0 = full warp toward vanishing point.
#  - min_roi_height (int): Minimum pixel height of a generated ROI polygon.
CALIB_PARAMS = {
    'Left': {'vanishing_point': (381, 214), 'near_y': 406, 'far_y': 274,
             'left_x': 0, 'right_x': 640, 'n_spots': 6, 'n_rows': 1, 'fisheye_distortion': 0.0,
             'perspective_strength': 0.16, 'min_roi_height': 63},
    'Right': {'vanishing_point': (277, 223), 'near_y': 451, 'far_y': 337,
              'left_x': 0, 'right_x': 640, 'n_spots': 5, 'n_rows': 1, 'fisheye_distortion': 0.03,
              'perspective_strength': 0.32, 'min_roi_height': 70},
}

for cam, spots in ROIS.items():
    SPOT_MASK[cam] = []
    for spot in spots:
        m = np.zeros((DISP_H, DISP_W), dtype=np.uint8)
        cv2.fillPoly(m, [np.array(spot['poly'], dtype=np.int32)], 1)
        SPOT_MASK[cam].append(m.astype(bool))
## @var OCCUPIED_COLOR
#  @brief BGR colour used to fill occupied spot overlays (red tint).
OCCUPIED_COLOR = (0, 0, 220)
## @var EMPTY_COLOR
#  @brief BGR colour used to fill empty spot overlays (green tint).
EMPTY_COLOR = (0, 220, 80)
## @var SPOT_ALPHA
#  @brief Opacity of the spot overlay blend. 0.0 = invisible, 1.0 = fully opaque.
SPOT_ALPHA = 0.25

## @var store_lock
#  @brief Threading lock protecting all shared state variables (spot_states, _last_boxes, etc.)
#         from concurrent reads/writes by the inference and display threads.
store_lock = threading.Lock()

## @var spot_states
#  @brief Per-camera list of occupancy strings for each spot. Values are 'Empty' or 'Car'.
#         Written by check_parking_spots(), read by the Flask API and display loop.
spot_states: dict[str, list[str]] = {
    cam: ['Empty'] * len(spots) for cam, spots in ROIS.items()
}

## @var RAW_SHM_OBJS
#  @brief SharedMemory objects holding raw (unannotated) camera frames.
RAW_SHM_OBJS: list[SharedMemory] = []
## @var ANN_SHM_OBJS
#  @brief SharedMemory objects holding annotated frames written by the inference thread.
ANN_SHM_OBJS: list[SharedMemory] = []
## @var RAW_BUFS
#  @brief NumPy arrays mapped onto RAW_SHM_OBJS for zero copy frame access.
RAW_BUFS: list[np.ndarray] = []
## @var ANN_BUFS
#  @brief NumPy arrays mapped onto ANN_SHM_OBJS for zero copy annotated frame access.
ANN_BUFS: list[np.ndarray] = []
## @var ANN_LOCKS
#  @brief Per camera multiprocessing Locks protecting ANN_BUFS entries.
ANN_LOCKS: list = []
## @var RAW_LOCKS
#  @brief Per-camera multiprocessing Locks protecting RAW_BUFS entries.
RAW_LOCKS: list = []
## @var RAW_FRAME_ID
#  @brief Per camera shared Value('i') counters incremented each time a new frame is written.
#         Used by the inference thread to detect stale frames without locking.
RAW_FRAME_ID: list = []
## @var PROCESSES
#  @brief List of capture worker Process objects, kept for clean shutdown.
PROCESSES: list = []
## @var _MAIN_PID
#  @brief PID of the main process. Used in the atexit handler to avoid running
#         shutdown logic in child processes.
_MAIN_PID = 0
## @var STOP_EVENT
#  @brief Multiprocessing Event shared with all worker processes and threads.
#         Set to True to signal a clean shutdown across the entire system.
STOP_EVENT = Event()
## @var _last_boxes
#  @brief Per camera cache of the most recent raw bounding boxes (display coordinates).
#         Written by the inference thread, read by auto-calibration and the display loop.
_last_boxes: dict[str, list] = {cam: [] for cam in CAM_ORDER}
## @var _last_ann_boxes
#  @brief Per camera cache of (box, score, name) tuples from the last inference pass.
#         Used by the display loop to draw labels without re-running inference.
_last_ann_boxes: dict[str, list] = {cam: [] for cam in CAM_ORDER}
## @var _empty_streak
#  @brief Per camera, per spot counter of consecutive inference frames with no detection.
#         A spot transitions from 'Car' to 'Empty' only after EMPTY_CONFIRM_FRAMES consecutive
#         empty frames, preventing flickering from momentary missed detections.
_empty_streak: dict[str, list[int]] = {cam: [] for cam in CAM_ORDER}
## @var _manual_bounds
#  @brief Per camera list of manually set spot boundary X positions from the CalibWindow.
#         Blended into auto-calibrated boundaries with a small weight to stabilise results.
_manual_bounds: dict[str, list[float]] = {cam: [] for cam in CAM_ORDER}
## @var _debug_frames
#  @brief Per camera debug visualisation frames showing detected edges and boundary lines.
#         Only populated when the CALIB_DEBUG environment variable is set.
_debug_frames: dict[str, np.ndarray] = {}
## @var EMPTY_CONFIRM_FRAMES
#  @brief Number of consecutive inference frames a spot must appear empty before its
#         state is changed from 'Car' to 'Empty'. Reduces false-empty flicker.
EMPTY_CONFIRM_FRAMES = 3

## @brief Create a NumPy array view backed by a SharedMemory block.
#
#  No data is copied — the array directly references the shared memory buffer.
#  All processes sharing the same SharedMemory name will see the same bytes.
#
#  @param shm   An open SharedMemory object.
#  @param shape Desired NumPy shape tuple, e.g. (480, 640, 3).
#  @return      uint8 NumPy array backed by shm.buf.
def shm_ndarray(shm: SharedMemory, shape: tuple) -> np.ndarray:
    return np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)


app = Flask(__name__)
CORS(app)

## @brief Flask REST endpoint — returns current parking spot occupancy states.
#
#  Called by the front end or any HTTP client to poll occupancy without needing
#  direct access to the process. Acquires store_lock briefly to get a consistent
#  snapshot of spot_states.
#
#  @par Example response
#  @code{.json}
#  {
#    "timestamp": "14:32:01",
#    "spots": {
#      "Left":  ["Empty", "Car", "Empty", "Empty", "Car", "Empty"],
#      "Right": ["Car", "Car", "Empty", "Empty", "Car"]
#    }
#  }
#  @endcode
#
#  @return Flask JSON response with HTTP 200.
@app.route('/detections', methods=['GET'])
def get_detections():
    with store_lock:
        data = {
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            'spots': {cam: list(states) for cam, states in spot_states.items()},
        }
    return jsonify(data)

## @brief Determine occupancy state for each parking spot ROI on a given camera.
#
#  For every bounding box received from the inference thread, a boolean pixel mask
#  is created and compared against each pre-rasterised spot mask using bitwise AND.
#  A spot is marked 'Car' if the overlap pixel count exceeds INTERSECT_ALLOWANCE
#  multiplied by the total spot mask area. Each car is assigned to the single spot
#  it overlaps the most, preventing one large vehicle from claiming multiple spots.
#
#  To avoid flickering when a car is momentarily missed by the detector, a spot only
#  transitions back to 'Empty' after EMPTY_CONFIRM_FRAMES consecutive frames with
#  no detection, tracked via _empty_streak.
#
#  @param cam_name   Camera identifier string, e.g. "Left" or "Right".
#  @param disp_boxes List of [x1, y1, x2, y2] bounding boxes in display pixel coordinates.
def check_parking_spots(cam_name: str, disp_boxes: list):
    """
        Spot is occupied if bounding box overlaps 10% of its ROI
    """

    masks = SPOT_MASK.get(cam_name, [])
    min_h = MIN_BOX_H.get(cam_name, 0)
    if not masks:
        return

    if len(_empty_streak[cam_name]) != len(masks):
        _empty_streak[cam_name] = [0] * len(masks)

    car_pixels = []
    # occupied_pixels = np.zeros((DISP_H,DISP_W), dtype=bool)

    for box in disp_boxes:
        x1, y1, x2, y2 = box

        if (y2 - y1) < min_h:
            continue

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(DISP_W, int(x2)), min(DISP_H, int(y2))
        if x2 > x1 and y2 > y1:
            m = np.zeros((DISP_H, DISP_W), dtype=bool)
            m[y1:y2, x1:x2] = True
            car_pixels.append(m)

    spot_claimed_by = {}
    for car_mask in car_pixels:
        best_spot = -1
        best_overlap = 0
        for idx, mask in enumerate(masks):
            overlap = np.count_nonzero(car_mask & mask)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spot = idx
        if best_spot >= 0 and best_overlap >= masks[best_spot].sum() * INTERSECT_ALLOWANCE:
            if best_spot not in spot_claimed_by or best_overlap > spot_claimed_by[best_spot]:
                spot_claimed_by[best_spot] = best_overlap

    with store_lock:
        current_states = list(spot_states.get(cam_name, ['Empty'] * len(masks)))
        new_states = list(current_states)

        for idx in range(len(masks)):
            if idx in spot_claimed_by:
                new_states[idx] = 'Car'
                _empty_streak[cam_name][idx] = 0
            else:
                _empty_streak[cam_name][idx] += 1
                if _empty_streak[cam_name][idx] >= EMPTY_CONFIRM_FRAMES:
                    new_states[idx] = 'Empty'
        spot_states[cam_name] = new_states

## @brief Draw YOLO detection boxes and confidence labels directly onto a frame (in-place).
#
#  Iterates over the paired box/score/name lists and draws a filled rectangle label
#  above each bounding box. Boxes shorter than MIN_BOX_H are skipped to ignore
#  noise from distant or partial detections. Modifies the frame array in-place to
#  avoid an extra memory allocation.
#
#  @param frame      BGR uint8 NumPy array to annotate. Modified in-place.
#  @param disp_boxes List of [x1, y1, x2, y2] bounding boxes in display coordinates.
#  @param scores     Confidence score (float) for each box.
#  @param names      COCO class name string for each box (e.g. "car", "truck").
#  @param cam_name   Camera identifier used to look up the MIN_BOX_H threshold.
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

## @brief Blend semi transparent ROI polygon overlays onto a camera frame.
#
#  For each defined parking spot, a filled polygon is drawn onto a copy of the frame
#  using the occupancy-dependent colour (OCCUPIED_COLOR or EMPTY_COLOR), then blended
#  back onto the original using cv2.addWeighted with SPOT_ALPHA opacity. A spot ID
#  and state label is drawn at the polygon centroid.
#
#  @param frame    BGR uint8 NumPy array to draw onto. Not modified in-place;
#                  a blended copy is returned.
#  @param cam_name Camera identifier used to look up ROIS and spot_states.
#  @return         New BGR frame with spot overlays blended in.
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

## @brief Procedurally generate perspective correct parking spot ROI polygons.
#
#  Computes ROI polygons that match real-world lane markings
#  under camera perspective distortion. The algorithm works by interpolating
#  horizontal boundary positions between a near Y row and a far Y row using a
#  vanishing point to model convergence, then optionally applying a radial
#  fisheye correction. Three internal helpers handle the geometry:
#
#  horizontal_bounds(y): Returns the left/right X extents at a given Y row,
#  blending between a rectangular layout and full-perspective convergence
#  toward the vanishing point using perspective_strength.
#
#  scale_x_to_y(x_at_near, y): Projects a single X position from the near row
#  to any other Y row along the perspective gradient.
#
#  fisheye(x, y): Applies barrel/pincushion radial distortion correction.
#  A positive fisheye_distortion value corrects barrel distortion (wide-angle
#  lenses); negative corrects pincushion.
#
#  If spot_boundaries is provided as a list of (near_x, far_x) tuples, those
#  explicit positions are used instead of uniform division.
#
#  @param cam_id               Camera identifier string, used as spot ID prefix.
#  @param vanishing_point      (x, y) pixel coordinate of the perspective vanishing point.
#  @param near_y               Y pixel of the near (bottom) edge of the parking area.
#  @param far_y                Y pixel of the far (top) edge of the parking area.
#  @param left_x               Left boundary X pixel of the parking area.
#  @param right_x              Right boundary X pixel of the parking area.
#  @param n_spots              Number of parking spots per row (default 5).
#  @param n_rows               Number of rows of spots (default 1).
#  @param fisheye_distortion   Radial distortion coefficient. 0.0 disables correction.
#  @param perspective_strength Blend between rect (0.0) and full-perspective (1.0) layout.
#  @param name                 Optional spot ID prefix override (defaults to cam_id[0]).
#  @param spot_boundaries      Optional pre-computed boundary list. Length must be n_spots+1.
#  @return List of dicts, each with keys 'id' (str) and 'poly' (list of (x,y) int tuples).
def generate_rois(
        cam_id: str,
        vanishing_point: tuple,
        near_y: int,
        far_y: int,
        left_x: int,
        right_x: int,
        n_spots: int = 5,
        n_rows: int = 1,
        fisheye_distortion: float = 0.0,
        perspective_strength: float = 1.0,
        name: str = None,
        spot_boundaries: list = None,
        **kwargs,
) -> list[dict]:
    vpx, vpy = vanishing_point
    W, H = DISP_H, DISP_W
    cx, cy = W / 2, H / 2
    prefix = name or cam_id[0].upper()

    def horizontal_bounds(y):

        denominator = vpy - near_y
        if denominator == 0:
            denominator = far_y - near_y
        if denominator == 0:
            return left_x, right_x
        t = (y - near_y) / denominator
        t = max(0.0, min(t, 1.0))
        xl_perspective = left_x + (vpx - left_x) * t
        xr_perspective = right_x + (vpx - right_x) * t
        xl_rect = left_x
        xr_rect = right_x

        xl = xl_rect + (xl_perspective - xl_rect) * perspective_strength
        xr = xr_rect + (xr_perspective - xr_rect) * perspective_strength

        return xl, xr

    def scale_x_to_y(x_at_near, y):
        denominator = vpy - near_y
        if denominator == 0:
            denominator = far_y - near_y
        if denominator == 0:
            return x_at_near
        t = (y - near_y) / denominator
        t = max(0.0, min(t, 1.0))
        return x_at_near + (vpx - x_at_near) * t * perspective_strength

    def fisheye(x, y):

        if fisheye_distortion == 0.0:
            return x, y
        nx, ny = (x - cx) / cx, (y - cy) / cy
        r2 = nx * nx + ny * ny
        f = 1 + fisheye_distortion * r2
        return cx + nx * f * cx, cy + ny * f * cy

    rois = []
    for row in range(n_rows):
        t_near = row / n_rows
        t_far = (row + 1) / n_rows
        row_near_y = near_y + (far_y - near_y) * t_near
        row_far_y = near_y + (far_y - near_y) * t_far

        for s in range(n_spots):
            if spot_boundaries and len(spot_boundaries) == n_spots + 1:
                if isinstance(spot_boundaries[0], tuple):
                    x0_near, x0_far = spot_boundaries[s]
                    x1_near, x1_far = spot_boundaries[s + 1]
                else:
                    x0_near = scale_x_to_y(spot_boundaries[s], row_near_y)
                    x1_near = scale_x_to_y(spot_boundaries[s + 1], row_near_y)
                    x0_far = scale_x_to_y(spot_boundaries[s], row_far_y)
                    x1_far = scale_x_to_y(spot_boundaries[s + 1], row_far_y)
            else:
                near_xl, near_xr = horizontal_bounds(row_near_y)
                far_xl, far_xr = horizontal_bounds(row_far_y)
                t0, t1 = s / n_spots, (s + 1) / n_spots
                x0_near = near_xl + (near_xr - near_xl) * t0
                x1_near = near_xl + (near_xr - near_xl) * t1
                x0_far = far_xl + (far_xr - far_xl) * t0
                x1_far = far_xl + (far_xr - far_xl) * t1

            pts_raw = [
                (x0_near, row_near_y),
                (x1_near, row_near_y),
                (x1_far, row_far_y),
                (x0_far, row_far_y),
            ]

            poly = [
                (int(round(fx)), int(round(fy)))
                for x, y in pts_raw
                for fx, fy in [fisheye(x, y)]
            ]
            rois.append({'id': f'{prefix}{row * n_spots + s + 1}', 'poly': poly})
    return rois

## @brief Automatically recalibrate ROI polygons for a camera using edge detection.
#
#  Analyses a snapshot frame using the following pipeline:
#  1. Converts to greyscale and applies Canny edge detection.
#  2. Uses Probabilistic Hough Transform to find line segments.
#  3. Filters out horizontal lines near known car bounding box edges to avoid
#     using car rooftops/hoods as calibration references.
#  4. Clusters remaining horizontal lines to estimate the far_y boundary.
#  5. Clamps far_y drift to ±50px of the original value to prevent wild jumps.
#  6. Detects near-vertical lines and computes their pairwise intersections
#     to estimate the vanishing point.
#  7. Calls detect_spot_boundaries() to find individual spot divider positions,
#     optionally blending in any manually set boundaries from _manual_bounds.
#  8. Calls generate_rois() with the updated parameters to produce new polygons.
#
#  Updates CALIB_PARAMS in-place and syncs the CalibWindow trackbars if open.
#  Returns None and prints a warning if insufficient lines are detected.
#
#  @param cam_name       Camera identifier string, e.g. "Left".
#  @param frame          BGR display-resolution frame to analyse.
#  @param current_params Current CALIB_PARAMS dict for this camera.
#  @param detections     Optional list of [x1,y1,x2,y2] boxes to exclude car edges
#                        from the line analysis.
#  @return List of new ROI dicts (same format as generate_rois()), or None on failure.
def auto_caliberate_rois(cam_name: str, frame: np.ndarray, current_params: dict, detections: list = None) -> list[dict]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    roi_mask = np.zeros_like(edges)
    roi_mask[frame.shape[0] // 3:, :] = 255
    edges = cv2.bitwise_and(edges, roi_mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=20)
    left_x = current_params['left_x']
    right_x = current_params['right_x']
    near_y = current_params['near_y']
    far_y_ref = current_params.get('far_y', 0)
    if lines is None or len(lines) < 4:
        print(f"\033[1;91mWarning: [Auto Caliberation] Not enough lines detected, keeping current ROIS\033[0m")
        return None

    car_y_exclusions = []
    if detections:
        for x1, y1, x2, y2 in detections:
            car_cx = (x1 + x2) / 2
            car_cy = (y1 + y2) / 2
            if not (left_x <= car_cx <= right_x and far_y_ref <= car_cy <= near_y):
                continue
            pad = 15
            car_y_exclusions.append((int(y1) - pad, int(y1) + pad))
            car_y_exclusions.append((int(y2) - pad, int(y2) + pad))

    def is_near_car_edge(y):
        for low, hi in car_y_exclusions:
            if low <= y <= hi:
                return True
        return False

    h_lines = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 20 or angle > 160:
            mid_y = (y1 + y2) / 2
            if not is_near_car_edge(mid_y):
                h_lines.append((x1, y1, x2, y2))

    if len(h_lines) < 2:
        print(f"\033[1;91mWarning: [Auto Caliberation] Not enough horizontal lines, keeping current ROIS\033[0m")
        return None

    ys = np.array([(y1 + y2) / 2 for x1, y1, x2, y2 in h_lines])
    ys_sorted = np.sort(ys)

    gaps = np.diff(ys_sorted)
    big_gaps = np.where(gaps > 20)[0]

    cluster_centers = []
    prev = 0
    for g in big_gaps:
        cluster_centers.append(np.mean(ys_sorted[prev:g + 1]))
        prev = g + 1
    cluster_centers.append(np.mean(ys_sorted[prev:]))

    if len(cluster_centers) < 2:
        print(f"\033[1;91mWarning: [Auto Caliberation] Could not find row boundaries\033[0m")
        return None

    min_far_y = near_y - 250
    valid_clusters = [c for c in cluster_centers if min_far_y < c < near_y]
    if not valid_clusters:
        print(f"\033[1;91mWarning: [Auto Caliberation] no valid far_y found\033[0m")
        return None
    far_y = int(min(valid_clusters, key=lambda c: near_y - c))

    n_spots = current_params.get('n_spots', 5)
    n_rows = current_params.get('n_rows', 1)
    distortion = current_params.get('fisheye_distortion', 0.0)
    strength = current_params.get('perspective_strength', 1.0)

    MIN_ROI_HEIGHT = current_params.get('min_roi_height', 60)

    if near_y - far_y < MIN_ROI_HEIGHT:
        far_y = near_y - MIN_ROI_HEIGHT
        print(f"\033[1;93mWarning: [Auto Caliberation] far_y too close to near_y, clamped to {far_y}\033[0m")

    far_y = max(0, far_y)
    original_far_y = CALIB_PARAMS[cam_name].get('far_y', far_y)
    max_drift = 50
    far_y = max(far_y, original_far_y - max_drift)
    far_y = min(far_y, original_far_y + max_drift)
    vanish_lines = []
    for l in lines[:]:
        x1, y1, x2, y2 = l[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 10 < angle < 80 or 100 < angle < 170:
            vanish_lines.append((x1, y1, x2, y2))

    vanishing_point = current_params['vanishing_point']

    if len(vanish_lines) >= 2:
        intersections = []
        for i in range(min(len(vanish_lines), 8)):
            for j in range(i + 1, min(len(vanish_lines), 8)):
                x1, y1, x2, y2 = vanish_lines[i]
                x3, y3, x4, y4 = vanish_lines[j]
                d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(d) < 1e-6:
                    continue
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                if 0 < ix < DISP_W and 0 < iy < far_y + 20:
                    intersections.append((ix, iy))

        if intersections:
            vanishing_point = (
                int(np.median([p[0] for p in intersections])),
                int(np.median([p[1] for p in intersections])))

    CALIB_PARAMS[cam_name]['far_y'] = far_y
    CALIB_PARAMS[cam_name]['vanishing_point'] = vanishing_point

    if calib_window.open and calib_window.active_cam == cam_name:
        cv2.setTrackbarPos('Far Y', CalibWindow.WIN, far_y)
        cv2.setTrackbarPos('VP X', CalibWindow.WIN, vanishing_point[0])
        cv2.setTrackbarPos('VP Y', CalibWindow.WIN, vanishing_point[1])

    roi_detections = [
        box for box in (detections or [])
        if left_x <= (box[0] + box[2]) / 2 <= right_x
           and far_y <= (box[1] + box[3]) / 2 <= near_y
    ]
    boundaries, detected_n_spots = detect_spot_boundaries(frame, near_y, far_y, left_x, right_x, n_spots,
                                                          detections=roi_detections, cam_name=cam_name)
    manual = _manual_bounds.get(cam_name, [])
    if manual:
        boundaries = blend_boundaries(boundaries, manual, manual_weight=0.15)

    return generate_rois(cam_id=cam_name, vanishing_point=vanishing_point, near_y=near_y, far_y=far_y, left_x=left_x,
                         right_x=right_x, n_spots=detected_n_spots, n_rows=n_rows, fisheye_distortion=distortion,
                         perspective_strength=strength, spot_boundaries=boundaries)

## @brief Rebuild the in-memory ROI and mask tables for a camera after calibration.
#
#  Re-rasterises each new ROI polygon into a boolean pixel mask and atomically
#  replaces the global ROIS, SPOT_MASK, and spot_states entries under store_lock.
#  If the number of spots has changed, spot_states and _empty_streak are reset
#  to avoid index mismatches.
#
#  @param cam_name     Camera identifier string.
#  @param new_rois     New list of ROI dicts from generate_rois() or auto_caliberate_rois().
#  @param reset_states If True, forces spot_states back to all 'Empty' regardless of
#                      whether the spot count changed.
def rebuild_spot_masks(cam_name: str, new_rois: list[dict], reset_states: bool = False):
    global ROIS, SPOT_MASK, spot_states

    new_masks = []
    for spot in new_rois:
        m = np.zeros((DISP_H, DISP_W), dtype=np.uint8)
        cv2.fillPoly(m, [np.array(spot['poly'], dtype=np.int32)], 1)
        new_masks.append(m.astype(bool))
    with store_lock:
        ROIS[cam_name] = new_rois
        SPOT_MASK[cam_name] = new_masks

        if reset_states or len(new_rois) != len(spot_states.get(cam_name, [])):
            spot_states[cam_name] = ['Empty'] * len(new_rois)
            _empty_streak[cam_name] = [0] * len(new_rois)
    # else:
    # 	_empty_streak[cam_name] = [0] * len(new_rois)
    print(f"\033[1;91mWarning: [Auto Caliberation] {cam_name}: {len(new_rois)} spots rebuilt\033[0m")

## @brief Interactive polygon editor for defining parking spot ROIs at runtime.
#
#  Lives entirely in the display thread — all methods are called from display_loop()
#  and are not thread-safe. The editor is toggled on/off by pressing I in the
#  display window. While active, the user left-clicks to place polygon vertices on
#  the active camera panel, then presses Enter to print the completed polygon's
#  coordinates to the terminal in a copy-pasteable format for hardcoding into ROIS.
#
#  Tab cycles between camera panels so both Left and Right ROIs can be defined
#  without restarting. Points are stored in panel-local coordinates and converted
#  to stitched-window coordinates only for rendering.
#
#  @note This editor does NOT automatically apply the polygon to ROIS/SPOT_MASK.
#        Copy the printed output into the ROIS constant at the top of the file.
class ROIEditor:
    """
    Interactive polygon editor. Lives entirely in the display thread.

    Controls (active when editor_mode=True):
        Left click   - add point
        Right click  - undo last point
        Enter        - finish polygon, print to terminal
        Backspace    - clear current polygon
        Tab          - cycle active camera
        I            - toggle editor on/off
    """

    POINT_COLOR = (0, 255, 255)  # cyan dots
    LINE_COLOR = (0, 200, 255)  # cyan-ish edges
    CLOSE_COLOR = (180, 0, 255)  # purple closing edge
    FILL_COLOR = (0, 180, 255)  # translucent fill
    CURSOR_COLOR = (200, 200, 200)  # crosshair
    TEXT_COLOR = (255, 255, 255)
    SHADOW_COLOR = (0, 0, 0)

    def __init__(self):
        self.editor_mode: bool = False
        self.points: list[tuple[int, int]] = []
        self.mouse_pos: tuple[int, int] = (0, 0)
        # Which camera panel the user is editing (index into DISPLAY_ORDER)
        self.cam_panel_idx: int = 0
        # Running count per camera so auto-IDs don't repeat within a session
        self._spot_counters: dict[str, int] = {c: len(ROIS.get(c, [])) for c in CAM_ORDER}

    # ── property helpers ──────────────────────────────────────────────────────

    @property
    def active_cam(self) -> str:
        return CAM_ORDER[DISPLAY_ORDER[self.cam_panel_idx]]

    def _next_id(self, cam: str) -> str:
        self._spot_counters[cam] += 1
        prefix = cam[0].upper()
        return f"{prefix}{self._spot_counters[cam]}"

    # ── mouse callback ────────────────────────────────────────────────────────

    def mouse_cb(self, event, x, y, flags, param):
        """OpenCV mouse callback (runs in display thread)."""
        if not self.editor_mode:
            return

        # x is in the stitched window; clamp to the active panel column
        panel_w = DISP_W
        # left panel = DISPLAY_ORDER[0], right panel = DISPLAY_ORDER[1]
        # we only care about the panel that matches self.cam_panel_idx
        col_start = self.cam_panel_idx * panel_w
        col_end = col_start + panel_w

        # Track mouse regardless of which panel (for crosshair)
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if col_start <= x < col_end:
                # Convert to panel-local coordinates
                lx = x - col_start
                self.points.append((lx, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    # ── keyboard handling ─────────────────────────────────────────────────────

    def handle_key(self, key: int) -> bool:
        """
        Process a keypress.
        Returns True if the key was consumed (caller should not process further).
        Returns False if the key should be handled by the normal display loop.
        """
        if key == ord('i') or key == ord('I'):
            self.editor_mode = not self.editor_mode
            if self.editor_mode:
                print(f"\n[ROI Editor] ON  — camera: {self.active_cam}")
                print("  Left-click to add points  |  Right-click to undo")
                print("  Enter = finish polygon    |  Backspace = clear")
                print("  Tab = switch camera       |  I = exit editor\n")
            else:
                print("[ROI Editor] OFF")
                self.points.clear()
            return True

        if not self.editor_mode:
            return False

        # Enter — finish polygon
        if key in (13, 10):  # CR or LF
            self._finish_polygon()
            return True

        # Backspace — clear
        if key == 8:
            self.points.clear()
            print("[ROI Editor] Points cleared")
            return True

        # Tab — cycle camera
        if key == 9:
            self.cam_panel_idx = (self.cam_panel_idx + 1) % len(DISPLAY_ORDER)
            self.points.clear()
            print(f"[ROI Editor] Active camera: {self.active_cam}  (points cleared)")
            return True

        return False

    def _finish_polygon(self):
        if len(self.points) < 3:
            print(f"[ROI Editor] Need at least 3 points (have {len(self.points)})")
            return

        cam = self.active_cam
        sid = self._next_id(cam)
        poly = list(self.points)

        print(f"\n# ── ROI output ──────────────────────────────────────────")
        print(f"# Camera : {cam}   ID : {sid}")
        print(f"{{'id': '{sid}', 'poly': {poly}}},")
        print(f"# ─────────────────────────────────────────────────────────\n")

        self.points.clear()

    def draw_overlay(self, canvas: np.ndarray):
        if not self.editor_mode:
            return

        num_panels = len(DISPLAY_ORDER)
        panel_w = DISP_W
        col_start = self.cam_panel_idx * panel_w

        # ── translucent panel highlight ───────────────────────────────────────
        overlay = canvas.copy()
        cv2.rectangle(overlay, (col_start, 0), (col_start + panel_w, DISP_H),
                      (0, 60, 100), -1)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

        mx, my = self.mouse_pos
        cv2.line(canvas, (mx, 0), (mx, DISP_H), self.CURSOR_COLOR, 1, cv2.LINE_AA)
        cv2.line(canvas, (0, my), (canvas.shape[1], my), self.CURSOR_COLOR, 1, cv2.LINE_AA)

        canvas_pts = [(col_start + px, py) for (px, py) in self.points]

        if len(canvas_pts) >= 3:
            fill_overlay = canvas.copy()
            cv2.fillPoly(fill_overlay,
                         [np.array(canvas_pts, dtype=np.int32)],
                         self.FILL_COLOR)
            cv2.addWeighted(fill_overlay, 0.25, canvas, 0.75, 0, canvas)

        for i in range(1, len(canvas_pts)):
            cv2.line(canvas, canvas_pts[i - 1], canvas_pts[i],
                     self.LINE_COLOR, 2, cv2.LINE_AA)

        if len(canvas_pts) >= 3:
            p0, pn = canvas_pts[0], canvas_pts[-1]

            dx = p0[0] - pn[0];
            dy = p0[1] - pn[1]
            dist = max(1, int((dx ** 2 + dy ** 2) ** 0.5))
            segs = max(4, dist // 10)
            for s in range(segs):
                if s % 2 == 0:
                    t0 = s / segs
                    t1 = (s + 1) / segs
                    pt0 = (int(pn[0] + dx * t0), int(pn[1] + dy * t0))
                    pt1 = (int(pn[0] + dx * t1), int(pn[1] + dy * t1))
                    cv2.line(canvas, pt0, pt1, self.CLOSE_COLOR, 2, cv2.LINE_AA)

        for idx, (cx, cy) in enumerate(canvas_pts):
            cv2.circle(canvas, (cx, cy), 5, self.POINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)
            lbl = str(idx)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(canvas, lbl, (cx + 7, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.SHADOW_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, lbl, (cx + 7, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.POINT_COLOR, 1, cv2.LINE_AA)

        hud_lines = [
            f"ROI EDITOR  |  cam: {self.active_cam}  |  pts: {len(self.points)}",
            "LClick=add  RClick=undo  Enter=done  Bksp=clear  Tab=cam  I=exit",
        ]
        for li, line in enumerate(hud_lines):
            yy = DISP_H - 12 - (len(hud_lines) - 1 - li) * 18
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas,
                          (col_start + 4, yy - th - 4),
                          (col_start + tw + 10, yy + 4),
                          (10, 10, 10), -1)
            cv2.putText(canvas, line,
                        (col_start + 7, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.TEXT_COLOR, 1, cv2.LINE_AA)

        if col_start <= mx < col_start + panel_w:
            lx = mx - col_start
            tip = f"({lx}, {my})"
            (tw, th), _ = cv2.getTextSize(tip, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            tx = min(mx + 10, canvas.shape[1] - tw - 6)
            ty = max(my - 10, th + 4)
            cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2),
                          (20, 20, 20), -1)
            cv2.putText(canvas, tip, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.TEXT_COLOR, 1, cv2.LINE_AA)


roi_editor = ROIEditor()

## @brief Camera capture worker — runs as a separate Process per camera.
#
#  Opens the video source (V4L2 device or file), configures resolution and FPS,
#  then loops reading frames and writing them into the shared memory buffer.
#  After each write it increments raw_frame_id and sets frame_ready_event to
#  immediately wake the inference thread without polling.
#
#  For live cameras, YUYV format and a buffer size of 1 are set to always
#  deliver the most recent frame. For file sources, the video rewinds at EOF.
#  The process ignores SIGINT so Ctrl+C is handled only by the main process
#  via STOP_EVENT.
#
#  Pinned to CPU core (cam_id + 1) by the main process after spawning.
#
#  @param cam_id            Zero-based camera index.
#  @param src               V4L2 device index (int) or video file path (str).
#  @param raw_shm_name      Name of the shared memory block to write frames into.
#  @param raw_lock          Multiprocessing Lock protecting the shared buffer.
#  @param raw_frame_id      Shared Value('i') incremented on each new frame.
#  @param frame_ready_event Multiprocessing Event set after each frame write.
#  @param stop_event        Multiprocessing Event polled to trigger shutdown.
def capture_worker(cam_id: int, src: int, raw_shm_name: str, raw_lock, raw_frame_id, frame_ready_event, stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'

    # cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    for attempt in range(5):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            break
        time.sleep(0.2)

    if not cap.isOpened():
        print(f"[CAM {cam_id}] Failed to open source {src}")
        return

    is_live = isinstance(src, int)

    if is_live:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # only keep most recent frame

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = CAPTURE_FPS

    frame_delay = 1.0 / actual_fps
    print(f"[cam {cam_id}] Video FPS detected: {actual_fps}, frame_delay: {frame_delay:.4f}s")

    print(f'[cam {cam_id}] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
          f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} '
          f'@ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS')

    shm = SharedMemory(name=raw_shm_name)
    buf = shm_ndarray(shm, CAP_SHAPE)

    local_id = 0
    last_frame_time = time.perf_counter()

    while not stop_event.is_set():

        now = time.perf_counter()
        elapsed = now - last_frame_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0.001:
            time.sleep(sleep_time)
        last_frame_time = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            if not is_live:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_frame_time = time.perf_counter()
                continue
            print(f"[cam {cam_id}] read failed")
            break

        if frame.shape[:2] != (CAP_H, CAP_W):
            frame = cv2.resize(frame, (CAP_W, CAP_H), interpolation=cv2.INTER_NEAREST)

        with raw_lock:
            np.copyto(buf, frame)

        local_id += 1
        raw_frame_id.value = local_id

        frame_ready_event.set()  # Wake inference thread immediately to work on the frame

        if is_live:
            drift = time.perf_counter() - last_frame_time
            if drift > frame_delay * 2:
                last_frame_time = time.perf_counter()

    cap.release()
    shm.close()

## @brief YOLO inference thread — batches frames from all cameras and runs detection.
#
#  Runs as a daemon Thread pinned to CPU core 4. On startup it performs 5 warm-up
#  inference passes with dummy frames to prime the TensorRT engine and avoid a
#  latency spike on the first real frame.
#
#  Each iteration checks whether new frames are available by comparing raw_frame_id
#  against last_seen. If new frames exist they are resized to IMGSZ, padded to
#  MAX_BATCH with blank frames, and passed to the YOLO model in a single batched
#  call. Detections are scaled back to display resolution using sx/sy factors.
#
#  Results are processed per camera:
#  - check_parking_spots() updates occupancy state.
#  - _last_boxes and _last_ann_boxes are updated under store_lock.
#  - If need_annotations is True, annotated frames are written to ANN_BUFS.
#
#  Sleeps briefly and waits on frame_ready_event between inference cycles to
#  avoid busy-waiting while respecting the INFERENCEFPS cap.
#
#  @param need_annotations  If True, annotated frames are rendered and written to ANN_BUFS.
#  @param frame_ready_event Event set by capture workers when a new frame is available.
#  @param stop_event        Event polled to trigger shutdown.
def inference_loop(need_annotations: bool, frame_ready_event, stop_event):
    try:
        os.sched_setaffinity(0, {4})
    except Exception:
        pass

    model = YOLO(MODEL_PATH, task='detect')
    if not MODEL_PATH.endswith('.engine'):
        model.fuse()

    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    # model.predict([dummy] * MAX_BATCH, imgsz=IMGSZ, conf=CONF,device='cuda', verbose=False, half=True)
    for _ in range(5):
        model.predict([dummy] * MAX_BATCH, imgsz=IMGSZ, conf=CONF, device='cuda', verbose=False, half=True)
    # model.predict([dummy] * MAX_BATCH, imgsz=IMGSZ, conf=CONF, verbose=False, half=True)

    print('[Inference] warm-up done')

    sx = DISP_W / IMGSZ
    sy = DISP_H / IMGSZ

    last_seen = [0] * 2
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

        # padded = frames
        padded = list(frames)
        while len(padded) < MAX_BATCH:
            padded.append(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8))

        try:
            results = model.predict(padded, imgsz=IMGSZ, conf=CONF, device='cuda', classes=CLASSES, verbose=False,
                                    half=True)

            # results = model.predict(padded, imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False,
            # 						half=True)

            for result, i, raw_full in zip(results, valid_indices, raw_full_list):
                cam_name = CAM_ORDER[i]

                disp_boxes = [
                    [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                    for x1, y1, x2, y2 in result.boxes.xyxy.tolist()
                ]

                if cam_name in SPOT_CAMS:
                    check_parking_spots(cam_name, disp_boxes)
                    with store_lock:
                        _last_boxes[cam_name] = list(disp_boxes)

                scores = result.boxes.conf.tolist()
                names = [model.names[int(c)] for c in result.boxes.cls]
                with store_lock:
                    _last_ann_boxes[cam_name] = list(zip(disp_boxes, scores, names))

                if need_annotations:

                    disp = raw_full.copy()
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

## @brief Display loop — renders the live camera feed with overlays in a GUI window.
#
#  Runs in the main thread (or a dedicated thread) when -T / --test is passed.
#  Pinned to CPU core 5. Creates a single OpenCV window and stitches all camera
#  panels side by side each frame. For each panel it:
#  - Copies the latest raw frame from shared memory.
#  - Calls draw_spot_overlays() to blend ROI polygons.
#  - Draws cached detection boxes and labels from _last_ann_boxes.
#  - Overlays the camera name and occupancy count.
#
#  After stitching, roi_editor.draw_overlay() is called to paint the ROI editor
#  UI if it is active. An optional CALIB_DEBUG window shows edge detection output.
#
#  Keyboard bindings (in addition to ROI editor keys):
#  - U: Trigger a manual auto-calibration pass for all cameras.
#  - C: Toggle the CalibWindow trackbar panel for the active camera.
#  - Q: Set STOP_EVENT and exit.
#
#  @param stop_event Multiprocessing Event polled to exit the loop.
def display_loop(stop_event):
    """
    Displays what the cams see.
    Press I to toggle the interactive ROI editor.
    """

    try:
        os.sched_setaffinity(0, {5})
    except Exception:
        pass

    win = 'Parking Finder'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISP_W * len(DISPLAY_ORDER), DISP_H)
    cv2.setMouseCallback(win, roi_editor.mouse_cb)

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
            with RAW_LOCKS[i]:
                panel = RAW_BUFS[i].copy()

            if panel.shape[:2] != (DISP_H, DISP_W):
                panel = cv2.resize(panel, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)

            if cam_name in SPOT_CAMS:
                panel = draw_spot_overlays(panel, cam_name)

            with store_lock:
                ann_boxes = list(_last_ann_boxes.get(cam_name, []))

            min_h = MIN_BOX_H.get(cam_name, 0)
            for box, score, name in ann_boxes:
                x1, y1, x2, y2 = box
                if (y2 - y1) < min_h:
                    continue
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = (0, 150, 150)
                cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
                label = f'{name} {score:.2f}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                cv2.rectangle(panel, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(panel, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_COMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)

            cv2.putText(panel, cam_name, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 200), 2, cv2.LINE_AA)
            if cam_name in SPOT_CAMS:
                with store_lock:
                    states = spot_states.get(cam_name, [])
                occ = sum(1 for s in states if s == 'Car')
                cv2.putText(panel, f'{occ}/{len(states)} occupied', (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 230, 200), 1, cv2.LINE_AA)

            panels.append(panel)

        # Stitch panels side by side
        canvas = np.hstack(panels)

        roi_editor.draw_overlay(canvas)

        cv2.imshow(win, canvas)

        if os.environ.get('CALIB_DEBUG') and _debug_frames:
            debug_panels = []
            for cam in CAM_ORDER:
                p = _debug_frames.get(cam)
                if p is None:
                    p = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
                if p.shape[:2] != (DISP_H, DISP_W):
                    p = cv2.resize(p, (DISP_W, DISP_H))
                cv2.putText(p, f'DEBUG: {cam}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                debug_panels.append(p)
            cv2.imshow('boundary_debug', np.hstack(debug_panels))

        key = cv2.waitKey(1) & 0xFF

        if roi_editor.handle_key(key):
            continue

        if key == ord('u') or key == ord('U'):
            for active_idx in range(len(SOURCES)):
                cam_name = CAM_ORDER[active_idx]
                with RAW_LOCKS[active_idx]:
                    snap = RAW_BUFS[active_idx].copy()
                snap_disp = cv2.resize(snap, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
                with store_lock:
                    boxes = _last_boxes.get(cam_name, [])
                if len(boxes) > 2:
                    print(f"\033[1;93mWarning: [Auto Caliberation] {cam_name} has {len(boxes)} detections")
                new_rois = auto_caliberate_rois(cam_name, snap_disp, CALIB_PARAMS[cam_name], boxes)
                if new_rois:
                    rebuild_spot_masks(cam_name, new_rois)
            continue

        if key == ord('c') or key == ord('C'):
            calib_window.toggle(roi_editor.active_cam)
            continue
        if key == ord('q'):
            stop_event.set()
            break

        calib_window.tick()
        last_time = now

    calib_window.close()
    cv2.destroyAllWindows()

# @brief Blend detected spot boundaries with manually set ones from the CalibWindow.
#
#  Performs a weighted average between auto-detected and manually specified boundary
#  X positions. The manual weight is kept intentionally small (default 0.15) so the
#  auto-detection result dominates while the manual input provides a gentle stabilising
#  nudge. The perspective angle offset (far_x - near_x) of the detected boundary is
#  preserved through the blend.
#
#  Returns detected unchanged if manual is empty or the lists differ in length.
#
#  @param detected      List of detected boundary positions. Each entry is either
#                       a float (uniform) or a (near_x, far_x) tuple.
#  @param manual        List of manually specified boundary X positions (floats).
#  @param manual_weight Blend weight for the manual values (0.0–1.0, default 0.15).
#  @return              Blended boundary list as (near_x, far_x) tuples.
def blend_boundaries(detected: list[float], manual: list[float], manual_weight: float = 0.15) -> list[float]:
    if not manual or len(detected) != len(manual):
        return detected
    result = []
    for d, m in zip(detected, manual):
        dn = d[0] if isinstance(d, tuple) else d
        df = d[1] if isinstance(d, tuple) else d
        mn = m[0] if isinstance(m, tuple) else m

        angle_offset = df - dn
        blended_near = dn * (1 - manual_weight) + mn * manual_weight
        result.append((blended_near, blended_near + angle_offset))
    return result

## @brief Live calibration window with OpenCV trackbars for tuning CALIB_PARAMS.
#
#  Opened and closed by pressing C in the display window. Each trackbar maps to one
#  field in CALIB_PARAMS for the currently active camera. Float parameters
#  (fisheye_distortion, perspective_strength) are stored as integers scaled by 100
#  to work around OpenCV's integer-only trackbar API.
#
#  On every slider change, tick() reads all trackbar values, updates CALIB_PARAMS,
#  calls detect_spot_boundaries() on the latest camera snapshot, and then calls
#  generate_rois() to immediately rebuild and display the new ROIs — giving real-time
#  visual feedback while tuning.
#
#  suppress_auto is set to True while the window is open to prevent the background
#  auto-calibration thread from overwriting the manual adjustments.
class CalibWindow:
    """
    Separate OpenCV window with trackbars for live CALIB_PARAMS tuning.
    Opens when C is pressed, closes when C is pressed again.
    Automatically regenerates ROIs on any slider change.
    """
    WIN = 'Calibration'

    def __init__(self):
        self.open = False
        self.active_cam = 'Right'  # tracks which cam's params are shown
        self._last_vals = {}
        self._pending_update = False
        self.suppress_auto = False

    def _cam_idx(self):
        return CAM_ORDER.index(self.active_cam)

    def toggle(self, cam_name: str):
        self.active_cam = cam_name
        if self.open:
            self.close()
        else:
            self._build()

    def _build(self):
        p = CALIB_PARAMS[self.active_cam]
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, 500, 280)

        # Each trackbar: (name, min, max, initial)
        # Multiply floats by 100 to use int trackbars
        cv2.createTrackbar('VP X', self.WIN, p['vanishing_point'][0], DISP_W, lambda v: self._on_change())
        cv2.createTrackbar('VP Y', self.WIN, p['vanishing_point'][1], DISP_H, lambda v: self._on_change())
        cv2.createTrackbar('Near Y', self.WIN, p['near_y'], DISP_H, lambda v: self._on_change())
        cv2.createTrackbar('Far Y', self.WIN, p['far_y'], DISP_H, lambda v: self._on_change())
        cv2.createTrackbar('Left X', self.WIN, p['left_x'], DISP_W, lambda v: self._on_change())
        cv2.createTrackbar('Right X', self.WIN, p['right_x'], DISP_W, lambda v: self._on_change())
        cv2.createTrackbar('Spots', self.WIN, p['n_spots'], 20, lambda v: self._on_change())
        cv2.createTrackbar('Rows', self.WIN, p['n_rows'], 6, lambda v: self._on_change())
        # fisheye_distortion: range -0.30 to +0.30, stored as int -30..30 (divide by 100)
        fisheye_int = int(p['fisheye_distortion'] * 100) + 30  # shift so 0 maps to 30
        cv2.createTrackbar('Fisheye x100', self.WIN, fisheye_int, 60, lambda v: self._on_change())
        strength_int = int(p.get('perspective_strength', 1.0) * 100)
        cv2.createTrackbar('Persp x100', self.WIN, strength_int, 100, lambda v: self._on_change())
        cv2.createTrackbar('Min Height', self.WIN, p.get('min_roi_height', 60), DISP_H, lambda v: self._on_change())
        self.suppress_auto = True
        self.open = True
        print(f"[Calib] Window open for {self.active_cam}. Drag sliders to tune, C to close.")

    def _read(self) -> dict:
        fisheye_int = cv2.getTrackbarPos('Fisheye x100', self.WIN)
        return {
            'vanishing_point': (
                cv2.getTrackbarPos('VP X', self.WIN),
                cv2.getTrackbarPos('VP Y', self.WIN),
            ),
            'near_y': cv2.getTrackbarPos('Near Y', self.WIN),
            'far_y': cv2.getTrackbarPos('Far Y', self.WIN),
            'left_x': cv2.getTrackbarPos('Left X', self.WIN),
            'right_x': cv2.getTrackbarPos('Right X', self.WIN),
            'n_spots': max(1, cv2.getTrackbarPos('Spots', self.WIN)),
            'n_rows': max(1, cv2.getTrackbarPos('Rows', self.WIN)),
            'fisheye_distortion': (fisheye_int - 30) / 100.0,
            'perspective_strength': cv2.getTrackbarPos('Persp x100', self.WIN) / 100.0,
            'min_roi_height': max(20, cv2.getTrackbarPos('Min Height', self.WIN)),
        }

    def _on_change(self):
        if self.open:
            self._pending_update = True

    def tick(self):
        """Call once per display loop iteration to keep the window alive."""
        if not self.open:
            return

        if not self._pending_update:
            return
        self._pending_update = False

        try:
            params = self._read()
            near_y = params['near_y']
            far_y = params['far_y']
            vpy = params['vanishing_point'][1]

            if near_y == far_y or vpy >= near_y:
                return

            CALIB_PARAMS[self.active_cam] = params

            cam_idx = CAM_ORDER.index(self.active_cam)
            with RAW_LOCKS[cam_idx]:
                snap = RAW_BUFS[cam_idx].copy()
            snap_disp = cv2.resize(snap, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
            with store_lock:
                boxes = _last_boxes.get(self.active_cam, [])
            boundaries, detected_n = detect_spot_boundaries(
                snap_disp, near_y, far_y,
                params['left_x'], params['right_x'], params['n_spots'],
                detections=boxes, cam_name=self.active_cam
            )
            roi_params = {k: v for k, v in params.items() if k not in ('min_roi_height',)}
            new_rois = generate_rois(
                cam_id=self.active_cam,
                n_spots=detected_n,
                spot_boundaries=boundaries,
                **{k: v for k, v in roi_params.items() if k != 'n_spots'},
            )
            if new_rois:
                rebuild_spot_masks(self.active_cam, new_rois)
                n = params['n_spots']
                manual = [params['left_x'] + (params['right_x'] - params['left_x']) * i / n for i in range(n + 1)]
                _manual_bounds[self.active_cam] = manual
        except Exception as e:
            print(f"[Calib] tick error: {e}")

    def close(self):
        if self.open:
            try:
                cv2.destroyWindow(self.WIN)
            except Exception:
                pass
            self.open = False
            self.suppress_auto = False
            print(f"[Calib] Window closed. Final params for {self.active_cam}:")
            print(f"  CALIB_PARAMS['{self.active_cam}'] = {CALIB_PARAMS[self.active_cam]}")


calib_window = CalibWindow()

## @brief Recording loop — writes camera frames to timestamped MP4 files.
#
#  Runs as a daemon Thread when -r / --record is passed. Pinned to CPU core 5
#  (shared with the display loop, which is absent in headless record mode).
#  Creates one MP4 file per camera in the recordings/ directory, named with a
#  timestamp prefix and the camera name.
#
#  When use_anns is True, annotated frames from ANN_BUFS are written instead
#  of raw frames, burning in bounding boxes and ROI overlays permanently.
#  Note that ANN_BUFS are only populated when need_annotations is True, which
#  requires either -T or -r to be passed at startup.
#
#  @param stop_event  Multiprocessing Event polled to stop recording and flush files.
#  @param use_anns    If True, record annotated frames; otherwise record raw frames. a 
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

## @brief Background auto-calibration thread — periodically re-runs ROI calibration.
#
#  Runs as a daemon Thread pinned to CPU core 3. Sleeps for AUTO_CALIBRATE_INTERVAL
#  seconds between passes (set to 0 to disable). On each pass it iterates over all
#  SPOT_CAMS, skips cameras where the CalibWindow is open (to avoid fighting the
#  user's manual adjustments), and calls auto_caliberate_rois() on the latest
#  snapshot. If calibration succeeds, rebuild_spot_masks() is called to apply
#  the new ROIs immediately.
#
#  Skips cameras with more than 3 simultaneous detections, as a heavily occupied
#  lot provides poor lane-marking signal for edge detection.
#
#  @param stop_event Multiprocessing Event polled to exit the loop.
def auto_calibrate_loop(stop_event):
    try:
        os.sched_setaffinity(0, {3})
    except Exception:
        pass

    print('[Auto Calibrate] Background Calibration thread started')
    while not stop_event.is_set():
        time.sleep(AUTO_CALIBRATE_INTERVAL)

        if stop_event.is_set():
            break

        for idx, cam_name in enumerate(CAM_ORDER):
            if cam_name not in SPOT_CAMS:
                continue

            if calib_window.suppress_auto and calib_window.active_cam == cam_name:
                continue
            try:
                with RAW_LOCKS[idx]:
                    snap = RAW_BUFS[idx].copy()

                snap_disp = cv2.resize(snap, (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)

                with store_lock:
                    boxes = _last_boxes.get(cam_name, [])

                if len(boxes) > 3:
                    print(f'[Auto Calibrate] {cam_name}: skipping --- too many detections ({len(boxes)})')
                    continue

                new_rois = auto_caliberate_rois(cam_name, snap_disp, CALIB_PARAMS[cam_name], boxes)

                if new_rois:
                    rebuild_spot_masks(cam_name, new_rois)
                    print(f'[Auto Calibrate] {cam_name}: updated')

            except Exception as e:
                print(f'[Auto Calibrate] {cam_name} error: {e}')

## @brief Remove spot boundaries that would produce slots narrower than min_width pixels.
#
#  Iterates through the boundary list and skips any position that would create a
#  slot width below min_width relative to the previous kept boundary. This prevents
#  the ROI generator from producing tiny sliver polygons when line detection finds
#  spurious closely-spaced dividers. The last boundary is always forced to match
#  the original final value to preserve the overall parking area extent.
#
#  @param boundaries List of X boundary positions (floats), including left and right edges.
#  @param min_width  Minimum acceptable slot width in pixels (default 80.0).
#  @return           Filtered boundary list with narrow gaps merged away.
def merge_narrow_spots(boundaries: list[float], min_width: float = 80.0) -> list[float]:
    if len(boundaries) < 2:
        return boundaries

    merged = [boundaries[0]]
    for i in range(1, len(boundaries)):
        width = boundaries[i] - merged[-1]
        if width < min_width and len(merged) > 1:
            continue
        merged.append(boundaries[i])

    if merged[-1] != boundaries[-1]:
        merged[-1] = boundaries[-1]

    return merged

## @brief Detect parking spot divider X positions from lane markings in a camera frame.
#
#  Pipeline:
#  1. Thresholds the greyscale frame to isolate bright white lane markings.
#  2. Applies morphological closing to fill small gaps in the markings.
#  3. Runs Canny edge detection and masks to the parking area band.
#  4. Uses Probabilistic Hough Transform to find line segments.
#  5. Filters for near-vertical lines (45°–135°) and projects their X intercepts
#     to near_y and far_y to produce (near_x, far_x) boundary pairs.
#  6. Clusters nearby dividers (within 20px) and merges narrow gaps.
#  7. If the detected count matches n_spots+1, returns those boundaries.
#     If too many are found, picks the n_spots best-matching ones.
#     If too few, falls back to _try_from_cars().
#
#  When CALIB_DEBUG is set, saves an annotated debug frame to _debug_frames.
#
#  @param frame      BGR display-resolution frame to analyse.
#  @param near_y     Y pixel of the near edge of the parking area.
#  @param far_y      Y pixel of the far edge of the parking area.
#  @param left_x     Left boundary X pixel.
#  @param right_x    Right boundary X pixel.
#  @param n_spots    Expected number of parking spots.
#  @param detections Optional list of [x1,y1,x2,y2] detection boxes for fallback.
#  @param cam_name   Camera name used for debug frame storage.
#  @return           Tuple of (boundaries, n_spots) where boundaries is a list of
#                    (near_x, far_x) tuples or plain floats, and n_spots is the
#                    detected spot count.
def detect_spot_boundaries(frame: np.ndarray, near_y: int, far_y: int, left_x: int, right_x: int, n_spots: int,
                           detections: list = None, cam_name: str = None) -> tuple[list[float], int]:
    uniform = [left_x + (right_x - left_x) * i / n_spots for i in range(n_spots + 1)]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    # blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(white_mask, 50, 150)

    roi_mask = np.zeros_like(edges)
    band_top = max(0, far_y - 10)
    roi_mask[band_top:near_y + 10, left_x:right_x] = 255
    edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=25, maxLineGap=30)
    divider_xs = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 45 < angle < 135:

            if y2 != y1:
                t = (near_y - y1) / (y2 - y1)
                x_at_near = x1 + t * (x2 - x1)
                t_far = (far_y - y1) / (y2 - y1)
                x_far = x1 + t_far * (x2 - x1)
            else:
                x_at_near = x_far = (x1 + x2) / 2
            if left_x <= x_at_near <= right_x:
                divider_xs.append((x_at_near, x_far))
    if os.environ.get('CALIB_DEBUG'):
        dbg = frame.copy()
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if 45 < angle < 135:
                    cv2.line(dbg, (x1, y1), (x2, y2), (255, 0, 0), 1)
        for xn, xf in divider_xs:
            cv2.line(dbg, (int(xn), near_y), (int(xf), far_y), (0, 255, 0), 2)
        cv2.line(dbg, (left_x, near_y), (right_x, near_y), (0, 200, 255), 1)
        cv2.line(dbg, (left_x, far_y), (right_x, far_y), (0, 200, 255), 1)
        if cam_name:
            _debug_frames[cam_name] = dbg

    if lines is None or len(divider_xs) < 2:
        return _try_from_cars(detections, near_y, left_x, right_x, n_spots, uniform)

    divider_xs.sort(key=lambda d: d[0])
    clustered = []
    group = [divider_xs[0]]
    for x in divider_xs[1:]:
        if x[0] - group[0][0] < 20:
            group.append(x)
        else:
            clustered.append((np.mean([g[0] for g in group]), np.mean([g[1] for g in group])))
            group = [x]
    clustered.append((np.mean([g[0] for g in group]), np.mean([g[1] for g in group])))

    xs_near = [left_x] + [c[0] for c in clustered] + [right_x]
    xs_far = [left_x] + [c[1] for c in clustered] + [right_x]
    all_bounds = sorted(zip(xs_near, xs_far), key=lambda p: p[0])
    xs_near = [p[0] for p in all_bounds]
    xs_far = [p[1] for p in all_bounds]

    paired = list(zip(xs_near, xs_far))
    merged_pairs = [paired[0]]
    for i in range(1, len(paired)):
        width = paired[i][0] - merged_pairs[-1][0]
        if width >= 80.0 or len(merged_pairs) == 1:
            merged_pairs.append(paired[i])
        else:
            merged_pairs[-1] = paired[i]
    xs_far = [p[1] for p in merged_pairs]
    xs_near = [p[0] for p in merged_pairs]
    if len(xs_near) == n_spots + 1:
        return list(zip(xs_near, xs_far)), len(xs_near) - 1

    if len(xs_near) > n_spots + 1:
        expected = (right_x - left_x) / n_spots
        best_near, best_far = [left_x], [left_x]
        interior = list(zip(xs_near[1:-1], xs_far[1:-1]))
        for i in range(1, n_spots):
            target = left_x + expected * i
            closest = min(interior, key=lambda p: abs(p[0] - target))
            best_near.append(closest[0])
            best_far.append(closest[1])
        best_near.append(right_x)
        best_far.append(right_x)
        return list(zip(best_near, best_far)), len(best_near) - 1

    return _try_from_cars(detections, near_y, left_x, right_x, n_spots, uniform)

## @brief Fallback boundary estimator that infers spot dividers from car center positions.
#
#  Used when lane marking detection finds too few divider lines. Takes the horizontal
#  centers of detected cars that are close to near_y and places divider boundaries
#  at the midpoints between adjacent cars. Falls back to a uniform layout (after
#  merge_narrow_spots cleaning) if fewer than 2 car centers are available or if the
#  resulting boundary count does not match n_spots+1.
#
#  @param detections  List of [x1,y1,x2,y2] detection boxes, or None.
#  @param near_y      Y pixel of the near edge used to filter relevant detections.
#  @param left_x      Left boundary X pixel.
#  @param right_x     Right boundary X pixel.
#  @param n_spots     Expected number of parking spots.
#  @param fallback    Uniform boundary list used when car-based estimation fails.
#  @return            Tuple of (boundaries, n_spots).
def _try_from_cars(detections, near_y, left_x, right_x, n_spots, fallback):
    if not detections:
        fallback = merge_narrow_spots(fallback)
        return fallback, len(fallback) - 1

    car_centers = []
    for x1, y1, x2, y2 in detections:
        if abs(y2 - near_y) < 80:
            car_centers.append((x1 + x2) / 2)

    car_centers = sorted(c for c in car_centers if left_x < c < right_x)

    if len(car_centers) < 2:
        fallback = merge_narrow_spots(fallback)
        return fallback, len(fallback) - 1

    boundaries = [left_x]
    for i in range(len(car_centers) - 1):
        boundaries.append((car_centers[i] + car_centers[i + 1]) / 2)
    boundaries.append(right_x)

    if len(boundaries) == n_spots + 1:
        boundaries = merge_narrow_spots(boundaries)
        return boundaries, len(boundaries) - 1
    fallback = merge_narrow_spots(fallback)
    return [(x, x) for x in fallback], len(fallback) - 1

## @brief atexit handler — gracefully shuts down all workers and frees shared memory.
#
#  Registered with atexit so it runs automatically on normal exit, KeyboardInterrupt,
#  or unhandled exceptions. Only executes in the main process (guarded by _MAIN_PID)
#  to prevent child capture processes from triggering a double-shutdown.
#
#  Sets STOP_EVENT, joins all capture processes with a 3-second timeout, unlinks
#  all shared memory blocks, and restores Jetson clock settings.
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
    set_jetson_clocks(False)
    print('Done')


if __name__ == "__main__":
    args = parse_args()
    _MAIN_PID = os.getpid()
    frame_ready_event = Event()
    set_jetson_clocks(True)
    if args.annotate and not args.record:
        print("\033[1;91mWarning: -a/-A/--annotate has no effect without -r/-R/--record\033[0m")
    need_annotation = args.test or args.record
    os.system('v4l2-ctl --list-devices > ./utils/camInfo.txt')
    with open('./utils/camInfo.txt') as f:
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
            args=(i, src, RAW_SHM_OBJS[i].name, RAW_LOCKS[i], RAW_FRAME_ID[i], frame_ready_event, STOP_EVENT),
            daemon=False,
        )
        p.start()
        try:
            os.sched_setaffinity(p.pid, {i + 1})
        except Exception:
            pass
        PROCESSES.append(p)

    print("Waiting for cameras...")
    deadline = time.time() + 15
    for i in range(len(SOURCES)):
        while RAW_FRAME_ID[i].value == 0:
            if not PROCESSES[i].is_alive():
                raise RuntimeError(
                    f"Camera {i} ({CAM_ORDER[i]}) process died before sending a frame."
                )
            if time.time() > deadline:
                raise RuntimeError(
                    f"Camera {i} ({CAM_ORDER[i]}) timed out waiting for the first frame"
                )
            time.sleep(0.1)
        print(f'Camera {i} ({CAM_ORDER[i]}) ready')
    auto_calibrate_t = Thread(
        target=auto_calibrate_loop,
        args=(STOP_EVENT,),
        daemon=True,
    )
    auto_calibrate_t.start()
    inf_t = Thread(
        target=inference_loop,
        args=(need_annotation, frame_ready_event, STOP_EVENT),
        daemon=True
    )
    inf_t.start()
    Thread(
        target=lambda: app.run(host='localhost', port=5001, debug=False, use_reloader=False, threaded=True),
        daemon=True,
    ).start()
    print('Flask Api -> http:/localhost/detections')

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