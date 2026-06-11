"""Configuration constants for the lane-assignment app."""
from PySide6.QtGui import QColor

# ── detection ────────────────────────────────────────────────────────────────
# YOLOv8 (not v11): the venv is Python 3.7 (pinned by the CARLA egg), so it's
# stuck on ultralytics 8.0.x, which predates YOLO11's C3k2 block. yolo11*.pt
# will fail to load here — use yolov8*.pt (swap n→s/m for more accuracy).
YOLO_MODEL      = "models/yolov8n.pt"   # static-image path only (3.7 in-process)

# Detection confidence for the live per-frame detector. Moderate: with no
# tracker to stabilise things, very low conf just spawns background false boxes
# that inflate demand. Per-frame flicker is absorbed by smoothing the demand
# signal (DEMAND_SMOOTHING), not by tracking individual cars.
DETECT_CONF     = 0.15

# Demand smoothing: the per-arm weighted demand is the signal that feeds signal
# timing, so it must be stable even though raw per-frame detections flicker.
# Exponential moving average — weight of each new sample (0..1]; lower = smoother
# but laggier. demand = α·now + (1-α)·previous.
DEMAND_SMOOTHING = 0.2

# Perceived-car-count stabilisation: the integer count jumps as detections blink
# on/off, so we report the MEDIAN of the last N frames instead of the raw count.
# Median rejects brief dropout frames (a lane that's usually 3 reads a steady 3)
# without sagging toward the average the way an EMA would. Larger = steadier but
# slower to reflect a car genuinely arriving/leaving. In frames (≈ N / fps secs).
PERCEPTION_WINDOW = 7

# Model for the live out-of-process inference service. It runs py3.12 + latest
# ultralytics, so the py3.7 "v8 only" limit does NOT apply — use the bigger,
# more accurate yolo11 weights. Bump down (yolo11l/m/s/n) if GPU throughput is
# the bottleneck. Path is relative to the repo root (the service's cwd).
SERVICE_MODEL   = "models/yolo11m.pt"
CONF            = 0.25                  # detection confidence threshold
VEHICLE_CLASSES = {1, 2, 3, 5, 7}       # COCO: bicycle, car, motorcycle, bus, truck
PERSON_CLASSES  = {0}                   # COCO: person — counted as pedestrians, not cars

# ── live tracking / analysis ───────────────────────────────────────────────────
TRACKER           = "bytetrack.yaml"    # Ultralytics built-in tracker config
ANALYSIS_INTERVAL = 0.15                # seconds between live analysis passes (~6-7 Hz)

# ── camera capture ─────────────────────────────────────────────────────────────
# Capture resolution is decoupled from the on-screen tile: cameras render at
# this (high) resolution into an off-screen buffer, the inference client crops
# the lane ROI from it at full detail, and the display tiles just downscale it.
# Raising this improves recall on distant queued cars (the ROI crop carries more
# pixels) at the cost of CARLA GPU render time. NOTE: lane calibration polygons
# live in capture-pixel coordinates — change this and you must re-calibrate.
CAPTURE_W = 1920
CAPTURE_H = 1080

# Minimum sim-seconds between camera captures. The app runs CARLA in async mode,
# so without this each 1080p camera re-renders on *every* server frame — 4 of
# them starve the GPU, the server FPS craters, and the async Traffic Manager
# goes unstable (cars zigzag / drive off the road). We only display ~10 Hz and
# analyse ~2 Hz, so capping capture rate here costs nothing and frees the sim.
SENSOR_TICK = 0.1   # 10 Hz

# Max wall-clock rate at which the worker pushes frames to the GUI. The sync loop
# advances sim time as fast as the machine allows, so when the GPU isn't busy the
# sim ticks far faster than real time. Emitting four 1080p frames to the GUI on
# every sensor frame then floods the cross-thread (QueuedConnection) event queue
# faster than the GUI can paint it: the backlog grows unbounded (each event pins
# ~24 MB of ndarrays), FPS collapses, and the app eventually OOM-crashes. Cap the
# *display* push to this real-time cadence; the sim + capture keep their own rate.
FRAME_EMIT_INTERVAL = 1.0 / 30   # seconds (≈30 Hz display)

# Fixed simulation step for synchronous mode. The worker drives CARLA with
# world.tick() so physics advances this many sim-seconds per tick regardless of
# how long rendering + GPU inference take — the sim slows in wall-clock when the
# GPU is busy but stays physically correct (no FPS-coupled Traffic Manager
# zigzag / off-road). 0.05 = a 20 Hz sim step (CARLA's recommended default).
SIM_FIXED_DELTA = 0.05

# Inference resolution: YOLO resizes the lane crop so its longest side == this
# before detecting. MUST scale up with CAPTURE_* — capturing more pixels does
# nothing for distant cars if YOLO then shrinks the crop back to 640. Higher =
# better small/distant recall, but slower (cost grows ~quadratically). 0 = let
# ultralytics use its default (640).
INFER_IMGSZ = 1280

# Hard cap on the EFFECTIVE inference resolution of any single lane crop. A
# calibration whose lanes span most of the frame yields a near-full-frame crop
# per arm; four of those inferred at INFER_IMGSZ each pass is the GPU-memory
# load that evicts CARLA's VRAM into system RAM (GTT/shmem) and OOMed the box
# on 2026-06-11 — MemoryMax can't see those pages. Crops are inferred at their
# real size (never upscaled) and never above this; crops more than twice this
# wide are decimated before shipping so the socket payload shrinks too.
CROP_MAX_SIDE = 960

# The pedestrian (crossing) crop is shipped and inferred SEPARATELY from the
# lane crop: the crossing zone sits on the far sidewalk, so one bbox around
# lanes+crossings would span nearly the whole frame and void the ROI saving
# (that mistake made every crop full-frame and pushed the 16 GB box into
# swap-thrash). The crossing strip is wide but its pedestrians stand close to
# the pole camera (large in the image), so it tolerates being decimated before
# sending and inferred at a lower resolution than the lane crop.
PED_CROP_DECIMATE = 2     # keep every Nth pixel of the crossing crop (1 = off)
PED_INFER_IMGSZ   = 960   # YOLO long-side for the (decimated) crossing crop


def distance_weight(d):
    """Weight a vehicle by its normalised distance from the stop line.

    `d` in [0,1]: 0 at the stop line, 1 at the far end of the lane. Linear —
    a car at the line counts 1.0, one at the far end ~0.0. Swap for a bucketed
    or exponential curve to bias the timing logic differently.
    """
    return 1.0 - d

# ── lanes ────────────────────────────────────────────────────────────────────
DEFAULT_PHASE = "approach_A"
DIRECTIONS    = ("incoming", "outgoing")

# colour-code by direction so both are highlighted and distinguishable
INCOMING_PALETTE = ["#2ca02c", "#17becf", "#1f9e89", "#66c2a5", "#006d2c"]   # greens/teals
OUTGOING_PALETTE = ["#ff7f0e", "#d62728", "#e377c2", "#bcbd22", "#8c564b"]   # oranges/reds
IGNORED_COLOR    = QColor(150, 150, 150)

# ── misc ─────────────────────────────────────────────────────────────────────
DEFAULT_IMAGE = "data/test1.png"
