"""Configuration constants for the lane-assignment app."""
from PySide6.QtGui import QColor

# ── detection ────────────────────────────────────────────────────────────────
YOLO_MODEL      = "models/yolo11n.pt"          # swap for yolo11s/m for more accuracy
CONF            = 0.25                  # detection confidence threshold
VEHICLE_CLASSES = {1, 2, 3, 5, 7}       # COCO: bicycle, car, motorcycle, bus, truck

# ── lanes ────────────────────────────────────────────────────────────────────
DEFAULT_PHASE = "approach_A"
DIRECTIONS    = ("incoming", "outgoing")

# colour-code by direction so both are highlighted and distinguishable
INCOMING_PALETTE = ["#2ca02c", "#17becf", "#1f9e89", "#66c2a5", "#006d2c"]   # greens/teals
OUTGOING_PALETTE = ["#ff7f0e", "#d62728", "#e377c2", "#bcbd22", "#8c564b"]   # oranges/reds
IGNORED_COLOR    = QColor(150, 150, 150)

# ── misc ─────────────────────────────────────────────────────────────────────
DEFAULT_IMAGE = "data/test1.png"
