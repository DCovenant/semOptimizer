"""Run detection + lane assignment for an arm, headless (no Qt).

Given a loaded YOLO model and an Arm (its camera image + lane calibration), detect
the vehicles, drop each foot point into a lane polygon, and tally per-lane counts,
per-direction totals, the ignored/parked leftover, and per-phase incoming demand.
"""
import numpy as np
from PIL import Image

from app.config import DEFAULT_PHASE
from app.core.detection import detect_vehicles


def point_in_polygon(x, y, poly):
    """Ray-casting point-in-polygon. `poly` is a list of (x, y)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def assign_lane(foot, calibration):
    """Return the name of the lane whose polygon contains `foot`, else None."""
    for name, pts in calibration.get("lanes", {}).items():
        if len(pts) >= 3 and point_in_polygon(foot[0], foot[1], pts):
            return name
    return None


def analyze_arm(model, arm):
    """Detect + assign for one arm. Returns a result dict:

        {lanes: {name: count}, incoming, outgoing, ignored, total,
         phase_demand: {phase: count}}
    """
    img = np.asarray(Image.open(arm.image).convert("RGB"))
    detections = detect_vehicles(model, img)

    cal = arm.calibration or {}
    directions = cal.get("directions", {})
    phases = cal.get("phases", {})

    lane_counts = {name: 0 for name in cal.get("lanes", {})}
    incoming = outgoing = ignored = 0
    phase_demand = {}

    for d in detections:
        name = assign_lane(d["foot"], cal)
        if name is None:
            ignored += 1
            continue
        lane_counts[name] += 1
        if directions.get(name) == "incoming":
            incoming += 1
            ph = phases.get(name, DEFAULT_PHASE)
            phase_demand[ph] = phase_demand.get(ph, 0) + 1
        else:
            outgoing += 1

    return {
        "lanes": lane_counts,
        "directions": directions,
        "incoming": incoming,
        "outgoing": outgoing,
        "ignored": ignored,
        "total": len(detections),
        "phase_demand": phase_demand,
    }
