"""Run detection + lane assignment for an arm, headless (no Qt).

Given a loaded YOLO model and an Arm (its camera image + lane calibration), detect
the vehicles, drop each foot point into a lane polygon, and tally per-lane counts,
per-direction totals, the ignored/parked leftover, and per-phase incoming demand.
"""
import numpy as np
from PIL import Image

from app.config import DEFAULT_PHASE, distance_weight
from app.core.detection import detect_vehicles
from app.core.lane_distance import stop_line_distance


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


def _run_detection(model, img: np.ndarray, arm) -> dict:
    """Core detection + lane-assignment logic shared by both public functions."""
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


def analyze_arm(model, arm) -> dict:
    """Detect + assign for one arm using arm.image (file path).

        Returns {lanes, directions, incoming, outgoing, ignored, total, phase_demand}.
    """
    img = np.asarray(Image.open(arm.image).convert("RGB"))
    return _run_detection(model, img, arm)


def analyze_arm_frame(model, arm, frame_array: np.ndarray) -> dict:
    """Like analyze_arm but accepts a pre-captured numpy RGB array (H, W, 3 uint8).

    Used by CarlaWorker to run YOLO on live camera frames without disk I/O.
    """
    return _run_detection(model, np.ascontiguousarray(frame_array), arm)


def analyze_tracks(arm, tracked_dets: list) -> dict:
    """Distance-weighted demand for one arm from already-tracked detections.

    `tracked_dets` are dicts from detection.track_vehicles ({box, foot, id, …}).
    Each detection inside an incoming lane gets a normalised distance from the
    stop line and a weight (config.distance_weight). The per-arm demand is the
    sum of those weights — a car at the line counts ~1.0, one far up the queue
    ~0.0. Detections outside every lane are ignored/parked.

    Returns:
        {tracks, count, weighted_demand, phase_demand, ignored}
    where `tracks` carries per-vehicle overlay data (id, box, foot, lane,
    distance, weight) for the lane-assigned vehicles plus the ignored ones.
    """
    cal = arm.calibration or {}
    lanes = cal.get("lanes", {})
    phases = cal.get("phases", {})

    tracks = []
    count = ignored = 0
    weighted_demand = 0.0
    phase_demand: dict = {}

    for d in tracked_dets:
        name = assign_lane(d["foot"], cal)
        if name is None:
            ignored += 1
            tracks.append({"id": d.get("id"), "box": d["box"], "foot": d["foot"],
                           "lane": None, "distance": None, "weight": 0.0})
            continue
        dist = stop_line_distance(lanes[name], d["foot"])
        w = distance_weight(dist)
        count += 1
        weighted_demand += w
        ph = phases.get(name, DEFAULT_PHASE)
        phase_demand[ph] = phase_demand.get(ph, 0.0) + w
        tracks.append({"id": d.get("id"), "box": d["box"], "foot": d["foot"],
                       "lane": name, "distance": dist, "weight": w})

    return {
        "tracks": tracks,
        "count": count,
        "weighted_demand": weighted_demand,
        "phase_demand": phase_demand,
        "ignored": ignored,
    }
