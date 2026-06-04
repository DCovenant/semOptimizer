"""Background thread that runs continuous YOLO tracking on the live feeds.

This is a thin *client* of the out-of-process inference service: it does no
torch/YOLO itself. The CARLA egg pins this app to Python 3.7, but ROCm torch
needs 3.10+, so detection lives in inference/service.py (Python 3.12 + the GPU)
behind a Unix socket. See inference/service.py for the ownership split.

Per pass, for each arm: crop the live frame to the lane ROI (so the GPU only
sees the lane region at full detail — high effective resolution where demand is
measured, a fraction of the pixels), ship the crop to the service, offset the
returned boxes back into full-frame coordinates, then run the pure-Python
lane-assignment + distance-weighting (analyze_tracks) here.

Calibrations + ROIs are snapshotted at construction; re-calibrating an arm means
restarting the worker (the UI does this on the Run-YOLO toggle).
"""
import socket
import statistics
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.config import (ANALYSIS_INTERVAL, DEMAND_SMOOTHING, DETECT_CONF,
                        INFER_IMGSZ, PERCEPTION_WINDOW)
from app.core.analysis import analyze_tracks

# protocol.py lives in inference/ at the repo root; it is pure-Python and
# 3.7-safe by design, so the 3.7 app can import it directly.
from inference.protocol import recv_message, send_message

# extra margin (fraction of the ROI's larger side) so a vehicle whose foot is in
# the lane but whose body extends past the polygon is still fully in the crop.
_ROI_PAD = 0.12


class _ArmProxy:
    """Minimal stand-in carrying just `calibration` for analyze_tracks."""

    def __init__(self, calibration):
        self.calibration = calibration


def _lane_bbox(calibration):
    """Bounding box (x1, y1, x2, y2 ints) of all lane polygons, padded.

    Returns None if the calibration has no usable polygons — caller then sends
    the whole frame for that arm.
    """
    pts = []
    for poly in calibration.get("lanes", {}).values():
        if len(poly) >= 2:
            pts.extend(poly)
    if not pts:
        return None
    arr = np.asarray(pts, dtype=float)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    pad = _ROI_PAD * max(x2 - x1, y2 - y1)
    return (int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad))


class AnalysisWorker(QThread):
    analysis_ready = Signal(object)   # {arm: result dict from analyze_tracks}
    status_message = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, frame_getter, calibrations: dict, socket_path: str,
                 parent=None):
        """frame_getter() -> {arm: np.ndarray RGB}; calibrations: {arm: cal dict}."""
        super().__init__(parent)
        self._frame_getter = frame_getter
        self._socket_path = socket_path
        self._arms = {arm: _ArmProxy(cal)
                      for arm, cal in calibrations.items() if cal}
        self._rois = {arm: _lane_bbox(cal)
                      for arm, cal in calibrations.items() if cal}
        self._running = False
        self._sock = None
        self._last_frame = {}   # arm -> last ndarray inferred (frame dedup)
        self._results = {}      # arm -> last analyze_tracks result (stable demand)
        self._smooth = {}       # arm -> {"wd": float, "pd": {phase: float}} EMA
        self._count_hist = {}   # arm -> deque of recent raw counts (median window)

    def stop(self):
        self._running = False
        # interrupt a blocking recv so the thread can exit promptly — otherwise
        # it sits in the socket timeout and gets destroyed mid-run on app close
        # ("QThread: Destroyed while thread is still running" → abort).
        s = self._sock
        if s is not None:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    # ── service connection ────────────────────────────────────────────────────
    def _connect(self):
        """Connect to the inference service, retrying while it warms up."""
        deadline = time.time() + 30.0
        last_err = None
        while time.time() < deadline and self._running:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(60.0)   # generous: first GPU pass JITs kernels (~13s)
                s.connect(self._socket_path)
                self._sock = s
                send_message(s, {"type": "reset"})
                recv_message(s)
                return True
            except (ConnectionError, OSError) as e:
                last_err = e
                time.sleep(0.5)
        if self._running:
            self.error_occurred.emit(
                "Could not reach inference service at %s: %s"
                % (self._socket_path, last_err))
        return False

    def _crop(self, arm, frame):
        """Crop a frame to the arm's lane ROI. Returns (crop, origin_x, origin_y)."""
        roi = self._rois.get(arm)
        h, w = frame.shape[:2]
        if roi is None:
            return np.ascontiguousarray(frame), 0, 0
        x1 = max(0, min(roi[0], w - 1)); y1 = max(0, min(roi[1], h - 1))
        x2 = max(x1 + 1, min(roi[2], w)); y2 = max(y1 + 1, min(roi[3], h))
        return np.ascontiguousarray(frame[y1:y2, x1:x2]), x1, y1

    def _detect_batch(self, frames):
        """One request for all arms whose frame is NEW since the last pass.

        Frame dedup matters: in sync mode the sim emits frames slower than this
        loop runs, so without it we'd re-infer the same stale frame repeatedly —
        boxes freeze on the old image while burning GPU that CARLA needs to make
        the next frame. We compare frame identity (the cache hands out the same
        ndarray until CARLA emits a new one) and skip unchanged arms.

        Returns {arm: [det, ...]} only for arms re-inferred this pass.
        """
        crops_meta, payload_parts, origins = [], [], {}
        for arm in self._arms:
            frame = frames.get(arm)
            if frame is None:
                continue
            if frame is self._last_frame.get(arm):   # unchanged → skip
                continue
            self._last_frame[arm] = frame
            crop, ox, oy = self._crop(arm, frame)
            crops_meta.append({"arm": arm, "h": crop.shape[0], "w": crop.shape[1]})
            payload_parts.append(crop.tobytes())
            origins[arm] = (ox, oy)
        if not crops_meta:
            return {}

        header = {"type": "detect_batch", "conf": DETECT_CONF, "crops": crops_meta}
        if INFER_IMGSZ:
            header["imgsz"] = INFER_IMGSZ
        send_message(self._sock, header, b"".join(payload_parts))
        reply, _ = recv_message(self._sock)
        if reply.get("type") != "detections":
            raise RuntimeError(reply.get("message", "bad reply from service"))

        out = {}
        for arm, (ox, oy) in origins.items():
            dets = []
            for d in reply["results"].get(arm, []):
                bx = d["box"]
                dets.append({
                    "box": (bx[0] + ox, bx[1] + oy, bx[2] + ox, bx[3] + oy),
                    "foot": (d["foot"][0] + ox, d["foot"][1] + oy),
                    "cls": d["cls"], "conf": d["conf"],
                })
            out[arm] = dets
        return out

    def _smooth_demand(self, arm, res):
        """EMA-smooth this arm's weighted/phase demand in place. The boxes and
        live count stay raw (per-frame); only the timing signal is smoothed so it
        doesn't jump as detections flicker."""
        a = DEMAND_SMOOTHING
        prev = self._smooth.get(arm)
        wd, pd = res["weighted_demand"], res["phase_demand"]
        if prev is None:
            swd, spd = wd, dict(pd)
        else:
            swd = a * wd + (1 - a) * prev["wd"]
            spd = {k: a * pd.get(k, 0.0) + (1 - a) * prev["pd"].get(k, 0.0)
                   for k in set(pd) | set(prev["pd"])}
        self._smooth[arm] = {"wd": swd, "pd": spd}
        res["weighted_demand"] = swd
        res["phase_demand"] = spd

    def _stabilize_count(self, arm, res):
        """Replace the raw per-frame car count with the median of the last
        PERCEPTION_WINDOW frames, so brief detection dropouts don't make the
        count jump (3 → 0/1/2 → 3). Boxes stay live; only the number steadies."""
        hist = self._count_hist.get(arm)
        if hist is None:
            hist = self._count_hist[arm] = deque(maxlen=PERCEPTION_WINDOW)
        hist.append(res["count"])
        res["count"] = int(round(statistics.median(hist)))

    def run(self):
        self._running = True
        if not self._connect():
            return
        self.status_message.emit(
            "Live YOLO running on %d arm(s): %s"
            % (len(self._arms), ", ".join(sorted(self._arms))))

        try:
            while self._running:
                t0 = time.time()
                frames = self._frame_getter() or {}
                detections = self._detect_batch(frames)
                # merge fresh per-arm results over the last set, so dedup-skipped
                # arms keep their last value and the demand display stays complete.
                for arm, dets in detections.items():
                    res = analyze_tracks(self._arms[arm], dets)
                    self._smooth_demand(arm, res)
                    self._stabilize_count(arm, res)
                    self._results[arm] = res
                if detections and self._results:
                    self.analysis_ready.emit(dict(self._results))
                dt = time.time() - t0
                if self._running and dt < ANALYSIS_INTERVAL:
                    time.sleep(ANALYSIS_INTERVAL - dt)
        except (ConnectionError, OSError) as e:
            if self._running:
                self.error_occurred.emit("Inference service connection lost: %s" % e)
        except Exception as e:
            self.error_occurred.emit("Analysis failed: %s" % e)
        finally:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
            self.status_message.emit("Live YOLO stopped.")
