"""Background thread that runs continuous YOLO tracking on the live feeds.

This is a thin *client* of the out-of-process inference service: it does no
torch/YOLO itself. The CARLA egg pins this app to Python 3.7, but ROCm torch
needs 3.10+, so detection lives in inference/service.py (Python 3.12 + the GPU)
behind a Unix socket. See inference/service.py for the ownership split.

Per pass, for each arm: crop the live frame to the lane ROI (so the GPU only
sees the lane region at full detail — high effective resolution where demand is
measured, a fraction of the pixels) plus a SEPARATE crossing ROI for
pedestrians (decimated — see config), ship both to the service tagged with the
classes each wants, map the returned boxes back into full-frame coordinates,
then run the pure-Python lane-assignment + distance-weighting (analyze_tracks)
here.

Calibrations + ROIs are snapshotted at construction; re-calibrating an arm means
restarting the worker (the UI does this on the Run-YOLO toggle).
"""
import socket
import statistics
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.config import (ANALYSIS_INTERVAL, CROP_MAX_SIDE, DEMAND_SMOOTHING,
                        DETECT_CONF, INFER_IMGSZ, PED_CROP_DECIMATE,
                        PED_INFER_IMGSZ, PERCEPTION_WINDOW)
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


def _imgsz_for(crop, cap):
    """YOLO long-side for this crop: its real size rounded up to the /32 the
    model needs, but never above `cap` — a crop smaller than the configured
    inference resolution is detected at native size, not upscaled."""
    side = (max(crop.shape[:2]) + 31) // 32 * 32
    return min(cap, side)


def _group_bbox(calibration, group):
    """Bounding box (x1, y1, x2, y2 ints) of one polygon group, padded.

    The lane and crossing groups are boxed SEPARATELY on purpose: the crossing
    zone sits on the far sidewalk, so a single box around both spans nearly the
    whole frame and the ROI saving evaporates. Returns None if the group has no
    usable polygons — for "lanes" the caller then sends the whole frame, for
    "crossings" it skips the pedestrian crop entirely.
    """
    pts = []
    for poly in calibration.get(group, {}).values():
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
        self._rois = {arm: {"veh": _group_bbox(cal, "lanes"),
                            "ped": _group_bbox(cal, "crossings")}
                      for arm, cal in calibrations.items() if cal}
        self._running = False
        self._sock = None
        self._last_frame = {}   # arm -> last ndarray inferred (frame dedup)
        self._results = {}      # arm -> last analyze_tracks result (stable demand)
        self._smooth = {}       # arm -> {"wd": float, "pd": {phase: float}} EMA
        self._count_hist = {}   # arm -> deque of recent raw counts (median window)
        self._ped_hist = {}     # arm -> deque of recent raw ped counts (median window)

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

    @staticmethod
    def _crop(frame, roi, step=1):
        """Crop a frame to `roi`, keeping every `step`-th pixel (decimation).
        Returns (crop, origin_x, origin_y); roi None = the whole frame."""
        h, w = frame.shape[:2]
        if roi is None:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1 = max(0, min(roi[0], w - 1)); y1 = max(0, min(roi[1], h - 1))
            x2 = max(x1 + 1, min(roi[2], w)); y2 = max(y1 + 1, min(roi[3], h))
        return np.ascontiguousarray(frame[y1:y2:step, x1:x2:step]), x1, y1

    def _detect_batch(self, frames):
        """One request for all arms whose frame is NEW since the last pass.

        Frame dedup matters: in sync mode the sim emits frames slower than this
        loop runs, so without it we'd re-infer the same stale frame repeatedly —
        boxes freeze on the old image while burning GPU that CARLA needs to make
        the next frame. We compare frame identity (the cache hands out the same
        ndarray until CARLA emits a new one) and skip unchanged arms.

        Each fresh arm contributes up to TWO crops, each asking only for the
        classes it can contain:
          * lane ROI    → vehicles, at full INFER_IMGSZ (distant queued cars
            need the detail);
          * crossing ROI → persons, decimated by PED_CROP_DECIMATE and inferred
            at PED_INFER_IMGSZ (waiting pedestrians stand near the camera, so
            they survive the downscale and the wide crossing strip stays cheap).
        Boxes come back in crop-local pixels; ×step then +origin maps them to
        full-frame coordinates.

        Returns {arm: [det, ...]} only for arms re-inferred this pass.
        """
        crops_meta, payload_parts, crop_src = [], [], {}

        def add_crop(key, crop, ox, oy, arm, want, imgsz, step):
            crops_meta.append({"key": key, "h": crop.shape[0], "w": crop.shape[1],
                               "want": want, "imgsz": imgsz})
            payload_parts.append(crop.tobytes())
            crop_src[key] = (arm, ox, oy, step)

        for arm in self._arms:
            frame = frames.get(arm)
            if frame is None:
                continue
            if frame is self._last_frame.get(arm):   # unchanged → skip
                continue
            self._last_frame[arm] = frame
            rois = self._rois.get(arm) or {}
            crop, ox, oy = self._crop(frame, rois.get("veh"))
            # Cap the lane crop (CROP_MAX_SIDE): a near-full-frame ROI is
            # decimated before shipping, and YOLO sees at most the cap — never
            # an upscale of a small crop. _imgsz_for keeps the GPU tensor
            # bounded even when decimation alone can't reach the cap.
            step = max(1, max(crop.shape[:2]) // CROP_MAX_SIDE)
            if step > 1:
                crop, ox, oy = self._crop(frame, rois.get("veh"), step)
            add_crop(arm + "#veh", crop, ox, oy, arm, "vehicles",
                     _imgsz_for(crop, min(INFER_IMGSZ, CROP_MAX_SIDE)), step)
            ped_roi = rois.get("ped")
            if ped_roi is not None:     # no crossings drawn → no pedestrian crop
                step = max(1, PED_CROP_DECIMATE)
                crop, ox, oy = self._crop(frame, ped_roi, step)
                add_crop(arm + "#ped", crop, ox, oy, arm, "persons",
                         _imgsz_for(crop, PED_INFER_IMGSZ), step)
        if not crops_meta:
            return {}

        header = {"type": "detect_batch", "conf": DETECT_CONF, "crops": crops_meta}
        send_message(self._sock, header, b"".join(payload_parts))
        reply, _ = recv_message(self._sock)
        if reply.get("type") != "detections":
            raise RuntimeError(reply.get("message", "bad reply from service"))

        out = {}
        for key, (arm, ox, oy, step) in crop_src.items():
            dets = out.setdefault(arm, [])
            for d in reply["results"].get(key, []):
                bx = d["box"]
                dets.append({
                    "box": (bx[0] * step + ox, bx[1] * step + oy,
                            bx[2] * step + ox, bx[3] * step + oy),
                    "foot": (d["foot"][0] * step + ox, d["foot"][1] * step + oy),
                    "cls": d["cls"], "conf": d["conf"],
                })
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

    def _stabilize_ped(self, arm, res):
        """Median-filter the waiting-pedestrian count the same way as the car
        count, so brief detection dropouts don't make the ponder's patience timer
        flicker (peds blinking 2 → 0 → 2 must not look like 'peds left')."""
        hist = self._ped_hist.get(arm)
        if hist is None:
            hist = self._ped_hist[arm] = deque(maxlen=PERCEPTION_WINDOW)
        hist.append(res["ped_count"])
        res["ped_count"] = int(round(statistics.median(hist)))

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
                    self._stabilize_ped(arm, res)
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
