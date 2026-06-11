"""Out-of-process YOLO inference service (Python 3.12 + ROCm torch).

Why this exists: the CARLA egg pins the app to Python 3.7, but ROCm PyTorch
needs 3.10+. They can't share an interpreter, so detection lives here, behind a
Unix-domain socket, and the 3.7 app talks to it as a thin client. This also
sandboxes ROCm: if the GPU wedges, it crashes this process, not CARLA.

Ownership split:
  * service (here): pixels -> detections   (YOLO + torch, the only torch in play)
  * app (3.7):       detections -> demand   (lane assignment, distance weighting)

The client crops each frame before sending — the incoming-lane ROI (vehicles,
high detail) and a separate, decimated crossing ROI (persons) — so we only ever
run YOLO on the regions that matter, each asking just for the classes it can
contain. Boxes come back in crop-local coordinates; the client maps them into
full-frame space.

Per-frame detection only — no tracking. At the achievable frame rate ByteTrack
couldn't follow fast cars (boxes anchored to a spot, re-triggered by passing
traffic), so we detect each frame independently and the client smooths the
aggregate DEMAND signal instead of tracking individual cars.
"""
import argparse
import os
import socket
import sys
import time

# Must be set before torch is first imported (ultralytics pulls it in). Keeps
# the ROCm caching allocator from hoarding freed blocks: those live partly in
# GTT (shmem carved from system RAM, invisible to RSS/MemoryMax) and on the
# 16 GB box that hoard is what swap-thrashed the desktop during recording.
# setdefault → an explicit env var from the launcher still wins.
os.environ.setdefault(
    "PYTORCH_HIP_ALLOC_CONF",
    "garbage_collection_threshold:0.7,max_split_size_mb:128")

import numpy as np

# protocol.py sits beside this file; make it importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import DEFAULT_SOCKET_PATH, recv_message, send_message  # noqa: E402

# Detection params live with the service — it owns detection. Keep in sync with
# app/config.py (DETECT_CONF / VEHICLE_CLASSES) if those change there.
DEFAULT_MODEL = "models/yolov8n.pt"
DEFAULT_CONF = 0.20
VEHICLE_CLASSES = {1, 2, 3, 5, 7}       # COCO: bicycle, car, motorcycle, bus, truck
PERSON_CLASSES = {0}                    # COCO: person — pedestrians
# Everything we return; the client routes by the cls NAME ("person" vs vehicle).
KEEP_CLASSES = VEHICLE_CLASSES | PERSON_CLASSES
# Per-crop class filter: each crop's header says what it WANTS ("vehicles" for
# the lane ROI, "persons" for the crossing ROI), so a car silhouette on the
# sidewalk or a person between cars never crosses into the wrong count.
WANT_CLASSES = {"vehicles": VEHICLE_CLASSES, "persons": PERSON_CLASSES}


class InferenceServer:
    def __init__(self, model_path, device):
        self._model_path = model_path
        self._device = device
        self._model = None             # one shared detector

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
        return self._model

    def reset(self):
        """No per-run state in the stateless per-frame detector."""

    # ── request handlers ────────────────────────────────────────────────────
    def _detect_batch(self, header, payload):
        """Detect vehicles in every arm's crop (one request, looped inference).

        All arms arrive in ONE request (one socket round-trip), inference run
        sequentially per crop — NOT a torch batch: a single ~960px inference
        already saturates the GPU, and ultralytics' batch path letterboxes every
        image to a square, so batching does more compute and is ~2× slower than
        looping rectangular crops. Coords stay crop-local; the client offsets.
        """
        crops_meta = header["crops"]
        conf = header.get("conf", DEFAULT_CONF)
        imgsz = header.get("imgsz")

        model = self._load()
        names = model.names
        out, off = {}, 0
        for c in crops_meta:
            n = c["h"] * c["w"] * 3
            crop = np.frombuffer(payload[off:off + n], dtype=np.uint8) \
                .reshape((c["h"], c["w"], 3))
            off += n
            keep = WANT_CLASSES.get(c.get("want"), KEEP_CLASSES)
            pkw = {"conf": conf, "device": self._device, "verbose": False,
                   "classes": sorted(keep)}   # filter at NMS, not after
            if c.get("imgsz") or imgsz:       # per-crop override, header fallback
                pkw["imgsz"] = c.get("imgsz") or imgsz
            res = model.predict(crop, **pkw)[0]   # rectangular letterbox = fast
            dets = []
            for b in res.boxes:
                cls = int(b.cls)
                if cls not in keep:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append({
                    "box": [x1, y1, x2, y2],
                    "foot": [(x1 + x2) / 2.0, y2],
                    "cls": names[cls],
                    "conf": float(b.conf),
                })
            out[c.get("key", c.get("arm"))] = dets
        return {"type": "detections", "results": out}

    def handle(self, header, payload):
        mtype = header.get("type")
        if mtype == "detect_batch":
            return self._detect_batch(header, payload)
        if mtype == "reset":
            self.reset()
            return {"type": "ok"}
        if mtype == "ping":
            return {"type": "pong", "device": self._device}
        return {"type": "error", "message": "unknown message type: %r" % mtype}

    # ── serve loop ────────────────────────────────────────────────────────────
    def serve(self, sock_path):
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        srv.listen(1)
        print("inference service listening on %s (device=%s)"
              % (sock_path, self._device), flush=True)
        try:
            while True:
                conn, _ = srv.accept()
                self.reset()
                print("client connected", flush=True)
                try:
                    while True:
                        header, payload = recv_message(conn)
                        reply = self.handle(header, payload)
                        send_message(conn, reply)
                except (ConnectionError, OSError):
                    print("client disconnected", flush=True)
                finally:
                    conn.close()
        finally:
            srv.close()
            if os.path.exists(sock_path):
                os.unlink(sock_path)


def _resolve_device(requested):
    """Map auto/cuda/cpu to an actual torch device, reporting what we got."""
    import torch
    # ROCm builds of torch report the GPU under the 'cuda' API.
    have_gpu = torch.cuda.is_available()
    if requested == "cpu" or (requested == "auto" and not have_gpu):
        return "cpu"
    if not have_gpu:
        print("WARNING: GPU requested but torch.cuda.is_available() is False; "
              "falling back to CPU.", flush=True)
        return "cpu"
    name = torch.cuda.get_device_name(0)
    print("GPU visible to torch: %s (torch %s)" % (name, torch.__version__),
          flush=True)
    return "cuda"


def main():
    ap = argparse.ArgumentParser(description="YOLO inference service")
    ap.add_argument("--socket", default=DEFAULT_SOCKET_PATH)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    device = _resolve_device(args.device)
    InferenceServer(args.model, device).serve(args.socket)


if __name__ == "__main__":
    main()
