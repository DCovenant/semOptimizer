"""YOLO vehicle detection — no Qt, easy to unit-test or reuse headless."""
from app.config import CONF, TRACKER, VEHICLE_CLASSES, YOLO_MODEL


def load_model(name=YOLO_MODEL):
    """Load a YOLO model. Imported lazily so the app starts without torch loaded."""
    from ultralytics import YOLO
    return YOLO(name)


def detect_vehicles(model, np_img, conf=CONF):
    """Run detection and return a list of vehicle dicts:

        {box: (x1, y1, x2, y2), foot: (x, y), cls: str, conf: float}

    `foot` is the bottom-centre of the box — the wheels-on-road contact point
    that gets tested against the lane polygons.
    """
    res = model(np_img, conf=conf, verbose=False)[0]
    dets = []
    for b in res.boxes:
        cls = int(b.cls)
        if cls not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append({"box": (x1, y1, x2, y2),
                     "foot": ((x1 + x2) / 2.0, y2),
                     "cls": res.names[cls],
                     "conf": float(b.conf)})
    return dets


def track_vehicles(model, np_img, conf=CONF, tracker=TRACKER):
    """Like detect_vehicles but with persistent multi-frame tracking.

    Uses Ultralytics' built-in tracker (ByteTrack/BoT-SORT) via model.track with
    persist=True, so each vehicle keeps a stable `id` across calls. Tracker state
    lives on the `model` object — use one model instance per camera stream so
    independent approaches don't share an ID space.

    Returns the same dicts as detect_vehicles plus `id` (int, or None if the
    tracker hasn't assigned one yet).
    """
    res = model.track(np_img, conf=conf, persist=True, tracker=tracker,
                      verbose=False)[0]
    ids = res.boxes.id
    ids = ids.int().tolist() if ids is not None else None
    dets = []
    for i, b in enumerate(res.boxes):
        cls = int(b.cls)
        if cls not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append({"box": (x1, y1, x2, y2),
                     "foot": ((x1 + x2) / 2.0, y2),
                     "cls": res.names[cls],
                     "conf": float(b.conf),
                     "id": ids[i] if ids is not None else None})
    return dets
