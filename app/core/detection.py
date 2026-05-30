"""YOLO vehicle detection — no Qt, easy to unit-test or reuse headless."""
from app.config import CONF, VEHICLE_CLASSES, YOLO_MODEL


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
