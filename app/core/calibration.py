"""Read/write the lane calibration as JSON. Pure data — no Qt types here.

Format (compatible with map_lane_assignment.ipynb, which reads the `lanes` key)::

    {
      "image": "data/test1.png",
      "size": [W, H],
      "lanes":      {"lane_0": [[x, y], ...], ...},
      "directions": {"lane_0": "incoming" | "outgoing", ...},
      "phases":     {"lane_0": "approach_A", ...}
    }
"""
import json


def save_calibration(path, image, size, lanes, directions, phases, signal_state=None):
    """`lanes` maps name -> list of (x, y); `directions`/`phases` map name -> str."""
    data = {
        "image": image,
        "size": list(size),
        "lanes": {name: [[float(x), float(y)] for x, y in pts]
                  for name, pts in lanes.items()},
        "directions": dict(directions),
        "phases": dict(phases),
        "signal_state": signal_state,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_calibration(path):
    """Return the parsed dict. Missing optional keys default to empty dicts."""
    with open(path) as f:
        data = json.load(f)
    data.setdefault("lanes", {})
    data.setdefault("directions", {})
    data.setdefault("phases", {})
    return data
