"""Intersection template: a 4-way junction made of 4 arms (N/E/S/W).

Each arm references its own fixed camera image and holds the lane calibration
drawn on that image (the same {lanes, directions, phases} dict the lane tool and
the notebook use). The whole template serialises to one intersection.json.
"""
import json

ARMS = ("N", "E", "S", "W")
ARM_NAMES = {"N": "North", "E": "East", "S": "South", "W": "West"}


class Arm:
    def __init__(self, key):
        self.key = key
        self.enabled = True
        self.image = None              # path to this arm's camera frame
        self.lanes_in = 1              # expected incoming lanes (template hint)
        self.lanes_out = 1            # expected outgoing lanes
        self.signal_state = "red"      # semaphore for this approach: red|yellow|green
        self.calibration = None        # {image, size, lanes, directions, phases}

    @property
    def calibrated(self):
        return bool(self.calibration and self.calibration.get("lanes"))

    @property
    def lane_count(self):
        if self.calibration:
            return len(self.calibration.get("lanes", {}))
        return 0

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "image": self.image,
            "lanes_in": self.lanes_in,
            "lanes_out": self.lanes_out,
            "signal_state": self.signal_state,
            "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, key, d):
        arm = cls(key)
        arm.enabled = d.get("enabled", True)
        arm.image = d.get("image")
        arm.lanes_in = d.get("lanes_in", 1)
        arm.lanes_out = d.get("lanes_out", 1)
        arm.signal_state = d.get("signal_state", "red")
        arm.calibration = d.get("calibration")
        return arm


class Intersection:
    def __init__(self):
        self.arms = {k: Arm(k) for k in ARMS}

    @property
    def all_calibrated(self):
        return all(a.calibrated for a in self.arms.values() if a.enabled)

    def save(self, path):
        data = {"type": "4-way", "arms": {k: a.to_dict() for k, a in self.arms.items()}}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        inter = cls()
        for k, d in data.get("arms", {}).items():
            if k in inter.arms:
                inter.arms[k] = Arm.from_dict(k, d)
        return inter
