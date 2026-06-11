"""Headless verification of PhaseController's pedestrian 'ponder' logic.

Stubs the carla module so carla_intersection imports without the simulator,
then drives the controller tick-by-tick through the scenarios that matter.
"""
import os
import sys
import types

# ── stub carla ───────────────────────────────────────────────────────────────
carla = types.ModuleType("carla")

class _State:
    Green, Yellow, Red = "Green", "Yellow", "Red"

carla.TrafficLightState = _State
carla.Location = lambda **kw: types.SimpleNamespace(**kw)
carla.Vector3D = lambda **kw: types.SimpleNamespace(**kw)
carla.Transform = lambda *a, **kw: None
carla.Rotation = lambda **kw: None
carla.WalkerControl = lambda **kw: None
sys.modules["carla"] = carla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carla_intersection import PhaseController  # noqa: E402


class FakeLight:
    def __init__(self):
        self.state = None
    def freeze(self, v):
        pass
    def set_state(self, s):
        self.state = s


def make_ctrl(ns_green=30.0, ew_green=30.0):
    ns = [FakeLight(), FakeLight()]
    ew = [FakeLight(), FakeLight()]
    return PhaseController(ns, ew, ns_green, ew_green), ns, ew


def run_until(ctrl, phase, dt=0.05, limit=600.0):
    """Tick until phase_name() == phase; return elapsed sim-seconds."""
    t = 0.0
    while t < limit:
        ctrl.tick(dt)
        t += dt
        if ctrl.phase_name() == phase:
            return t
    return None


DT = 0.05
fails = []

def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        fails.append(name)


# ── A. light traffic + peds waiting → PED_CROSS at next all-red ─────────────
ctrl, ns, ew = make_ctrl()
ctrl.set_counts(1, 1)          # total 2 <= PED_LOW_TRAFFIC
ctrl.set_ped_demand(3)
t = run_until(ctrl, "PED_CROSS")
check("A1: PED_CROSS granted under light traffic", t is not None)
# Patience gap-out: at 25 s wait the near-empty NS green (1 car <= low-traffic)
# is force-offed, then full yellow (3) + all-red (2) → crossing at ~30 s.
check("A2: patience gap-out ends green early; crossing after yellow+all-red (~30 s)",
      t is not None and 29.5 <= t <= 30.5)
check("A3: every light red during the crossing",
      all(l.state == "Red" for l in ns + ew))
# duration = base 7 + 0.5*3 = 8.5 s, then cycle resumes at EW_GREEN
t2 = 0.0
while ctrl.phase_name() == "PED_CROSS":
    ctrl.tick(DT); t2 += DT
check("A4: crossing lasts BASE + 0.5/ped = 8.5 s", 8.3 <= t2 <= 8.7)
check("A5: cycle resumes with the next phase (EW_GREEN)",
      ctrl.phase_name() == "EW_GREEN")
check("A6: EW green actually applied to lights",
      all(l.state == "Green" for l in ew) and all(l.state == "Red" for l in ns))

# ── B. heavy balanced traffic → peds wait until the patience cap ────────────
ctrl, ns, ew = make_ctrl()
ctrl.set_counts(6, 6)          # heavy, balanced: not low, not lopsided
ctrl.set_ped_demand(2)
t = run_until(ctrl, "PED_CROSS", limit=120.0)
# _ped_wait reaches 25 s during NS_GREEN(30); next insertion point is the
# all-red ending at 35 s.
check("B1: heavy traffic defers the crossing past the first all-red... "
      "until patience (served at ~35 s, not never)",
      t is not None and 34.5 <= t <= 35.5)

# heavy traffic but patience NOT yet reached at the first all-red:
ctrl, ns, ew = make_ctrl(ns_green=15.0, ew_green=15.0)
ctrl.set_counts(6, 6)
ctrl.set_ped_demand(2)
# first all-red ends at 15+3+2 = 20 s; _ped_wait = 20 < 25 → NOT served there
t = run_until(ctrl, "PED_CROSS", limit=120.0)
check("B2: not served at first all-red (wait 20 s < 25 s patience)",
      t is not None and t > 20.5)
# second insertion point: 20 + 15+3+2 = 40 s; wait is 40 >= 25 → served
check("B3: served at the following all-red once patience exceeded (~40 s)",
      t is not None and 39.5 <= t <= 40.5)

# ── C. no pedestrians → PED_CROSS never inserted ─────────────────────────────
ctrl, ns, ew = make_ctrl()
ctrl.set_counts(0, 0)
ctrl.set_ped_demand(0)
t = run_until(ctrl, "PED_CROSS", limit=200.0)
check("C1: no peds → no crossing in 200 s", t is None)

# ── D. lopsided traffic (one axis empty) → crossing is cheap, grant it ──────
ctrl, ns, ew = make_ctrl()
ctrl.set_counts(8, 0)          # heavy NS, empty EW → lopsided
ctrl.set_ped_demand(1)
t = run_until(ctrl, "PED_CROSS", limit=120.0)
check("D1: lopsided demand grants the crossing at the first all-red",
      t is not None and 34.5 <= t <= 35.5)

# ── E. patience + near-empty green → force-off reaches the crossing sooner ──
ctrl, ns, ew = make_ctrl(ns_green=60.0, ew_green=10.0)
ctrl.set_counts(1, 6)          # green axis (NS) nearly empty, EW heavier
ctrl.set_ped_demand(2)
t = run_until(ctrl, "PED_CROSS", limit=120.0)
# force-off can end NS green at MIN_GREEN=10 (EW leads by >deadband), then
# yellow 3 + all-red 2 → crossing possible at ~15 s if patience... wait is only
# 15 < 25, but counts: total 7 > low-traffic, not lopsided → must wait.
# At 15 s wait<25 → not served; cycle continues EW_GREEN(10)+Y(3)+R(2) → 30 s,
# wait=30 >= 25 → served.
check("E1: gap-out path reaches the crossing by the second all-red (~30 s)",
      t is not None and t <= 31.0)

# ── F. ped demand cleared after serving (no immediate re-grant) ─────────────
ctrl, ns, ew = make_ctrl()
ctrl.set_counts(0, 0)
ctrl.set_ped_demand(2)
run_until(ctrl, "PED_CROSS")
while ctrl.phase_name() == "PED_CROSS":
    ctrl.tick(DT)
ctrl.set_ped_demand(0)         # they crossed; perception now sees nobody
t = run_until(ctrl, "PED_CROSS", limit=200.0)
check("F1: after serving + demand cleared, no further crossing", t is None)

# ── G. crossing duration capped at PED_CLEAR_MAX ─────────────────────────────
ctrl, ns, ew = make_ctrl()
ctrl.set_counts(0, 0)
ctrl.set_ped_demand(100)       # 7 + 50 → capped at 20
run_until(ctrl, "PED_CROSS")
t2 = 0.0
while ctrl.phase_name() == "PED_CROSS":
    ctrl.tick(DT); t2 += DT
check("G1: crossing capped at PED_CLEAR_MAX = 20 s", 19.8 <= t2 <= 20.2)

print()
if fails:
    print("FAILURES:", len(fails))
    sys.exit(1)
print("all pedestrian-controller scenarios pass")
