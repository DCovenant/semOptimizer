"""Headless check of the perception side (analyze_tracks person routing) and
the PedestrianManager wait→cross→despawn state machine, with stub actors."""
import math
import os
import sys
import types

carla = types.ModuleType("carla")

class _State:
    Green, Yellow, Red = "Green", "Yellow", "Red"

class _Loc:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

class _Vec(_Loc):
    pass

carla.TrafficLightState = _State
carla.Location = _Loc
carla.Vector3D = _Vec
carla.Transform = lambda loc, rot=None: (loc, rot)
carla.Rotation = lambda **kw: types.SimpleNamespace(**kw)
carla.WalkerControl = lambda direction=None, speed=0.0: (direction, speed)
sys.modules["carla"] = carla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fails = []

def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        fails.append(name)


# ── analyze_tracks: persons → crossings, never vehicles ─────────────────────
from app.core.analysis import analyze_tracks  # noqa: E402

class ArmProxy:
    calibration = {
        "lanes": {"lane_0": [[0, 0], [100, 0], [100, 100], [0, 100]]},
        "directions": {"lane_0": "incoming"},
        "phases": {"lane_0": "approach_A"},
        "crossings": {"crossing_0": [[200, 0], [300, 0], [300, 100], [200, 100]]},
    }

dets = [
    {"box": (10, 10, 30, 50), "foot": (20, 50), "cls": "car", "conf": 0.9},
    {"box": (240, 10, 260, 60), "foot": (250, 60), "cls": "person", "conf": 0.8},
    {"box": (250, 10, 270, 60), "foot": (260, 60), "cls": "person", "conf": 0.8},
    {"box": (400, 10, 420, 60), "foot": (410, 60), "cls": "person", "conf": 0.8},
]
res = analyze_tracks(ArmProxy(), dets)
check("P1: car counted as vehicle", res["count"] == 1)
check("P2: 2 persons inside the crossing → ped_count 2", res["ped_count"] == 2)
check("P3: person outside any crossing ignored",
      res["ped_per_crossing"] == {"crossing_0": 2})
check("P4: persons never inflate the car count / demand",
      res["count"] == 1 and len(res["tracks"]) == 1)
check("P5: ped overlay data carries crossing tag",
      [p["crossing"] for p in res["peds"]] == ["crossing_0", "crossing_0", None])

# kerb tolerance: a person standing a few px OUTSIDE the polygon border (at the
# kerb line) still counts; far away still doesn't
near = analyze_tracks(ArmProxy(), [
    {"box": (300, 10, 330, 60), "foot": (315, 60), "cls": "person", "conf": .8}])
check("P6: foot within CROSSING_MARGIN_PX of the border counts",
      near["ped_count"] == 1)
far = analyze_tracks(ArmProxy(), [
    {"box": (380, 10, 410, 60), "foot": (395, 60), "cls": "person", "conf": .8}])
check("P7: foot well outside the margin still ignored", far["ped_count"] == 0)


# ── PedestrianManager: spawn → wait → cross on cue → despawn ────────────────
import carla_intersection as ci  # noqa: E402

class FakeActor:
    def __init__(self, loc):
        self.loc = _Loc(loc.x, loc.y, loc.z)
        self.is_alive = True
        self.physics = True
        self.controls = []
    def get_location(self):
        return self.loc
    def set_location(self, loc):
        self.loc = _Loc(loc.x, loc.y, loc.z)
    def set_transform(self, transform):
        loc, _ = transform
        self.set_location(loc)
    def set_simulate_physics(self, on):
        self.physics = bool(on)
    def apply_control(self, ctl):
        direction, speed = ctl
        if direction is not None:
            # integrate one 0.05 s step so tick() actually moves the walker
            self.loc.x += direction.x * speed * 0.05
            self.loc.y += direction.y * speed * 0.05
            self.controls.append(ctl)
    def destroy(self):
        self.is_alive = False

class FakeBP:
    def has_attribute(self, name):
        return False

class FakeBPLib:
    def filter(self, pat):
        return [FakeBP()]

class FakeLight:
    def __init__(self, x, y):
        self._loc = _Loc(x, y, 0.0)
    def get_location(self):
        return self._loc

class FakeWorld:
    def __init__(self):
        self.spawned = []
    def get_blueprint_library(self):
        return FakeBPLib()
    def try_spawn_actor(self, bp, transform):
        loc, _ = transform
        a = FakeActor(loc)
        self.spawned.append(a)
        return a

center = _Loc(0.0, 0.0, 0.0)
lights = {"N": FakeLight(3.5, 18.0), "S": FakeLight(-3.5, -18.0),
          "E": FakeLight(18.0, -3.5), "W": FakeLight(-18.0, 3.5)}

world = FakeWorld()
pm = ci.PedestrianManager(world, lights, center)
check("M1: a crosswalk built for each arm", set(pm.crosswalks) == set(lights))

# force a deterministic spawn
pm._next_spawn_in = 0.0
pm.maybe_spawn(0.05)
check("M2: walker spawned and waiting",
      len(pm.peds) == 1 and pm.peds[0]["state"] == "waiting")
check("M3: waiting_counts ground truth sees it",
      sum(pm.waiting_counts().values()) == 1)

# held at the kerb while not allowed
walker = pm.peds[0]["actor"]
for _ in range(50):
    pm.tick(0.05)
check("M4: held while walk not allowed (no movement, still waiting)",
      pm.peds[0]["state"] == "waiting" and not walker.controls)

# release: PED_CROSS
pm.set_walk_allowed(True)

# a walker arriving AFTER the window opened must NOT be released into a window
# that may be about to close (rising-edge release only)
pm._next_spawn_in = 0.0
pm.maybe_spawn(0.05)
late = [p for p in pm.peds if p["actor"] is not walker][0]

pm.tick(0.05)
check("M5: released on cue → crossing", pm.peds[0]["state"] == "crossing")
check("M6: crossing no longer counts as waiting (late arrival still waits)",
      sum(pm.waiting_counts().values()) == 1)
check("M8: walker spawned mid-window stays at the kerb for the next one",
      late["state"] == "waiting")

# integrate until arrival (crossing span is ~2*half = manageable). Walkers are
# never destroyed mid-run — the actor is PARKED (pooled) for reuse, because
# walker spawn/destroy churn leaks memory inside the CARLA server.
in_peds = lambda a: any(p["actor"] is a for p in pm.peds)
for _ in range(20000):
    if in_peds(walker):
        pm.tick(0.05)
check("M7: walker reaches the far kerb and is parked (pooled, not destroyed)",
      not in_peds(walker) and walker in pm._idle
      and walker.is_alive and not walker.physics
      and walker.loc.z < -10)

# stuck-crossing cull: a walker whose movement is blocked (run over / pinned)
# is parked after PED_CROSS_TIMEOUT instead of lingering forever
pm.set_walk_allowed(False)
pm.set_walk_allowed(True)              # rising edge releases the late walker
stuck = late["actor"]
stuck.apply_control = lambda ctl: None  # never moves
t = 0.0
while late in pm.peds and t < ci.PED_CROSS_TIMEOUT + 1.0:
    pm.tick(0.05); t += 0.05
check("M9: stuck crossing walker parked at PED_CROSS_TIMEOUT",
      late not in pm.peds and stuck in pm._idle
      and abs(t - ci.PED_CROSS_TIMEOUT) < 0.5)

# pool reuse: the next arrival must reuse a parked walker, not spawn a new one
pm.set_walk_allowed(False)
spawned_before = len(world.spawned)
pm._next_spawn_in = 0.0
pm.maybe_spawn(0.05)
reused = pm.peds[-1]
check("M11: next arrival reuses a pooled walker (no new server spawn)",
      len(world.spawned) == spawned_before
      and reused["actor"].physics and reused["actor"].loc.z > -10)

# waiting give-up: never-served walkers (e.g. adaptive off) leave after
# PED_WAIT_GIVEUP instead of saturating the spawn cap forever
quitter = reused
t = 0.0
while quitter in pm.peds and t < ci.PED_WAIT_GIVEUP + 1.0:
    pm.tick(0.05); t += 0.05
check("M10: unserved waiting walker gives up (parked) at PED_WAIT_GIVEUP",
      quitter not in pm.peds and quitter["actor"] in pm._idle
      and abs(t - ci.PED_WAIT_GIVEUP) < 0.5)

# teardown is the only place walkers are actually destroyed — active AND pooled
pm._next_spawn_in = 0.0
pm.maybe_spawn(0.05)
everyone = [p["actor"] for p in pm.peds] + list(pm._idle)
pm.destroy_all()
check("M12: destroy_all destroys active and pooled walkers",
      everyone and all(not a.is_alive for a in everyone)
      and not pm.peds and not pm._idle)

# sync-mode spawn lag: a fresh walker reports is_alive=False until the next
# world snapshot reaches the client. The worker spawns and ticks in the same
# loop iteration, so the manager must NOT drop it on its spawn tick (that bug
# orphaned every walker: standing at the kerb forever, detected by YOLO,
# triggering crossings nobody used).
pm3 = ci.PedestrianManager(FakeWorld(), lights, center)
pm3._next_spawn_in = 0.0
pm3.maybe_spawn(0.05)
lagger = pm3.peds[0]["actor"]
lagger.is_alive = False          # snapshot hasn't caught up yet
pm3.tick(0.05)
kept = len(pm3.peds) == 1
lagger.is_alive = True           # next tick: snapshot arrived
pm3.tick(0.05)
check("M13: walker not dropped while spawn snapshot lags (sync mode)",
      kept and len(pm3.peds) == 1 and pm3.peds[0]["state"] == "waiting")
lagger.is_alive = False          # genuinely gone (despawned server-side)
for _ in range(25):
    pm3.tick(0.05)
check("M14: genuinely dead walker still culled after the grace second",
      not pm3.peds)

print()
if fails:
    print("FAILURES:", len(fails))
    sys.exit(1)
print("perception + pedestrian-manager scenarios pass")
