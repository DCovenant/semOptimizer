#!/usr/bin/env python3
"""
CARLA Intersection Data Capture

Connects to a running CARLA server, loads a minimal map, finds a 4-way
intersection, mounts a static pole camera, spawns traffic, and captures
RGB + semantic segmentation frames for training data.

Usage:
    1. Start CARLA:  ./carla/CarlaUE4.sh -vulkan
    2. Run this:     python carla_intersection.py

Press Ctrl+C to stop and clean up.
"""

import math
import os
import sys
import time
import queue
import signal
import random
import argparse
import numpy as np

# Add the CARLA Python API to path
CARLA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carla")
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

import carla


class CarlaSyncMode:
    """Synchronizes world ticks with sensor data collection."""

    def __init__(self, world, *sensors, fps=20):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# ── custom OpenDRIVE map + geofenced respawn demand generator ─────────────────
# Road ids in maps/loop_intersection.xodr (from simple_intersection.xodr).
ARM_ROAD_ID = {"W": 0, "E": 1, "S": 2, "N": 3}
INBOUND_LANES = (-1,)           # lane -1 is the only inbound lane (1+1 road)
SPAWN_S = 14.0                  # metres from an arm's outer end to place spawns
_OUTBOUND_RECYCLE_S = 25.0      # recycle outbound car when s < this (25 m from arm outer tip)


# Arm outer-end positions match the generator (ARM_LENGTH=250, junction at ±5)
_ARM_GEO = {
    "W": (-255.0,   0.0,  0.0),
    "E": ( 255.0,   0.0,  math.pi),
    "S": (   0.0, -255.0, math.pi / 2),
    "N": (   0.0,  255.0, -math.pi / 2),
}
_SIG_S_FROM_JUNCTION = 15.0   # metres — matches generator
_POLE_H               = 5.0    # metres
_LANE_W               = 3.5
_SIDEWALK_W           = 2.0
# Perpendicular distance from a road's centerline out to the middle of its
# sidewalk (one driving lane + half the sidewalk). A pedestrian crosswalk runs
# between the two sidewalks, so its kerb endpoints sit ±this from the centerline.
_CROSSWALK_HALF_SPAN = _LANE_W + _SIDEWALK_W / 2.0


def _pole_camera_transforms(arm_length=250.0, cam_pitch=-15.0):
    """Compute world-space camera transforms for each signal pole.

    Each camera sits at the kerb face of the signal pole and looks toward the
    junction along the arm (in the +s / direction-of-travel direction).
    """
    t_val = -(_LANE_W + _SIDEWALK_W / 2)          # −4.5 m (right / kerb side)
    s_sig = arm_length - _SIG_S_FROM_JUNCTION      # 244 m from outer end

    transforms = {}
    for arm, (ox, oy, hdg) in _ARM_GEO.items():
        # Point along arm at signal s-position
        px = ox + s_sig * math.cos(hdg)
        py = oy + s_sig * math.sin(hdg)
        # OpenDRIVE: positive t = left of travel → left unit = (−sin, cos)
        cx = px + t_val * (-math.sin(hdg))
        cy = py + t_val * ( math.cos(hdg))
        yaw = math.degrees(hdg)   # face in s-direction (toward junction)
        transforms[arm] = carla.Transform(
            carla.Location(x=cx, y=cy, z=_POLE_H),
            carla.Rotation(pitch=cam_pitch, yaw=yaw),
        )
    return transforms


class CameraGrid:
    """2×2 pygame window showing all four signal-pole camera feeds live.

    Layout:
        N  |  E
       ────┼────
        S  |  W
    """

    CELL_W = 640
    CELL_H = 360
    _LAYOUT = {"N": (0, 0), "E": (1, 0), "S": (0, 1), "W": (1, 1)}

    def __init__(self, world, cam_transforms):
        try:
            import pygame as _pg
        except ImportError:
            raise RuntimeError("pygame not found — install it with: pip install pygame")
        self._pg = _pg
        _pg.init()
        _pg.display.set_caption("Intersection — Pole Cameras")
        self._screen = _pg.display.set_mode((self.CELL_W * 2, self.CELL_H * 2))
        self._font   = _pg.font.SysFont("monospace", 18, bold=True)

        # Grey placeholder per cell
        self._surfs = {arm: _pg.Surface((self.CELL_W, self.CELL_H))
                       for arm in cam_transforms}

        self._cameras = {}
        self._queues  = {}

        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.CELL_W))
        bp.set_attribute("image_size_y", str(self.CELL_H))
        bp.set_attribute("fov", "90")

        for arm, t in cam_transforms.items():
            q = queue.Queue()
            cam = world.spawn_actor(bp, t)
            cam.listen(q.put)
            self._cameras[arm] = cam
            self._queues[arm]  = q

        print(f"Camera grid ready — {len(self._cameras)} pole cameras spawned.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _to_surface(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        arr = arr[:, :, ::-1]                          # BGR → RGB
        return self._pg.surfarray.make_surface(arr.swapaxes(0, 1))

    # ── public API ────────────────────────────────────────────────────────────

    def tick(self, phase_name, counts):
        """Drain queues, refresh display. Returns False if window was closed."""
        for ev in self._pg.event.get():
            if ev.type == self._pg.QUIT:
                return False

        for arm, q in self._queues.items():
            latest = None
            try:
                while True:
                    latest = q.get_nowait()
            except Exception:
                pass
            if latest is not None:
                self._surfs[arm] = self._to_surface(latest)

        for arm, (col, row) in self._LAYOUT.items():
            x, y = col * self.CELL_W, row * self.CELL_H
            self._screen.blit(self._surfs[arm], (x, y))
            lbl = self._font.render(
                f" {arm}  q:{counts.get(arm, 0):2d}  {phase_name}", True, (255, 220, 0))
            self._screen.blit(lbl, (x + 6, y + 6))

        cw, ch = self.CELL_W, self.CELL_H
        self._pg.draw.line(self._screen, (60, 60, 60), (cw, 0),  (cw, ch * 2), 2)
        self._pg.draw.line(self._screen, (60, 60, 60), (0,  ch), (cw * 2, ch), 2)
        self._pg.display.flip()
        return True

    def destroy(self):
        for cam in self._cameras.values():
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        self._pg.quit()


_PHASE_NAMES = [
    "NS_GREEN", "NS_YELLOW", "ALL_RED",
    "EW_GREEN", "EW_YELLOW", "ALL_RED",
]


class PhaseController:
    """Two-phase signal controller with all-red clearance.

    Phase sequence:
      0  NS_GREEN   (ns_green s)
      1  NS_YELLOW  (3 s)
      2  ALL_RED    (2 s)
      3  EW_GREEN   (ew_green s)
      4  EW_YELLOW  (3 s)
      5  ALL_RED    (2 s)

    set_green_times() is the optimizer's hook to change durations mid-run.
    """

    YELLOW  = 3.0
    ALL_RED = 2.0
    MIN_GREEN = 10.0
    MAX_GREEN = 60.0   # starvation guard — no axis stays green longer than this
    DEADBAND  = 1      # ignore axis count gaps this small (cars): 5 vs 4 → hold

    # ── pedestrian crossing ("the ponder") ──────────────────────────────────
    # When pedestrians are waiting we insert an EXCLUSIVE all-red crossing phase
    # (every approach red so people can cross safely). It is granted
    # opportunistically — when traffic is light or lopsided so the cost is low —
    # but never deferred past PED_MAX_WAIT, so pedestrians are never starved.
    # This is the "ponder": adaptive timing of the crossing, not a fixed cycle.
    PED_CLEAR_BASE    = 7.0    # base all-red crossing seconds (time to walk across)
    PED_CLEAR_PER_PED = 0.5    # extra seconds per waiting pedestrian
    PED_CLEAR_MAX     = 20.0   # cap on a single crossing window
    PED_MAX_WAIT      = 25.0   # patience: serve regardless once peds have waited this long
    PED_LOW_TRAFFIC   = 2      # total cars at/below which a crossing is "cheap" to insert

    def __init__(self, ns_lights, ew_lights, ns_green=30.0, ew_green=30.0):
        self.ns = ns_lights
        self.ew = ew_lights
        self.ns_green = float(ns_green)
        self.ew_green = float(ew_green)
        # Total green budget to keep constant when reallocating by demand, so the
        # cycle length stays ~fixed and only the NS/EW split moves (see
        # set_demand_split). Captured from the initial fixed times.
        self._budget  = self.ns_green + self.ew_green
        self._ns_count = 0          # latest perceived cars on the NS axis
        self._ew_count = 0          # latest perceived cars on the EW axis
        self._phase   = 0
        self._elapsed = 0.0
        # Pedestrian "ponder" state (see PED_* constants).
        self._ped_request   = 0     # latest perceived waiting-pedestrian count
        self._ped_wait      = 0.0   # sim-seconds peds have waited unserved
        self._ped_active    = False # currently running the all-red crossing
        self._ped_remaining = 0.0   # countdown of the active crossing
        for l in self.ns + self.ew:
            l.freeze(True)
        self._apply()

    # ── internal ──────────────────────────────────────────────────────────────

    def _duration(self):
        return [self.ns_green, self.YELLOW, self.ALL_RED,
                self.ew_green, self.YELLOW, self.ALL_RED][self._phase]

    def _apply(self):
        G = carla.TrafficLightState.Green
        Y = carla.TrafficLightState.Yellow
        R = carla.TrafficLightState.Red
        if self._ped_active:                       # exclusive pedestrian crossing
            for l in self.ns + self.ew:
                l.set_state(R)
            return
        ns_s, ew_s = [
            (G, R), (Y, R), (R, R),
            (R, G), (R, Y), (R, R),
        ][self._phase]
        for l in self.ns: l.set_state(ns_s)
        for l in self.ew: l.set_state(ew_s)

    # ── public API ────────────────────────────────────────────────────────────

    def tick(self, dt):
        # Pedestrians accumulate wait while unserved (drives the patience cap).
        if self._ped_request > 0 and not self._ped_active:
            self._ped_wait += dt

        # An active all-red crossing just counts down, then resumes the cycle by
        # advancing out of the all-red phase it paused on.
        if self._ped_active:
            self._ped_remaining -= dt
            if self._ped_remaining <= 0.0:
                self._ped_active = False
                self._ped_wait   = 0.0
                self._elapsed    = 0.0
                self._phase      = (self._phase + 1) % 6
                self._apply()
            return

        self._elapsed += dt
        # Actuated early termination ("force-off"): if we're green on the lighter
        # axis, have already served MIN_GREEN (driver-reaction safety), and the
        # cross axis is favoured beyond the deadband, end this green now. The
        # normal advance still runs the full yellow + all-red before cars move.
        if self._should_force_off():
            self._elapsed = self._duration()
        if self._elapsed >= self._duration():
            # At the end of an all-red clearance, the junction is already empty —
            # the cheapest moment to insert a pedestrian crossing. Ponder it here.
            if self._phase in (2, 5) and self._should_serve_peds():
                self._ped_active    = True
                self._ped_remaining = min(
                    self.PED_CLEAR_MAX,
                    self.PED_CLEAR_BASE + self.PED_CLEAR_PER_PED * self._ped_request)
                self._apply()       # hold every approach red for the crossing
                return
            self._elapsed = 0.0
            self._phase   = (self._phase + 1) % 6
            self._apply()

    def _should_serve_peds(self):
        """The ponder: is now a good moment to grant the all-red crossing?

        Yes when pedestrians are waiting AND either traffic is light, demand is
        lopsided (one axis empty, so its green serves nobody), or they have waited
        past the patience cap (fairness — never starve them)."""
        if self._ped_request <= 0:
            return False
        total = self._ns_count + self._ew_count
        lopsided = (self._ns_count == 0) != (self._ew_count == 0)
        return (total <= self.PED_LOW_TRAFFIC
                or lopsided
                or self._ped_wait >= self.PED_MAX_WAIT)

    def _should_force_off(self):
        if self._elapsed < self.MIN_GREEN:
            return False
        # Pedestrians waiting past patience: end a light current green early so we
        # reach the all-red crossing sooner (still pays full yellow + all-red).
        if self._ped_request > 0 and self._ped_wait >= self.PED_MAX_WAIT:
            cur = {0: self._ns_count, 3: self._ew_count}.get(self._phase)
            if cur is not None and cur <= self.PED_LOW_TRAFFIC:
                return True
        if self._phase == 0:        # NS green — switch early toward a busier EW?
            return (self._ew_count - self._ns_count) > self.DEADBAND
        if self._phase == 3:        # EW green — switch early toward a busier NS?
            return (self._ns_count - self._ew_count) > self.DEADBAND
        return False

    def phase_name(self):
        if self._ped_active:
            return "PED_CROSS"
        return _PHASE_NAMES[self._phase]

    def set_green_times(self, ns_green, ew_green):
        """Optimizer hook — takes effect at the start of the next cycle."""
        self.ns_green = max(self.MIN_GREEN, ns_green)
        self.ew_green = max(self.MIN_GREEN, ew_green)

    def set_counts(self, ns_count, ew_count):
        """Latest perceived per-axis car counts. Drives both reallocate() and the
        in-tick force-off. Pass (0, 0) to disable actuation (adaptive off)."""
        self._ns_count = ns_count
        self._ew_count = ew_count

    def set_ped_demand(self, n):
        """Latest perceived count of pedestrians waiting to cross (any approach).
        Drives the ponder in tick(). Pass 0 to disable pedestrian crossings (e.g.
        adaptive off)."""
        self._ped_request = max(0, int(n))

    def reallocate(self):
        """Re-split the fixed green budget from the latest per-axis counts.

        Deadband: within DEADBAND cars the axes count as balanced → even split,
        nothing favoured (this is the "cars shared equally, do nothing" case).
        Outside it, split the budget in proportion to the counts, each phase
        clamped to [MIN_GREEN, MAX_GREEN] so the lighter axis is never starved
        and the heavier one can't run away. Call at the start of a green phase."""
        ns, ew = self._ns_count, self._ew_count
        total = ns + ew
        if total <= 0 or abs(ns - ew) <= self.DEADBAND:
            half = self._budget / 2.0
            self.set_green_times(half, half)
            return
        ns_green = self._budget * (ns / total)
        ns_green = min(self.MAX_GREEN, max(self.MIN_GREEN, ns_green))
        ew_green = min(self.MAX_GREEN, max(self.MIN_GREEN, self._budget - ns_green))
        self.set_green_times(ns_green, ew_green)

    def unfreeze(self):
        for l in self.ns + self.ew:
            try:
                l.freeze(False)
            except Exception:
                pass


def group_traffic_lights(world, junction_center):
    """Classify traffic lights into {arm: actor} and NS/EW phase lists.

    Uses position relative to junction_center so the result is correct
    regardless of where CARLA placed the world origin.
    """
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    by_arm = {}
    for light in lights:
        loc = light.get_location()
        dx = loc.x - junction_center.x
        dy = loc.y - junction_center.y
        if abs(dx) >= abs(dy):
            arm = "E" if dx > 0 else "W"
        else:
            arm = "N" if dy > 0 else "S"
        by_arm[arm] = light

    ns = [by_arm[a] for a in ("N", "S") if a in by_arm]
    ew = [by_arm[a] for a in ("E", "W") if a in by_arm]
    print(f"Traffic lights — by arm: {list(by_arm)}  NS:{len(ns)}  EW:{len(ew)}")
    return by_arm, ns, ew


def _cameras_from_lights(lights_by_arm, junction_center, pitch=-25.0):
    """Camera transforms derived from actual traffic light world positions.

    Each camera sits at the light's pole and looks AWAY from the junction
    (toward the approaching queue), avoiding all coordinate-system guesswork.
    """
    cam_setback = 6.0   # metres to slide camera toward junction along the road axis
    transforms = {}
    for arm, light in lights_by_arm.items():
        loc = light.get_location()
        # Unit vector pointing outward from junction (= current look direction)
        dx = loc.x - junction_center.x
        dy = loc.y - junction_center.y
        dist = math.sqrt(dx * dx + dy * dy)
        nx, ny = dx / dist, dy / dist
        # Slide position toward junction (opposite of outward direction)
        cam_loc = carla.Location(
            x=loc.x - cam_setback * nx,
            y=loc.y - cam_setback * ny,
            z=loc.z + _POLE_H,
        )
        yaw = math.degrees(math.atan2(dy, dx)) + 25.0
        transforms[arm] = carla.Transform(
            cam_loc, carla.Rotation(pitch=pitch, yaw=yaw))
    return transforms


def clean_nav_cache():
    """Delete generated-OpenDRIVE walker-nav leftovers (Nav/OpenDriveMap.obj/.bin).

    generate_opendrive_world() writes these next to the server's content as a
    side effect of building the walker navmesh. A LATER server boot re-processes
    a leftover .obj (Recast tile build during startup) and a stale/corrupt one
    segfaults CarlaUE4 before it even listens on port 2000. They are pure caches
    we never use (walkers run on manual WalkerControl), so clear them whenever
    the server isn't mid-generation: before loading a map and at app teardown."""
    nav = os.path.join(CARLA_ROOT, "CarlaUE4", "Content", "Carla", "Maps", "Nav")
    for name in ("OpenDriveMap.obj", "OpenDriveMap.bin"):
        try:
            os.remove(os.path.join(nav, name))
        except OSError:
            pass


def load_opendrive_map(client, xodr_path):
    """Load a standalone OpenDRIVE .xodr as a runtime CARLA world (no UE cook)."""
    clean_nav_cache()
    with open(xodr_path) as f:
        xodr = f.read()
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=50.0, wall_height=0.0,
        additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True)
    print(f"Generating world from {xodr_path} ...")
    return client.generate_opendrive_world(xodr, params)


def parse_demand(spec):
    """'N:3,S:3,E:1,W:1' -> per-arm spawn weights. Missing arms default to 1."""
    weights = {"N": 1.0, "E": 1.0, "S": 1.0, "W": 1.0}
    if spec:
        for part in spec.split(","):
            k, _, v = part.partition(":")
            k = k.strip().upper()
            if k in weights and v:
                weights[k] = float(v)
    return weights


_VEH_PARK_Z      = -60.0  # idle pooled vehicles are hidden this far under the map
                          # (below the walker pool's -50 so the two never mingle)
_VEH_SPAWN_CLEAR = 8.0    # a spawn point is free if no active vehicle is within
                          # this many metres (try_spawn_actor used to do this check)


class DemandManager:
    """Keep a fixed number of vehicles recirculating through the junction.

    Spawns cars on the inbound lanes of each arm (weighted by per-arm demand),
    lets Traffic Manager drive them through, and recycles any car that reaches an
    arm tip back onto an inbound lane. Traffic loops forever and you control the
    arrival mix — no closed-loop road geometry needed.
    """

    def __init__(self, world, client, world_map, center, weights, target):
        self.world = world
        self.client = client
        self.map = world_map
        self.center = center
        self.weights = weights
        self.target = target
        self.tm = client.get_trafficmanager()
        self.tm_port = self.tm.get_port()
        self.tm.global_percentage_speed_difference(-30)  # 30 % above speed limit
        bps = world.get_blueprint_library().filter("vehicle.*")
        self.bps = [b for b in bps
                    if int(b.get_attribute("number_of_wheels")) == 4] or list(bps)
        self.spawns = self._build_spawns()
        self.vehicles = []
        # Idle vehicle pool. Vehicles are NEVER destroyed mid-run: repeated
        # vehicle spawn/destroy cycles leak memory inside the CARLA server the
        # same way walker churn does (UE4 never fully releases the meshes —
        # see the pool note in PedestrianManager.__init__), so the server RSS
        # grows without bound for as long as traffic circulates. Recycled cars
        # are parked under the map (physics off, TM unregistered) and
        # teleported back onto an inbound lane by fill() — after the initial
        # fill the vehicle population is constant and the server does no
        # vehicle churn at all.
        self._idle = []

    def _build_spawns(self):
        spawns = {}
        for arm, rid in ARM_ROAD_ID.items():
            transforms = []
            for lane in INBOUND_LANES:
                wp = self.map.get_waypoint_xodr(rid, lane, SPAWN_S)
                if wp is not None:
                    t = wp.transform
                    t.location.z += 0.3
                    transforms.append(t)
            spawns[arm] = transforms
        n = sum(len(v) for v in spawns.values())
        print(f"Built {n} inbound spawn points across {len(spawns)} arms.")
        return spawns

    def _pick_arm(self):
        arms = [a for a in self.weights if self.spawns.get(a)]
        return random.choices(arms, weights=[self.weights[a] for a in arms])[0]

    def prefill(self):
        """Spawn all target vehicles underground at startup so fill() never
        calls try_spawn_actor mid-run. Use inbound spawn points one at a time
        (immediately teleporting each underground after spawn) so they don't
        block each other."""
        park_x = self.center.x
        park_y = self.center.y
        park_z = self.center.z + _VEH_PARK_Z
        all_points = [(arm, t) for arm, ts in self.spawns.items() for t in ts]
        if not all_points:
            return
        spawned = 0
        tries = 0
        max_tries = self.target * 8
        while spawned < self.target and tries < max_tries:
            _, t = all_points[tries % len(all_points)]
            tries += 1
            bp = random.choice(self.bps)
            if bp.has_attribute("color"):
                bp.set_attribute(
                    "color", random.choice(bp.get_attribute("color").recommended_values))
            v = self.world.try_spawn_actor(bp, t)
            if v is None:
                continue
            try:
                v.set_simulate_physics(False)
                v.set_transform(carla.Transform(carla.Location(
                    x=park_x + spawned * 5.0, y=park_y, z=park_z)))
                self._idle.append(v)
                spawned += 1
            except Exception:
                try:
                    v.destroy()
                except Exception:
                    pass
        print(f"DemandManager prefill: {spawned}/{self.target} vehicles staged below map.")

    def _spawn_one(self):
        """Spawn a fresh vehicle directly onto the road. Only called by fill()
        as a recovery path when the pool is empty (CARLA killed an actor mid-run
        or prefill() staged fewer than target). Normal operation uses _release_one()."""
        arm = self._pick_arm()
        opts = list(self.spawns[arm])
        random.shuffle(opts)
        for t in opts:
            bp = random.choice(self.bps)
            if bp.has_attribute("color"):
                bp.set_attribute(
                    "color", random.choice(bp.get_attribute("color").recommended_values))
            v = self.world.try_spawn_actor(bp, t)
            if v is not None:
                v.set_autopilot(True, self.tm_port)
                self.vehicles.append(v)
                print(f"DemandManager recovery spawn on arm {arm} (pool was empty).")
                return v
        return None

    def fill(self):
        """Top active traffic back up to target. Prefers releasing pooled
        vehicles (teleport only); falls back to spawning a fresh actor only
        when the pool is empty (CARLA killed an actor or prefill staged fewer
        than target). After a healthy prefill(), _spawn_one() should never fire
        during normal operation."""
        self._idle = [v for v in self._idle if v.is_alive]
        tries = 0
        while len(self.vehicles) < self.target and tries < self.target * 4:
            ok = (self._release_one() if self._idle
                  else self._spawn_one() is not None)
            if not ok:
                tries += 1

    def tick(self):
        """Recycle vehicles that have exited the junction and reached the outbound arm tip.

        Uses waypoint lane/s checks instead of distance so that inbound cars
        stopped at a red light are never incorrectly recycled.  A car is
        recycled when it is on an outbound arm lane (lane_id == 1) and within
        _OUTBOUND_RECYCLE_S metres of the arm's outer end (s < threshold).
        Cars that go completely off-road (no waypoint) are also recycled.
        """
        _ARM_IDS = set(ARM_ROAD_ID.values())
        stale = []
        for v in list(self.vehicles):
            if not v.is_alive:
                self.vehicles.remove(v)
                continue
            try:
                loc = v.get_location()
            except RuntimeError:
                # the server already destroyed it (collision / cleanup); drop it
                self.vehicles.remove(v)
                continue
            wp = self.map.get_waypoint(
                loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None:
                stale.append(v); self.vehicles.remove(v); continue
            if (wp.road_id in _ARM_IDS
                    and wp.lane_id == 1          # outbound lane
                    and wp.s < _OUTBOUND_RECYCLE_S):   # near arm outer tip
                stale.append(v); self.vehicles.remove(v)
        for v in stale:
            self._park(v)
        self.fill()

    def _park(self, v):
        """Return a recycled vehicle to the idle pool instead of destroying it
        (see the pool note in __init__). Autopilot goes off FIRST so the TM
        unregisters the actor before it is moved; then physics off so it
        doesn't fall, and hide it under the map until fill() reuses it."""
        try:
            v.set_autopilot(False, self.tm_port)
            v.set_simulate_physics(False)
            v.set_transform(carla.Transform(carla.Location(
                x=self.center.x, y=self.center.y,
                z=self.center.z + _VEH_PARK_Z)))
            self._idle.append(v)
        except Exception:
            self._destroy([v])      # parking failed — last resort

    def _spawn_clear(self, t):
        """True if no active vehicle is near the spawn transform. Replaces the
        occupancy check try_spawn_actor did implicitly before pooling."""
        for v in self.vehicles:
            try:
                if v.get_location().distance(t.location) < _VEH_SPAWN_CLEAR:
                    return False
            except RuntimeError:
                continue
        return True

    def _release_one(self):
        """Teleport an idle pooled vehicle onto a free inbound spawn point of a
        demand-weighted arm. Returns True if a vehicle re-entered traffic."""
        arm = self._pick_arm()
        opts = list(self.spawns[arm])
        random.shuffle(opts)
        for t in opts:
            if not self._spawn_clear(t):
                continue
            while self._idle:
                cand = self._idle.pop()
                try:
                    cand.set_transform(t)
                    cand.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    cand.set_simulate_physics(True)
                    cand.set_autopilot(True, self.tm_port)
                except Exception:   # half-dead actor — drop it from the pool
                    self._destroy([cand])
                    continue
                self.vehicles.append(cand)
                return True
            return False            # pool empty
        return False                # no free spawn on that arm this tick

    def _destroy(self, vehicles):
        """Remove vehicles via a batch DestroyActor command rather than per-actor
        actor.destroy(). In synchronous mode destroying a Traffic-Manager-driven
        vehicle directly can leave a dangling reference in the TM thread, which
        then throws 'trying to operate on a destroyed actor' as an uncaught C++
        exception (std::terminate → the whole process aborts). The batch command
        unregisters the actor from the TM atomically on the server, avoiding it."""
        self.client.apply_batch(
            [carla.command.DestroyActor(v) for v in vehicles])

    def detect_inbound(self):
        """Count vehicles per arm that are on the inbound lane (lane -1) only.
        Vehicles on outbound lanes (lane 1) or junction roads are ignored."""
        counts = {arm: 0 for arm in ARM_ROAD_ID}
        for v in self.vehicles:
            if not v.is_alive:
                continue
            wp = self.map.get_waypoint(
                v.get_location(), project_to_road=True,
                lane_type=carla.LaneType.Driving)
            if wp is None or wp.lane_id != -1:
                continue
            for arm, rid in ARM_ROAD_ID.items():
                if wp.road_id == rid:
                    counts[arm] += 1
                    break
        return counts

    def destroy_all(self):
        alive = [v for v in self.vehicles + self._idle if v.is_alive]
        if alive:
            self._destroy(alive)
        self.vehicles = []
        self._idle = []


# ── pedestrians ────────────────────────────────────────────────────────────────
# Simulation knobs for the walker generator. Kept here (not in app/config) so
# carla_intersection.py stays importable standalone, without the Qt app.
PED_SPEED         = 1.4    # m/s walking speed while crossing (~average adult)
PED_SPAWN_MIN_GAP = 6.0    # min sim-seconds between pedestrian arrivals
PED_SPAWN_MAX_GAP = 18.0   # max sim-seconds between pedestrian arrivals
PED_MAX_ACTIVE    = 8      # cap on simultaneous walkers in the world
PED_WAIT_GIVEUP   = 90.0   # waiting sim-seconds after which an unserved walker
                           # leaves (despawns) — without this, in fixed mode the
                           # cap fills with permanent statues and spawning stops
PED_CROSS_TIMEOUT = 30.0   # crossing sim-seconds after which a walker is culled.
                           # A stuck or run-over walker (a dead walker still
                           # reports is_alive) otherwise lingers forever: it costs
                           # two RPCs per tick, holds a PED_MAX_ACTIVE slot, and a
                           # corpse on the road makes TM traffic brake for it
                           # forever — the sim degrades a little more with each
                           # one. Crossing normally takes ~7 s, so 30 s is stuck.
_PED_PARK_Z        = -50.0  # idle pooled walkers are hidden this far under the map
_PED_SPAWN_Z       = 1.0    # spawn this far above ground; physics settles them
_PED_REACH         = 1.5    # within this many metres of the far kerb → arrived
_CROSSWALK_SETBACK = 3.0    # place the crosswalk this far junction-ward of the stop line
_PED_WAIT_FORWARD  = 1.25   # nudge the waiting walker this far up-arm toward oncoming
                            # cars (away from the junction), so the pole camera that
                            # watches the approach frames the pedestrian better
_PED_CROSS_SPREAD  = 0.7    # ± random offset along the arm axis applied to both
                            # start and target so concurrent crossers take parallel
                            # lines instead of converging on the same point
_PED_SPAWN_CLEAR   = 1.8    # don't spawn within this many metres of another walker


class PedestrianManager:
    """Spawn walkers on the sidewalk, hold them, and walk them across on cue.

    Mirrors DemandManager but for pedestrians. At random intervals a walker is
    spawned at one kerb of a random arm's crosswalk and stands (state "waiting").
    When set_walk_allowed(True) (the controller's all-red PED_CROSS phase) the
    waiting walkers start crossing to the far kerb under manual WalkerControl, and
    are destroyed once they arrive — "appear, wait, cross, disappear".

    The signal logic perceives them via the cameras like any object; this class
    only generates the scenario. waiting_counts() is the ground-truth tally.
    """

    def __init__(self, world, lights_by_arm, center):
        self.world = world
        self.center = center
        self.bps = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
        self.crosswalks = self._build_crosswalks(lights_by_arm, center)
        self.peds = []                 # [{actor, arm, target(carla.Location), state}]
        # Idle walker pool. Walkers are NEVER destroyed mid-run: repeated walker
        # spawn/destroy cycles leak memory inside the CARLA server (UE4 never
        # fully releases the skeletal meshes), so the server gets progressively
        # slower the longer the app runs and can hang on shutdown. Finished
        # walkers are parked under the map (physics off) and teleported back to
        # a kerb for a later arrival — after warm-up the walker population is
        # constant and the server does no walker churn at all.
        self._idle = []                # parked walker actors awaiting reuse
        self._walk_allowed = False
        self._next_spawn_in = random.uniform(PED_SPAWN_MIN_GAP, PED_SPAWN_MAX_GAP)

    def prefill(self):
        """Spawn PED_MAX_ACTIVE walkers underground at startup so maybe_spawn()
        never calls try_spawn_actor mid-run. Stagger park positions slightly so
        CARLA doesn't reject overlapping no-physics actors."""
        if not self.bps or not self.crosswalks:
            return
        arms = list(self.crosswalks)
        park_z = self.center.z + _PED_PARK_Z
        spawned = 0
        tries = 0
        max_tries = PED_MAX_ACTIVE * 4
        while spawned < PED_MAX_ACTIVE and tries < max_tries:
            tries += 1
            arm = arms[spawned % len(arms)]
            a, b, _ = self.crosswalks[arm]
            yaw = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
            transform = carla.Transform(a, carla.Rotation(yaw=yaw))
            bp = random.choice(self.bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is None:
                continue
            try:
                actor.set_simulate_physics(False)
                actor.set_location(carla.Location(
                    x=self.center.x + spawned * 2.0,
                    y=self.center.y,
                    z=park_z))
                self._idle.append(actor)
                spawned += 1
            except Exception:
                try:
                    actor.destroy()
                except Exception:
                    pass
        print(f"PedestrianManager prefill: {spawned}/{PED_MAX_ACTIVE} walkers staged below map.")

    # ── geometry ────────────────────────────────────────────────────────────
    def _build_crosswalks(self, lights_by_arm, center):
        """Two endpoints per arm, one on each sidewalk across the roadway.

        The arms are axis-aligned, so each arm runs along a cardinal axis and the
        junction centre lies on the road centerline. The crosswalk sits a little
        junction-ward of the stop line (the light's along-arm position) and its two
        endpoints are ±_CROSSWALK_HALF_SPAN to either side of the *centerline* —
        i.e. mid-sidewalk on the left and right, NOT the kerb where the pole is.

        Each arm also stores `fwd`, a unit vector pointing up-arm toward oncoming
        traffic (away from the junction); maybe_spawn() nudges the waiting walker
        along it so the approach's pole camera frames the pedestrian better."""
        crosswalks = {}
        half = _CROSSWALK_HALF_SPAN
        z = center.z + _PED_SPAWN_Z
        for arm, light in lights_by_arm.items():
            loc = light.get_location()
            dx, dy = loc.x - center.x, loc.y - center.y
            if abs(dx) >= abs(dy):
                # E/W arm: road runs along X, crosswalk spans Y (N & S sidewalks).
                x = loc.x - math.copysign(_CROSSWALK_SETBACK, dx)
                a = carla.Location(x=x, y=center.y + half, z=z)
                b = carla.Location(x=x, y=center.y - half, z=z)
                fwd = carla.Vector3D(x=math.copysign(1.0, dx), y=0.0, z=0.0)
            else:
                # N/S arm: road runs along Y, crosswalk spans X (E & W sidewalks).
                y = loc.y - math.copysign(_CROSSWALK_SETBACK, dy)
                a = carla.Location(x=center.x + half, y=y, z=z)
                b = carla.Location(x=center.x - half, y=y, z=z)
                fwd = carla.Vector3D(x=0.0, y=math.copysign(1.0, dy), z=0.0)
            crosswalks[arm] = (a, b, fwd)
        return crosswalks

    # ── public API ────────────────────────────────────────────────────────────
    def set_walk_allowed(self, allowed):
        """Release (True) or hold (False) waiting pedestrians. Called by the worker
        from the controller's PED_CROSS phase.

        Only the walkers already waiting when the window OPENS are released
        (rising edge). A walker arriving mid-window stays at the kerb for the
        next one — released with e.g. one second of all-red left it ends up
        mid-road when traffic goes green, gets hit or blocked, and lingers as a
        stuck actor (see PED_CROSS_TIMEOUT)."""
        allowed = bool(allowed)
        if allowed and not self._walk_allowed:
            for p in self.peds:
                if p["state"] == "waiting":
                    p["release"] = True
        self._walk_allowed = allowed

    def _ped_clear(self, loc):
        """True if no active pedestrian is within _PED_SPAWN_CLEAR metres of loc."""
        for p in self.peds:
            if not p["actor"].is_alive:
                continue
            try:
                pl = p["actor"].get_location()
                if math.sqrt((pl.x - loc.x) ** 2 + (pl.y - loc.y) ** 2) < _PED_SPAWN_CLEAR:
                    return False
            except Exception:
                continue
        return True

    def maybe_spawn(self, dt):
        """Count down to the next random arrival and spawn one walker when due."""
        self._next_spawn_in -= dt
        if self._next_spawn_in > 0:
            return
        self._next_spawn_in = random.uniform(PED_SPAWN_MIN_GAP, PED_SPAWN_MAX_GAP)
        if not self.bps or not self.crosswalks:
            return
        if len([p for p in self.peds if p["actor"].is_alive]) >= PED_MAX_ACTIVE:
            return

        # Try arms in random order; pick the first arm+side with a clear waiting spot.
        # Apply a random spread along the arm axis to both start and target so that
        # concurrent crossers take parallel lines instead of converging on the same
        # point (which causes walkers to block each other mid-road).
        arms = list(self.crosswalks)
        random.shuffle(arms)
        chosen = None
        for arm in arms:
            a, b, fwd = self.crosswalks[arm]
            spread = random.uniform(-_PED_CROSS_SPREAD, _PED_CROSS_SPREAD)
            for s_base, t_base in ((a, b), (b, a)):
                s = carla.Location(x=s_base.x + fwd.x * spread,
                                   y=s_base.y + fwd.y * spread,
                                   z=s_base.z)
                t = carla.Location(x=t_base.x + fwd.x * spread,
                                   y=t_base.y + fwd.y * spread,
                                   z=t_base.z)
                wait = carla.Location(x=s.x + fwd.x * _PED_WAIT_FORWARD,
                                      y=s.y + fwd.y * _PED_WAIT_FORWARD,
                                      z=s.z)
                if self._ped_clear(wait):
                    chosen = (arm, wait, t)
                    break
            if chosen:
                break

        if chosen is None:
            return  # every waiting spot occupied — skip this arrival

        arm, start, target = chosen
        yaw = math.degrees(math.atan2(target.y - start.y, target.x - start.x))
        transform = carla.Transform(start, carla.Rotation(yaw=yaw))
        actor = None
        while self._idle and actor is None:     # reuse a parked walker first
            cand = self._idle.pop()
            try:
                cand.set_transform(transform)
                cand.set_simulate_physics(True)
                actor = cand
            except Exception:                   # half-dead actor — replace it
                try:
                    cand.destroy()
                except Exception:
                    pass
        if actor is None:                       # pool empty → skip this arrival
            return
        self.peds.append({"actor": actor, "arm": arm,
                          "target": target, "state": "waiting",
                          "age": 0.0})

    def _park(self, actor):
        """Return a walker to the idle pool instead of destroying it (see the
        pool note in __init__). Halt it, switch physics off so it doesn't fall,
        and hide it under the map until the next arrival reuses it."""
        try:
            actor.apply_control(carla.WalkerControl(speed=0.0))
            actor.set_simulate_physics(False)
            actor.set_location(carla.Location(
                x=self.center.x, y=self.center.y, z=self.center.z + _PED_PARK_Z))
            self._idle.append(actor)
        except Exception:
            try:                               # parking failed — last resort
                actor.destroy()
            except Exception:
                pass

    def tick(self, dt):
        """Advance crossing walkers toward their target and recycle finished ones.

        `dt` is the sim-seconds since the last call; it ages each walker so the
        two lifecycle guards work in sim time. Released walkers get a manual
        control toward their far kerb each tick and are destroyed on arrival.
        Two guards keep the population healthy over long runs:
          * waiting longer than PED_WAIT_GIVEUP → the walker leaves (despawns);
          * crossing longer than PED_CROSS_TIMEOUT → stuck/run-over, culled.
        Without them stuck walkers accumulate to PED_MAX_ACTIVE and stay forever
        (per-tick RPCs, blocked traffic, no new arrivals) — the sim and the app
        degrade progressively the longer the run."""
        for p in list(self.peds):
            actor = p["actor"]
            p["age"] += dt
            if not actor.is_alive:
                # In synchronous mode a freshly spawned actor reports
                # is_alive=False until the next world.tick() delivers the
                # episode snapshot that contains it. The worker calls
                # maybe_spawn() and tick() in the SAME loop iteration, so
                # without this grace every new walker would be dropped from
                # tracking on its spawn tick — leaving an orphan standing at
                # the kerb forever: never released across, never despawned,
                # yet still detected by the cameras (so the controller keeps
                # granting crossings nobody uses, and the server accumulates
                # walkers for the rest of the session).
                if p["age"] >= 1.0:
                    self.peds.remove(p)
                continue
            if p["state"] == "waiting":
                if self._walk_allowed and p.get("release"):
                    p["state"] = "crossing"
                    p["age"] = 0.0          # now counts time-in-crossing
                elif p["age"] >= PED_WAIT_GIVEUP:
                    self._park(actor)
                    self.peds.remove(p)
                    continue
                else:
                    continue
            # crossing
            if p["age"] >= PED_CROSS_TIMEOUT:
                self._park(actor)
                self.peds.remove(p)
                continue
            loc = actor.get_location()
            tgt = p["target"]
            dx, dy = tgt.x - loc.x, tgt.y - loc.y
            if math.sqrt(dx * dx + dy * dy) <= _PED_REACH:
                self._park(actor)
                self.peds.remove(p)
                continue
            d = math.sqrt(dx * dx + dy * dy) or 1.0
            actor.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(x=dx / d, y=dy / d, z=0.0),
                speed=PED_SPEED))

    def waiting_counts(self):
        """Ground-truth per-arm count of pedestrians currently waiting to cross."""
        counts = {arm: 0 for arm in self.crosswalks}
        for p in self.peds:
            if p["state"] == "waiting" and p["actor"].is_alive:
                counts[p["arm"]] += 1
        return counts

    def destroy_all(self):
        """Teardown only — the one place walkers are actually destroyed.

        Sweeps every walker actor in the world, not just the tracked ones:
        this manager is the only walker source in the sim, and any stray that
        slipped out of tracking (e.g. orphans from before the is_alive grace
        fix) would otherwise survive the run and bog the server down."""
        for actor in [p["actor"] for p in self.peds] + self._idle:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.peds = []
        self._idle = []
        try:
            for actor in self.world.get_actors().filter("walker.*"):
                try:
                    actor.destroy()
                except Exception:
                    pass
        except Exception:
            pass


def run_demand_mode(world, client, world_map, center, args):
    """Run the intersection with a phase-controlled semaphore and inbound detection."""
    weights = parse_demand(args.demand)
    dm = DemandManager(world, client, world_map, center, weights, args.vehicles)

    # Group traffic lights — positions relative to junction_center handle any world offset
    lights_by_arm, ns_lights, ew_lights = group_traffic_lights(world, center)
    phase_ctrl = PhaseController(
        ns_lights, ew_lights,
        ns_green=getattr(args, "ns_green", 30.0),
        ew_green=getattr(args, "ew_green", 30.0),
    )

    # Top-down spectator
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=center.x, y=center.y, z=center.z + 120.0),
        carla.Rotation(pitch=-90.0)))

    # Pole cameras — derived from actual light actor positions, looking outward
    cam_grid = CameraGrid(world, _cameras_from_lights(lights_by_arm, center))

    # Pedestrians — random arrivals served by the controller's all-red ponder.
    ped_mgr = PedestrianManager(world, lights_by_arm, center)

    dm.fill()
    print(f"\nRecirculating up to {args.vehicles} vehicles. Demand: {weights}")
    print(f"Cycle: NS {phase_ctrl.ns_green:.0f}s green / "
          f"EW {phase_ctrl.ew_green:.0f}s green / "
          f"yellow {PhaseController.YELLOW:.0f}s / all-red {PhaseController.ALL_RED:.0f}s")
    print("Press Ctrl+C or close the camera window to stop.\n")

    counts = {a: 0 for a in ARM_ROAD_ID}
    try:
        last_tick   = time.time()
        last_report = time.time()
        while True:
            world.wait_for_tick()
            now = time.time()

            dt = now - last_tick
            ped_mgr.maybe_spawn(dt)
            # The standalone script has no perception, so feed the controller the
            # ground-truth car + waiting-ped counts and release walkers during the
            # all-red PED_CROSS phase.
            phase_ctrl.set_counts(counts["N"] + counts["S"], counts["E"] + counts["W"])
            phase_ctrl.set_ped_demand(sum(ped_mgr.waiting_counts().values()))
            phase_ctrl.tick(dt)
            ped_mgr.set_walk_allowed(phase_ctrl.phase_name() == "PED_CROSS")
            ped_mgr.tick(dt)
            last_tick = now

            if not cam_grid.tick(phase_ctrl.phase_name(), counts):
                break  # window closed

            if now - last_report >= 1.0:
                dm.tick()
                counts = dm.detect_inbound()
                ns_q = counts["N"] + counts["S"]
                ew_q = counts["E"] + counts["W"]
                peds = sum(ped_mgr.waiting_counts().values())
                print(
                    f"[{phase_ctrl.phase_name():<12}] "
                    f"N:{counts['N']} S:{counts['S']} (Σ{ns_q})  "
                    f"E:{counts['E']} W:{counts['W']} (Σ{ew_q})  "
                    f"peds:{peds}  total:{len(dm.vehicles)}"
                )
                last_report = now

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cam_grid.destroy()
        phase_ctrl.unfreeze()
        dm.destroy_all()
        ped_mgr.destroy_all()


def get_junction_center(world_map, junction_id):
    """Get the world-space center of a junction using its entry/exit waypoints."""
    topology = world_map.get_topology()

    for wp_start, _ in topology:
        if wp_start.is_junction and wp_start.junction_id == junction_id:
            junction = wp_start.get_junction()
            wp_pairs = junction.get_waypoints(carla.LaneType.Driving)
            all_locs = []
            for entry_wp, exit_wp in wp_pairs:
                all_locs.append(entry_wp.transform.location)
                all_locs.append(exit_wp.transform.location)
            if all_locs:
                avg_x = sum(l.x for l in all_locs) / len(all_locs)
                avg_y = sum(l.y for l in all_locs) / len(all_locs)
                avg_z = sum(l.z for l in all_locs) / len(all_locs)
                return carla.Location(x=avg_x, y=avg_y, z=avg_z)
    return None


def get_all_junctions(world_map):
    """Get all junctions with their connection counts and locations."""
    topology = world_map.get_topology()

    junction_ids = {}
    for wp_start, _ in topology:
        if wp_start.is_junction:
            jid = wp_start.junction_id
            junction_ids.setdefault(jid, 0)
            junction_ids[jid] += 1

    junctions = []
    for jid, count in junction_ids.items():
        loc = get_junction_center(world_map, jid)
        if loc:
            junctions.append({"id": jid, "count": count, "location": loc})

    junctions.sort(key=lambda j: j["count"], reverse=True)
    return junctions


def find_junction(world_map, junction_id=None):
    """Find a junction by ID, or pick the one with most connections."""
    if junction_id is not None:
        loc = get_junction_center(world_map, junction_id)
        if loc:
            return loc
        print(f"ERROR: Junction {junction_id} not found. Use --list-junctions to see available IDs.")
        sys.exit(1)

    junctions = get_all_junctions(world_map)
    if not junctions:
        print("WARNING: No junctions found, using a spawn point instead.")
        sp = world_map.get_spawn_points()[0]
        return sp.location

    return junctions[0]["location"]


def spawn_vehicles(client, world, num_vehicles, junction_center, radius=80.0):
    """Spawn vehicles near the intersection with autopilot enabled."""
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter("vehicle.*")

    # Filter to common car types (no bikes/bicycles for simplicity)
    car_bps = [bp for bp in vehicle_bps
               if int(bp.get_attribute("number_of_wheels")) == 4]
    if not car_bps:
        car_bps = list(vehicle_bps)

    spawn_points = world.get_map().get_spawn_points()

    # Filter spawn points near the junction
    nearby_spawns = []
    for sp in spawn_points:
        dist = sp.location.distance(junction_center)
        if dist < radius:
            nearby_spawns.append(sp)

    if not nearby_spawns:
        print(f"WARNING: No spawn points within {radius}m of junction, using all spawn points.")
        nearby_spawns = spawn_points

    # Spawn vehicles
    vehicles = []
    import random
    random.shuffle(nearby_spawns)

    batch = []
    for i in range(min(num_vehicles, len(nearby_spawns))):
        bp = random.choice(car_bps)
        if bp.has_attribute("color"):
            color = random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        batch.append(carla.command.SpawnActor(bp, nearby_spawns[i])
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True)))

    results = client.apply_batch_sync(batch, True)
    for result in results:
        if not result.error:
            vehicles.append(result.actor_id)

    print(f"Spawned {len(vehicles)}/{num_vehicles} vehicles near the intersection.")
    return vehicles


def setup_cameras(world, junction_center, image_width=1920, image_height=1080):
    """Mount static RGB and semantic segmentation cameras above the intersection."""
    blueprint_library = world.get_blueprint_library()

    # Camera position: 10m above the intersection, looking down
    camera_location = carla.Location(
        x=junction_center.x,
        y=junction_center.y,
        z=junction_center.z + 10.0
    )
    camera_rotation = carla.Rotation(pitch=-70.0, yaw=0.0, roll=0.0)
    camera_transform = carla.Transform(camera_location, camera_rotation)

    # RGB camera
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(image_width))
    rgb_bp.set_attribute("image_size_y", str(image_height))
    rgb_bp.set_attribute("fov", "110")
    camera_rgb = world.spawn_actor(rgb_bp, camera_transform)

    # Semantic segmentation camera (same position)
    semseg_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    semseg_bp.set_attribute("image_size_x", str(image_width))
    semseg_bp.set_attribute("image_size_y", str(image_height))
    semseg_bp.set_attribute("fov", "110")
    camera_semseg = world.spawn_actor(semseg_bp, camera_transform)

    print(f"Cameras mounted at ({camera_location.x:.1f}, {camera_location.y:.1f}, {camera_location.z:.1f})")
    print(f"  Resolution: {image_width}x{image_height}, FOV: 110, Pitch: {camera_rotation.pitch}")

    return camera_rgb, camera_semseg


def save_image(image, path):
    """Convert a CARLA image to numpy array and save as PNG."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    array = array[:, :, :3]  # Drop alpha → BGR
    array = array[:, :, ::-1]  # BGR → RGB

    from PIL import Image
    img = Image.fromarray(array)
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description="CARLA intersection data capture")
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--map", default="Town01", help="CARLA map to load (Town01, Town02, Town05, etc.)")
    parser.add_argument("--map-file", default=None,
                        help="Load a standalone OpenDRIVE .xodr (e.g. maps/loop_intersection.xodr)")
    parser.add_argument("--respawn", action="store_true",
                        help="Run the geofenced respawn demand generator (auto-on with --map-file)")
    parser.add_argument("--demand", default=None,
                        help="Per-arm demand weights, e.g. 'N:3,S:3,E:1,W:1'")
    parser.add_argument("--junction", type=int, default=None, help="Junction ID to use (run --list-junctions to see options)")
    parser.add_argument("--list-junctions", action="store_true", help="List all junctions in the map and exit")
    parser.add_argument("--vehicles", type=int, default=15, help="Number of vehicles to spawn")
    parser.add_argument("--ns-green", type=float, default=30.0, help="Green time for N+S phase (seconds)")
    parser.add_argument("--ew-green", type=float, default=30.0, help="Green time for E+W phase (seconds)")
    parser.add_argument("--frames", type=int, default=0, help="Number of frames to capture (0 = no capture, just run)")
    parser.add_argument("--fps", type=int, default=20, help="Simulation FPS")
    parser.add_argument("--skip", type=int, default=5, help="Capture every N-th frame (skip the rest)")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    # Create output directories
    rgb_dir = os.path.join(args.output, "rgb")
    semseg_dir = os.path.join(args.output, "semseg")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(semseg_dir, exist_ok=True)

    actors_to_destroy = []
    vehicle_ids = []
    original_settings = None

    try:
        # Connect
        print(f"Connecting to CARLA at {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(30.0)

        # Load map — either a standalone OpenDRIVE file or a built-in town
        if args.map_file:
            world = load_opendrive_map(client, args.map_file)
        else:
            map_name = args.map
            print(f"Loading {map_name}...")
            try:
                world = client.load_world(map_name)
            except RuntimeError:
                opt_name = f"{map_name}_Opt"
                print(f"{map_name} not found, trying {opt_name}...")
                world = client.load_world(opt_name)
        time.sleep(2)

        world = client.get_world()
        original_settings = world.get_settings()
        world_map = world.get_map()
        print(f"Map loaded: {world_map.name}")

        if args.list_junctions:
            junctions = get_all_junctions(world_map)
            print(f"\nFound {len(junctions)} junctions in {world_map.name}:\n")
            print(f"  {'ID':>4}  {'Connections':>11}  {'X':>8}  {'Y':>8}  {'Z':>6}")
            print(f"  {'─'*4}  {'─'*11}  {'─'*8}  {'─'*8}  {'─'*6}")
            for j in junctions:
                loc = j["location"]
                label = "← 4-way" if j["count"] >= 8 else ""
                print(f"  {j['id']:>4}  {j['count']:>11}  {loc.x:>8.1f}  {loc.y:>8.1f}  {loc.z:>6.1f}  {label}")
            print(f"\nUse --junction <ID> to select one.")
            return

        # Find intersection
        junction_center = find_junction(world_map, args.junction)
        print(f"Using junction at ({junction_center.x:.1f}, {junction_center.y:.1f}, {junction_center.z:.1f})")

        # Respawn / demand mode (the closed-loop test environment)
        if args.respawn or args.map_file:
            run_demand_mode(world, client, world_map, junction_center, args)
            return

        # Set up cameras
        camera_rgb, camera_semseg = setup_cameras(world, junction_center)
        actors_to_destroy.extend([camera_rgb, camera_semseg])

        # Spawn traffic
        print(f"Spawning {args.vehicles} vehicles...")
        vehicle_ids = spawn_vehicles(client, world, args.vehicles, junction_center)

        if args.frames > 0:
            # Capture mode
            print(f"\nCapturing {args.frames} frames (every {args.skip}th tick at {args.fps} FPS)...")
            print("Press Ctrl+C to stop.\n")

            captured = 0
            tick_count = 0

            with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=args.fps) as sync_mode:
                while captured < args.frames:
                    snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=5.0)
                    tick_count += 1

                    if tick_count % args.skip != 0:
                        continue

                    # Save RGB
                    rgb_path = os.path.join(rgb_dir, f"{captured:06d}.png")
                    save_image(image_rgb, rgb_path)

                    # Save semantic segmentation (with CityScapes palette for visualization)
                    image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                    semseg_path = os.path.join(semseg_dir, f"{captured:06d}.png")
                    save_image(image_semseg, semseg_path)

                    captured += 1
                    if captured % 50 == 0 or captured == 1:
                        print(f"  Captured {captured}/{args.frames} frames")

            print(f"\nDone! {captured} frames saved to {args.output}/")
        else:
            # View-only mode — no capture, just let traffic run
            print("\nRunning in view-only mode (no capture). Press Ctrl+C to stop.\n")
            while True:
                world.wait_for_tick()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        # Clean up
        print("Cleaning up actors...")
        if vehicle_ids:
            client.apply_batch([carla.command.DestroyActor(vid) for vid in vehicle_ids])
            print(f"  Destroyed {len(vehicle_ids)} vehicles")
        for actor in actors_to_destroy:
            if actor is not None and actor.is_alive:
                actor.destroy()
        if original_settings is not None:
            world.apply_settings(original_settings)
        print("Done.")


if __name__ == "__main__":
    main()
