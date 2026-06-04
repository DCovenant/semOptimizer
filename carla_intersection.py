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

    def __init__(self, ns_lights, ew_lights, ns_green=30.0, ew_green=30.0):
        self.ns = ns_lights
        self.ew = ew_lights
        self.ns_green = float(ns_green)
        self.ew_green = float(ew_green)
        self._phase   = 0
        self._elapsed = 0.0
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
        ns_s, ew_s = [
            (G, R), (Y, R), (R, R),
            (R, G), (R, Y), (R, R),
        ][self._phase]
        for l in self.ns: l.set_state(ns_s)
        for l in self.ew: l.set_state(ew_s)

    # ── public API ────────────────────────────────────────────────────────────

    def tick(self, dt):
        self._elapsed += dt
        if self._elapsed >= self._duration():
            self._elapsed = 0.0
            self._phase   = (self._phase + 1) % 6
            self._apply()

    def phase_name(self):
        return _PHASE_NAMES[self._phase]

    def set_green_times(self, ns_green, ew_green):
        """Optimizer hook — takes effect at the start of the next cycle."""
        self.ns_green = max(self.MIN_GREEN, ns_green)
        self.ew_green = max(self.MIN_GREEN, ew_green)

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


def load_opendrive_map(client, xodr_path):
    """Load a standalone OpenDRIVE .xodr as a runtime CARLA world (no UE cook)."""
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


class DemandManager:
    """Keep a fixed number of vehicles recirculating through the junction.

    Spawns cars on the inbound lanes of each arm (weighted by per-arm demand),
    lets Traffic Manager drive them through, and recycles any car that reaches an
    arm tip back onto an inbound lane. Traffic loops forever and you control the
    arrival mix — no closed-loop road geometry needed.
    """

    def __init__(self, world, client, world_map, center, weights, target):
        self.world = world
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

    def _spawn_one(self):
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
                return v
        return None

    def fill(self):
        tries = 0
        while len(self.vehicles) < self.target and tries < self.target * 4:
            if self._spawn_one() is None:
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
        for v in list(self.vehicles):
            if not v.is_alive:
                self.vehicles.remove(v)
                continue
            wp = self.map.get_waypoint(
                v.get_location(), project_to_road=True,
                lane_type=carla.LaneType.Driving)
            if wp is None:
                v.destroy(); self.vehicles.remove(v); continue
            if (wp.road_id in _ARM_IDS
                    and wp.lane_id == 1          # outbound lane
                    and wp.s < _OUTBOUND_RECYCLE_S):   # near arm outer tip
                v.destroy(); self.vehicles.remove(v)
        self.fill()

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
        for v in self.vehicles:
            if v.is_alive:
                v.destroy()
        self.vehicles = []


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

            phase_ctrl.tick(now - last_tick)
            last_tick = now

            if not cam_grid.tick(phase_ctrl.phase_name(), counts):
                break  # window closed

            if now - last_report >= 1.0:
                dm.tick()
                counts = dm.detect_inbound()
                ns_q = counts["N"] + counts["S"]
                ew_q = counts["E"] + counts["W"]
                print(
                    f"[{phase_ctrl.phase_name():<12}] "
                    f"N:{counts['N']} S:{counts['S']} (Σ{ns_q})  "
                    f"E:{counts['E']} W:{counts['W']} (Σ{ew_q})  "
                    f"total:{len(dm.vehicles)}"
                )
                last_report = now

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cam_grid.destroy()
        phase_ctrl.unfreeze()
        dm.destroy_all()


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
