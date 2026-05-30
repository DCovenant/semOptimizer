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

import os
import sys
import time
import queue
import signal
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
    parser.add_argument("--junction", type=int, default=None, help="Junction ID to use (run --list-junctions to see options)")
    parser.add_argument("--list-junctions", action="store_true", help="List all junctions in the map and exit")
    parser.add_argument("--vehicles", type=int, default=15, help="Number of vehicles to spawn")
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

        # Load map
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
