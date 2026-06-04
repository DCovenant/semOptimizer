"""Background QThread that owns the CARLA simulation and emits camera frames.

All CARLA imports are deferred to run() so the module loads cleanly even when
CARLA is unavailable and the Qt app starts unconditionally.
"""
import os
import queue
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from app.config import CAPTURE_H, CAPTURE_W, SENSOR_TICK, SIM_FIXED_DELTA

_PHASE_ARM_STATES = {
    "NS_GREEN":  {"N": "green",  "S": "green",  "E": "red",    "W": "red"},
    "NS_YELLOW": {"N": "yellow", "S": "yellow", "E": "red",    "W": "red"},
    "EW_GREEN":  {"N": "red",    "S": "red",    "E": "green",  "W": "green"},
    "EW_YELLOW": {"N": "red",    "S": "red",    "E": "yellow", "W": "yellow"},
    "ALL_RED":   {"N": "red",    "S": "red",    "E": "red",    "W": "red"},
}
_ALL_RED = {"N": "red", "S": "red", "E": "red", "W": "red"}


class CarlaWorker(QThread):
    frames_ready   = Signal(object)  # dict[str, np.ndarray] — {arm: H×W×3 RGB uint8}
    phase_changed  = Signal(object)  # dict[str, str]         — {arm: "green"|"yellow"|"red"}
    vehicle_counts = Signal(object)  # dict[str, int]         — {arm: count}
    status_message = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, host, port, map_file, vehicles,
                 demand_weights, ns_green, ew_green, parent=None):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._map_file = map_file
        self._vehicles = vehicles
        self._demand_weights = demand_weights
        self._ns_green = ns_green
        self._ew_green = ew_green
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        cameras: dict = {}
        phase_ctrl = None
        dm = None
        world = None
        original_settings = None

        # CARLA's C++ OpenDRIVE parser is locale-sensitive; keep numeric parsing
        # dot-based regardless of what Qt set the process locale to.
        import locale
        locale.setlocale(locale.LC_NUMERIC, "C")

        try:
            # ── deferred CARLA imports ──────────────────────────────────────
            import glob
            _proj = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
            # Force the official .egg ahead of any pip-installed carla wheel:
            # the wheel's OpenDRIVE parser asserts "distance > 0.0" in get_map()
            # while the .egg parses cleanly. (LD_LIBRARY_PATH=.compatlibs is set
            # by main.py's re-exec so the egg's libcarla can load.)
            for egg in glob.glob(os.path.join(
                    _proj, "carla", "PythonAPI", "carla", "dist",
                    "carla-*-py3.7-*.egg")):
                if egg not in sys.path:
                    sys.path.insert(0, egg)
                break
            if _proj not in sys.path:
                sys.path.insert(0, _proj)

            try:
                import carla
                from carla_intersection import (  # noqa: PLC0415
                    DemandManager, PhaseController,
                    _cameras_from_lights, find_junction,
                    group_traffic_lights, load_opendrive_map,
                )
            except ImportError as e:
                self.error_occurred.emit(f"CARLA not available: {e}")
                return

            # ── connect + load map ──────────────────────────────────────────
            self.status_message.emit("Connecting to CARLA …")
            client = carla.Client(self._host, self._port)
            client.set_timeout(30.0)

            self.status_message.emit(f"Loading map from {self._map_file} …")
            load_opendrive_map(client, self._map_file)
            time.sleep(2.0)
            world = client.get_world()
            world_map = world.get_map()
            center = find_junction(world_map, junction_id=None)

            # ── traffic lights + phase controller ───────────────────────────
            self.status_message.emit("Setting up traffic signals …")
            lights_by_arm, ns_lights, ew_lights = group_traffic_lights(world, center)
            phase_ctrl = PhaseController(
                ns_lights, ew_lights,
                ns_green=self._ns_green, ew_green=self._ew_green,
            )

            # ── pole cameras ────────────────────────────────────────────────
            self.status_message.emit("Spawning pole cameras …")
            cam_transforms = _cameras_from_lights(lights_by_arm, center)
            rgb_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            rgb_bp.set_attribute("image_size_x", str(CAPTURE_W))
            rgb_bp.set_attribute("image_size_y", str(CAPTURE_H))
            rgb_bp.set_attribute("fov", "90")
            # cap render rate so 4×1080p doesn't starve the async sim (see config)
            rgb_bp.set_attribute("sensor_tick", str(SENSOR_TICK))

            cam_queues: dict = {}
            for arm, t in cam_transforms.items():
                q: queue.Queue = queue.Queue()
                cam = world.spawn_actor(rgb_bp, t)
                cam.listen(q.put)
                cameras[arm] = cam
                cam_queues[arm] = q

            # ── demand manager + vehicles ───────────────────────────────────
            self.status_message.emit("Spawning vehicles …")
            dm = DemandManager(
                world, client, world_map, center,
                self._demand_weights, self._vehicles,
            )
            dm.fill()

            # ── synchronous mode ────────────────────────────────────────────
            # Async mode couples sim stability to server FPS; once YOLO shares
            # the GPU, FPS collapses and the Traffic Manager's steering goes
            # unstable (cars zigzag off the road). In sync mode the worker owns
            # the clock — each world.tick() advances a fixed delta regardless of
            # render/inference load, so the sim slows in wall-clock but stays
            # physically correct. The TM must be put in sync mode too, else it
            # desyncs from the stepped world.
            original_settings = world.get_settings()
            sync_settings = world.get_settings()
            sync_settings.synchronous_mode = True
            sync_settings.fixed_delta_seconds = SIM_FIXED_DELTA
            world.apply_settings(sync_settings)
            dm.tm.set_synchronous_mode(True)

            self.status_message.emit("CARLA simulation running (synchronous).")

            # ── main loop ───────────────────────────────────────────────────
            # We own the clock now: world.tick() advances exactly SIM_FIXED_DELTA
            # of sim time. Phase timing + recycle cadence are paced in sim-seconds
            # (not wall-clock), so green durations stay correct even if the GPU
            # makes each tick slow in real time.
            sim_time = 0.0
            last_report_sim = 0.0
            last_phase: str | None = None

            while self._running:
                try:
                    world.tick()
                except Exception:
                    if not self._running:
                        break
                    continue
                sim_time += SIM_FIXED_DELTA

                phase_ctrl.tick(SIM_FIXED_DELTA)
                pn = phase_ctrl.phase_name()
                if pn != last_phase:
                    last_phase = pn
                    self.phase_changed.emit(
                        dict(_PHASE_ARM_STATES.get(pn, _ALL_RED)))

                frames: dict = {}
                for arm, q in cam_queues.items():
                    latest = None
                    try:
                        while True:
                            latest = q.get_nowait()
                    except queue.Empty:
                        pass
                    if latest is not None:
                        arr = np.frombuffer(latest.raw_data, dtype=np.uint8)
                        arr = arr.reshape((latest.height, latest.width, 4))
                        # BGRA → RGB; np.ascontiguousarray required before QImage
                        arr = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
                        frames[arm] = arr

                if frames:
                    self.frames_ready.emit(frames)

                if sim_time - last_report_sim >= 1.0:
                    dm.tick()
                    self.vehicle_counts.emit(dict(dm.detect_inbound()))
                    last_report_sim = sim_time

        except Exception as e:
            # full traceback to the terminal; concise message to the UI
            traceback.print_exc()
            self.error_occurred.emit(f"CARLA error: {e}")

        finally:
            # restore async first so we never leave the server frozen in sync
            # mode (a stepped world advances only on tick(); without this a later
            # standalone run or reconnect would hang).
            if world is not None and original_settings is not None:
                try:
                    if dm is not None:
                        dm.tm.set_synchronous_mode(False)
                    world.apply_settings(original_settings)
                except Exception:
                    pass
            for cam in cameras.values():
                try:
                    cam.stop()
                    cam.destroy()
                except Exception:
                    pass
            if phase_ctrl is not None:
                try:
                    phase_ctrl.unfreeze()
                except Exception:
                    pass
            if dm is not None:
                try:
                    dm.destroy_all()
                except Exception:
                    pass
            self.status_message.emit("CARLA disconnected.")
