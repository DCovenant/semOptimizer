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

from app.config import (CAPTURE_H, CAPTURE_W, FRAME_EMIT_INTERVAL, SENSOR_TICK,
                        SIM_FIXED_DELTA)

_PHASE_ARM_STATES = {
    "NS_GREEN":  {"N": "green",  "S": "green",  "E": "red",    "W": "red"},
    "NS_YELLOW": {"N": "yellow", "S": "yellow", "E": "red",    "W": "red"},
    "EW_GREEN":  {"N": "red",    "S": "red",    "E": "green",  "W": "green"},
    "EW_YELLOW": {"N": "red",    "S": "red",    "E": "yellow", "W": "yellow"},
    "ALL_RED":   {"N": "red",    "S": "red",    "E": "red",    "W": "red"},
    "PED_CROSS": {"N": "red",    "S": "red",    "E": "red",    "W": "red"},
}
_ALL_RED = {"N": "red", "S": "red", "E": "red", "W": "red"}


class CarlaWorker(QThread):
    frames_ready   = Signal(object)  # dict[str, np.ndarray] — {arm: H×W×3 RGB uint8}
    phase_changed  = Signal(object)  # dict[str, str]         — {arm: "green"|"yellow"|"red"}
    vehicle_counts = Signal(object)  # dict[str, int]         — {arm: count}
    ped_counts     = Signal(object)  # dict[str, int]         — {arm: waiting peds} (ground truth)
    ped_status     = Signal(object)  # {perceived: int, serving: bool}
    timing_changed = Signal(object)  # {mode, ns_green, ew_green, ns_demand, ew_demand}
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
        # Adaptive control state, written from the GUI thread and read in run().
        # Plain attribute assignment is atomic enough under the GIL — the loop
        # only ever reads the latest snapshot at a cycle boundary.
        self._adaptive = False
        self._counts: dict = {}       # {arm: perceived car count}
        self._ped_counts: dict = {}   # {arm: perceived waiting-pedestrian count}
        self._frames_pending = False  # a frames_ready batch is queued, unprocessed

    def stop(self):
        self._running = False

    def set_adaptive(self, enabled: bool):
        """Toggle perceived-demand-driven green times. When off, the controller
        reverts to the configured fixed ns/ew green at the next cycle."""
        self._adaptive = bool(enabled)

    def update_counts(self, per_arm_counts: dict):
        """Push the latest per-arm perceived car counts (from AnalysisWorker)."""
        self._counts = dict(per_arm_counts)

    def update_ped_counts(self, per_arm_ped_counts: dict):
        """Push the latest per-arm perceived waiting-pedestrian counts."""
        self._ped_counts = dict(per_arm_ped_counts)

    def frames_displayed(self):
        """GUI acknowledgment that the last frames_ready batch was processed.

        Re-arms the frame emitter (see the backpressure note in run()). Called
        from the main thread by the last frames_ready slot; plain bool store,
        atomic enough under the GIL."""
        self._frames_pending = False

    def run(self):
        self._running = True
        self._frames_pending = False
        cameras: dict = {}
        phase_ctrl = None
        dm = None
        ped_mgr = None
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
                    DemandManager, PedestrianManager, PhaseController,
                    _cameras_from_lights, clean_nav_cache, find_junction,
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
            # Pre-spawn all actors underground first so no new try_spawn_actor
            # calls happen mid-run. This lets us confirm whether CARLA's memory
            # growth comes from the spawn lifecycle or something else entirely.
            self.status_message.emit("Pre-spawning all actors below map …")
            dm = DemandManager(
                world, client, world_map, center,
                self._demand_weights, self._vehicles,
            )
            dm.prefill()

            # ── pedestrians ─────────────────────────────────────────────────
            ped_mgr = PedestrianManager(world, lights_by_arm, center)
            ped_mgr.prefill()

            self.status_message.emit("Releasing actors onto road …")
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
            # newest undelivered RGB frame per arm; emitted to the GUI at a
            # throttled wall-clock cadence (see the frame block below).
            pending_frames: dict = {}
            last_emit = 0.0
            # wall-clock deadline for the next tick, used to pace the loop to at
            # most real time (see the sleep at the bottom of the loop).
            next_tick = time.monotonic()

            tick_failures = 0
            while self._running:
                try:
                    world.tick()
                    tick_failures = 0
                except Exception:
                    if not self._running:
                        break
                    # A dead/wedged server makes every tick() raise (each after
                    # carla's internal timeout). Without a give-up the loop spins
                    # on failures forever — the app looks hung, the user force-
                    # kills it, and teardown never restores async mode, leaving
                    # the server frozen. Bail out cleanly instead.
                    tick_failures += 1
                    if tick_failures >= 5:
                        self.error_occurred.emit(
                            "CARLA stopped responding — disconnecting.")
                        break
                    continue
                sim_time += SIM_FIXED_DELTA

                # Feed the controller the latest per-axis counts every tick so its
                # in-tick force-off can react mid-phase. Zeros when adaptive is off
                # disable actuation, leaving fixed timing untouched.
                if self._adaptive:
                    c = self._counts
                    ns_count = c.get("N", 0) + c.get("S", 0)
                    ew_count = c.get("E", 0) + c.get("W", 0)
                    phase_ctrl.set_counts(ns_count, ew_count)
                    phase_ctrl.set_ped_demand(sum(self._ped_counts.values()))
                else:
                    phase_ctrl.set_counts(0, 0)
                    phase_ctrl.set_ped_demand(0)   # peds served only in adaptive mode

                # spawn random arrivals; release/hold + advance walkers each tick
                ped_mgr.maybe_spawn(SIM_FIXED_DELTA)
                phase_ctrl.tick(SIM_FIXED_DELTA)
                pn = phase_ctrl.phase_name()
                ped_mgr.set_walk_allowed(pn == "PED_CROSS")
                ped_mgr.tick(SIM_FIXED_DELTA)
                if pn != last_phase:
                    last_phase = pn
                    # At each green-phase start, size it: adaptive → re-split from
                    # the perceived counts (deadband → even); fixed → the
                    # configured times (so toggling adaptive off recovers them).
                    if pn in ("NS_GREEN", "EW_GREEN"):
                        if self._adaptive:
                            phase_ctrl.reallocate()
                            ns_count = self._counts.get("N", 0) + self._counts.get("S", 0)
                            ew_count = self._counts.get("E", 0) + self._counts.get("W", 0)
                            diff = ns_count - ew_count
                            if abs(diff) <= phase_ctrl.DEADBAND:
                                decision = "balanced — held"
                            else:
                                decision = "favouring N–S" if diff > 0 else "favouring E–W"
                        else:
                            phase_ctrl.set_green_times(self._ns_green, self._ew_green)
                            ns_count = ew_count = None
                            decision = None
                        # publish exactly what the controller will run this phase
                        self.timing_changed.emit({
                            "mode": "adaptive" if self._adaptive else "fixed",
                            "ns_green": phase_ctrl.ns_green,
                            "ew_green": phase_ctrl.ew_green,
                            "ns_count": ns_count,
                            "ew_count": ew_count,
                            "decision": decision,
                            "ped_count": sum(self._ped_counts.values()),
                        })
                    self.phase_changed.emit(
                        dict(_PHASE_ARM_STATES.get(pn, _ALL_RED)))
                    # reflect crossing start/stop immediately on phase changes
                    self.ped_status.emit({
                        "perceived": sum(self._ped_counts.values()),
                        "serving": pn == "PED_CROSS",
                    })

                # Drain the camera queues every tick and convert the newest image
                # per arm to numpy *now* — a carla.Image must be read on the tick
                # it was delivered; its buffer can be recycled on the next
                # world.tick(), so holding it across ticks risks reading freed
                # memory (native abort).
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
                        pending_frames[arm] = np.ascontiguousarray(
                            arr[:, :, :3][:, :, ::-1])

                # Throttle the emit→paint path: a wall-clock cadence cap PLUS
                # backpressure. The cadence alone is not enough — each queued
                # frames_ready event pins ~25 MB of ndarrays, and if the GUI
                # thread is slower per batch than the emit interval (heavy
                # overlay repaints, YOLO + CARLA hogging the machine) the
                # cross-thread event queue grows without bound: memory climbs,
                # paints lag further and further behind, and the app eventually
                # OOM-crashes. So at most ONE batch is ever in flight: emit only
                # after the GUI acked the previous one (frames_displayed()).
                # The GUI then simply displays at whatever rate it can sustain,
                # and a slow machine drops frames instead of accumulating them.
                now = time.monotonic()
                if (pending_frames and not self._frames_pending
                        and now - last_emit >= FRAME_EMIT_INTERVAL):
                    self._frames_pending = True
                    self.frames_ready.emit(dict(pending_frames))
                    pending_frames.clear()
                    last_emit = now

                if sim_time - last_report_sim >= 1.0:
                    dm.tick()
                    self.vehicle_counts.emit(dict(dm.detect_inbound()))
                    self.ped_counts.emit(dict(ped_mgr.waiting_counts()))
                    last_report_sim = sim_time

                # Pace the loop to at most real time. Each tick advances
                # SIM_FIXED_DELTA of sim time, so without this a fast machine
                # spins world.tick() flat out — 100 % CPU on this thread, which
                # starves the GUI thread (the app feels laggy) and runs the sim
                # faster than real time. When rendering/inference makes a step
                # slower than real time we just fall behind: correct, only slower
                # in wall-clock (no catch-up burst, which would zigzag the TM).
                next_tick += SIM_FIXED_DELTA
                slack = next_tick - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -1.0:
                    next_tick = time.monotonic()   # fell >1 s behind → resync

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
            if ped_mgr is not None:
                try:
                    ped_mgr.destroy_all()
                except Exception:
                    pass
            # Clear the generated-map walker-nav leftovers so the NEXT server
            # boot doesn't re-process (and segfault on) a stale OpenDriveMap.obj.
            try:
                clean_nav_cache()
            except Exception:
                pass
            self.status_message.emit("CARLA disconnected.")
