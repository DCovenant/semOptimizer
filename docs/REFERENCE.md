# Implementation Reference

> **What this is.** A reference "dictionary" for the *current* code: architecture,
> the runtime data flow, every module's role, the signal-timing logic in depth, the
> design decisions and their *why*, data formats, config knobs, and a glossary.
> When something in the code is unclear, start here.
>
> Scope note: the high-level vision, the model-compression pipeline, and the CARLA
> dataset-generation ideas live in [`README.md`](README.md); future/deferred ideas
> live in [`../brainstorming.md`](../brainstorming.md). This file documents what is
> actually implemented and running today.

---

## 1. Big picture

The app turns four pole-mounted intersection cameras into a live, demand-driven
traffic-signal controller, validated in CARLA simulation:

```
 CARLA sim ── 4 pole cameras ──► YOLO detection ──► lane assignment ──►
   per-arm car counts ──► adaptive signal timing ──► CARLA traffic lights
                                   ▲
                                   └── one button toggles adaptive vs fixed
```

Two things run the show:

- **`carla_intersection.py`** — the simulation library: loads a map, finds the
  junction, groups traffic lights, mounts cameras, spawns/recycles traffic, and
  holds the **`PhaseController`** (the actual signal logic). It is both a
  standalone script *and* an import target for the app.
- **`app/`** — the PySide6 desktop app: a planner UI, per-camera lane calibration,
  live camera tiles, the perception readout, and the threads that drive CARLA and
  run inference.

Detection itself runs in a **third, separate process** (`inference/`) for Python
version reasons (see §3).

---

## 2. Repository map

```
semOptimizer/
├── carla_intersection.py     # CARLA sim library + PhaseController (signal logic)
├── app/
│   ├── main.py               # entry point; re-execs with a CARLA-safe env
│   ├── config.py             # all tunables (detection, capture, timing, weights)
│   ├── core/
│   │   ├── intersection.py           # Intersection/Arm data model + JSON template
│   │   ├── calibration.py            # lane-calibration JSON read/write
│   │   ├── detection.py              # YOLO detect/track (in-process, 3.7 path)
│   │   ├── analysis.py               # detection → lane assignment → demand
│   │   ├── lane_distance.py          # normalised distance from the stop line
│   │   ├── carla_worker.py           # QThread: owns CARLA + PhaseController
│   │   ├── analysis_worker.py        # QThread: live YOLO client (per-frame)
│   │   └── inference_service_manager.py  # launches/supervises the infer process
│   ├── graphics/
│   │   ├── plan_view.py              # top-down intersection schematic
│   │   ├── semaphore.py             # one 3-light signal scene item
│   │   ├── camera_panel.py          # 2×2 live camera tiles + overlays
│   │   ├── lane.py                  # editable lane polygon (calibration)
│   │   └── canvas.py                # calibration drawing canvas
│   └── ui/
│       ├── intersection_window.py   # main window; wires everything together
│       └── lane_window.py           # per-camera lane calibration tool
├── inference/
│   ├── service.py            # out-of-process YOLO server (3.12 + ROCm torch)
│   └── protocol.py           # Unix-socket wire protocol (shared by both sides)
├── docs/REFERENCE.md         # this file
├── captures/                 # per-arm reference frames (N/E/S/W.png)
├── maps/                     # OpenDRIVE (.xodr) maps
└── models/                   # YOLO weights (yolov8n.pt, yolo11m.pt, …)
```

---

## 3. The three-interpreter split (read this first)

This is the single most surprising thing about the project. There are **three
Python environments**, on purpose:

| Process            | Venv          | Python | Why                                                        |
|--------------------|---------------|--------|------------------------------------------------------------|
| Desktop app + CARLA| `.venv`       | 3.7    | The official CARLA `.egg` pins us to **Python 3.7**.       |
| Inference service  | `.venv-infer` | 3.12   | ROCm PyTorch needs **3.10+**; can't share the 3.7 venv.    |

The app is launched with `.venv/bin/python app/main.py`.

Consequences that explain a lot of the code:

- **Detection is out-of-process** behind a Unix socket. The app (3.7) is a thin
  client; all torch/YOLO lives in `inference/service.py` (3.12). See
  [`inference/service.py`](../inference/service.py) header for the ownership split:
  *service = pixels→detections, app = detections→demand.*
- **`inference/protocol.py` must stay pure-Python and 3.7-safe** — it's imported by
  both sides across the version gap.
- **GPU is sandboxed**: if ROCm wedges, it crashes the service process, not CARLA.

See also the memory notes: *CARLA egg vs pip build*, *CARLA needs C numeric
locale*, *YOLOv8 not v11 on py3.7*, *ROCm GPU inference works*.

### CARLA launch gotchas (why `app/main.py` re-execs)

`app/main.py:_ensure_carla_env()` relaunches the interpreter once with a fixed
environment because three things must be true *before* CARLA loads:

1. **`LC_NUMERIC=C`** — `QApplication` runs `setlocale(LC_ALL, "")`; on a
   decimal-comma locale (e.g. pt_PT) CARLA's C++ OpenDRIVE parser then misreads
   floats (`"3.5"`→`3.0`) and `get_map()` asserts `distance > 0.0`.
2. **The official `.egg` ahead of any pip `carla` wheel** — the wheel's parser
   asserts on our map; the egg parses cleanly.
3. **`.compatlibs` on `LD_LIBRARY_PATH`** — so the egg's `libcarla` can dlopen its
   old deps (libtiff5, libjpeg62, …).

`carla_worker.py` re-asserts `LC_NUMERIC=C` and re-inserts the egg path inside its
thread for the same reasons.

---

## 4. Runtime data flow

Everything below the UI runs on background `QThread`s; results cross back to the
main thread via **queued** Qt signals (safe for cross-thread GUI updates).

```
                          ┌───────────────────────── main thread (Qt) ─────────────────────────┐
                          │  IntersectionWindow                                                  │
                          │   • owns CarlaWorker, AnalysisWorker, InferenceServiceManager        │
                          │   • _latest_frames cache   • Perception + Signal-logic docks         │
                          └───▲───────────────▲───────────────────────────▲───────────┬─────────┘
        frames_ready /        │               │ analysis_ready             │ timing_   │ update_counts
        phase_changed /       │               │                            │ changed   │ set_adaptive
        timing_changed        │               │                            │           ▼
   ┌──────────────────────────┴───┐   ┌───────┴──────────────┐    (same CarlaWorker; called from main)
   │ CarlaWorker  (QThread)       │   │ AnalysisWorker (QThr.)│
   │  • CARLA world + sync tick   │   │  • pulls latest frames│
   │  • cameras → frames          │   │  • crops lane ROI     │
   │  • DemandManager (traffic)   │   │  • sends to service ──┼───► inference/service.py (proc)
   │  • PhaseController ◄─counts   │   │  • analyze_tracks     │◄──── detections  (3.12 + ROCm)
   │    (set_counts/reallocate)   │   │  • smooth + stabilise │
   └──────────────────────────────┘   └───────────────────────┘
```

### The adaptive control loop, end to end

1. `CarlaWorker` renders 4 camera frames each sim tick → emits `frames_ready`.
2. `IntersectionWindow._cache_latest_frames` stores them; `get_latest_frames()`
   hands the latest snapshot to `AnalysisWorker`.
3. `AnalysisWorker` crops each frame to its **lane ROI**, ships the crops to the
   inference service, offsets returned boxes back to full-frame coords, then runs
   `analyze_tracks` → per-arm `count`, `weighted_demand`, `phase_demand`. It
   **stabilises the count** (median over `PERCEPTION_WINDOW`) and **EMA-smooths the
   demand** (`DEMAND_SMOOTHING`). Emits `analysis_ready`.
4. `IntersectionWindow._on_analysis` updates the camera overlays + perception dock,
   then pushes **per-arm car counts** to the worker via `CarlaWorker.update_counts`.
5. Inside `CarlaWorker.run()` each tick: it feeds the counts to the
   `PhaseController` (`set_counts`), ticks the controller, and at each green-phase
   start calls `reallocate()`. The controller drives CARLA's traffic lights and the
   worker emits `phase_changed` (→ plan-view semaphores) and `timing_changed` (→
   the Signal-logic panel).

> **Control input = car counts, not weighted demand.** `weighted_demand` (distance
> weighted) is still computed and shown per-arm, but the *timing decision* uses the
> raw stabilised **counts**, because the policy is expressed in cars ("NS has 5,
> EW has 4"). See §6.

---

## 5. Module reference

Grouped by area. For each: what it is, the symbols that matter, and gotchas.

### Entry / configuration

**`app/main.py`** — entry point. `_ensure_carla_env()` re-execs once with the
CARLA-safe env (§3); `main()` builds the `QApplication`, re-forces `LC_NUMERIC=C`,
opens `IntersectionWindow`. Run with `.venv312/bin/python -m app.main [template]`.

**`app/config.py`** — every tunable in one place. See the full table in §8.
`distance_weight(d)` lives here: the `1 - d` curve that turns a car's normalised
distance from the stop line into a demand weight.

### Core data model

**`app/core/intersection.py`** — `Arm` and `Intersection`.
- `Arm`: `enabled`, `image` (camera frame path), `lanes_in/out`, `signal_state`,
  `calibration` (the lane dict). `calibrated` = has lanes; `lane_count`.
- `Intersection`: `arms = {N,E,S,W}`, `all_calibrated`, `save()/load()` to the
  `intersection.json` template (§7).

**`app/core/calibration.py`** — pure JSON read/write for one camera's lane
calibration (`save_calibration`/`load_calibration`). Format in §7.

### Perception (detection → demand)

**`app/core/detection.py`** — `load_model`, `detect_vehicles`, `track_vehicles`.
Each detection is `{box:(x1,y1,x2,y2), foot:(x,y), cls, conf[, id]}`. **`foot`** is
the bottom-centre of the box — the wheels-on-road point tested against lane
polygons. *In-process YOLO; used for the static-image path. The live path uses the
out-of-process service instead.*

**`app/core/analysis.py`** — the detection→demand math (no Qt, unit-testable).
- `point_in_polygon`, `assign_lane(foot, calibration)` — drop a foot point into a
  lane polygon.
- `analyze_arm` / `analyze_arm_frame` — static counts per lane/direction/phase.
- **`analyze_tracks(arm, dets)`** — the live one. For each detection in an incoming
  lane: normalised stop-line distance → `distance_weight` → summed into
  `weighted_demand` and per-`phase_demand`. Returns
  `{tracks, count, weighted_demand, phase_demand, ignored}`. `count` is the integer
  used by the signal logic; `tracks` carries per-vehicle overlay data.

**`app/core/lane_distance.py`** — `stop_line_distance(polygon, foot) → [0,1]`.
0 at the stop line, 1 at the far end. Finds the lane's principal axis (covariance
eigenvector), projects the foot onto it. **Geometry assumption:** cameras look
*outward toward the queue*, so the stop line is the polygon end with the **largest
image-y** (nearest the camera). Calibration-free; a metric upgrade would use a
homography to a known 3.5 m lane width.

### Threads

**`app/core/carla_worker.py`** — `CarlaWorker(QThread)`. Owns the CARLA world,
cameras, `DemandManager`, and the `PhaseController`. Runs CARLA in **synchronous
mode** (it owns the clock via `world.tick()`), so timing is paced in *sim-seconds*
and stays correct even when the GPU makes each tick slow.
- **Signals out:** `frames_ready`, `phase_changed`, `vehicle_counts`,
  `timing_changed` (`{mode, ns_green, ew_green, ns_count, ew_count, decision}`),
  `status_message`, `error_occurred`.
- **Frame backpressure:** at most ONE `frames_ready` batch is ever in flight —
  the worker re-arms only when the GUI acks via `frames_displayed()` (called by
  the last connected slot, `_cache_latest_frames`). The wall-clock throttle
  alone is not enough: each queued batch pins ~25 MB of ndarrays, so a GUI
  slower per batch than the emit interval otherwise accumulates events without
  bound (growing lag → OOM crash). With the ack, a slow GUI drops frames
  instead of queueing them.
- **Called from main thread:** `set_adaptive(bool)`, `update_counts({arm:int})`,
  `stop()`.
- **Loop responsibilities:** feed counts to the controller every tick (zeros when
  adaptive is off, which disables actuation); tick the controller; on a green-phase
  start, `reallocate()` (adaptive) or restore fixed times; emit `timing_changed`;
  emit `phase_changed` on any phase change; emit `vehicle_counts` (ground truth)
  once per sim-second.
- **Teardown order matters:** restore async settings *first* (a stepped world only
  advances on `tick()`; leaving it in sync mode would hang a later run), then stop
  cameras, unfreeze lights, destroy vehicles.

**`app/core/analysis_worker.py`** — `AnalysisWorker(QThread)`. Thin *client* of the
inference service (no torch here). Per pass: crop each new frame to its lane ROI,
batch all arms into one socket round-trip, offset boxes back, `analyze_tracks`,
then `_smooth_demand` (EMA) + `_stabilize_count` (median). **Frame dedup**: skips
arms whose frame object is unchanged since last pass (sync mode emits frames slower
than this loop runs). Calibrations/ROIs are snapshotted at construction —
recalibrating means restarting the worker (the Run-YOLO toggle does this).
**Inference sizing:** `_imgsz_for(crop, cap=CROP_MAX_SIDE)` rounds the crop's long
side up to the nearest 32-multiple but never above the cap — small ROIs run at
native resolution (no upscale inflation), and large ROIs are pre-decimated before
the socket send so the GPU tensor never exceeds `CROP_MAX_SIDE` (640 by default;
`SEM_CROP_MAX_SIDE=960` for accuracy-first runs — see the config table in §8).

**`app/core/inference_service_manager.py`** — `InferenceServiceManager`. Finds
`.venv-infer/bin/python`, launches `inference/service.py` as a subprocess bound to
`/tmp/sem_inference.sock`, keeps it warm across Run-YOLO toggles (the first GPU
pass JITs kernels ~13 s), `_kill_stale()` clears orphans from a prior crash, and
`stop()` tears it down on exit.

### Inference service (separate process)

**`inference/service.py`** — `InferenceServer`. Loads YOLO once, serves a Unix
socket. `detect_batch`: all arms arrive in **one** request, inference looped per
crop (**not** a torch batch — a single ~960px inference already saturates the GPU,
and ultralytics' batch path letterboxes to squares, ~2× slower). **Per-frame only,
no tracking** — at the achievable frame rate ByteTrack mis-followed fast cars, so
we detect each frame and smooth the *aggregate demand* instead. Run with
`--socket --model --device {auto,cuda,cpu}`. Sets `PYTORCH_HIP_ALLOC_CONF`
(garbage-collect at 70 %, 128 MB max split) before torch loads, so the ROCm
caching allocator returns freed blocks instead of hoarding GTT — an explicit
env var from the launcher still wins (`setdefault`).

**`inference/protocol.py`** — the wire protocol (§7). `send_message`/`recv_message`
frame `[4-byte len][JSON header][raw payload]`. Pure-Python, 3.7-safe by mandate.

### UI

**`app/ui/intersection_window.py`** — `IntersectionWindow`, the main window and the
integration hub. Builds the toolbar, the four docks (Perception, Arm setup,
Detection results, CARLA Cameras), owns the workers, and wires every signal with
`QueuedConnection`. Key handlers:
- `on_connect_carla` / `on_disconnect_carla` — build/start/stop `CarlaWorker`,
  connect its signals (incl. `timing_changed → _update_signal_logic`).
- `on_toggle_yolo` — start/stop `AnalysisWorker` (ensures the inference service is
  up first).
- **`on_toggle_adaptive`** — the 🧠 button; flips `_adaptive_enabled` and calls
  `CarlaWorker.set_adaptive`.
- `_on_analysis` — overlays + perception dock, and pushes **counts** to the worker.
- `_build_signal_logic_panel` / `_update_signal_logic` / `_set_mode_badge` — the
  Signal-logic readout (mode badge, per-axis cars→green table, proportional split
  bar, verdict).
- `_CarlaConnectDialog` — host/port/map/vehicles/demand-weights/green-times.

**`app/ui/lane_window.py`** — `LaneCalibrationWindow`. Per-camera tool to draw the
lane polygons, tag each lane incoming/outgoing, assign a phase, set the signal
state, and save the calibration back onto the arm (via the `on_done` callback).

**`app/graphics/plan_view.py`** — `PlanView`, the top-down schematic. Clickable
arm tiles (select / double-click to calibrate), coloured by calibration state,
with a `Semaphore` per arm. `update_semaphore_states(states)` recolours the lights
from a `phase_changed` payload without a full rebuild.

**`app/graphics/semaphore.py`** — `Semaphore`, a 3-circle signal scene item;
click cycles red→green→yellow (manual mode).

**`app/graphics/camera_panel.py`** — `CameraGridWidget`, the 2×2 live feed. `update_frames` (RGB arrays), `update_counts` (queue number), `set_overlays`
(tracked boxes/feet/ids + pedestrians together — one tile repaint per analysis
pass, not two). Overlay coords are original-frame pixels scaled to the tile.

**`app/graphics/lane.py` / `canvas.py`** — the calibration drawing primitives:
an editable polygon with draggable vertex handles, and the canvas that hosts them.

### Simulation library

**`carla_intersection.py`** — besides `PhaseController` (§6):
- `load_opendrive_map`, `find_junction`/`get_junction_center` — load a standalone
  `.xodr` and locate the junction centre (relative coords, so world origin doesn't
  matter).
- `group_traffic_lights(world, center)` → `{arm:light}`, plus NS and EW lists
  (classified by position relative to the centre).
- `_cameras_from_lights` — camera transforms at each signal pole, looking *outward
  toward the queue* (this is what `lane_distance.py` assumes).
- `DemandManager` — keeps a fixed vehicle count recirculating: pool-based (teleport
  only, no mid-run spawn/destroy). `prefill()` pre-spawns all target vehicles
  underground at startup; `fill()` releases pooled vehicles onto inbound lanes
  (weighted per-arm demand) by teleport; `tick()` recycles cars that reach the
  outbound arm tip back underground. `_spawn_one()` fires only as a recovery path
  when CARLA unexpectedly destroys an actor (logged to stdout). `detect_inbound()`
  is the **ground-truth** per-arm count (lane −1 only).
- `PedestrianManager` — random walker arrivals on the sidewalk that wait, cross on
  cue (`set_walk_allowed`, driven by the `PED_CROSS` phase), and return to the pool
  on arrival. Crosswalk geometry is derived from the light positions like
  `_cameras_from_lights`. `waiting_counts()` is the ground-truth per-arm waiting tally.
  See §6.
  **Walker pool:** walkers are *never destroyed mid-run*. `prefill()` pre-spawns
  `PED_MAX_ACTIVE` walkers underground at startup (same rationale as vehicles: UE4
  never fully releases skeletal meshes). `_park()` hides finished walkers under the
  map (`_PED_PARK_Z`); `maybe_spawn()` teleports a pooled walker to the kerb.
  `destroy_all()` (teardown only) sweeps every `walker.*` actor in the world.
  **Spawn placement:** `maybe_spawn()` shuffles arms and tries each kerb endpoint;
  `_ped_clear()` rejects a spot if another walker is within `_PED_SPAWN_CLEAR` (1.8 m).
  A random offset of ±`_PED_CROSS_SPREAD` (0.7 m) along the arm axis is applied to
  both the start *and* target of every walker so concurrent crossers take parallel
  paths rather than converging on the same point (which causes physics blocking).
  **Lifecycle guards:** release is rising-edge only (walkers arriving mid-window wait
  for the next one); `PED_CROSS_TIMEOUT` (30 s) recycles stuck/run-over walkers;
  `PED_WAIT_GIVEUP` (90 s) releases unserved waiters in fixed mode. `tick(dt)` uses
  the sim-step for the ages.
  **Sync-mode spawn grace:** a freshly spawned actor reports `is_alive=False` until
  the next `world.tick()` snapshot; `tick(dt)` only treats it as gone after 1 s.

---

## 6. Signal-timing logic (the heart)

All in `PhaseController` (`carla_intersection.py`), driven by `CarlaWorker`.

### Phase machine

Fixed two-phase cycle with all-red clearance:

| # | Phase       | Duration   |
|---|-------------|------------|
| 0 | `NS_GREEN`  | `ns_green` |
| 1 | `NS_YELLOW` | `YELLOW` 3 s |
| 2 | `ALL_RED`   | `ALL_RED` 2 s |
| 3 | `EW_GREEN`  | `ew_green` |
| 4 | `EW_YELLOW` | `YELLOW` 3 s |
| 5 | `ALL_RED`   | `ALL_RED` 2 s |

`tick(dt)` accumulates elapsed sim-time and advances at the phase duration.
`phase_name()` reports the current phase; `CarlaWorker` maps it to per-arm
red/yellow/green for the UI.

### Constants

| Const       | Value | Meaning                                                        |
|-------------|-------|----------------------------------------------------------------|
| `MIN_GREEN` | 10 s  | floor on any green (driver-reaction safety + fairness)         |
| `MAX_GREEN` | 60 s  | ceiling on any green (**starvation guard** for the other axis) |
| `YELLOW`    | 3 s   | amber clearance — always run, never skipped                    |
| `ALL_RED`   | 2 s   | all-red clearance between phases                               |
| `DEADBAND`  | 1 car | axis count gaps this small are ignored (`5 vs 4` → hold)       |
| `_budget`   | ns+ew | total green budget, captured at construction (default 60 s)    |

### The adaptive policy

Inputs: per-axis **car counts** `NS = N+S`, `EW = E+W` (the stabilised integer
`count` from `analyze_tracks`, pushed via `update_counts`). Two mechanisms:

**A. Re-split at each green start — `reallocate()`**
- If `total == 0` **or** `|NS − EW| ≤ DEADBAND` → **balanced**: even split
  (`budget/2` each). This is the "cars shared equally → do nothing" case.
- Otherwise split the budget in proportion to counts, each phase clamped to
  `[MIN_GREEN, MAX_GREEN]`:
  `ns_green = clamp(budget · NS / (NS+EW))`, `ew_green = clamp(budget − ns_green)`.
  The clamps mean the lighter axis is never starved and the heavier can't run away.

**B. Early switch mid-phase — force-off (`_should_force_off`, in `tick`)**
- Counts are fed every tick. If we're green on the **lighter** axis, have already
  served `MIN_GREEN`, and the other axis leads by more than `DEADBAND`, the current
  green ends *now* (`elapsed` snapped to its duration) → the normal advance then
  runs the **full yellow + all-red** before the cross street moves.
- This is "immediate change, with precaution": the soonest *safe* switch is
  `remaining MIN_GREEN (if any) + 3 s yellow + 2 s all-red` — never a snap.

**Mode off (fixed):** `CarlaWorker` feeds `set_counts(0, 0)` (force-off can't
trigger) and calls `set_green_times(configured_ns, configured_ew)` at each green
start, so toggling adaptive off cleanly restores the dialog's fixed timing.

### What the panel shows

`timing_changed` is emitted at each green-phase start with
`{mode, ns_green, ew_green, ns_count, ew_count, decision, ped_count}`;
`_update_signal_logic` renders:
- **ADAPTIVE / FIXED** badge
- Three-row table: N–S and E–W count → green seconds; **People** row showing the
  perceived pedestrian waiting count → `phase` (purple, `_PED_CROSS` will be
  inserted this cycle) or `—`
- Proportional NS / EW split bar (blue / orange)
- Verdict: `balanced — held` / `favouring N–S` / `favouring E–W`, with
  `· ped phase` appended when `ped_count > 0`

On toggle the badge flips immediately and the verdict shows `applies next green…`
because the green seconds can only legally change at the next green start.

### Pedestrian crossing — the "ponder"

Pedestrians are served by an **exclusive all-red phase** (`PED_CROSS`): every approach
red so people cross safely. It is **not** on a fixed timer — the controller *ponders*
when to grant it so it costs little throughput, but never starves pedestrians.

- **Input:** `set_ped_demand(n)` — the perceived count of pedestrians waiting (summed
  across crossings), pushed each tick (`CarlaWorker`, adaptive mode only). `_ped_wait`
  accumulates the sim-seconds they've waited unserved.
- **The ponder — `_should_serve_peds()`:** grant a crossing when peds are waiting **and**
  any of: total cars ≤ `PED_LOW_TRAFFIC` (light traffic), exactly one axis empty
  (lopsided — its green serves nobody), or `_ped_wait ≥ PED_MAX_WAIT` (fairness cap).
- **Insertion:** only at the end of an `ALL_RED` clearance (phases 2/5) — the junction
  is already empty, the cheapest moment. The phase then holds all-red for
  `PED_CLEAR_BASE + PED_CLEAR_PER_PED·peds` (capped at `PED_CLEAR_MAX`), after which the
  normal cycle resumes. `force-off` also gaps out a near-empty current green when peds
  have waited past patience, to reach the crossing sooner.
- **Constants:** `PED_CLEAR_BASE` 7 s, `PED_CLEAR_PER_PED` 0.5 s, `PED_CLEAR_MAX` 20 s,
  `PED_MAX_WAIT` 25 s, `PED_LOW_TRAFFIC` 2 cars.
- **Scope:** pedestrian serving is part of **adaptive mode**. In fixed mode the worker
  feeds `set_ped_demand(0)`, so peds wait (documented limitation). The standalone
  `run_demand_mode` drives it from ground-truth counts instead.

The simulation side is `PedestrianManager` (`carla_intersection.py`): random walker
arrivals on the sidewalk → wait → cross under manual `WalkerControl` while `PED_CROSS`
is active (`set_walk_allowed`) → despawn on arrival. `waiting_counts()` is ground truth.

### Deferred edge case

The two-phase controller treats `N+S` as one demand, so `NS=5` with 4 cars on N and
1 on S looks the same as a balanced 3/2 — it can't favour N over S (they always go
green together). Fixing per-approach fairness needs protected/4-phase control; this
is logged in [`../brainstorming.md`](../brainstorming.md).

---

## 7. Data formats

### Lane calibration (per camera) — `app/core/calibration.py`

```json
{
  "image": "captures/N.png",
  "size": [1920, 1080],
  "lanes":      { "lane_0": [[x, y], ...], ... },     // polygon, image px
  "directions": { "lane_0": "incoming" | "outgoing" },
  "phases":     { "lane_0": "approach_A" },
  "signal_state": "red" | "yellow" | "green",
  "crossings":  { "crossing_0": [[x, y], ...], ... }  // pedestrian zones, image px
}
```
`crossings` are the manually-drawn pedestrian crossing polygons (drawn in the lane
tool with "Add crossing zone"). A detected **person** whose foot lands in a crossing
— or within `CROSSING_MARGIN_PX` (30 px) of its border, since someone waiting AT the
kerb stands exactly on the drawn line — counts as a waiting pedestrian for that arm
(see `analysis.assign_crossing`); persons elsewhere are ignored. Missing key ⇒ no
pedestrian sensing for that camera.
Polygons are in **capture-pixel coordinates** — change `CAPTURE_W/H` and you must
re-calibrate. Compatible with the `map_lane_assignment.ipynb` notebook (reads
`lanes`).

### Intersection template — `app/core/intersection.py` (`intersection.json`)

```json
{
  "type": "4-way",
  "arms": {
    "N": { "enabled": true, "image": "captures/N.png",
           "lanes_in": 1, "lanes_out": 1, "signal_state": "red",
           "calibration": { ...the calibration dict above... } },
    "E": { ... }, "S": { ... }, "W": { ... }
  }
}
```

### Inference wire protocol — `inference/protocol.py`

One message = `[4-byte big-endian header length][UTF-8 JSON header][raw payload]`.
`header["nbytes"]` gives the payload length (0 if absent). Request/response types:

| Type           | Direction | Header / payload                                              |
|----------------|-----------|--------------------------------------------------------------|
| `detect_batch` | →service  | `{conf, imgsz, crops:[{arm,h,w}]}` + concatenated RGB crops  |
| `detections`   | ←service  | `{results:{arm:[{box,foot,cls,conf}]}}` (crop-local coords)  |
| `reset`/`ok`   | both      | clears per-connection state (no-op for the stateless detector)|
| `ping`/`pong`  | both      | health check; `pong` carries the device                      |

---

## 8. Config reference (`app/config.py`)

| Constant            | Default            | What it controls                                                                 |
|---------------------|--------------------|----------------------------------------------------------------------------------|
| `YOLO_MODEL`        | `models/yolov8n.pt`| in-process model (static-image path; 3.7 → v8 only)                              |
| `DETECT_CONF`       | `0.15`             | live per-frame confidence (low; flicker absorbed by demand smoothing)            |
| `CONF`              | `0.25`             | confidence for the in-process detector                                           |
| `VEHICLE_CLASSES`   | `{1,2,3,5,7}`      | COCO ids kept: bicycle, car, motorcycle, bus, truck                              |
| `DEMAND_SMOOTHING`  | `0.2`              | EMA weight on new demand sample (lower = smoother, laggier)                      |
| `PERCEPTION_WINDOW` | `7`                | frames in the **median** car-count window (rejects dropout blinks)               |
| `SERVICE_MODEL`     | `models/yolo11m.pt`| out-of-process model (3.12 → may use yolo11)                                     |
| `TRACKER`           | `bytetrack.yaml`   | tracker config for the in-process `track_vehicles` (live path is per-frame)      |
| `ANALYSIS_INTERVAL` | `0.15` s           | min seconds between live analysis passes (~6–7 Hz)                               |
| `CAPTURE_W/H`       | `1920×1080`        | camera render resolution (calibration is in these px)                            |
| `SENSOR_TICK`       | `0.1` s            | min sim-seconds between camera captures (10 Hz)                                   |
| `SIM_FIXED_DELTA`   | `0.05` s           | fixed sync-mode physics step (20 Hz)                                              |
| `INFER_IMGSZ`       | `1280`             | baseline YOLO inference long-side (used by the in-process static path)           |
| `CROP_MAX_SIDE`     | `640` (env `SEM_CROP_MAX_SIDE`) | hard cap for live lane-crop inference; never upscaled past this. 640 leaves RAM headroom for screen recording; `SEM_CROP_MAX_SIDE=960` for accuracy-first runs |
| `DEFAULT_PHASE`     | `approach_A`       | phase tag for lanes that don't specify one                                       |

**`carla_intersection.py` constants:** `ARM_ROAD_ID={W:0,E:1,S:2,N:3}`,
`INBOUND_LANES=(-1,)`, `SPAWN_S=14.0`, `_OUTBOUND_RECYCLE_S=25.0`, `_POLE_H=5.0`,
`_SIG_S_FROM_JUNCTION=15.0`, `_LANE_W=3.5`. `PhaseController` constants in §6.

---

## 9. Glossary

- **Arm / approach** — one of the four legs (N/E/S/W) feeding the junction.
- **Axis** — a signal phase group: **NS** = North+South, **EW** = East+West (they
  go green together in the two-phase controller).
- **Foot point** — bottom-centre of a detection box; the wheels-on-road point
  matched against lane polygons.
- **Incoming/outgoing lane** — calibration tag; only **incoming** lanes generate
  demand.
- **Count** — stabilised integer number of cars in an arm's incoming lanes; the
  **control input**.
- **Weighted demand** — sum of `distance_weight` over in-lane cars (a car at the
  stop line ≈ 1.0, far back ≈ 0.0). Computed and displayed, but *not* the control
  input today.
- **Phase** — a stage in the signal cycle (`NS_GREEN`, `NS_YELLOW`, `ALL_RED`, …).
- **Budget** — total green seconds (`ns_green + ew_green`) preserved across an
  adaptive re-split.
- **Deadband** — count gap small enough to ignore (hold the even split).
- **Force-off / gap-out** — ending a green early (after MIN_GREEN) because the
  other axis is busier; still pays full yellow + all-red.
- **ROI** — the lane region cropped from a frame before inference (high detail,
  fewer pixels).
- **Ground truth** (`DemandManager.detect_inbound`) — CARLA's exact per-arm count,
  independent of perception; used for the camera-tile queue number, available for
  future before/after comparison.

---

## 10. Running it

See [`README.md — Running`](README.md#running) for the full launch guide including
the `systemd-run` memory-capped CARLA invocation, flag explanations, and ROCm
inference setup. Short form:

```bash
# Terminal 1 — CARLA in a 8 GB memory scope, headless, medium quality
# (use -quality-level=Low when screen-recording — see README "Memory budget")
systemd-run --user --scope -p MemoryMax=8G \
    ./carla/CarlaUE4.sh -quality-level=Medium -nosound -RenderOffScreen

# Terminal 2 — app (Python 3.7 + CARLA egg)
# (SEM_CROP_MAX_SIDE=960 prefix for accuracy-first runs; default is 640)
.venv/bin/python app/main.py
```

Then: **Connect CARLA…** → calibrate each arm's incoming lane → **▶ Run YOLO
(live)** → **🧠 Adaptive signals**. Watch the per-arm counts (left dock) and the
Signal-logic panel: greens shift toward the busier axis, a strong lean force-offs
the current green early, and the People row lights up purple when a pedestrian
crossing phase is triggered. Toggle the button off to fall back to fixed timing.

If something misbehaves (stream-ID error spam, desktop freeze, recovery-spawn
lines), see the troubleshooting table in
[`README.md — Troubleshooting`](README.md#troubleshooting).

---

## 11. Known limitations / next

- **Within-axis fairness** — see §6 deferred edge case (`brainstorming.md`).
- **No before/after metric yet** — the perception→timing path is live; a
  fixed-vs-adaptive comparison (avg wait / throughput, from `DemandManager` ground
  truth) is the planned next step.
- **Per-frame detection** — no tracking in the live path by design (§5); demand is
  stabilised at the aggregate level instead.
- **Segmentation** — road/sidewalk segmentation is intentionally *not* in the core
  path; the map prior supplies drivable area (see `brainstorming.md`).
