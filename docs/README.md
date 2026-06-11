# SemOptimizer

[Demo.webm](https://github.com/user-attachments/assets/7baffe54-7f37-4349-9f0c-812bfd65f969)
[ArmCalibrationDemo.webm](https://github.com/user-attachments/assets/3648e05c-7ac3-4794-9408-bb3e7332356b)


A computer vision system that watches all four approaches of a signalised intersection through pole-mounted cameras and continuously adapts the signal cycle to actual traffic demand — giving more green time to the busier axis, inserting pedestrian crossings when people are waiting, and cutting short a green that is serving nobody.

Validated end-to-end inside a CARLA simulation. Designed from the ground up for eventual deployment on an edge device (microcomputer + tiny distilled model) at a real intersection.

---

## The problem

Most intersections run fixed cycles programmed decades ago for average demand. The result is predictable: the N–S axis sits on red while the E–W light goes green for an empty road, every cycle, for hours.

Adaptive control exists in expensive proprietary systems. This project builds an open, camera-only alternative that needs no loop detectors or infrastructure beyond the cameras already bolted to signal poles.

---

## How it works

Four cameras — one per traffic light pole — each face their own inbound lane. Each frame is processed by a YOLO detector that counts vehicles and pedestrians per approach. Those counts feed an adaptive controller that reallocates green time proportionally, penalises the lopsided axis, and inserts a pedestrian crossing phase when people have been waiting.

```
 ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
 │  CAM  N  │  │  CAM  S  │  │  CAM  E  │  │  CAM  W  │
 │  YOLO    │  │  YOLO    │  │  YOLO    │  │  YOLO    │
 │  count   │  │  count   │  │  count   │  │  count   │
 └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
      └──────────────┴──────┬──────┴──────────────┘
                             │
                   ┌─────────▼──────────┐
                   │  PhaseController   │
                   │                    │
                   │  N:5  S:3  E:1  W:0│
                   │  NS=8,  EW=1       │
                   │  → NS gets 48 s    │
                   │  → EW gets 12 s    │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │  CARLA traffic     │
                   │  lights / real     │
                   │  signal hardware   │
                   └────────────────────┘
```

### Signal logic in brief

The controller runs a two-phase cycle (NS green → NS yellow → all-red → EW green → …). At the start of each green phase it re-splits the total green budget proportionally to the current car counts, clamped to a minimum (10 s — safety) and maximum (60 s — starvation guard). A running force-off mechanism also cuts short the current green mid-phase if the waiting axis has outgrown the running one by more than the deadband (1 car), once the minimum has been served.

Pedestrians are served in an exclusive all-red phase inserted at the cheapest moment (the natural all-red clearance between phases). The controller grants it opportunistically — when traffic is light, one axis is empty, or pedestrians have waited too long — so it costs minimal throughput.

See [`REFERENCE.md §6`](REFERENCE.md#6-signal-timing-logic-the-heart) for the full policy.

---

## Architecture

### Three-process split

The system runs across three Python environments, intentionally:

| Process | Python | Why |
|---|---|---|
| App + CARLA (`app/`) | 3.7 | The official CARLA `.egg` pins to Python 3.7 |
| Inference service (`inference/`) | 3.12 | ROCm PyTorch requires 3.10+; can't share the 3.7 env |
| Tools / notebooks | varies | |

Detection is out-of-process behind a Unix socket. The app (3.7) is a thin client; all YOLO/torch lives in `inference/service.py` (3.12). If the GPU wedges, it crashes the inference process, not the CARLA session.

### Runtime data flow

```
  Main thread (Qt)
  IntersectionWindow
  ├── CarlaWorker (QThread)
  │    • owns CARLA world + synchronous tick
  │    • 4 pole cameras → frames
  │    • DemandManager  (vehicle pool, teleport-based recycling)
  │    • PedestrianManager (walker pool, crosswalk lifecycle)
  │    • PhaseController  (signal timing)
  │    signals → frames_ready, phase_changed, timing_changed
  │
  └── AnalysisWorker (QThread)
       • crops lane ROI from each frame
       • sends crops to inference/service.py over Unix socket ──► GPU (ROCm)
       • maps detections back to full-frame coords
       • stabilises car count (median, 7-frame window)
       • smooths demand (EMA)
       • pushes counts → CarlaWorker → PhaseController
```

### Vehicle and pedestrian pools

All actors are pre-spawned underground at startup (`prefill()`) and only ever teleported to road positions — never spawned or destroyed mid-run. This eliminates the UE4 memory leak from repeated spawn/destroy churn. Vehicles recirculate: inbound spawn → drive through → outbound tip → park underground → release to road again. Walkers: kerb → wait → cross on green → park underground → return to next kerb.

### Project structure

```
semOptimizer/
├── app/
│   ├── main.py                    # entry point; fixes CARLA env then re-execs
│   ├── config.py                  # all tunables (detection, timing, geometry)
│   ├── core/
│   │   ├── carla_worker.py        # QThread: CARLA world + PhaseController
│   │   ├── analysis_worker.py     # QThread: perception pipeline client
│   │   ├── analysis.py            # detection → lane assignment → demand math
│   │   ├── calibration.py         # lane calibration JSON read/write
│   │   ├── intersection.py        # Intersection/Arm data model
│   │   ├── lane_distance.py       # normalised stop-line distance
│   │   └── inference_service_manager.py
│   ├── graphics/                  # plan view, camera tiles, lane editor
│   └── ui/
│       ├── intersection_window.py # main window, wires everything
│       └── lane_window.py         # per-camera lane calibration tool
├── inference/
│   ├── service.py                 # out-of-process YOLO server (3.12 + ROCm)
│   └── protocol.py                # Unix-socket wire protocol (3.7-safe)
├── carla_intersection.py          # CARLA library: map, cameras, traffic, PhaseController
├── maps/                          # OpenDRIVE (.xodr) maps
├── models/                        # YOLO weights
├── captures/                      # per-arm reference frames (N/E/S/W.png)
├── tools/                         # memwatch.sh, etc.
└── docs/
    ├── README.md                  # this file
    └── REFERENCE.md               # full technical reference
```

---

## System requirements

**Developed and tested on:**

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 5 7500F |
| GPU | AMD RX 9060 XT 16 GB (gfx1200) |
| RAM | 16 GB DDR5 |
| Motherboard | Gigabyte B650 EAGLE AX |
| GPU compute | ROCm 6.4 |

- **CARLA 0.9.15** with the official `.egg` (not the pip wheel)
- **Python 3.7** for the app and CARLA integration (venv at `.venv/`)
- **Python 3.12** for the inference service (venv at `.venv-infer/`)
- **ROCm 6.4** for AMD GPU inference; CUDA also works; CPU fallback is available but slow (~10× slower per frame)

---

## Running

### First-time setup

Create the inference venv (Python 3.12 + ROCm PyTorch):

```bash
python3.12 -m venv .venv-infer
.venv-infer/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm6.4
.venv-infer/bin/pip install ultralytics lapx
```

### Every session

**Step 1 — Launch CARLA inside a memory-capped systemd scope.**

CARLA + Unreal Engine 4 loads all map assets into system RAM regardless of VRAM. On 16 GB this leaves almost no headroom once Python and the UI are also running. The `systemd-run` wrapper confines CARLA to 8 GB and prevents it from swapping the whole machine to death:

```bash
systemd-run --user --scope -p MemoryMax=8G \
    ./carla/CarlaUE4.sh -quality-level=Medium -nosound -RenderOffScreen
```

| Flag | Why |
|---|---|
| `--user --scope` | Runs CARLA in a transient user scope; the kernel enforces `MemoryMax` on the whole UE4 process tree |
| `-p MemoryMax=8G` | Hard RSS cap — the kernel OOM-kills CARLA before it can swap-thrash the rest of the system |
| `-quality-level=Medium` | Cuts texture and mesh detail; saves ~1–2 GB RAM with no effect on camera output |
| `-nosound` | Disables the audio subsystem; saves ~200–400 MB |
| `-RenderOffScreen` | Disables the UE4 render window entirely; saves ~3–4 GB RAM and frees GPU bandwidth for inference |

To monitor GPU memory (GTT/shmem) during the run:

```bash
./tools/memwatch.sh &
```

**Step 2 — Launch the app.**

```bash
.venv/bin/python app/main.py
```

The app auto-starts the inference service (`inference/service.py`) in the `.venv-infer` Python 3.12 environment when **▶ Run YOLO** is toggled. YOLO inference runs on the RX 9060 XT via ROCm at ~14 ms per frame.

### Memory budget (screen recording, demos)

The 16 GB box runs close to its limit with CARLA + the app + ROCm inference all
up, and most of the GPU working set lives in **GTT** — shared memory carved
from system RAM that is invisible to RSS and to `MemoryMax`. Anything extra
(a screen recorder, a browser) can push the machine into swap-thrash: the
desktop freezes and all cores peg at 100 % (that's the kernel reclaiming
memory, not the app). Two settings keep the budget in check:

- **`CROP_MAX_SIDE` defaults to 640** (GPU tensor memory scales with the
  *square* of the crop side, so 640 needs ~2.2× less than 960). For
  accuracy-first runs with nothing else open:
  `SEM_CROP_MAX_SIDE=960 .venv/bin/python app/main.py`
- **`inference/service.py` sets `PYTORCH_HIP_ALLOC_CONF`** so the ROCm caching
  allocator returns freed blocks instead of hoarding them in GTT. An explicit
  env var from the launcher still overrides it.

When recording: launch CARLA with `-quality-level=Low`, and close the browser
first — it is typically holding 1–2 GB you will need.

### In the UI

1. **Connect CARLA…** — choose host, map, vehicle count, demand weights, and green times
2. **Double-click each arm** in the plan view → draw the inbound lane polygon → Save
3. **▶ Run YOLO (live)** — starts the inference service and the analysis worker
4. **🧠 Adaptive signals** — enables adaptive timing; watch green seconds shift as queues build

The **Signal logic** panel shows in real time what the controller decided each cycle: per-axis car and pedestrian counts, the resulting green seconds, whether a pedestrian crossing phase was triggered, and the verdict (favouring N–S / favouring E–W / balanced).

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| CARLA terminal spams `ERROR: Invalid session: no stream available with id N` | A previous app instance is still alive and its camera listeners keep retrying against the freshly restarted server (IDs 2–5 are the four pole cameras) | `pkill -f "app/main.py"`, then reconnect |
| Desktop freezes, all CPU cores at 100 % | RAM exhaustion — GTT/shmem pressure from CARLA + inference (+ recorder); the pegged cores are kernel reclaim, not the app | See *Memory budget* above; close other apps, keep `CROP_MAX_SIDE` at 640 |
| `DemandManager recovery spawn …` lines during a run | The vehicle pool ran dry, so the manager fell back to a fresh spawn — harmless, but frequent recovery spawns reintroduce the UE4 spawn-churn leak | Expected occasionally; if constant, lower the vehicle target |
| `time-out of 30000ms while waiting for the simulator` | CARLA still booting, or it was OOM-killed by the `MemoryMax` scope | `systemctl --user status run-*.scope`; relaunch CARLA |
| Walker navigation crash on server boot | Stale `OpenDriveMap.obj` nav cache from a previous generated map | The app cleans this on exit; delete `carla/CarlaUE4/Content/Carla/Maps/Nav/*` manually if it crashed before cleanup |

---

## Further reading

- [`REFERENCE.md`](REFERENCE.md) — full technical reference: every module, the signal-timing algorithm in depth, all data formats, config knobs, and a glossary.
- [`brainstorming.md`](../brainstorming.md) — deferred ideas, open questions, per-approach fairness, RL alternatives.
