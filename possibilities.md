# Possibilities — deployment feasibility & alternatives

> Strategy / options doc. Sibling to [`brainstorming.md`](brainstorming.md) (loose
> ideas) and [`docs/REFERENCE.md`](docs/REFERENCE.md) (how the code works today).
> This one answers: *can this actually run on small/cheap hardware, and what are
> the alternatives to the heavy ML path?*

---

## 1. The framing: this is a split system, not one blob to shrink

The honest correction to "can we shrink the whole project onto a microcontroller":
the project was never meant to deploy as one piece. Most of it is **lab / dev
tooling** that never leaves the development machine. Only a thin slice is the edge
node.

```
   ┌─────────────── DEV / LAB (never deployed) ───────────────┐    ┌──── EDGE (ships) ────┐
   │  CARLA sim · PySide6 app · training · big YOLO teacher    │    │  camera + tiny model │
   │  → generate data, calibrate, validate, monitor           │ ─► │  + control logic     │
   └──────────────────────────────────────────────────────────┘    └──────────────────────┘
```

So the real question is narrow: **what does one pole/intersection node need to
run**, not "fit CARLA + PySide6 + yolo11m on an ESP32" (which is nonsense and
isn't the goal).

## 2. What never ships to the edge

- **CARLA** — a data generator. Replaced in the field by real cameras.
- **The PySide6 desktop app** — monitoring/calibration tool for humans.
- **Training + the big YOLO "teacher"** (`yolo11m`, ~20M params, ~40 MB) — runs on
  the dev machine / a central server; generates labels and ground truth.

## 3. The control logic is a non-issue

`PhaseController` (phases, deadband, `reallocate()`, force-off) is a few dozen
arithmetic operations on four integers. It runs on literally any microcontroller,
in microseconds, in a few KB. This part is free — ignore it in the sizing debate.

## 4. Perception is the real question — and why a *bare MCU* is hard

The current detectors will **not** run on a bare microcontroller, and the reason is
subtler than "the file is too big":

- **Activation RAM is the binding constraint, not weight size.** A 640 px YOLO needs
  a multi-MB *tensor arena* for intermediate activations. The biggest MCUs have
  ~256 KB–1 MB SRAM (an ESP32-S3 has 512 KB + up to 8 MB PSRAM). Even if `yolov8n`'s
  ~3 MB of INT8 weights fit in flash, the working RAM for its detection head at
  usable resolution blows the budget.
- `yolo11m`: server/GPU only.
- `yolov8n` (~3.2M params): needs an **SBC or an NPU**, not an MCU.

## 5. Deployment tiers (a spectrum, not yes/no)

| Tier | Hardware (~cost) | Perception that fits | Output quality |
|---|---|---|---|
| **Bare MCU** | ESP32-S3, Cortex-M55 + Ethos-U micro-NPU (< $10) | A *distilled* tiny occupancy classifier (the README's `TinyCarCounter`: ~96–160 px zone crop → 3 conv → INT8, ~150–250 KB, a few Hz) **or** a classical virtual-loop detector (no NN, see §8) | Coarse: *empty / few / many*. No exact count, no tracking, no distance weighting |
| **Edge SBC / NPU** | Raspberry Pi 5, Jetson Orin Nano, Coral TPU, Hailo-8 ($50–250) | **`yolov8n` INT8 real-time** on all 4 feeds + control | Full per-vehicle boxes → exact counts, the demand signal we have now |
| **Central server** | any GPU box | `yolo11m` / ensembles, retraining, fleet monitoring | Reference accuracy; not a per-pole device |

The README's `< 150 KB / < 100 ms` target is the right instinct — but be honest
about what that budget *buys*: on an MCU it gets you **occupancy classification**,
not bounding-box detection. They are different problems.

## 6. The catch: model size ↔ control granularity

This is the crux most people miss. The current logic compares axes with a **1-car
deadband** (`carla_intersection.py` → `PhaseController`). A bucketed *empty/few/many*
MCU model **cannot feed that** — it would force a coarser control policy (e.g.
compare per-zone *occupancy ratios* instead of integer counts).

So the model-size decision is really a **control-resolution decision**:

- *Coarse demand is enough* (which axis is busier, roughly) → MCU tier works.
- *You want the distance-weighted, near-the-stop-line demand we built* → SBC tier.

Match the model to how fine-grained the timing actually needs to be.

## 7. Recommended architecture

Per-pole node = camera + a small **SBC/NPU** running a quantized detector (or a
distilled counter) that outputs **just one number per approach**; a central
featherweight controller fuses the four numbers and runs the phase logic.

```
  pole N ─[cam + detector]─►  count_N ┐
  pole E ─[cam + detector]─►  count_E ├─► fuse ─► PhaseController ─► lights
  pole S ─[cam + detector]─►  count_S ┤
  pole W ─[cam + detector]─►  count_W ┘
```

This matches the README's "4 cameras each report a count, fuse → decide" design,
keeps each node ~$50–100, and doesn't pretend a bare MCU can do YOLO. The network
traffic per node is a handful of bytes per second.

## 8. Alternative: the classical virtual-loop detector (no neural net)

Worth taking seriously: for a **fixed** pole camera, you may not need a neural net
at all to know "is there a queue and how long." This is how a lot of real traffic
cameras already work, and it runs on a potato (even an MCU).

### Where it comes from

Traditional intersections use **inductive loop detectors** — wire coils buried in
the asphalt that sense the metal mass of a car via a change in inductance. A
**virtual loop** (a.k.a. virtual induction loop, VLD) reproduces that with a
camera: you *draw* the loop as a polygon on the image and decide, with plain image
processing, whether a vehicle is occupying it. No buried hardware, no model.

> **Tie-in:** this project already has the geometry a virtual loop needs — the lane
> calibration polygons *are* virtual loops, and `lane_distance.stop_line_distance`
> already orders points from the stop line. A VLD variant is an *addition*, not a
> rewrite.

### How it works, step by step

**1. Define the loop(s).** Draw one or more polygons over the lane on the fixed
frame: one at the stop line for *presence*, or a ladder up the lane for *queue
length*.

```
   camera view of one approach          a ladder of virtual loops
   ┌───────────────────────────┐        ┌───────────────────────────┐
   │   ░░░ far end of lane ░░░  │        │  [ loop 4 ]   (far)       │
   │                           │        │  [ loop 3 ]               │
   │        🚗  🚗             │        │  [ loop 2 ] ← occupied    │
   │     🚗  🚗  🚗            │        │  [ loop 1 ] ← occupied    │
   │  ═══ stop line ═══════════│        │  ═══ stop line ═══════════│
   └───────────────────────────┘        └───────────────────────────┘
                                          queue ≈ consecutive occupied
                                          loops from the stop line back
```

**2. Decide occupancy** — is a vehicle inside the loop. Classic methods:

- **Background subtraction (best for *presence*, incl. stopped cars).** Keep a model
  of the empty road (the *background*). Each frame, subtract it, threshold the
  difference → a *foreground mask* (what's new vs. the empty road). Count foreground
  pixels inside the loop polygon; if the fraction exceeds a threshold → **occupied**.
  The background model can be a running average/median, or an adaptive one
  (OpenCV's `MOG2` / `KNN`) that tracks gradual lighting change.
- **Frame differencing (motion → *counting* passages).** Diff consecutive frames to
  detect motion. Great for counting cars crossing a mid-lane loop, **bad for a car
  stopped at red** (no frame-to-frame change → it "disappears"). Use it for flow,
  not for queue presence.
- **Edge / texture density.** Empty asphalt has low edge content; a vehicle adds
  edges (Canny/Sobel). High edge density in the loop → occupied. More robust to
  global brightness shifts than raw intensity.

**3. Turn occupancy into traffic metrics.**

- *Presence:* binary per loop — is a car sitting on it.
- *Count / flow:* count occupied→empty→occupied transitions on a mid-lane loop =
  vehicles passing. Two loops a known distance apart → **speed** and direction.
- *Queue length (the demand signal we want):* count the consecutive occupied loops
  from the stop line back, or measure how far up the lane the foreground extends.

**4. Pseudocode (the whole thing).**

```python
bg = BackgroundModel()                 # e.g. cv2.createBackgroundSubtractorMOG2()
for frame in stream:
    fg = bg.apply(frame)               # foreground mask (0/255)
    fg = remove_shadows(fg, frame)     # optional: HSV/Lab shadow suppression
    queue = 0
    for loop in loops_from_stopline:   # ordered near → far
        frac = fg[loop.mask].mean() / 255.0
        occupied = frac > loop.threshold
        if not occupied:
            break                      # queue ends at first empty loop
        queue += 1
    report(approach, queue)            # ← feeds the same control logic
```

**5. Why it's not trivial (the failure modes).**

- **Shadows** (vehicle or cloud) read as foreground → false occupancy. Mitigate with
  HSV/Lab shadow removal (shadows change luma, not much chroma) or use edge density.
- **Lighting / day–night** — the background must adapt; headlight glare and
  reflections at night are hard.
- **Camera shake** (wind on a pole) misaligns the background → false foreground.
  Needs stabilisation or robust ROIs.
- **Background "absorbing" stopped cars** — adaptive models slowly learn a car
  stopped at red *into* the background, so it vanishes. Lower the learning rate, or
  freeze adaptation while a loop is occupied.
- **Perspective / occlusion** — tall vehicles or adjacent lanes spill into a loop;
  distant cars cover few pixels. Per-loop thresholds and placement matter.
- **Rain / snow / fog** — degrades classical CV too (this is the same domain-gap
  worry as [`brainstorming.md`](brainstorming.md) §1.1.2), arguably worse than a
  trained net.

**6. Compute cost.** Trivial — background subtraction + averaging pixels in a small
polygon is a few ops per ROI pixel. Hundreds of fps on a Pi; feasible on an MCU with
a small ROI and a simple background model. No training, no weights, no tensor arena.

### NN vs. virtual loop — the real trade

| | Virtual loop (classical CV) | Neural net (YOLO / distilled) |
|---|---|---|
| Compute | near-zero (MCU-OK) | SBC/NPU (detector) or MCU (tiny classifier) |
| Setup | per-camera tuning (thresholds, background) | train once, generalises |
| Robustness | brittle to shadow/night/weather | robust if trained on those conditions |
| Output | occupancy / queue length | per-vehicle boxes, classes, tracking |
| Failure mode | silent false positives (shadows) | misses on out-of-distribution scenes |

The NN earns its keep almost entirely on **robustness to night/rain/shadow** — the
exact conditions classical CV struggles with. If your deployment is daytime/clear,
a virtual-loop detector + the existing control logic may already get most of the win
at a fraction of the cost.

## 9. Hybrid (probably the smart answer)

Combine them: the **virtual loop** is the cheap, always-on detector; the **NN** is a
periodic verifier / recalibrator (matches the README's "recalibrate at night" idea),
or is only invoked when the loop result is ambiguous. You get MCU-class average cost
with NN-class robustness when it matters.

## 10. Decision checklist

1. **How fine does the timing need to be?** Coarse axis comparison → MCU tier;
   distance-weighted demand → SBC tier (§6).
2. **What conditions must it survive?** Daytime/clear → virtual loop may suffice;
   night/rain → NN or hybrid (§8–§9).
3. **Per-pole budget?** < $10 forces MCU + tiny model / VLD; $50–250 unlocks
   real-time `yolov8n` (§5).
4. **Where does perception run?** Per-pole (report one number) vs. one box per
   intersection on all 4 feeds (§7).
