# Smart Traffic Signal Optimizer

A computer vision-based traffic signal optimization system. Uses **4 cameras** — one mounted on each traffic light at a 4-way intersection — to independently detect vehicle queues per approach and collectively decide, in real time, whether the signal cycle should be adjusted.

## The Problem

At many intersections, traffic signals operate on fixed cycles. This leads to situations where one side has 10+ cars stopped at a red light while the green side is completely empty — unnecessary congestion.

## The Solution

Each traffic light pole at the intersection is equipped with a camera facing its incoming lane. Each camera independently runs vehicle detection and reports the queue size for its direction. The **4 outputs are fused** into a single decision that controls the signal cycle.

![Intersection layout](docs/images/main_road_idea.png)

## Architecture

### Multi-camera setup

The intersection has 4 approaches (North, South, East, West). Each approach has a traffic light with a camera mounted on top, facing the incoming traffic:

```
                    ▲ North approach
                    │
            ┌───────┤
            │  [CAM] │ ← Camera on North semaphore
            │       │       (faces North, sees incoming cars)
  ──────────┘       └──────────
  West approach                 East approach
  ──────────┐       ┌──────────
  [CAM] →   │       │   ← [CAM]
            │       │
            │ [CAM] │
            └───┤───┘
                │
                ▼ South approach
```

Each camera sees **only its own lane** — the vehicles waiting or approaching from that direction. This is more realistic than a single overhead camera because:
- Cameras mount directly on existing traffic light poles (no special infrastructure)
- Each camera has a clear, unobstructed view of its lane
- The system is modular — a single camera failure doesn't blind the whole intersection

### Decision pipeline

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ CAM North│  │ CAM South│  │ CAM East │  │ CAM West │
│  YOLOv8n │  │  YOLOv8n │  │  YOLOv8n │  │  YOLOv8n │
│  → count │  │  → count │  │  → count │  │  → count │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │              │
     └──────────────┴──────┬───────┴──────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   FUSION / DECISION │
                │                     │
                │  N:8  S:2  E:0  W:1 │
                │                     │
                │  → N/S axis: GREEN  │
                │  → E/W axis: RED    │
                └─────────────────────┘
```

### Recalibration

Each camera should be recalibrated when:
- The camera is moved or repositioned
- Road construction changes the layout
- Conditions change drastically (snow covering lane markings)

Recalibration can be scheduled to run periodically (e.g., once per day at night) as a safety net.

## Tech Stack

| Component | Technology | Size | Speed (CPU) |
|---|---|---|---|
| Road segmentation | SegFormer-B0 (Cityscapes) | ~14MB | ~2-5s per image |
| Vehicle detection | YOLOv8n (COCO) | ~6MB | ~100-500ms per frame |
| Image processing | OpenCV | - | - |
| Optimized inference | ONNX Runtime | - | 2-3x faster than PyTorch |

## Project Structure

```
semOptimizer/
├── README.md
├── docs/
│   └── images/
│       └── main_road_idea.png     # intersection layout diagram
├── config/
│   └── intersection_config.json   # camera positions + signal mapping
├── notebooks/
│   ├── road_edge_detection.ipynb  # experiment with road segmentation
│   └── traffic_signal_optimizer.ipynb  # experiment with vehicle detection
├── src/
│   ├── camera_detector.py        # per-camera YOLO detection + vehicle count
│   ├── signal_decision.py        # fuses 4 camera outputs → signal decision
│   └── visualize.py              # visualization and debug functions
├── carla_intersection.py          # CARLA simulation: 4-camera data generation
├── models/
│   └── yolov8n.onnx              # YOLO exported to ONNX
└── tests/
    └── test_with_sample_images/
```

## intersection_config.json

```json
{
  "cameras": {
    "north": {"position": "north_semaphore", "faces": "north_incoming"},
    "south": {"position": "south_semaphore", "faces": "south_incoming"},
    "east":  {"position": "east_semaphore",  "faces": "east_incoming"},
    "west":  {"position": "west_semaphore",  "faces": "west_incoming"}
  },
  "signal_groups": {
    "ns_axis": ["north", "south"],
    "ew_axis": ["east", "west"]
  },
  "calibration_date": "2026-05-23"
}
```

## Decision Logic

Each camera produces a vehicle count for its approach. The decision fuses all 4 counts (v1):

```
ns_demand = count_north + count_south
ew_demand = count_east  + count_west
```

- If current green axis has 0 vehicles and red axis has vehicles → **switch immediately**
- If `red_demand - green_demand >= threshold` → **suggest switch**
- Otherwise → **keep current cycle**

Example: N=8, S=2, E=0, W=1 → `ns_demand=10`, `ew_demand=1` → N/S axis gets green.

Future improvements:
- **Temporal smoothing** — require N consecutive frames to agree before switching
- **Tracking** — use `model.track()` to distinguish stopped vs moving vehicles
- **Minimum green time** — never switch before X seconds (safety)
- **Priority** — give more weight to buses/emergency vehicles
- **RL** — replace fixed rules with reinforcement learning trained on SynTraC
- **Per-camera failure handling** — degrade gracefully if one camera goes down

## Useful Datasets

- **SynTraC** — synthetic dataset (CARLA) for traffic signal control with RL, 86K+ images
- **UA-DETRAC** — 140K real traffic frames with 1.21M bounding boxes
- **Cityscapes** — urban segmentation (what trained SegFormer)
- **BDD100K** — 100K driving videos with segmentation

## Synthetic Data Generation (CARLA + Augmentations)

The models (YOLOv8, SegFormer) come pre-trained, but to validate and fine-tune on intersection scenarios, we can generate unlimited synthetic data with the CARLA simulator.

### Weather conditions in CARLA

CARLA exposes independent parameters via the Python API, allowing varied scenario creation:

```python
import carla

weather = carla.WeatherParameters(
    cloudiness=90.0,              # cloud cover (0-100%)
    precipitation=80.0,           # rain (0-100%)
    precipitation_deposits=60.0,  # puddles on the ground (0-100%)
    wind_intensity=70.0,          # wind (0-100%)
    fog_density=50.0,             # fog density (0-100%)
    fog_distance=10.0,            # fog distance (meters)
    wetness=100.0,                # wet road (0-100%)
    sun_altitude_angle=-30.0      # night (< 0 = below horizon)
)
world.set_weather(weather)
```

Available presets: ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, MidRainyNoon, HardRainNoon, SoftRainNoon, ClearSunset, CloudySunset, WetSunset, HardRainSunset, SoftRainSunset.

Night mode activates automatically when `sun_altitude_angle < 0`, turning on street lights and vehicle headlights.

### Post-processing with Augmentations

CARLA doesn't simulate camera artifacts (grain, motion blur). For that, apply augmentations in post-processing with Albumentations:

```python
import albumentations as A

augment = A.Compose([
    A.GaussNoise(var_limit=(10, 50)),                          # grain / sensor noise
    A.MotionBlur(blur_limit=7),                                 # motion blur
    A.RandomBrightnessContrast(p=0.5),                          # lighting variation
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3),        # extra fog
    A.RandomSunFlare(src_radius=100, p=0.3),                    # sun reflections
    A.ImageCompression(quality_lower=40, quality_upper=80),      # JPEG compression (cheap camera)
])

augmented = augment(image=frame)["image"]
```

### Generation pipeline

The CARLA + Albumentations combination covers virtually all real-world conditions:

| Condition | Source |
|---|---|
| Rain, fog, night, sunset | CARLA (native) |
| Puddles, wet road, wind | CARLA (native) |
| Grain / sensor noise | Albumentations (post) |
| Motion blur, defocus | Albumentations (post) |
| Sun reflections / lens flare | Albumentations (post) |
| Cheap camera compression | Albumentations (post) |
| Brightness / contrast variations | Albumentations (post) |

This allows generating large, varied datasets without leaving the desk, with perfect ground truth (bounding boxes, segmentation) generated automatically by the simulator.

## Model Optimization for Edge Deployment

The end goal is to compress the model as much as possible for lightweight inference. Training happens on the development machine (Ryzen 5 7500F + RX 9060 XT 16GB via ROCm), and the optimized model targets resource-constrained environments.

### Philosophy: Train big, compress maximally

The optimization pipeline follows a progressive compression chain. Each step reduces the model while preserving as much accuracy as possible.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT MACHINE                          │
│                  (Ryzen 5 7500F + RX 9060 XT)                   │
│                                                                 │
│   1. TRAIN TEACHER                                              │
│      Full YOLOv8n → large but accurate model                    │
│      ~6MB, FP32, ~3.2M parameters                               │
│                          │                                      │
│                          ▼                                      │
│   2. KNOWLEDGE DISTILLATION                                     │
│      Train "student" that mimics the teacher                    │
│      MobileNetV3-Small or custom CNN                            │
│      ~500KB-1MB, FP32, ~100-500K parameters                     │
│                          │                                      │
│                          ▼                                      │
│   3. PRUNING                                                    │
│      Remove neurons and connections that contribute little       │
│      50-80% weight reduction with <5% accuracy loss              │
│      ~200-500KB                                                 │
│                          │                                      │
│                          ▼                                      │
│   4. QUANTIZATION                                               │
│      FP32 (32 bits) → INT8 (8 bits)                             │
│      ~4x size reduction                                         │
│      ~50-150KB                                                  │
│                          │                                      │
│                          ▼                                      │
│   5. EXPORT                                                     │
│      Convert to optimized format (ONNX / TFLite)                │
│      Final model: ~50-150KB, INT8, ready for deployment         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1 — Train the Teacher (YOLOv8n)

The "teacher" model is YOLOv8n trained/fine-tuned on the development machine. It's too large for constrained environments but serves as an accuracy reference and to generate labels automatically.

```python
from ultralytics import YOLO

# Train on RX 9060 XT via ROCm
# Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
model = YOLO("yolov8n.pt")
model.train(data="intersection_dataset.yaml", epochs=100, imgsz=640, device=0)
```

The teacher generates "soft labels" — instead of "car" or "not car", it produces probabilities like "92% car, 5% truck, 3% background". These probabilities contain rich information about what the model learned.

### Step 2 — Knowledge Distillation

The student is a much smaller model that learns to mimic the teacher's probabilities, not the original data. This works better than training the student directly because the teacher's soft labels encode inter-class relationships that hard labels (0 or 1) don't capture.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCarCounter(nn.Module):
    """
    Minimal student model to classify: 0, 1-3, 4+ cars per zone.
    Architecture: input → 3 conv layers → global avg pool → 3 classes
    ~50-200K parameters (vs 3.2M in YOLOv8n)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 3)  # 3 classes: 0, 1-3, 4+

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def distillation_loss(student_logits, teacher_logits, true_labels, temperature=3.0, alpha=0.7):
    """
    Combines two learning signals:
    - soft_loss: mimic the teacher's probabilities (knowledge transfer)
    - hard_loss: get the real labels right (ground truth)
    alpha controls the relative weight: higher = more focus on the teacher
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean"
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(student_logits, true_labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

The key here is **problem simplification**. YOLOv8n does full object detection (bounding boxes + classes). The student only does classification: given a crop of an intersection zone, how many cars are there? This reduces complexity by orders of magnitude.

### Step 3 — Pruning

After distillation, many neurons in the student model contribute little to the final result. Pruning removes them.

```python
import torch.nn.utils.prune as prune

# Unstructured pruning — removes individual weights (more flexible)
for name, module in student_model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.5)  # remove 50%

# Structured pruning — removes entire filters (more hardware-efficient)
for name, module in student_model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.3, n=1, dim=0)

# Make pruning permanent (remove the mask and shrink the model)
for name, module in student_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.remove(module, "weight")
```

Pruning types and trade-offs:

| Type | What it removes | Typical reduction | Accuracy impact |
|---|---|---|---|
| Unstructured | Individual weights (scattered zeros) | 50-90% of weights | Low |
| Structured | Entire filters/channels | 30-70% of filters | Medium |
| Iterative | Cycles of prune → retrain → prune | Maximum possible | Controlled |

Iterative pruning is the most effective: prune 20% → retrain 10 epochs → prune another 20% → retrain → repeat. In each cycle, the model readapts to the new structure.

### Step 4 — Quantization

Convert from FP32 (floating point, 32 bits) to INT8 (integer, 8 bits). Edge devices are much faster with integer arithmetic.

```python
import tensorflow as tf

# Convert PyTorch → ONNX → TFLite (most robust path for edge deployment)

# 1. Export to ONNX
torch.onnx.export(student_model, dummy_input, "student.onnx", opset_version=13)

# 2. Convert ONNX → TFLite with INT8 quantization
# (using onnx2tf or ai-edge-torch)

# Post-training quantization with calibration dataset
converter = tf.lite.TFLiteConverter.from_saved_model("student_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Calibration dataset — the quantizer needs real samples
# to compute activation ranges for each layer
def representative_dataset():
    for image in calibration_images[:100]:
        yield [image.astype(np.float32)]

converter.representative_dataset = representative_dataset

# Force full INT8 (no float fallback)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("student_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Final size: {len(tflite_model) / 1024:.1f} KB")
```

Quantization impact:

| Precision | Size per weight | 200K params model | Speed |
|---|---|---|---|
| FP32 | 4 bytes | ~800 KB | Baseline |
| FP16 | 2 bytes | ~400 KB | ~1.5x |
| INT8 | 1 byte | ~200 KB | ~3-4x |
| INT4 | 0.5 bytes | ~100 KB | ~5-6x (limited) |

### Label generation pipeline

The teacher (YOLOv8) generates the labels for the student automatically:

```python
# 1. Run YOLOv8 on all training images
teacher = YOLO("best_teacher.pt")

# 2. For each image, count cars per zone and generate label
labels = []
for img_path in training_images:
    results = teacher(img_path, verbose=False)[0]
    car_count = sum(1 for box in results.boxes if int(box.cls[0]) in VEHICLE_CLASSES)

    if car_count == 0:
        label = 0     # empty
    elif car_count <= 3:
        label = 1     # few
    else:
        label = 2     # many

    labels.append((img_path, label))

# 3. Train the student with these labels
#    (+ distillation with teacher's soft labels)
```

### Success metrics

The model is ready for deployment when:

| Metric | Target |
|---|---|
| Model size (.tflite / .onnx) | < 150 KB |
| RAM required (tensor arena) | < 200 KB |
| Accuracy | > 85% on all 3 classes |
| Inference time | < 100ms |

### Useful tools

- **Netron** — visualize model architectures (ONNX, TFLite, PyTorch). Essential for understanding what you're compressing.
- **Edge Impulse** — web platform for training and deploying lightweight ML. Good for quick prototyping.
- **ONNX Runtime Mobile** — alternative to TFLite, easier to convert from PyTorch.
- **ai-edge-torch** — Google tool to convert PyTorch → TFLite directly.

## TODO

### CARLA Simulation
- [x] Set up CARLA 0.9.15 simulation environment
- [x] Create intersection data capture script (`carla_intersection.py`)
- [ ] Update `carla_intersection.py` to use 4 semaphore-mounted cameras instead of 1 overhead
- [ ] Generate dataset with weather variations (sun, rain, fog, night)
- [ ] Apply augmentations (grain, blur, compression) to generated dataset

### Core Pipeline
- [ ] Implement `camera_detector.py` — per-camera YOLO vehicle detection + count
- [ ] Implement `signal_decision.py` — fuse 4 camera outputs into signal decision
- [ ] Export models to ONNX
- [ ] Add temporal smoothing to decision logic
- [ ] Add tracking to distinguish stopped vs moving vehicles
- [ ] Per-camera failure handling (degrade gracefully)

### Model Optimization
- [ ] Define and train student model architecture (TinyCarCounter)
- [ ] Implement knowledge distillation pipeline (teacher → student)
- [ ] Apply iterative pruning with retraining
- [ ] Quantize model to INT8 with calibration dataset
- [ ] Export to optimized format (ONNX / TFLite)
- [ ] Benchmark: size < 150KB, latency < 100ms, accuracy > 85%

### Future
- [ ] Web interface for monitoring (FastAPI + Vue.js)
- [ ] Explore RL with SynTraC as an alternative to fixed rules
