# Ideas

- Objective: optimize semaphores by knowing how much cars are in each and use it has an input to the timing.

- Needs:
1. Detect if the car is in the lane that is relevant to the semaphore
2. Detect how much cars there are.

1.1. Need to know where the road starts and ends, sidewalks, and the limits of the lanes.
1.1.1. What if there are lane splitting lines that are not visible? The driver knows how it works because of previous knowledge of how to drive and signals in the entry of the road. Should this be entered in the question? adding the knowledge that a road is one way only or both way? Not rely only on the lane splitting?

1.1.2. Pretrained SegFormer (Cityscapes) confuses sidewalk with road when they share the same color, and the dashcam lane models (UFLD/CULane) perform badly. Root cause is **viewpoint/domain mismatch, not model size** — Cityscapes/CULane are ego-perspective (forward-facing, car height) while our cameras are pole-mounted looking down. Scaling the model up does NOT fix a domain gap and fights the edge-deployment size budget (<150KB, <100ms).
   Fix: fine-tune SegFormer-B0 on our own viewpoint using the semantic-segmentation ground truth that `carla_intersection.py` already captures (free, pixel-perfect labels in the exact camera geometry). The model leans on curb/elevation/context cues, not color — so giving it our actual geometry matters far more than capacity.
   Note: we may not even need road/sidewalk segmentation for the core task — drivable area and lane polygons come from the map prior (see 1.1.1). Only fine-tune segmentation if a downstream step genuinely needs per-pixel road/sidewalk (e.g. pedestrian-on-sidewalk filtering).
