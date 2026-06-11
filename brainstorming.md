# Ideas

- Objective: optimize semaphores by knowing how much cars are in each and use it has an input to the timing.

- Needs:
1. Detect if the car is in the lane that is relevant to the semaphore
2. Detect how much cars there are.

1.1. Need to know where the road starts and ends, sidewalks, and the limits of the lanes.
1.1.1. What if there are lane splitting lines that are not visible? The driver knows how it works because of previous knowledge of how to drive and signals in the entry of the road. Should this be entered in the question? adding the knowledge that a road is one way only or both way? Not rely only on the lane splitting?

- Take images of the road continuously when there is no car or is there to, in the background train a simple model to detect the road boundries, and then detect if the camera needs repositioning







● Done. The lane-crop inference size is now capped at CROP_MAX_SIDE = 960 (app/config.py:83), and no crop is ever
  upscaled.

  What changed in app/core/analysis_worker.py:

  - New _imgsz_for(crop, cap) helper — each crop is inferred at its own real size, rounded up to the multiple of 32
  YOLO needs, but never above the cap. Previously every lane crop ran at a fixed imgsz=1280 even if the crop was
  500px (upscaled!) or full-frame.
  - Lane crop: if its long side is ≥2× the cap (e.g., a near-full-frame 1920px ROI), it's decimated before shipping —
  same step mechanism the ped crop already used, and box coordinates map back through the existing ×step + origin
  path. The per-crop imgsz then enforces the hard 960 bound even when decimation alone can't reach it.
  - Ped crop: keeps its decimation, but now also gets the no-upscale treatment instead of a fixed imgsz=960.

  Verified with the app's Python (3.7 venv) against representative ROI sizes — worst-case GPU tensor drops from 1280
  to 960 long-side (~44% fewer pixels per lane crop ×4 arms), small ROIs run at native resolution instead of being
  inflated to 1280, and a full-frame ROI's socket payload shrinks 4× from the decimation. The trade-off to know
  about: lanes calibrated wider than 960px lose some effective resolution on distant queued cars (960 vs 1280); if
  recall on far cars degrades, raise CROP_MAX_SIDE — it's one knob.

  This cuts the inference side of the VRAM contention, but CARLA's renderer is still the other half — keep the
  hardened systemd-run launch (with MemorySwapMax/ManagedOOMSwap) and run tools/memwatch.sh on the next session so we
  can confirm GTT/shmem stays flat.

✻ Baked for 4m 10s

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯ ❯ user@user-B650-EAGLE-AX:~/Documents/projects/semOptimizer$ ./tools/memwatch.sh &                                   
  [1] 41514                                                                                                          
  user@user-B650-EAGLE-AX:~/Documents/projects/semOptimizer$ ./tools/memwatch.sh &                                   
  [2] 42117                                                                                                          
  user@user-B650-EAGLE-AX:~/Documents/projects/semOptimizer$ ./tools/memwatch.sh &                                   
  [3] 43435                                                                                                          
  user@user-B650-EAGLE-AX:~/Documents/projects/semOptimizer$                                                         
  , if i countinued to do the script the number would be different every time. the memory was again on the way to    
  the max, one thing must be made all of the persons and cars must all be imediatly spawned below the road and only  
  be teletransported to the positions, this must happen to know if the leak is from the spawning or something else, the memory was again on the way to the max, one thing must be made all of the persons
  and cars must all be imediatly spawned below the road and only be teletransported to the positions, this must
  happen to know if the leak is from the spawning or something else

