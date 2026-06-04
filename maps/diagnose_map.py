#!/usr/bin/env python3
"""Confirm whether CARLA's get_map() fails when driven off the main thread.

All map-string variants PASS on the main thread, so the suspect is now the
background thread the app uses. This runs the identical connect→generate→
get_map sequence (a) on the main thread and (b) inside a worker thread.

Usage (CARLA server running):
    LD_LIBRARY_PATH=.compatlibs \
      PYTHONPATH=carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg \
      .venv/bin/python maps/diagnose_map.py
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.append(os.path.join(ROOT, "carla", "PythonAPI", "carla"))

import carla  # noqa: E402

XODR = os.path.join(HERE, "loop_intersection.xodr")


def _params():
    return carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=50.0, wall_height=0.0,
        additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True)


def load_and_getmap(label):
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(30.0)
        xodr = open(XODR).read()
        client.generate_opendrive_world(xodr, _params())
        time.sleep(2.0)
        client.get_world().get_map()
        print(f"  [PASS] {label}")
    except Exception as e:
        print(f"  [FAIL] {label}  →  {e}")


def main():
    print("Same sequence, main thread vs worker thread:\n")

    load_and_getmap("MAIN thread")
    time.sleep(0.5)

    t = threading.Thread(target=load_and_getmap, args=("WORKER thread",))
    t.start()
    t.join()

    print("\nIf MAIN passes and WORKER fails, CARLA must run on the main "
          "thread (or in its own process) — not a QThread.")


if __name__ == "__main__":
    main()
