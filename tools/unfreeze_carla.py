"""Recover a CARLA server stuck in synchronous mode.

If the app dies without running its teardown (crash, force-kill), the server is
left in synchronous mode: a stepped world only advances on world.tick(), so the
sim looks frozen and CarlaUE4.sh can hang on close. This connects, restores
asynchronous mode, and destroys leftover vehicles/walkers/sensors so the next
run starts clean — no server restart needed.

Run with the app's venv (it has the CARLA egg on its path via main.py's logic):

    .venv312/bin/python tools/unfreeze_carla.py [host] [port]
"""
import glob
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for egg in glob.glob(os.path.join(_PROJ, "carla", "PythonAPI", "carla", "dist",
                                  "carla-*-py3.7-*.egg")):
    sys.path.insert(0, egg)
    break

import carla  # noqa: E402


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    # Clear generated-map nav leftovers FIRST — this also fixes the case where
    # the server segfaults at boot ("Saving 128 tiles" then Signal 11) because
    # a stale Nav/OpenDriveMap.obj from a crashed session is re-processed at
    # startup. Works even when no server is reachable.
    nav = os.path.join(_PROJ, "carla", "CarlaUE4", "Content", "Carla", "Maps", "Nav")
    for name in ("OpenDriveMap.obj", "OpenDriveMap.bin"):
        path = os.path.join(nav, name)
        if os.path.exists(path):
            os.remove(path)
            print("removed stale nav cache:", name)

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    was_sync = settings.synchronous_mode
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    try:
        client.get_trafficmanager().set_synchronous_mode(False)
    except Exception:
        pass
    print("synchronous mode was %s -> restored async" % ("ON" if was_sync else "off"))

    leftovers = [a for a in world.get_actors()
                 if a.type_id.startswith(("vehicle.", "walker.", "sensor."))]
    if leftovers:
        client.apply_batch([carla.command.DestroyActor(a) for a in leftovers])
        print("destroyed %d leftover actor(s)" % len(leftovers))
    else:
        print("no leftover actors")


if __name__ == "__main__":
    main()
