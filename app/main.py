#!/usr/bin/env python3
"""Entry point for the lane-assignment desktop app.

Run from the project root with either:
    .venv312/bin/python -m app.main [image_path]
    .venv312/bin/python app/main.py [image_path]
"""
import glob
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_carla_env():
    """Re-exec once with a CARLA-safe environment (locale + .egg + .compatlibs).

    Three things must be true before CARLA is used, and all are easiest to set
    by relaunching the interpreter once (guarded by a flag):

    1. LC_NUMERIC=C — QApplication runs setlocale(LC_ALL, ""), which on a
       decimal-comma system locale (e.g. pt_PT) makes CARLA's C++ OpenDRIVE
       parser misread floats ("3.5" → 3.0) → "distance > 0.0" in get_map().
       Forcing LC_NUMERIC=C keeps float parsing dot-based even after Qt's call.
    2. The official .egg ahead of any pip-installed `carla` wheel (different
       libcarla builds; the wheel's parser also asserts on our map).
    3. .compatlibs on LD_LIBRARY_PATH so the .egg's libcarla can load
       (libtiff5, libjpeg62, …); must be set before libcarla is dlopened.
    """
    if os.environ.get("_SEMOPT_CARLA_ENV") == "1":
        return
    env = dict(os.environ)
    env["_SEMOPT_CARLA_ENV"] = "1"
    env["LC_NUMERIC"] = "C"
    env.pop("LC_ALL", None)
    eggs = glob.glob(os.path.join(
        _PROJ, "carla", "PythonAPI", "carla", "dist", "carla-*-py3.7-*.egg"))
    if eggs:  # egg first so it wins over any pip-installed carla
        env["PYTHONPATH"] = os.pathsep.join(
            [eggs[0]] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    compat = os.path.join(_PROJ, ".compatlibs")
    if os.path.isdir(compat):
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [compat] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_ensure_carla_env()

# allow `python app/main.py` (not just `-m app.main`) by putting the repo root on sys.path
sys.path.insert(0, _PROJ)

from PySide6.QtWidgets import QApplication

from app.ui.intersection_window import IntersectionWindow

DEFAULT_TEMPLATE = "intersection.json"


def main():
    app = QApplication(sys.argv)
    # QApplication() ran setlocale(LC_ALL, ""); make sure numeric parsing is
    # dot-based (C) again so CARLA's OpenDRIVE float parsing stays correct.
    import locale
    locale.setlocale(locale.LC_NUMERIC, "C")
    template = sys.argv[1] if len(sys.argv) > 1 else \
        (DEFAULT_TEMPLATE if os.path.exists(DEFAULT_TEMPLATE) else None)
    win = IntersectionWindow(template)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
