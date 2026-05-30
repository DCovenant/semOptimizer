#!/usr/bin/env python3
"""Entry point for the lane-assignment desktop app.

Run from the project root with either:
    .venv312/bin/python -m app.main [image_path]
    .venv312/bin/python app/main.py [image_path]
"""
import os
import sys

# allow `python app/main.py` (not just `-m app.main`) by putting the repo root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

from app.ui.intersection_window import IntersectionWindow

DEFAULT_TEMPLATE = "intersection.json"


def main():
    app = QApplication(sys.argv)
    template = sys.argv[1] if len(sys.argv) > 1 else \
        (DEFAULT_TEMPLATE if os.path.exists(DEFAULT_TEMPLATE) else None)
    win = IntersectionWindow(template)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
