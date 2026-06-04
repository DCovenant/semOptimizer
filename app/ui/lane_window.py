"""Per-camera lane calibration window: canvas + YOLO + lanes/demand dock.

Used two ways:
  * standalone — open an image, draw lanes, save/load calibration JSON
  * from the intersection planner — opened for one arm with a preloaded image and
    calibration; the "Done" button hands the drawn calibration back via `on_done`.
"""
import os

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QHeaderView,
                               QLabel, QMainWindow, QMessageBox, QTableWidget,
                               QTableWidgetItem, QToolBar, QVBoxLayout, QWidget)

from app.config import DEFAULT_PHASE, YOLO_MODEL
from app.core.calibration import load_calibration, save_calibration
from app.core.detection import detect_vehicles, load_model
from app.graphics.canvas import Canvas
from app.graphics.lane import Lane
from app.graphics.semaphore import Semaphore


class LaneCalibrationWindow(QMainWindow):
    def __init__(self, image_path=None, calibration=None, arm_label=None,
                 on_done=None, signal_state="red"):
        super().__init__()
        title = "Lane Assignment — fixed camera"
        if arm_label:
            title = f"Calibrate arm {arm_label} — {title}"
        self.setWindowTitle(title)
        self.resize(1320, 780)
        self.model = None
        self.detections = []
        self.on_done = on_done
        self.signal_state = (calibration or {}).get("signal_state", signal_state) or "red"
        self.semaphore = None

        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)
        self._build_toolbar()
        self._build_dock()

        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
        if calibration:
            self._apply_calibration(calibration)
        self._place_semaphore()

    # semaphore (top-left of the canvas) ------------------------------------
    def _place_semaphore(self):
        if self.semaphore is not None:
            self.semaphore.remove()
        w = self.canvas.image_size()[0] or 400
        r = max(8, round(w * 0.012))
        self.semaphore = Semaphore(
            self.canvas.scene, 12, 12, r=r, state=self.signal_state,
            on_click=lambda state: setattr(self, "signal_state", state))

    # ui scaffolding --------------------------------------------------------
    def _build_toolbar(self):
        tb = QToolBar("Main")
        self.addToolBar(tb)

        def act(text, slot):
            a = QAction(text, self)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        act("Open image", self.on_open_image)
        tb.addSeparator()
        act("Add incoming lane", self.on_add_lane)
        act("Finish lane", self.on_finish_lane)
        act("Delete lane", self.on_delete_lane)
        tb.addSeparator()
        act("Run YOLO", self.on_run_yolo)
        tb.addSeparator()
        act("Save calib", self.on_save)
        act("Load calib", self.on_load)
        if self.on_done is not None:
            tb.addSeparator()
            act("✓ Done", self.on_done_clicked)

    def _build_dock(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Lane", "Phase", "Count"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellChanged.connect(self.on_cell_edited)
        lay.addWidget(self.table)

        self.demand_label = QLabel("incoming demand: —")
        self.demand_label.setWordWrap(True)
        lay.addWidget(self.demand_label)
        self.extra_label = QLabel("ignored/parked: 0")
        self.extra_label.setWordWrap(True)
        lay.addWidget(self.extra_label)

        dock = QDockWidget("Lanes / demand", self)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    # image -----------------------------------------------------------------
    def load_image(self, path):
        img = np.asarray(Image.open(path).convert("RGB"))
        self.canvas.set_image(img)
        self.image_path = path
        if self.semaphore is not None:           # rescale to the new frame
            self._place_semaphore()
        self.statusBar().showMessage(f"Loaded {path}  ({img.shape[1]}x{img.shape[0]})")

    def on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open camera frame", "",
                                              "Images (*.png *.jpg *.jpeg)")
        if path:
            self.load_image(path)

    # lanes -----------------------------------------------------------------
    def on_add_lane(self):
        if self.canvas.pixmap_item is None:
            QMessageBox.information(self, "No image", "Open a camera frame first.")
            return
        if self.canvas.mode == "add":
            self.on_finish_lane()
        name = f"lane_{len(self.canvas.lanes)}"
        self.canvas.start_lane(name, "incoming")
        self.statusBar().showMessage(
            f"Drawing {name}: left-click corners, then 'Finish lane'. "
            f"Anything outside an incoming lane is counted as parked/ignored.")

    def on_finish_lane(self):
        lane = self.canvas.finish_lane()
        if lane is None:
            self.statusBar().showMessage("Lane discarded (needs >= 3 corners).")
        else:
            self.statusBar().showMessage(f"Finished {lane.name} ({lane.direction}).")
        self.recompute()

    def on_delete_lane(self):
        if not self.canvas.lanes:
            return
        self.canvas.remove_lane(self.canvas.lanes[-1])
        self.recompute()

    # detection -------------------------------------------------------------
    def _ensure_model(self):
        if self.model is None:
            self.statusBar().showMessage(f"Loading {YOLO_MODEL}…")
            QApplication.processEvents()
            self.model = load_model(YOLO_MODEL)
        return self.model

    def on_run_yolo(self):
        if self.canvas._buf is None:
            QMessageBox.information(self, "No image", "Open a camera frame first.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            model = self._ensure_model()
            self.detections = detect_vehicles(model, self.canvas._buf)
        finally:
            QApplication.restoreOverrideCursor()
        self.recompute()
        self.statusBar().showMessage(f"Detected {len(self.detections)} vehicles.")

    # assignment + counts ---------------------------------------------------
    def recompute(self):
        for d in self.detections:
            d["lane"] = self.canvas.classify(*d["foot"])
        self.canvas.show_detections(self.detections)
        self.refresh_table()

    def refresh_table(self):
        self.table.blockSignals(True)
        counts = {l.name: 0 for l in self.canvas.lanes}
        ignored = 0
        for d in self.detections:
            if d.get("lane"):
                counts[d["lane"].name] += 1
            else:
                ignored += 1
        self.table.setRowCount(len(self.canvas.lanes))
        for r, lane in enumerate(self.canvas.lanes):
            self.table.setItem(r, 0, _ro_item(lane.name))
            self.table.setItem(r, 1, QTableWidgetItem(lane.phase))       # editable
            self.table.setItem(r, 2, _ro_item(str(counts[lane.name])))
        self.table.blockSignals(False)
        self.update_demand(counts, ignored)

    def on_cell_edited(self, row, col):
        if row >= len(self.canvas.lanes):
            return
        lane = self.canvas.lanes[row]
        if col == 1:   # phase
            lane.phase = self.table.item(row, 1).text().strip() or DEFAULT_PHASE
        self.recompute()

    def update_demand(self, counts, ignored):
        demand = {}
        for lane in self.canvas.lanes:        # every lane is incoming
            n = counts.get(lane.name, 0)
            demand[lane.phase] = demand.get(lane.phase, 0) + n
        txt = "  ".join(f"{p}: {n}" for p, n in sorted(demand.items())) if demand else "—"
        self.demand_label.setText(f"incoming demand → {txt}")
        self.extra_label.setText(f"ignored/parked: {ignored}")

    # calibration in/out ----------------------------------------------------
    def current_calibration(self):
        w, h = self.canvas.image_size()
        return {
            "image": getattr(self, "image_path", None),
            "size": [w, h],
            "lanes": {l.name: [[float(x), float(y)] for x, y in l.points()]
                      for l in self.canvas.lanes},
            "directions": {l.name: l.direction for l in self.canvas.lanes},
            "phases": {l.name: l.phase for l in self.canvas.lanes},
            "signal_state": self.signal_state,
        }

    def _apply_calibration(self, data):
        phases = data.get("phases", {})
        self.canvas.clear_lanes()
        for name, pts in data.get("lanes", {}).items():
            lane = Lane(name, self.canvas,
                        "incoming",                 # incoming-only calibration
                        phases.get(name, DEFAULT_PHASE))
            self.canvas.lanes.append(lane)
            for x, y in pts:
                lane.add_vertex(QPointF(x, y))
        self.canvas.recolor_lanes()
        self.canvas.update_ignored()
        self.recompute()

    # save / load (file) ----------------------------------------------------
    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration",
                                              "lane_calibration.json", "JSON (*.json)")
        if path:
            self.save(path)

    def save(self, path):
        c = self.current_calibration()
        save_calibration(path, c["image"], c["size"], c["lanes"],
                         c["directions"], c["phases"], c["signal_state"])
        self.statusBar().showMessage(f"Saved {len(self.canvas.lanes)} lanes -> {path}")

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration", "", "JSON (*.json)")
        if path:
            self._apply_calibration(load_calibration(path))
            self.statusBar().showMessage(f"Loaded lanes from {path}")

    # arm mode --------------------------------------------------------------
    def on_done_clicked(self):
        if self.on_done is not None:
            self.on_done(self.current_calibration())
        self.close()


def _ro_item(text):
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it
