"""Intersection planner — the app's first screen.

Top-down preset 4-way cross (PlanView) plus a side panel to configure each arm
(enable, camera image, expected lanes in/out) and launch the per-camera lane
calibration tool. Saves/loads the whole template as intersection.json.
"""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QCheckBox, QDockWidget, QFileDialog,
                               QFormLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPushButton, QSpinBox, QToolBar,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from app.config import YOLO_MODEL
from app.core.analysis import analyze_arm
from app.core.detection import load_model
from app.core.intersection import ARM_NAMES, ARMS, Intersection
from app.graphics.plan_view import PlanView
from app.ui.lane_window import LaneCalibrationWindow


class IntersectionWindow(QMainWindow):
    def __init__(self, template_path=None):
        super().__init__()
        self.setWindowTitle("Intersection Planner — 4-way")
        self.resize(1100, 740)
        self.template_path = "intersection.json"
        self.intersection = Intersection()
        self._lane_windows = []          # keep refs so they aren't GC'd
        self.model = None
        self.results = {}

        self.plan = PlanView(self.intersection, self)
        self.setCentralWidget(self.plan)
        self._build_toolbar()
        self._build_dock()
        self._build_results_dock()

        if template_path and os.path.exists(template_path):
            self.load_template(template_path)
        self.on_arm_selected("N")

    # ui scaffolding --------------------------------------------------------
    def _build_toolbar(self):
        tb = QToolBar("Main")
        self.addToolBar(tb)
        for text, slot in [("New", self.on_new),
                           ("Save template", self.on_save),
                           ("Load template", self.on_load)]:
            a = QAction(text, self)
            a.triggered.connect(slot)
            tb.addAction(a)
        tb.addSeparator()
        run = QAction("▶ Run YOLO (all arms)", self)
        run.triggered.connect(self.on_run_all)
        tb.addAction(run)

    def _build_dock(self):
        panel = QWidget()
        outer = QVBoxLayout(panel)
        self.arm_title = QLabel("Arm —")
        self.arm_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        outer.addWidget(self.arm_title)

        form = QFormLayout()
        self.enabled_cb = QCheckBox("arm exists")
        self.enabled_cb.stateChanged.connect(self.on_enabled_changed)
        form.addRow(self.enabled_cb)

        self.image_edit = QLineEdit()
        self.image_edit.setReadOnly(True)
        browse = QPushButton("Choose image…")
        browse.clicked.connect(self.on_choose_image)
        form.addRow("Camera image", self.image_edit)
        form.addRow("", browse)

        self.in_spin = QSpinBox(); self.in_spin.setRange(0, 8)
        self.out_spin = QSpinBox(); self.out_spin.setRange(0, 8)
        self.in_spin.valueChanged.connect(self.on_lanes_changed)
        self.out_spin.valueChanged.connect(self.on_lanes_changed)
        form.addRow("Lanes incoming", self.in_spin)
        form.addRow("Lanes outgoing", self.out_spin)

        self.status_label = QLabel("—")
        form.addRow("Status", self.status_label)
        outer.addLayout(form)

        self.calib_btn = QPushButton("Calibrate this arm →")
        self.calib_btn.clicked.connect(lambda: self.calibrate_arm(self._selected))
        outer.addWidget(self.calib_btn)

        outer.addStretch(1)
        self.overall_label = QLabel("")
        outer.addWidget(self.overall_label)

        dock = QDockWidget("Arm setup", self)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._selected = "N"

    def _build_results_dock(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        self.results_tree = QTreeWidget()
        self.results_tree.setColumnCount(3)
        self.results_tree.setHeaderLabels(["Arm / lane", "Dir", "Count"])
        lay.addWidget(self.results_tree)
        self.demand_overall = QLabel("intersection demand → —")
        self.demand_overall.setWordWrap(True)
        self.demand_overall.setStyleSheet("font-weight: bold;")
        lay.addWidget(self.demand_overall)

        dock = QDockWidget("Detection results", self)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    # run detection on every arm -------------------------------------------
    def _ensure_model(self):
        if self.model is None:
            self.statusBar().showMessage(f"Loading {YOLO_MODEL}…")
            QApplication.processEvents()
            self.model = load_model(YOLO_MODEL)
        return self.model

    def on_run_all(self):
        arms = [a for a in self.intersection.arms.values() if a.enabled and a.image]
        if not arms:
            QMessageBox.information(self, "Nothing to run",
                                    "Enable arms and set a camera image first.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            model = self._ensure_model()
            self.results = {a.key: analyze_arm(model, a) for a in arms}
        finally:
            QApplication.restoreOverrideCursor()

        self.plan.results = self.results
        self.plan.refresh()
        self._fill_results_tree()

        total = sum(r["total"] for r in self.results.values())
        self.statusBar().showMessage(
            f"Ran YOLO on {len(arms)} arm(s): {total} vehicles detected.")

    def _fill_results_tree(self):
        self.results_tree.clear()
        demand = {}
        for key in ARMS:
            res = self.results.get(key)
            if res is None:
                continue
            top = QTreeWidgetItem([
                f"{ARM_NAMES[key]} ({key})", "",
                f"{res['total']} cars"])
            for name, cnt in res["lanes"].items():
                top.addChild(QTreeWidgetItem(
                    [name, res["directions"].get(name, ""), str(cnt)]))
            top.addChild(QTreeWidgetItem(
                ["⌁ ignored/parked", "", str(res["ignored"])]))
            self.results_tree.addTopLevelItem(top)
            for ph, n in res["phase_demand"].items():
                demand[ph] = demand.get(ph, 0) + n
        self.results_tree.expandAll()
        txt = ", ".join(f"{p}: {n}" for p, n in sorted(demand.items())) or "—"
        self.demand_overall.setText(f"intersection demand (incoming) → {txt}")

    # arm selection / editing ----------------------------------------------
    def on_arm_selected(self, key):
        self._selected = key
        arm = self.intersection.arms[key]
        self.arm_title.setText(f"Arm {ARM_NAMES[key]} ({key})")
        for w in (self.enabled_cb, self.in_spin, self.out_spin):
            w.blockSignals(True)
        self.enabled_cb.setChecked(arm.enabled)
        self.image_edit.setText(arm.image or "")
        self.in_spin.setValue(arm.lanes_in)
        self.out_spin.setValue(arm.lanes_out)
        for w in (self.enabled_cb, self.in_spin, self.out_spin):
            w.blockSignals(False)
        self._refresh_status()
        self._update_overall()

    def _refresh_status(self):
        arm = self.intersection.arms[self._selected]
        if not arm.enabled:
            s = "off"
        elif arm.calibrated:
            s = f"✓ calibrated — {arm.lane_count} lanes drawn"
        elif arm.image:
            s = "image set — not calibrated"
        else:
            s = "no image yet"
        self.status_label.setText(s)
        self.calib_btn.setEnabled(arm.enabled)

    def _update_overall(self):
        done = sum(1 for a in self.intersection.arms.values() if a.enabled and a.calibrated)
        total = sum(1 for a in self.intersection.arms.values() if a.enabled)
        ok = "✓ all arms calibrated" if self.intersection.all_calibrated and total else ""
        self.overall_label.setText(f"Calibrated {done}/{total} arms.  {ok}")

    def on_enabled_changed(self):
        self.intersection.arms[self._selected].enabled = self.enabled_cb.isChecked()
        self.plan.refresh(); self._refresh_status(); self._update_overall()

    def on_lanes_changed(self):
        arm = self.intersection.arms[self._selected]
        arm.lanes_in = self.in_spin.value()
        arm.lanes_out = self.out_spin.value()
        self.plan.refresh()

    def on_choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose camera frame for this arm",
                                              "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.intersection.arms[self._selected].image = path
            self.image_edit.setText(path)
            self._refresh_status()

    # calibration drill-down ------------------------------------------------
    def calibrate_arm(self, key):
        arm = self.intersection.arms[key]
        if not arm.enabled:
            return
        if not arm.image:
            QMessageBox.information(self, "No image",
                                    f"Choose a camera image for arm {key} first.")
            return

        def on_done(calibration):
            arm.calibration = calibration
            arm.signal_state = calibration.get("signal_state", arm.signal_state)
            self.plan.refresh()
            if key == self._selected:
                self._refresh_status()
            self._update_overall()

        win = LaneCalibrationWindow(image_path=arm.image,
                                    calibration=arm.calibration,
                                    arm_label=key,
                                    on_done=on_done,
                                    signal_state=arm.signal_state)
        self._lane_windows.append(win)
        win.show()

    # template save / load --------------------------------------------------
    def _clear_results(self):
        self.results = {}
        self.plan.results = {}
        self.results_tree.clear()
        self.demand_overall.setText("intersection demand → —")

    def on_new(self):
        self.intersection = Intersection()
        self.plan.intersection = self.intersection
        self._clear_results()
        self.plan.refresh()
        self.on_arm_selected("N")

    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save intersection template",
                                              self.template_path, "JSON (*.json)")
        if path:
            self.intersection.save(path)
            self.template_path = path
            self.statusBar().showMessage(f"Saved template -> {path}")

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load intersection template",
                                              "", "JSON (*.json)")
        if path:
            self.load_template(path)

    def load_template(self, path):
        self.intersection = Intersection.load(path)
        self.plan.intersection = self.intersection
        self._clear_results()
        self.plan.refresh()
        self.template_path = path
        self.on_arm_selected("N")
        self.statusBar().showMessage(f"Loaded template from {path}")
