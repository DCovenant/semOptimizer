"""Intersection planner — the app's first screen.

Top-down preset 4-way cross (PlanView) plus a side panel to configure each arm
(enable, camera image, expected lanes in/out) and launch the per-camera lane
calibration tool. Saves/loads the whole template as intersection.json.
"""
import os

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
                               QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPushButton, QSpinBox, QToolBar,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from app.core.analysis_worker import AnalysisWorker
from app.core.carla_worker import CarlaWorker
from app.core.inference_service_manager import InferenceServiceManager
from app.graphics.camera_panel import CameraGridWidget

from app.config import SERVICE_MODEL, YOLO_MODEL
from app.core.analysis import analyze_arm
from app.core.detection import load_model
from app.core.intersection import ARM_NAMES, ARMS, Intersection
from app.graphics.plan_view import PlanView
from app.ui.lane_window import LaneCalibrationWindow

CAPTURES_DIR = "captures"


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

        self._carla_worker: CarlaWorker | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._inference_mgr = InferenceServiceManager(model=SERVICE_MODEL)
        self._captured_arms: set = set()   # arms whose reference frame was grabbed this connection
        self._latest_frames: dict = {}     # arm -> latest live RGB frame (for analysis)

        self.plan = PlanView(self.intersection, self)
        self.setCentralWidget(self.plan)
        self._build_toolbar()
        self._build_perception_dock()
        self._build_dock()
        self._build_results_dock()
        self._build_camera_dock()

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
        self._yolo_action = QAction("▶ Run YOLO (live)", self)
        self._yolo_action.triggered.connect(self.on_toggle_yolo)
        tb.addAction(self._yolo_action)

        tb.addSeparator()
        self._connect_action = QAction("⏵ Connect CARLA…", self)
        self._connect_action.triggered.connect(self.on_connect_carla)
        tb.addAction(self._connect_action)

        self._disconnect_action = QAction("⏹ Disconnect", self)
        self._disconnect_action.triggered.connect(self.on_disconnect_carla)
        self._disconnect_action.setEnabled(False)
        tb.addAction(self._disconnect_action)

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

    def _build_perception_dock(self):
        """Left-side live readout of how many cars are perceived per arm.

        Driven by AnalysisWorker results: `count` (vehicles detected in this
        arm's lanes) and the EMA-smoothed `weighted_demand` that feeds the
        timing logic. Populated by _update_perception on every analysis tick.
        """
        panel = QWidget()
        lay = QVBoxLayout(panel)
        title = QLabel("Perceived cars")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        lay.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self._perc_count = {}
        self._perc_demand = {}
        for row, key in enumerate(ARMS):
            card = QFrame()
            card.setFrameShape(QFrame.Shape.StyledPanel)
            cg = QGridLayout(card)
            cg.setContentsMargins(8, 4, 8, 4)
            name = QLabel(f"{ARM_NAMES[key]} ({key})")
            name.setStyleSheet("font-weight: bold;")
            cnt = QLabel("0")
            cnt.setStyleSheet("font-size: 22px; font-weight: bold; color: #2ca02c;")
            cnt.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            cars_lbl = QLabel("cars")
            cars_lbl.setStyleSheet("color: #888;")
            cars_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            dem = QLabel("demand 0.0")
            dem.setStyleSheet("color: #888;")
            cg.addWidget(name, 0, 0)
            cg.addWidget(cnt, 0, 1)
            cg.addWidget(dem, 1, 0)
            cg.addWidget(cars_lbl, 1, 1)
            grid.addWidget(card, row, 0)
            self._perc_count[key] = cnt
            self._perc_demand[key] = dem
        lay.addLayout(grid)
        lay.addStretch(1)

        dock = QDockWidget("Perception", self)
        dock.setWidget(panel)
        dock.setMinimumWidth(180)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _update_perception(self, results: dict):
        """Refresh the per-arm perceived-car counts + demand from analysis."""
        for key in ARMS:
            res = results.get(key)
            if res is None:
                continue
            self._perc_count[key].setText(str(res.get("count", 0)))
            self._perc_demand[key].setText(
                "demand %.1f" % res.get("weighted_demand", 0.0))

    def _reset_perception(self):
        for key in ARMS:
            self._perc_count[key].setText("0")
            self._perc_demand[key].setText("demand 0.0")

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

    # carla connection ------------------------------------------------------
    def _build_camera_dock(self):
        self._camera_panel = CameraGridWidget()
        dock = QDockWidget("CARLA Cameras", self)
        dock.setWidget(self._camera_panel)
        dock.setMinimumHeight(300)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

    def on_connect_carla(self):
        dlg = _CarlaConnectDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if not dlg.map_file or not os.path.exists(dlg.map_file):
            QMessageBox.warning(self, "No map file",
                                "Choose a valid OpenDRIVE (.xodr) map file.")
            return
        worker = CarlaWorker(
            host=dlg.host,
            port=dlg.port,
            map_file=dlg.map_file,
            vehicles=dlg.vehicles,
            demand_weights=_parse_demand(dlg.demand),
            ns_green=dlg.ns_green,
            ew_green=dlg.ew_green,
        )
        self._captured_arms = set()
        self._latest_frames = {}
        worker.frames_ready.connect(
            self._camera_panel.update_frames,
            Qt.ConnectionType.QueuedConnection)
        worker.frames_ready.connect(
            self._capture_reference_frames,
            Qt.ConnectionType.QueuedConnection)
        worker.frames_ready.connect(
            self._cache_latest_frames,
            Qt.ConnectionType.QueuedConnection)
        worker.vehicle_counts.connect(
            self._camera_panel.update_counts,
            Qt.ConnectionType.QueuedConnection)
        worker.phase_changed.connect(
            self.plan.update_semaphore_states,
            Qt.ConnectionType.QueuedConnection)
        worker.status_message.connect(
            self.statusBar().showMessage,
            Qt.ConnectionType.QueuedConnection)
        worker.error_occurred.connect(
            lambda msg: QMessageBox.critical(self, "CARLA Error", msg),
            Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(
            self._on_carla_worker_finished,
            Qt.ConnectionType.QueuedConnection)
        self._carla_worker = worker
        self._connect_action.setEnabled(False)
        self._disconnect_action.setEnabled(True)
        worker.start()

    def _capture_reference_frames(self, frames: dict):
        """Save the first live frame of each arm as its calibration reference.

        Runs once per arm per connection. Overwrites the stored screenshot
        (the static pole view is unchanged) but leaves any existing
        calibration intact.
        """
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        for key, arr in frames.items():
            if key in self._captured_arms or key not in self.intersection.arms:
                continue
            path = os.path.abspath(os.path.join(CAPTURES_DIR, f"{key}.png"))
            try:
                Image.fromarray(arr).save(path)
            except Exception as e:
                self.statusBar().showMessage(f"Could not save {key} frame: {e}")
                continue
            self._captured_arms.add(key)
            arm = self.intersection.arms[key]
            arm.image = path
            if key == self._selected:
                self.image_edit.setText(path)
                self._refresh_status()
        self.statusBar().showMessage(
            f"Captured reference frames: {', '.join(sorted(self._captured_arms))}")

    def _cache_latest_frames(self, frames: dict):
        """Keep the most recent live frame per arm for the analysis worker."""
        self._latest_frames.update(frames)

    def get_latest_frames(self) -> dict:
        """Thread-safe-enough snapshot of the latest frames for AnalysisWorker."""
        return dict(self._latest_frames)

    # live YOLO (continuous tracking) ---------------------------------------
    def on_toggle_yolo(self):
        if self._analysis_worker is not None:
            self._stop_analysis()
            return
        if self._carla_worker is None:
            QMessageBox.information(self, "Not connected",
                                    "Connect to CARLA before running live YOLO.")
            return
        calibrations = {k: a.calibration for k, a in self.intersection.arms.items()
                        if a.enabled and a.calibrated}
        if not calibrations:
            QMessageBox.information(self, "Nothing calibrated",
                                    "Calibrate at least one arm's incoming lane first.")
            return
        ok, msg = self._inference_mgr.ensure_running()
        self.statusBar().showMessage(msg)
        if not ok:
            QMessageBox.critical(self, "Inference service", msg)
            return
        worker = AnalysisWorker(self.get_latest_frames, calibrations,
                                self._inference_mgr.socket_path)
        worker.analysis_ready.connect(self._on_analysis,
                                      Qt.ConnectionType.QueuedConnection)
        worker.status_message.connect(self.statusBar().showMessage,
                                      Qt.ConnectionType.QueuedConnection)
        worker.error_occurred.connect(
            lambda msg: QMessageBox.warning(self, "Analysis error", msg),
            Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._on_analysis_finished,
                                Qt.ConnectionType.QueuedConnection)
        self._analysis_worker = worker
        self._yolo_action.setText("⏹ Stop YOLO")
        worker.start()

    def _stop_analysis(self):
        if self._analysis_worker is not None:
            self._analysis_worker.stop()
        self._yolo_action.setText("▶ Run YOLO (live)")

    def _on_analysis_finished(self):
        if self._analysis_worker is not None:
            self._analysis_worker.deleteLater()
            self._analysis_worker = None
        self._yolo_action.setText("▶ Run YOLO (live)")
        for key in ARMS:                 # drop stale overlays so they don't trail
            self._camera_panel.set_overlay(key, [])
        self._reset_perception()

    def _on_analysis(self, results: dict):
        """Per-arm tracked results → camera overlays + live weighted demand."""
        for key, res in results.items():
            self._camera_panel.set_overlay(key, res.get("tracks", []))
        self._update_perception(results)
        self._show_live_demand(results)

    def _show_live_demand(self, results: dict):
        self.results_tree.clear()
        demand = {}
        for key in ARMS:
            res = results.get(key)
            if res is None:
                continue
            top = QTreeWidgetItem([
                f"{ARM_NAMES[key]} ({key})", "",
                f"{res['count']} cars  ·  demand {res['weighted_demand']:.1f}"])
            for ph, w in sorted(res["phase_demand"].items()):
                top.addChild(QTreeWidgetItem([f"  {ph}", "", f"{w:.1f}"]))
                demand[ph] = demand.get(ph, 0.0) + w
            top.addChild(QTreeWidgetItem(["  ⌁ ignored/parked", "", str(res["ignored"])]))
            self.results_tree.addTopLevelItem(top)
        self.results_tree.expandAll()
        txt = ", ".join(f"{p}: {w:.1f}" for p, w in sorted(demand.items())) or "—"
        self.demand_overall.setText(f"live weighted demand → {txt}")

    def on_disconnect_carla(self):
        self._disconnect_action.setEnabled(False)
        self._stop_analysis()
        if self._carla_worker is not None:
            self._carla_worker.stop()

    def _on_carla_worker_finished(self):
        if self._carla_worker is not None:
            self._carla_worker.deleteLater()
            self._carla_worker = None
        self._camera_panel.clear()
        self._connect_action.setEnabled(True)
        self._disconnect_action.setEnabled(False)

    def closeEvent(self, event):
        # stop analysis first (it shuts its socket so recv unblocks), then CARLA.
        if self._analysis_worker is not None:
            self._analysis_worker.stop()
            self._analysis_worker.wait(5000)
        if self._carla_worker is not None:
            self._carla_worker.stop()
            self._carla_worker.wait(8000)   # sync-mode tick can be slow under load
        self._inference_mgr.stop()
        super().closeEvent(event)

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


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_demand(spec: str) -> dict:
    """'N:3,S:3,E:1,W:1' → {"N": 3.0, ...}. Missing arms default to 1.0."""
    weights = {"N": 1.0, "E": 1.0, "S": 1.0, "W": 1.0}
    if spec:
        for part in spec.split(","):
            k, _, v = part.partition(":")
            k = k.strip().upper()
            if k in weights and v:
                weights[k] = float(v)
    return weights


class _CarlaConnectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect to CARLA")
        self.setMinimumWidth(380)

        form = QFormLayout()

        self._host_edit = QLineEdit("localhost")
        form.addRow("Host", self._host_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(2000)
        form.addRow("Port", self._port_spin)

        self._map_edit = QLineEdit()
        _default_map = os.path.join(os.getcwd(), "maps", "loop_intersection.xodr")
        if os.path.exists(_default_map):
            self._map_edit.setText(_default_map)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_map)
        form.addRow("Map file (.xodr)", self._map_edit)
        form.addRow("", browse)

        self._vehicles_spin = QSpinBox()
        self._vehicles_spin.setRange(1, 200)
        self._vehicles_spin.setValue(15)
        form.addRow("Vehicles", self._vehicles_spin)

        self._demand_edit = QLineEdit("N:1,E:1,S:1,W:1")
        form.addRow("Demand (arm:weight)", self._demand_edit)

        self._ns_spin = QDoubleSpinBox()
        self._ns_spin.setRange(10.0, 300.0)
        self._ns_spin.setValue(30.0)
        self._ns_spin.setSuffix(" s")
        form.addRow("N/S green time", self._ns_spin)

        self._ew_spin = QDoubleSpinBox()
        self._ew_spin.setRange(10.0, 300.0)
        self._ew_spin.setValue(30.0)
        self._ew_spin.setSuffix(" s")
        form.addRow("E/W green time", self._ew_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse_map(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select OpenDRIVE map", "", "OpenDRIVE (*.xodr)")
        if path:
            self._map_edit.setText(path)

    @property
    def host(self) -> str:
        return self._host_edit.text().strip() or "localhost"

    @property
    def port(self) -> int:
        return self._port_spin.value()

    @property
    def map_file(self) -> str:
        return self._map_edit.text().strip()

    @property
    def vehicles(self) -> int:
        return self._vehicles_spin.value()

    @property
    def demand(self) -> str:
        return self._demand_edit.text().strip()

    @property
    def ns_green(self) -> float:
        return self._ns_spin.value()

    @property
    def ew_green(self) -> float:
        return self._ew_spin.value()
