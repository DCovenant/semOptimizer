"""2×2 live camera feed display widget.

Each cell shows the RGB frame from one signal-pole camera (N/E/S/W) with a
small overlay showing the arm name and current inbound vehicle queue count.
Frames are delivered as numpy RGB uint8 arrays via the update_frames() slot.
"""
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import (QColor, QFont, QImage, QPainter, QPen, QPixmap)
from PySide6.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

_LANE_COLOR = QColor(60, 220, 90)        # vehicle in the incoming lane
_IGNORED_COLOR = QColor(150, 150, 150)   # parked / outside the lane

_ARMS = ("N", "E", "S", "W")
_GRID = {"N": (0, 0), "E": (0, 1), "S": (1, 0), "W": (1, 1)}
_PH_W, _PH_H = 160, 90


class CameraGridWidget(QWidget):
    """Shows four live camera feeds in a 2×2 grid (N top-left, E top-right,
    S bottom-left, W bottom-right). Feeds are updated via Qt slots so this
    widget can safely receive signals from a background QThread."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #111;")

        layout = QGridLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        self._labels: dict = {}
        self._pixmaps: dict = {}
        self._last_counts: dict = {arm: 0 for arm in _ARMS}
        self._overlays: dict = {arm: [] for arm in _ARMS}   # arm -> [track dicts]

        for arm in _ARMS:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Expanding)
            lbl.setMinimumSize(_PH_W, _PH_H)
            lbl.setStyleSheet("background: #1a1a1a;")
            row, col = _GRID[arm]
            layout.addWidget(lbl, row, col)
            self._labels[arm] = lbl

        self.clear()

    # ── public slots ────────────────────────────────────────────────────────

    @Slot(object)
    def update_frames(self, frames: dict) -> None:
        """Receive a {arm: np.ndarray H×W×3 RGB uint8} dict and refresh labels."""
        for arm, arr in frames.items():
            if arm not in self._labels:
                continue
            h, w = arr.shape[:2]
            # arr must be C-contiguous (guaranteed by CarlaWorker)
            qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
            # .copy() detaches the QImage from the numpy buffer
            self._pixmaps[arm] = QPixmap.fromImage(qimg.copy())
            self._refresh_label(arm)

    @Slot(object)
    def update_counts(self, counts: dict) -> None:
        """Receive a {arm: int} vehicle count dict and refresh overlays."""
        self._last_counts.update(counts)
        for arm in counts:
            if arm in self._pixmaps:
                self._refresh_label(arm)

    def set_overlay(self, arm: str, tracks: list) -> None:
        """Store the latest tracked vehicles for an arm and redraw its tile."""
        if arm not in self._labels:
            return
        self._overlays[arm] = tracks or []
        if arm in self._pixmaps:
            self._refresh_label(arm)

    def clear(self) -> None:
        """Reset all cells to grey placeholders."""
        self._pixmaps.clear()
        for arm in _ARMS:
            self._last_counts[arm] = 0
            self._overlays[arm] = []
            ph = QPixmap(_PH_W, _PH_H)
            ph.fill(QColor(50, 50, 50))
            p = QPainter(ph)
            p.setPen(QColor(100, 100, 100))
            p.setFont(QFont("", 12))
            p.drawText(ph.rect(), Qt.AlignmentFlag.AlignCenter, arm)
            p.end()
            self._labels[arm].setPixmap(ph)

    # ── internals ───────────────────────────────────────────────────────────

    def _refresh_label(self, arm: str) -> None:
        label = self._labels[arm]
        pm = self._pixmaps.get(arm)
        if pm is None:
            return
        scaled = pm.scaled(label.size(),
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.FastTransformation)
        # track coords are in original-frame pixels; map them to the scaled tile
        scale = scaled.width() / pm.width() if pm.width() else 1.0
        label.setPixmap(self._with_overlay(scaled, arm, scale))

    def _with_overlay(self, pm: QPixmap, arm: str, scale: float = 1.0) -> QPixmap:
        out = pm.copy()
        p = QPainter(out)

        # tracked vehicles (boxes + id + distance), scaled into the tile
        self._draw_tracks(p, self._overlays.get(arm, []), scale)

        # arm label + queue count, top-left
        p.setFont(QFont("monospace", 10, QFont.Weight.Bold))
        text = f" {arm}  q:{self._last_counts.get(arm, 0):2d} "
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        p.fillRect(4, 4, tw + 4, th + 2, QColor(0, 0, 0, 160))
        p.setPen(QColor(255, 220, 0))
        p.drawText(6, 4 + th - 1, text)
        p.end()
        return out

    def _draw_tracks(self, p: QPainter, tracks: list, scale: float) -> None:
        p.setFont(QFont("monospace", 8))
        for t in tracks:
            in_lane = t.get("lane") is not None
            col = _LANE_COLOR if in_lane else _IGNORED_COLOR
            x1, y1, x2, y2 = (v * scale for v in t["box"])
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)        # outline only — no fill
            p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            fx, fy = t["foot"][0] * scale, t["foot"][1] * scale
            p.setBrush(col)                          # filled dot at the foot point
            p.drawEllipse(int(fx - 3), int(fy - 3), 6, 6)
            tid = t.get("id")
            label = "#%s" % tid if tid is not None else ""
            if in_lane and t.get("distance") is not None:
                label += ("%s d=%.2f" % (" " if label else "", t["distance"]))
            if label:
                p.setPen(col)
                p.drawText(int(x1), max(8, int(y1) - 2), label)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        for arm in _ARMS:
            if arm in self._pixmaps:
                self._refresh_label(arm)
