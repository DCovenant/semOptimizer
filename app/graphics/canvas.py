"""The image canvas: shows the frame, hosts lane polygons, the ignored region,
and the detection overlays. Pure view/scene management — no app logic."""
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen,
                           QPixmap)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsPathItem,
                               QGraphicsRectItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView)

from app.config import IGNORED_COLOR, INCOMING_PALETTE, OUTGOING_PALETTE
from app.graphics.lane import Lane


class Canvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.pixmap_item = None
        self._buf = None
        self.lanes = []
        self.det_items = []
        self.ignored_item = None
        self.ignored_label = None
        self.mode = "idle"            # "idle" | "add"
        self.current_lane = None

    # image -----------------------------------------------------------------
    def set_image(self, np_img):
        self._buf = np.ascontiguousarray(np_img)
        h, w = self._buf.shape[:2]
        qimg = QImage(self._buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg.copy())
        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)
        self.pixmap_item = self.scene.addPixmap(pm)
        self.pixmap_item.setZValue(0)
        self.setSceneRect(QRectF(pm.rect()))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.update_ignored()

    def image_size(self):
        if self._buf is None:
            return (0, 0)
        h, w = self._buf.shape[:2]
        return (w, h)

    # lane drawing ----------------------------------------------------------
    def start_lane(self, name, direction):
        self.current_lane = Lane(name, self, direction)
        self.lanes.append(self.current_lane)
        self.recolor_lanes()
        self.mode = "add"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def finish_lane(self):
        lane = self.current_lane
        self.mode = "idle"
        self.current_lane = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        if lane is not None and len(lane.handles) < 3:
            lane.remove()
            self.lanes.remove(lane)
            lane = None
        self.recolor_lanes()
        self.update_ignored()
        return lane

    def remove_lane(self, lane):
        lane.remove()
        self.lanes.remove(lane)
        self.recolor_lanes()
        self.update_ignored()

    def clear_lanes(self):
        for lane in list(self.lanes):
            lane.remove()
        self.lanes = []
        self.update_ignored()

    def recolor_lanes(self):
        idx = {"incoming": 0, "outgoing": 0}
        for lane in self.lanes:
            pal = INCOMING_PALETTE if lane.direction == "incoming" else OUTGOING_PALETTE
            lane.color = QColor(pal[idx[lane.direction] % len(pal)])
            idx[lane.direction] += 1
            lane.apply_color()

    # ignored / parked region = image − union(lanes) ------------------------
    def update_ignored(self):
        if self.ignored_item is not None:
            self.scene.removeItem(self.ignored_item)
            self.ignored_item = None
        if self._buf is None:
            return
        w, h = self.image_size()
        path = QPainterPath()
        path.addRect(QRectF(0, 0, w, h))
        for lane in self.lanes:
            if len(lane.handles) >= 3:
                lp = QPainterPath()
                lp.addPolygon(lane.polygon())
                lp.closeSubpath()
                path = path.subtracted(lp)
        self.ignored_item = QGraphicsPathItem(path)
        self.ignored_item.setBrush(QBrush(QColor(120, 120, 120, 110),
                                           Qt.BrushStyle.BDiagPattern))
        self.ignored_item.setPen(QPen(QColor(90, 90, 90, 140), 1))
        self.ignored_item.setZValue(2)         # above image, below lanes
        self.scene.addItem(self.ignored_item)
        if self.ignored_label is None:
            self.ignored_label = QGraphicsSimpleTextItem("ignored / parked")
            self.ignored_label.setBrush(QBrush(QColor(230, 230, 230)))
            self.ignored_label.setZValue(16)
            self.scene.addItem(self.ignored_label)
        self.ignored_label.setPos(8, max(6, h - 22))   # bottom-left; top-left is the semaphore

    # classification: returns the Lane, or None => ignored/parked -----------
    def classify(self, x, y):
        return next((l for l in self.lanes if l.contains(x, y)), None)

    # detections ------------------------------------------------------------
    def show_detections(self, detections):
        for it in self.det_items:
            self.scene.removeItem(it)
        self.det_items = []
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            lane = d.get("lane")
            col = lane.color if lane else IGNORED_COLOR
            rect = QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
            rect.setPen(QPen(col, 2))
            rect.setZValue(10)
            self.scene.addItem(rect)
            self.det_items.append(rect)
            fx, fy = d["foot"]
            dot = QGraphicsEllipseItem(fx - 5, fy - 5, 10, 10)
            dot.setBrush(QBrush(col))
            dot.setPen(QPen(QColor("black"), 1))
            dot.setZValue(12)
            self.scene.addItem(dot)
            self.det_items.append(dot)

    # interaction -----------------------------------------------------------
    def mousePressEvent(self, event):
        if self.mode == "add" and self.current_lane is not None \
                and event.button() == Qt.MouseButton.LeftButton:
            sp = self.mapToScene(event.position().toPoint())
            self.current_lane.add_vertex(sp)
            event.accept()
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item is not None:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
