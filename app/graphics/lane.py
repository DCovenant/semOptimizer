"""Editable lane polygon: a filled polygon with draggable corner handles."""
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QBrush, QColor, QPen, QPolygonF
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsPolygonItem, QGraphicsSimpleTextItem)

from app.config import DEFAULT_PHASE


class VertexHandle(QGraphicsEllipseItem):
    """A draggable corner of a lane polygon."""
    R = 6

    def __init__(self, lane):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self.lane = lane
        self.setBrush(QBrush(QColor("white")))
        self.setPen(QPen(QColor("black"), 1))
        self.setZValue(20)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            self.lane.rebuild()
        return super().itemChange(change, value)


class Lane:
    """One lane: filled polygon + draggable handles + label. Has a direction."""

    def __init__(self, name, canvas, direction="incoming", phase=DEFAULT_PHASE):
        self.name, self.canvas = name, canvas
        self.scene = canvas.scene
        self.direction = direction
        self.phase = phase
        self.color = QColor("#2ca02c")
        self.handles = []

        self.poly_item = QGraphicsPolygonItem()
        self.poly_item.setZValue(5)
        self.scene.addItem(self.poly_item)

        self.label = QGraphicsSimpleTextItem()
        self.label.setBrush(QBrush(QColor("white")))
        self.label.setZValue(15)
        self.scene.addItem(self.label)
        self.apply_color()

    # geometry --------------------------------------------------------------
    def add_vertex(self, scene_pos):
        h = VertexHandle(self)
        h.setPos(scene_pos)
        self.scene.addItem(h)
        self.handles.append(h)
        self.rebuild()

    def points(self):
        return [(h.scenePos().x(), h.scenePos().y()) for h in self.handles]

    def polygon(self):
        return QPolygonF([h.scenePos() for h in self.handles])

    def contains(self, x, y):
        return len(self.handles) >= 3 and \
            self.polygon().containsPoint(QPointF(x, y), Qt.FillRule.OddEvenFill)

    def rebuild(self):
        pts = [h.scenePos() for h in self.handles]
        self.poly_item.setPolygon(QPolygonF(pts))
        if pts:
            cx = sum(p.x() for p in pts) / len(pts)
            cy = sum(p.y() for p in pts) / len(pts)
            self.label.setPos(cx, cy)
        self.canvas.update_ignored()

    # appearance ------------------------------------------------------------
    def apply_color(self):
        self.poly_item.setPen(QPen(self.color, 2))
        fill = QColor(self.color); fill.setAlpha(55)
        self.poly_item.setBrush(QBrush(fill))
        self._update_label()

    def _update_label(self):
        tag = "in" if self.direction == "incoming" else "out"
        self.label.setText(f"{self.name} ({tag})")

    def set_direction(self, direction):
        self.direction = direction
        self._update_label()

    def remove(self):
        for h in self.handles:
            self.scene.removeItem(h)
        self.scene.removeItem(self.poly_item)
        self.scene.removeItem(self.label)
