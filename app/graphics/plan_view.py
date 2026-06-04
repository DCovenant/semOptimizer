"""Top-down schematic of the 4-way intersection (the planner canvas).

Draws a preset cross: a central junction plus four arms (N/E/S/W). Each arm is a
clickable road tile — black edges, yellow dashed centreline — coloured by whether
that arm has been calibrated yet. Click selects an arm; double-click calibrates it.
"""
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QGraphicsRectItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView)

from app.core.intersection import ARM_NAMES
from app.graphics.semaphore import Semaphore

SCENE = 640                 # scene is SCENE x SCENE
JUNC = 150                  # central junction square side
EDGE_PEN = QPen(QColor("black"), 4)
CENTER_PEN = QPen(QColor("#f4c20d"), 3, Qt.PenStyle.DashLine)
DONE_FILL = QColor(76, 175, 80, 110)        # green-ish
PENDING_FILL = QColor(150, 150, 150, 90)    # gray
DISABLED_FILL = QColor(60, 60, 60, 40)
SELECTED_PEN = QPen(QColor("#1e88e5"), 4)


class ArmItem(QGraphicsRectItem):
    """A clickable road tile for one arm."""

    def __init__(self, key, rect, view):
        super().__init__(rect)
        self.key = key
        self.view = view
        self.setAcceptHoverEvents(True)
        self.setZValue(5)

    def mousePressEvent(self, event):
        self.view.select_arm(self.key)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.view.calibrate_arm(self.key)
        super().mouseDoubleClickEvent(event)


class PlanView(QGraphicsView):
    """Renders the intersection model and routes arm clicks back to the window."""

    def __init__(self, intersection, window):
        super().__init__()
        self.intersection = intersection
        self.window = window           # IntersectionWindow (has select_arm/calibrate_arm)
        self.selected = None
        self.results = {}              # {arm_key: analyze_arm result}
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSceneRect(0, 0, SCENE, SCENE)
        self.refresh()

    # routing ---------------------------------------------------------------
    def select_arm(self, key):
        self.selected = key
        self.window.on_arm_selected(key)
        self.refresh()

    def calibrate_arm(self, key):
        self.window.calibrate_arm(key)

    # geometry --------------------------------------------------------------
    def _arm_rect(self, key, road_w):
        c = SCENE / 2
        half = road_w / 2
        end = JUNC / 2
        if key == "N":
            return QRectF(c - half, 0, road_w, c - end)
        if key == "S":
            return QRectF(c - half, c + end, road_w, c - end)
        if key == "W":
            return QRectF(0, c - half, c - end, road_w)
        if key == "E":
            return QRectF(c + end, c - half, c - end, road_w)

    def _road_width(self, arm):
        lanes = max(1, arm.lanes_in) + max(1, arm.lanes_out)
        return max(60, min(JUNC, 24 * lanes))

    # drawing ---------------------------------------------------------------
    def refresh(self):
        self.scene.clear()
        self._semaphores_by_arm: dict = {}
        c = SCENE / 2

        # central junction
        j = self.scene.addRect(QRectF(c - JUNC / 2, c - JUNC / 2, JUNC, JUNC),
                               EDGE_PEN, QBrush(QColor(200, 200, 200, 120)))
        j.setZValue(4)

        for key, arm in self.intersection.arms.items():
            rect = self._arm_rect(key, self._road_width(arm))
            item = ArmItem(key, rect, self)
            if not arm.enabled:
                item.setBrush(QBrush(DISABLED_FILL))
                item.setPen(QPen(QColor(120, 120, 120), 1, Qt.PenStyle.DashLine))
            else:
                item.setBrush(QBrush(DONE_FILL if arm.calibrated else PENDING_FILL))
                item.setPen(SELECTED_PEN if key == self.selected else EDGE_PEN)
            self.scene.addItem(item)

            # yellow dashed centreline
            if arm.enabled:
                self._centerline(key, rect)

            # label + status
            self._label(key, arm, rect)

            # semaphore on the incoming side, near the junction (like the sketch)
            if arm.enabled:
                self._place_semaphore(key, arm)

        self.fitInView(self.sceneRect().adjusted(-20, -20, 20, 20),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def _place_semaphore(self, key, arm):
        c = SCENE / 2
        j = JUNC / 2
        r = 7
        w = 2 * r + 2 * max(2, round(r * 0.7))      # housing width
        h = 3 * (2 * r) + 4 * max(2, round(r * 0.7))  # housing height
        # right-hand side of incoming travel, just outside the junction
        if key == "N":      # incoming heads south; signal to the west, above junction
            x, y = c - 44, c - j - h - 4
        elif key == "S":    # incoming heads north; signal to the east, below junction
            x, y = c + 44 - w, c + j + 4
        elif key == "E":    # incoming heads west; signal to the north, right of junction
            x, y = c + j + 4, c - 44
        else:               # W — incoming heads east; signal to the south, left of junction
            x, y = c - j - w - 4, c + 44 - h

        def on_click(state, key=key):
            self.intersection.arms[key].signal_state = state

        sem = Semaphore(self.scene, x, y, r=r, state=arm.signal_state, on_click=on_click)
        self._semaphores_by_arm[key] = sem

    def _centerline(self, key, r):
        if key in ("N", "S"):
            x = r.center().x()
            line = self.scene.addLine(x, r.top(), x, r.bottom(), CENTER_PEN)
        else:
            y = r.center().y()
            line = self.scene.addLine(r.left(), y, r.right(), y, CENTER_PEN)
        line.setZValue(6)

    def _label(self, key, arm, r):
        status = "✓ calibrated" if arm.calibrated else ("— pending" if arm.enabled else "off")
        text = f"{ARM_NAMES[key]} ({key})\n{arm.lanes_in} in / {arm.lanes_out} out\n{status}"
        res = self.results.get(key)
        if res is not None:
            text += f"\n▶ {res['incoming']} in / {res['outgoing']} out  ({res['total']} cars)"
        t = QGraphicsSimpleTextItem(text)
        t.setFont(QFont("", 9))
        t.setBrush(QBrush(QColor("black")))
        t.setZValue(10)
        br = t.boundingRect()
        # place the label near the outer end of the arm
        if key == "N":
            t.setPos(r.center().x() - br.width() / 2, r.top() + 6)
        elif key == "S":
            t.setPos(r.center().x() - br.width() / 2, r.bottom() - br.height() - 6)
        elif key == "W":
            t.setPos(r.left() + 6, r.center().y() - br.height() / 2)
        else:
            t.setPos(r.right() - br.width() - 6, r.center().y() - br.height() / 2)
        self.scene.addItem(t)

    def update_semaphore_states(self, states: dict) -> None:
        """Update semaphore colours from a phase dict without a full scene rebuild.

        states: {"N": "green"|"yellow"|"red", "E": ..., "S": ..., "W": ...}
        Called from the main thread via a queued Qt signal connection.
        """
        for key, state in states.items():
            sem = self._semaphores_by_arm.get(key)
            if sem is not None:
                sem.set_state(state)
            if key in self.intersection.arms:
                self.intersection.arms[key].signal_state = state

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect().adjusted(-20, -20, 20, 20),
                       Qt.AspectRatioMode.KeepAspectRatio)
