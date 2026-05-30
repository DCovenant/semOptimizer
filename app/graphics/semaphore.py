"""A traffic-signal ('semaphore') scene item: 3 stacked circles (red/yellow/green)
in a dark housing. Click it to cycle red → green → yellow. State is read/written
through the optional `on_click` callback."""
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem

STATES = ("red", "yellow", "green")
LIT = {"red": QColor("#e53935"), "yellow": QColor("#fdd835"), "green": QColor("#43a047")}
DIM = {"red": QColor(70, 25, 25), "yellow": QColor(70, 65, 25), "green": QColor(25, 70, 30)}
CYCLE = ("red", "green", "yellow")


class _Housing(QGraphicsRectItem):
    """Captures clicks for the whole semaphore (circles pass clicks through)."""
    def __init__(self, rect, sem):
        super().__init__(rect)
        self.sem = sem

    def mousePressEvent(self, event):
        self.sem.cycle()
        event.accept()


class Semaphore:
    def __init__(self, scene, x, y, r=7, state="red", on_click=None, z=30):
        self.scene = scene
        self.state = state if state in STATES else "red"
        self.on_click = on_click
        pad = max(2, round(r * 0.7))
        self.width = 2 * r + 2 * pad
        self.height = 3 * (2 * r) + 4 * pad

        self.housing = _Housing(QRectF(x, y, self.width, self.height), self)
        self.housing.setBrush(QBrush(QColor(30, 30, 30)))
        self.housing.setPen(QPen(QColor(10, 10, 10), 1))
        self.housing.setZValue(z)
        scene.addItem(self.housing)

        self.circles = {}
        for i, st in enumerate(STATES):
            cx = x + pad
            cy = y + pad + i * (2 * r + pad)
            c = QGraphicsEllipseItem(cx, cy, 2 * r, 2 * r)
            c.setPen(QPen(QColor(10, 10, 10), 1))
            c.setZValue(z + 1)
            c.setAcceptedMouseButtons(Qt.MouseButton.NoButton)   # click falls to housing
            scene.addItem(c)
            self.circles[st] = c

        self.set_state(self.state)

    def set_state(self, state):
        self.state = state if state in STATES else "red"
        for st, c in self.circles.items():
            c.setBrush(QBrush(LIT[st] if st == self.state else DIM[st]))

    def cycle(self):
        self.set_state(CYCLE[(CYCLE.index(self.state) + 1) % len(CYCLE)])
        if self.on_click:
            self.on_click(self.state)

    def remove(self):
        self.scene.removeItem(self.housing)
        for c in self.circles.values():
            self.scene.removeItem(c)
