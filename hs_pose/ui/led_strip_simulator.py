from __future__ import annotations

from typing import Iterable, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


class LedStripSimulatorWidget(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixels: list[Tuple[int, int, int]] = []
        self.setMinimumHeight(70)

    def set_pixels(self, pixels: Iterable[Tuple[int, int, int]]) -> None:
        self._pixels = list(pixels)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        painter.fillRect(self.rect(), QtGui.QColor(20, 20, 20))

        if not self._pixels:
            painter.setPen(QtGui.QColor(180, 180, 180))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "LED strip idle")
            return

        margin = 10
        spacing = 2
        led_height = max(12, self.height() - (margin * 2))
        available_width = max(1, self.width() - (margin * 2))
        count = max(1, len(self._pixels))
        led_width = max(1, (available_width - spacing * (count - 1)) / count)

        x_pos = float(margin)
        y_pos = float((self.height() - led_height) / 2)
        for red, green, blue in self._pixels:
            color = QtGui.QColor(red, green, blue)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(QtGui.QColor(50, 50, 50), 1))
            painter.drawRoundedRect(
                QtCore.QRectF(x_pos, y_pos, led_width, float(led_height)),
                2.0,
                2.0,
            )
            x_pos += led_width + spacing
