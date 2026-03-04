import cv2
from PyQt5 import QtGui


def to_qimage_bgr(frame) -> QtGui.QImage:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    return QtGui.QImage(
        rgb_frame.data,
        width,
        height,
        bytes_per_line,
        QtGui.QImage.Format_RGB888,
    ).copy()
