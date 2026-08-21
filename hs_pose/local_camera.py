import platform

import cv2

try:
    from PyQt5 import QtMultimedia
except ImportError:
    QtMultimedia = None


def _camera_backend() -> int:
    """Prefer DirectShow on Windows, where UVC and OBS virtual cameras register."""
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def available_cameras(maximum: int = 10) -> list[tuple[int, str]]:
    """Return DirectShow camera indices paired with their system display names."""
    if QtMultimedia is not None:
        named_cameras = QtMultimedia.QCameraInfo.availableCameras()
        if named_cameras:
            return [
                (index, camera.description() or f"Camera {index}")
                for index, camera in enumerate(named_cameras)
            ]

    cameras = []
    backend = _camera_backend()
    for index in range(maximum):
        capture = cv2.VideoCapture(index, backend)
        try:
            if capture.isOpened():
                ok, _frame = capture.read()
                if ok:
                    cameras.append((index, f"Camera {index}"))
        finally:
            capture.release()
    return cameras


class LocalCameraCapture:
    """OpenCV-backed capture for UVC, OBS Virtual Camera, and similar devices."""

    active_transport = "Camera"
    active_decoder = "OpenCV"

    def __init__(self, camera_index: int) -> None:
        self.camera_index = int(camera_index)
        self.capture = None
        self._prefetched_frame = None

    def start(self) -> None:
        self.capture = cv2.VideoCapture(self.camera_index, _camera_backend())
        if not self.capture.isOpened():
            self.stop()
            raise RuntimeError(f"Unable to open camera {self.camera_index}.")

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.stop()
            raise RuntimeError(f"Camera {self.camera_index} did not provide video frames.")
        self._prefetched_frame = frame

    def read_latest(self, timeout_ms: int = 0):
        del timeout_ms
        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            return frame
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        return frame if ok else None

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self._prefetched_frame = None

    def poll_runtime_error(self):
        return None
