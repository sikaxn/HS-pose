import threading
import time

from PyQt5 import QtCore, QtGui

from hs_pose.constants import FRAME_WAIT_TIMEOUT_MS
from hs_pose.frame_buffer import LatestFrameBuffer
from hs_pose.gstreamer import GstRtspCapture
from hs_pose.ui.image_utils import to_qimage_bgr


class StreamWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    detected_changed = QtCore.pyqtSignal(str)
    waving_classes_changed = QtCore.pyqtSignal(dict)
    shirt_classes_changed = QtCore.pyqtSignal(dict)
    status_changed = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, rtsp_url: str, detector, transport: str = "tcp") -> None:
        super().__init__()
        self.rtsp_url = rtsp_url
        self.detector = detector
        self.frame_buffer = LatestFrameBuffer()
        self.capture = GstRtspCapture(rtsp_url, preferred_transport=transport)
        self._running = False
        self._capture_thread = None
        self._capture_fps = 0.0
        self._infer_fps = 0.0
        self._capture_count = 0
        self._infer_count = 0
        self._capture_report_started = time.perf_counter()
        self._infer_report_started = time.perf_counter()
        self._fps_lock = threading.Lock()

    def stop(self) -> None:
        self._running = False
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        self.wait(3000)

    def run(self) -> None:
        self._running = True
        try:
            self.status_changed.emit(f"Connecting to {self.rtsp_url}")
            self.capture.start()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="gst-capture", daemon=True
        )
        self._capture_thread.start()

        last_processed_timestamp = None

        while self._running:
            item = self.frame_buffer.get_latest(after_timestamp=last_processed_timestamp)
            if item is None:
                runtime_error = self.capture.poll_runtime_error()
                if runtime_error is not None:
                    self.error_occurred.emit(runtime_error)
                    break
                self.msleep(5)
                continue

            frame, frame_timestamp = item
            last_processed_timestamp = frame_timestamp

            try:
                (
                    annotated,
                    detection_count,
                    pose_count,
                    waving_count,
                    detected_items,
                    waving_by_class,
                    shirt_by_class,
                ) = self.detector.infer(frame)
            except Exception as exc:
                self.error_occurred.emit(f"Inference failed: {exc}")
                break

            self._record_infer_frame()
            self.frame_ready.emit(to_qimage_bgr(annotated))
            self.detected_changed.emit("\n".join(detected_items) if detected_items else "No detections")
            self.waving_classes_changed.emit(waving_by_class)
            self.shirt_classes_changed.emit(shirt_by_class)
            self.status_changed.emit(
                self._build_status(detection_count, pose_count, waving_count)
            )

        self._running = False
        self.capture.stop()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        self.status_changed.emit("Stopped")

    def _capture_loop(self) -> None:
        while self._running:
            runtime_error = self.capture.poll_runtime_error()
            if runtime_error is not None:
                self.error_occurred.emit(runtime_error)
                self._running = False
                return

            frame = self.capture.read_latest(timeout_ms=FRAME_WAIT_TIMEOUT_MS)
            if frame is None:
                continue

            self.frame_buffer.put(frame, time.monotonic())
            self._record_capture_frame()

    def _record_capture_frame(self) -> None:
        with self._fps_lock:
            self._capture_count += 1
            now = time.perf_counter()
            elapsed = now - self._capture_report_started
            if elapsed >= 1.0:
                self._capture_fps = self._capture_count / elapsed
                self._capture_count = 0
                self._capture_report_started = now

    def _record_infer_frame(self) -> None:
        with self._fps_lock:
            self._infer_count += 1
            now = time.perf_counter()
            elapsed = now - self._infer_report_started
            if elapsed >= 1.0:
                self._infer_fps = self._infer_count / elapsed
                self._infer_count = 0
                self._infer_report_started = now

    def _build_status(self, detection_count: int, pose_count: int, waving_count: int) -> str:
        with self._fps_lock:
            capture_fps = self._capture_fps
            infer_fps = self._infer_fps

        return (
            f"Running on {self.detector.device_name} | "
            f"Transport: {self.capture.active_transport} | "
            f"Decoder: {self.capture.active_decoder} | "
            f"Detections: {detection_count} | "
            f"Poses: {pose_count} | "
            f"Waving: {waving_count} | "
            f"Capture FPS: {capture_fps:.1f} | "
            f"Infer FPS: {infer_fps:.1f}"
        )
