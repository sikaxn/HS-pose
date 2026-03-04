import threading


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item = None

    def put(self, frame, timestamp_monotonic: float) -> None:
        with self._lock:
            self._item = (frame, timestamp_monotonic)

    def get_latest(self, after_timestamp: float | None = None):
        with self._lock:
            item = self._item

        if item is None:
            return None

        frame, timestamp_monotonic = item
        if after_timestamp is not None and timestamp_monotonic <= after_timestamp:
            return None

        return frame, timestamp_monotonic

    def clear(self) -> None:
        with self._lock:
            self._item = None
