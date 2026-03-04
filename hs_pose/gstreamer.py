import threading

import gi
import numpy as np

from hs_pose.constants import APPSINK_MAX_BUFFERS, FRAME_WAIT_TIMEOUT_MS, STREAM_CONNECT_TIMEOUT_MS


gi.require_version("Gst", "1.0")
from gi.repository import Gst


_GST_INIT_LOCK = threading.Lock()
_GST_INITIALIZED = False

DECODER_PRIORITY = ("nvh264dec", "d3d11h264dec", "avdec_h264")
DECODER_STAGES = {
    "nvh264dec": "nvh264dec max-display-delay=0 num-output-surfaces=1",
    "d3d11h264dec": "d3d11h264dec",
    "avdec_h264": "avdec_h264 max-threads=1",
}
TRANSPORT_PROTOCOLS = {"udp": "udp", "tcp": "tcp"}


def ensure_gst_initialized() -> None:
    global _GST_INITIALIZED
    with _GST_INIT_LOCK:
        if _GST_INITIALIZED:
            return
        Gst.init([])
        _GST_INITIALIZED = True


def build_pipeline_string(rtsp_url: str, transport: str, decoder: str) -> str:
    if transport not in TRANSPORT_PROTOCOLS:
        raise ValueError(f"Unsupported transport: {transport}")
    if decoder not in DECODER_STAGES:
        raise ValueError(f"Unsupported decoder: {decoder}")

    decoder_stage = DECODER_STAGES[decoder]
    common_tail = (
        "rtph264depay request-keyframe=true wait-for-keyframe=false ! "
        "h264parse ! "
        f"{decoder_stage} ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        f"appsink name=appsink emit-signals=false max-buffers={APPSINK_MAX_BUFFERS} "
        "leaky-type=downstream sync=false qos=false processing-deadline=0 "
        "enable-last-sample=false wait-on-eos=false"
    )

    if transport == "udp":
        return (
            f'rtspsrc location="{rtsp_url}" protocols=udp latency=0 buffer-mode=none '
            "drop-on-latency=true do-retransmission=false udp-buffer-size=262144 "
            "timeout=2000000 tcp-timeout=2000000 ! "
            + common_tail
        )

    return (
        f'rtspsrc location="{rtsp_url}" protocols=tcp latency=0 buffer-mode=none '
        "drop-on-latency=true tcp-timestamp=true timeout=2000000 tcp-timeout=2000000 ! "
        + common_tail
    )


class GstRtspCapture:
    def __init__(self, rtsp_url: str, preferred_transport: str = "tcp") -> None:
        self.rtsp_url = rtsp_url
        self.preferred_transport = preferred_transport.lower()
        self.pipeline = None
        self.appsink = None
        self.bus = None
        self.active_transport = None
        self.active_decoder = None
        self._prefetched_frame = None

    def start(self) -> None:
        ensure_gst_initialized()
        decoder = self._select_decoder()
        no_frame_seen = False

        for transport in self._transport_attempt_order():
            self.stop()
            self._start_pipeline(transport, decoder)
            frame = self.read_latest(timeout_ms=STREAM_CONNECT_TIMEOUT_MS)
            if frame is not None:
                self._prefetched_frame = frame
                self.active_transport = transport.upper()
                self.active_decoder = decoder
                return

            error_message = self._poll_bus_error()
            self.stop()

            if error_message is None:
                no_frame_seen = True

        if no_frame_seen:
            raise RuntimeError("Connected but no video frames were received.")
        raise RuntimeError("Unable to open RTSP stream.")

    def _transport_attempt_order(self):
        if self.preferred_transport == "udp":
            return ("udp", "tcp")
        if self.preferred_transport == "auto":
            return ("tcp", "udp")
        return ("tcp", "udp")

    def read_latest(self, timeout_ms: int = FRAME_WAIT_TIMEOUT_MS):
        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            return frame

        if self.appsink is None:
            return None

        timeout_ns = int(timeout_ms * 1_000_000)
        sample = self.appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            return None

        try:
            return self._sample_to_bgr_frame(sample)
        finally:
            sample = None

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        self.pipeline = None
        self.appsink = None
        self.bus = None
        self.active_transport = None
        self.active_decoder = None
        self._prefetched_frame = None

    def poll_runtime_error(self):
        error_message = self._poll_bus_error()
        if error_message is None:
            return None
        return f"Stream error: {error_message}"

    def _select_decoder(self) -> str:
        ensure_gst_initialized()
        for decoder in DECODER_PRIORITY:
            if Gst.ElementFactory.find(decoder):
                return decoder
        raise RuntimeError(
            "No supported H.264 decoder was found in the local GStreamer install."
        )

    def _start_pipeline(self, transport: str, decoder: str) -> None:
        pipeline_description = build_pipeline_string(self.rtsp_url, transport, decoder)
        pipeline = Gst.parse_launch(pipeline_description)
        appsink = pipeline.get_by_name("appsink")
        if appsink is None:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Unable to open RTSP stream.")

        self.pipeline = pipeline
        self.appsink = appsink
        self.bus = pipeline.get_bus()

        state_change = pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("Unable to open RTSP stream.")

    def _poll_bus_error(self):
        if self.bus is None:
            return None

        while True:
            message = self.bus.timed_pop_filtered(
                0, Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if message is None:
                return None

            if message.type == Gst.MessageType.ERROR:
                err, _debug = message.parse_error()
                return err.message
            if message.type == Gst.MessageType.EOS:
                return "EOS"

    @staticmethod
    def _sample_to_bgr_frame(sample):
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(map_info)

        return frame
