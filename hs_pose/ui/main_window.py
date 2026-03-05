from PyQt5 import QtCore, QtGui, QtWidgets

from hs_pose.config import load_config, save_config
from hs_pose.constants import DEFAULT_CONFIDENCE, DEFAULT_RTSP_TRANSPORT, MODEL_PATH
from hs_pose.detector import YoloV5Detector
from hs_pose.stream_worker import StreamWorker


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HS Pose RTSP Viewer")
        self.resize(1280, 800)

        self.config = load_config()
        self.detector = YoloV5Detector(MODEL_PATH)
        self.stream_worker = None
        self.detector.set_confidence(self.config["confidence"])

        self._build_ui()
        self._set_idle_frame()
        self.status_label.setText(
            f"Model: {MODEL_PATH.name} | Device: {self.detector.device_name} | "
            f"Confidence: {self.detector.confidence:.2f}"
        )

    def _build_ui(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        controls_layout = QtWidgets.QHBoxLayout()

        rtsp_label = QtWidgets.QLabel("RTSP URL")
        self.rtsp_input = QtWidgets.QLineEdit(self.config["rtsp_url"])
        self.rtsp_input.setPlaceholderText("rtsp://host:port/path")
        transport_label = QtWidgets.QLabel("Transport")
        self.transport_input = QtWidgets.QComboBox()
        self.transport_input.addItem("TCP", "tcp")
        self.transport_input.addItem("Auto", "auto")
        self.transport_input.addItem("UDP", "udp")
        selected_transport = self.config.get("transport", DEFAULT_RTSP_TRANSPORT)
        selected_index = self.transport_input.findData(selected_transport)
        if selected_index >= 0:
            self.transport_input.setCurrentIndex(selected_index)
        confidence_label = QtWidgets.QLabel("Confidence")
        self.confidence_input = QtWidgets.QDoubleSpinBox()
        self.confidence_input.setRange(0.0, 1.0)
        self.confidence_input.setSingleStep(0.05)
        self.confidence_input.setDecimals(2)
        self.confidence_input.setValue(self.config.get("confidence", DEFAULT_CONFIDENCE))

        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)

        controls_layout.addWidget(rtsp_label)
        controls_layout.addWidget(self.rtsp_input, 1)
        controls_layout.addWidget(transport_label)
        controls_layout.addWidget(self.transport_input)
        controls_layout.addWidget(confidence_label)
        controls_layout.addWidget(self.confidence_input)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet(
            "background-color: #111; color: #ddd; border: 1px solid #333;"
        )
        self.detected_title = QtWidgets.QLabel("Detected")
        self.detected_title.setStyleSheet("font-weight: 600; color: #ddd;")
        self.detected_text = QtWidgets.QPlainTextEdit()
        self.detected_text.setReadOnly(True)
        self.detected_text.setMinimumWidth(260)
        self.detected_text.setStyleSheet(
            "background-color: #111; color: #ddd; border: 1px solid #333;"
        )
        self.detected_text.setPlainText("No detections")

        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setStyleSheet("padding: 6px 0; color: #ddd;")

        main_layout.addLayout(controls_layout)
        content_layout = QtWidgets.QHBoxLayout()
        content_layout.addWidget(self.video_label, 1)
        detected_layout = QtWidgets.QVBoxLayout()
        detected_layout.addWidget(self.detected_title)
        detected_layout.addWidget(self.detected_text, 1)
        content_layout.addLayout(detected_layout)
        main_layout.addLayout(content_layout, 1)
        main_layout.addWidget(self.status_label)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)

    def _set_idle_frame(self) -> None:
        self.video_label.setText("No video")
        self.video_label.setPixmap(QtGui.QPixmap())

    def start_stream(self) -> None:
        rtsp_url = self.rtsp_input.text().strip()
        if not rtsp_url:
            QtWidgets.QMessageBox.warning(self, "Missing URL", "Enter an RTSP URL.")
            return

        confidence = self.confidence_input.value()
        transport = self.transport_input.currentData() or DEFAULT_RTSP_TRANSPORT
        self.detector.set_confidence(confidence)
        self.config["rtsp_url"] = rtsp_url
        self.config["confidence"] = confidence
        self.config["transport"] = transport
        save_config(self.config)

        self.stop_stream()

        self.stream_worker = StreamWorker(rtsp_url, self.detector, transport=transport)
        self.stream_worker.frame_ready.connect(self.update_frame)
        self.stream_worker.detected_changed.connect(self.detected_text.setPlainText)
        self.stream_worker.status_changed.connect(self.status_label.setText)
        self.stream_worker.error_occurred.connect(self.handle_stream_error)
        self.stream_worker.finished.connect(self.on_stream_finished)
        self.stream_worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_stream(self) -> None:
        if self.stream_worker and self.stream_worker.isRunning():
            self.stream_worker.stop()

    def update_frame(self, image: QtGui.QImage) -> None:
        pixmap = QtGui.QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def handle_stream_error(self, message: str) -> None:
        self.status_label.setText(message)
        if self.stream_worker and self.stream_worker.isRunning():
            self.stream_worker.stop()
        QtWidgets.QMessageBox.critical(self, "Stream Error", message)

    def on_stream_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.detected_text.setPlainText("No detections")
        self.status_label.setText(
            f"Model: {MODEL_PATH.name} | Device: {self.detector.device_name} | "
            f"Confidence: {self.detector.confidence:.2f}"
        )
        if self.video_label.pixmap() is None or self.video_label.pixmap().isNull():
            self._set_idle_frame()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_stream()
        super().closeEvent(event)
