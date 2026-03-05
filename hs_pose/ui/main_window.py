import time

from PyQt5 import QtCore, QtGui, QtWidgets

from hs_pose.config import load_config, save_config
from hs_pose.constants import DEFAULT_CONFIDENCE, DEFAULT_RTSP_TRANSPORT, MODEL_PATH
from hs_pose.detector import YoloV5Detector
from hs_pose.energy_game import CLOTH_ORDER, EnergyGameEngine, GameParams
from hs_pose.led_test_patterns import TEST_PALETTES, build_test_pixels
from hs_pose.sacn_sender import SacnSender
from hs_pose.stream_worker import StreamWorker
from hs_pose.ui.led_strip_simulator import LedStripSimulatorWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HS Pose RTSP Viewer")
        self.resize(1280, 860)

        self.config = load_config()
        self.detector = YoloV5Detector(MODEL_PATH)
        self.stream_worker = None
        self.detector.set_confidence(self.config["confidence"])

        self.latest_waving_counts = {cloth: 0 for cloth in CLOTH_ORDER}
        self.latest_shirt_counts = {cloth: 0 for cloth in CLOTH_ORDER}
        self.energy_engine = EnergyGameEngine(self._game_params_from_config())
        self.sacn_sender = SacnSender()
        self._game_timer = QtCore.QTimer(self)
        self._game_timer.timeout.connect(self._tick_game)
        self._last_game_tick = time.monotonic()
        self._started_at = time.monotonic()

        self._build_ui()
        self._set_idle_frame()
        self._apply_game_params(save=False)
        self._apply_output_params(save=False)
        self._game_timer.start()
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
            "background-color: #111; color: #ffffff; border: 1px solid #333;"
        )

        self.detected_title = QtWidgets.QLabel("Detected")
        self.detected_title.setStyleSheet("font-weight: 600; color: #000;")
        self.detected_text = QtWidgets.QPlainTextEdit()
        self.detected_text.setReadOnly(True)
        self.detected_text.setMinimumWidth(260)
        self.detected_text.setStyleSheet(
            "background-color: #fff; color: #000; border: 1px solid #999;"
        )
        self.detected_text.setPlainText("No detections")

        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setStyleSheet("padding: 6px 0; color: #000;")

        main_layout.addLayout(controls_layout)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.addWidget(self.video_label, 1)
        detected_layout = QtWidgets.QVBoxLayout()
        detected_layout.addWidget(self.detected_title)
        detected_layout.addWidget(self.detected_text, 1)
        content_layout.addLayout(detected_layout)
        main_layout.addLayout(content_layout, 1)

        game_group = QtWidgets.QGroupBox("Energy Game / LED Strip Simulator")
        game_group.setStyleSheet("QGroupBox { color: #000; }")
        game_layout = QtWidgets.QVBoxLayout(game_group)

        params_layout = QtWidgets.QHBoxLayout()
        game_cfg = self.config.get("game", {})

        self.pixel_count_input = QtWidgets.QSpinBox()
        self.pixel_count_input.setRange(8, 600)
        self.pixel_count_input.setValue(int(game_cfg.get("pixel_count", 120)))

        self.charge_rate_input = QtWidgets.QDoubleSpinBox()
        self.charge_rate_input.setRange(0.0, 10.0)
        self.charge_rate_input.setDecimals(2)
        self.charge_rate_input.setSingleStep(0.05)
        self.charge_rate_input.setValue(float(game_cfg.get("charge_rate", 1.0)))

        self.active_decay_input = QtWidgets.QDoubleSpinBox()
        self.active_decay_input.setRange(0.0, 10.0)
        self.active_decay_input.setDecimals(2)
        self.active_decay_input.setSingleStep(0.05)
        self.active_decay_input.setValue(float(game_cfg.get("active_decay_rate", 0.15)))

        self.idle_decay_input = QtWidgets.QDoubleSpinBox()
        self.idle_decay_input.setRange(0.0, 10.0)
        self.idle_decay_input.setDecimals(2)
        self.idle_decay_input.setSingleStep(0.05)
        self.idle_decay_input.setValue(float(game_cfg.get("idle_decay_rate", 0.35)))

        self.idle_drain_enabled_input = QtWidgets.QCheckBox("Enable Idle Drain")
        self.idle_drain_enabled_input.setChecked(
            bool(game_cfg.get("idle_drain_enabled", True))
        )

        self.takeover_decay_enabled_input = QtWidgets.QCheckBox("Enable Takeover Decay")
        self.takeover_decay_enabled_input.setChecked(
            bool(game_cfg.get("takeover_decay_enabled", True))
        )

        self.tick_hz_input = QtWidgets.QSpinBox()
        self.tick_hz_input.setRange(1, 120)
        self.tick_hz_input.setValue(int(game_cfg.get("tick_hz", 30)))

        self.apply_game_button = QtWidgets.QPushButton("Apply")
        self.reset_energy_button = QtWidgets.QPushButton("Reset Energy")

        params_layout.addWidget(QtWidgets.QLabel("Pixels"))
        params_layout.addWidget(self.pixel_count_input)
        params_layout.addWidget(QtWidgets.QLabel("Charge Speed"))
        params_layout.addWidget(self.charge_rate_input)
        params_layout.addWidget(QtWidgets.QLabel("Takeover Speed"))
        params_layout.addWidget(self.active_decay_input)
        params_layout.addWidget(QtWidgets.QLabel("Idle Drain"))
        params_layout.addWidget(self.idle_decay_input)
        params_layout.addWidget(self.idle_drain_enabled_input)
        params_layout.addWidget(self.takeover_decay_enabled_input)
        params_layout.addWidget(QtWidgets.QLabel("React Hz"))
        params_layout.addWidget(self.tick_hz_input)
        params_layout.addWidget(self.apply_game_button)
        params_layout.addWidget(self.reset_energy_button)
        params_layout.addStretch(1)

        game_layout.addLayout(params_layout)

        output_layout = QtWidgets.QHBoxLayout()
        sacn_cfg = self.config.get("sacn", {})

        self.sacn_enabled_input = QtWidgets.QCheckBox("Enable sACN (E1.31)")
        self.sacn_enabled_input.setChecked(bool(sacn_cfg.get("enabled", False)))
        self.sacn_ip_input = QtWidgets.QLineEdit(str(sacn_cfg.get("receiver_ip", "")))
        self.sacn_ip_input.setPlaceholderText("sACN receiver IP")
        self.sacn_universe_input = QtWidgets.QSpinBox()
        self.sacn_universe_input.setRange(1, 63999)
        self.sacn_universe_input.setValue(int(sacn_cfg.get("universe", 1)))
        self.sacn_start_input = QtWidgets.QSpinBox()
        self.sacn_start_input.setRange(1, 512)
        self.sacn_start_input.setValue(int(sacn_cfg.get("start_address", 1)))
        self.test_mode_enabled_input = QtWidgets.QCheckBox("LED Test Mode")
        self.test_mode_enabled_input.setChecked(bool(sacn_cfg.get("test_mode_enabled", False)))
        self.test_palette_input = QtWidgets.QComboBox()
        for palette in TEST_PALETTES:
            self.test_palette_input.addItem(palette)
        selected_palette = str(sacn_cfg.get("test_palette", "Manual RGB"))
        selected_palette_index = self.test_palette_input.findText(selected_palette)
        if selected_palette_index >= 0:
            self.test_palette_input.setCurrentIndex(selected_palette_index)

        self.test_r_input = QtWidgets.QSpinBox()
        self.test_r_input.setRange(0, 255)
        self.test_r_input.setValue(int(sacn_cfg.get("test_r", 255)))
        self.test_g_input = QtWidgets.QSpinBox()
        self.test_g_input.setRange(0, 255)
        self.test_g_input.setValue(int(sacn_cfg.get("test_g", 64)))
        self.test_b_input = QtWidgets.QSpinBox()
        self.test_b_input.setRange(0, 255)
        self.test_b_input.setValue(int(sacn_cfg.get("test_b", 64)))

        output_layout.addWidget(self.sacn_enabled_input)
        output_layout.addWidget(QtWidgets.QLabel("Receiver IP"))
        output_layout.addWidget(self.sacn_ip_input)
        output_layout.addWidget(QtWidgets.QLabel("Universe"))
        output_layout.addWidget(self.sacn_universe_input)
        output_layout.addWidget(QtWidgets.QLabel("Start Addr"))
        output_layout.addWidget(self.sacn_start_input)
        output_layout.addWidget(self.test_mode_enabled_input)
        output_layout.addWidget(QtWidgets.QLabel("Test Palette"))
        output_layout.addWidget(self.test_palette_input)
        output_layout.addWidget(QtWidgets.QLabel("R"))
        output_layout.addWidget(self.test_r_input)
        output_layout.addWidget(QtWidgets.QLabel("G"))
        output_layout.addWidget(self.test_g_input)
        output_layout.addWidget(QtWidgets.QLabel("B"))
        output_layout.addWidget(self.test_b_input)
        output_layout.addStretch(1)
        game_layout.addLayout(output_layout)

        self.led_strip_widget = LedStripSimulatorWidget()
        game_layout.addWidget(self.led_strip_widget)

        self.energy_status_label = QtWidgets.QLabel()
        self.energy_status_label.setStyleSheet("color: #000;")
        game_layout.addWidget(self.energy_status_label)

        main_layout.addWidget(game_group)
        main_layout.addWidget(self.status_label)

        self.start_button.clicked.connect(self.start_stream)
        self.stop_button.clicked.connect(self.stop_stream)
        self.apply_game_button.clicked.connect(lambda: self._apply_all_settings(save=True))
        self.reset_energy_button.clicked.connect(self._reset_energy)

    def _set_idle_frame(self) -> None:
        self.video_label.setText("No video")
        self.video_label.setPixmap(QtGui.QPixmap())

    def _game_params_from_config(self) -> GameParams:
        game_cfg = self.config.get("game", {})
        return GameParams(
            pixel_count=max(1, int(game_cfg.get("pixel_count", 120))),
            charge_rate=max(0.0, float(game_cfg.get("charge_rate", 1.0))),
            active_decay_rate=max(0.0, float(game_cfg.get("active_decay_rate", 0.15))),
            idle_decay_rate=max(0.0, float(game_cfg.get("idle_decay_rate", 0.35))),
            idle_drain_enabled=bool(game_cfg.get("idle_drain_enabled", True)),
            takeover_decay_enabled=bool(game_cfg.get("takeover_decay_enabled", True)),
        )

    def _apply_game_params(self, save: bool) -> None:
        tick_hz = max(1, int(self.tick_hz_input.value()))
        params = GameParams(
            pixel_count=max(1, int(self.pixel_count_input.value())),
            charge_rate=max(0.0, float(self.charge_rate_input.value())),
            active_decay_rate=max(0.0, float(self.active_decay_input.value())),
            idle_decay_rate=max(0.0, float(self.idle_decay_input.value())),
            idle_drain_enabled=self.idle_drain_enabled_input.isChecked(),
            takeover_decay_enabled=self.takeover_decay_enabled_input.isChecked(),
        )
        self.energy_engine.set_params(params)
        self._game_timer.setInterval(max(1, int(1000 / tick_hz)))

        self.config["game"] = {
            "pixel_count": params.pixel_count,
            "charge_rate": params.charge_rate,
            "active_decay_rate": params.active_decay_rate,
            "idle_decay_rate": params.idle_decay_rate,
            "idle_drain_enabled": params.idle_drain_enabled,
            "takeover_decay_enabled": params.takeover_decay_enabled,
            "tick_hz": tick_hz,
        }
        if save:
            save_config(self.config)

    def _apply_output_params(self, save: bool) -> None:
        self.config["sacn"] = {
            "enabled": self.sacn_enabled_input.isChecked(),
            "receiver_ip": self.sacn_ip_input.text().strip(),
            "universe": int(self.sacn_universe_input.value()),
            "start_address": int(self.sacn_start_input.value()),
            "test_mode_enabled": self.test_mode_enabled_input.isChecked(),
            "test_palette": self.test_palette_input.currentText(),
            "test_r": int(self.test_r_input.value()),
            "test_g": int(self.test_g_input.value()),
            "test_b": int(self.test_b_input.value()),
        }
        if save:
            save_config(self.config)

    def _apply_all_settings(self, save: bool) -> None:
        self._apply_game_params(save=False)
        self._apply_output_params(save=False)
        if save:
            save_config(self.config)

    def _reset_energy(self) -> None:
        self.energy_engine.reset()
        self.latest_waving_counts = {cloth: 0 for cloth in CLOTH_ORDER}
        self.latest_shirt_counts = {cloth: 0 for cloth in CLOTH_ORDER}
        self._last_game_tick = time.monotonic()
        self.led_strip_widget.set_pixels(self.energy_engine.update({}, 0.0))
        self._update_energy_status()

    def _canonical_cloth_name(self, class_name: str) -> str | None:
        name = class_name.strip().lower()
        aliases = {
            "green-socrates": "green-scorates",
            "green-scorates": "green-scorates",
            "yellow-mandela": "yellow-mandela",
            "red-teresa": "red-teresa",
            "blue-malala": "blue-malala",
        }
        return aliases.get(name)

    def _on_waving_classes_changed(self, raw_counts: dict) -> None:
        counts = {cloth: 0 for cloth in CLOTH_ORDER}
        for class_name, count in raw_counts.items():
            canonical = self._canonical_cloth_name(str(class_name))
            if canonical is None:
                continue
            counts[canonical] += max(0, int(count))
        self.latest_waving_counts = counts

    def _on_shirt_classes_changed(self, raw_counts: dict) -> None:
        counts = {cloth: 0 for cloth in CLOTH_ORDER}
        for class_name, count in raw_counts.items():
            canonical = self._canonical_cloth_name(str(class_name))
            if canonical is None:
                continue
            counts[canonical] += max(0, int(count))
        self.latest_shirt_counts = counts

    def _tick_game(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last_game_tick)
        self._last_game_tick = now

        if self.test_mode_enabled_input.isChecked():
            pixels = build_test_pixels(
                pixel_count=int(self.pixel_count_input.value()),
                palette=self.test_palette_input.currentText(),
                manual_rgb=(
                    int(self.test_r_input.value()),
                    int(self.test_g_input.value()),
                    int(self.test_b_input.value()),
                ),
                elapsed_seconds=max(0.0, now - self._started_at),
            )
        else:
            pixels = self.energy_engine.update(
                self.latest_waving_counts,
                dt,
                shirt_counts=self.latest_shirt_counts,
            )
        self.led_strip_widget.set_pixels(pixels)
        self._send_sacn_if_enabled(pixels)
        self._update_energy_status()

    def _send_sacn_if_enabled(self, pixels) -> None:
        if not self.sacn_enabled_input.isChecked():
            return
        ip = self.sacn_ip_input.text().strip()
        if not ip:
            return
        try:
            self.sacn_sender.send_pixels(
                destination_ip=ip,
                universe=int(self.sacn_universe_input.value()),
                start_address=int(self.sacn_start_input.value()),
                pixels=pixels,
                source_name="HS Pose",
            )
        except OSError:
            pass

    def _update_energy_status(self) -> None:
        energy = self.energy_engine.get_energy()
        total = sum(energy.values())
        parts = []
        for cloth in CLOTH_ORDER:
            percent = (energy[cloth] / total * 100.0) if total > 0 else 0.0
            active = self.latest_waving_counts.get(cloth, 0)
            parts.append(f"{cloth}: {percent:5.1f}% (waving={active})")
        self.energy_status_label.setText(" | ".join(parts))

    def start_stream(self) -> None:
        rtsp_url = self.rtsp_input.text().strip()
        if not rtsp_url:
            QtWidgets.QMessageBox.warning(self, "Missing URL", "Enter an RTSP URL.")
            return

        confidence = self.confidence_input.value()
        transport = self.transport_input.currentData() or DEFAULT_RTSP_TRANSPORT
        self.detector.set_confidence(confidence)
        self._apply_all_settings(save=False)

        self.config["rtsp_url"] = rtsp_url
        self.config["confidence"] = confidence
        self.config["transport"] = transport
        save_config(self.config)

        self.stop_stream()

        self.stream_worker = StreamWorker(rtsp_url, self.detector, transport=transport)
        self.stream_worker.frame_ready.connect(self.update_frame)
        self.stream_worker.detected_changed.connect(self.detected_text.setPlainText)
        self.stream_worker.waving_classes_changed.connect(self._on_waving_classes_changed)
        self.stream_worker.shirt_classes_changed.connect(self._on_shirt_classes_changed)
        self.stream_worker.status_changed.connect(self.status_label.setText)
        self.stream_worker.error_occurred.connect(self.handle_stream_error)
        self.stream_worker.finished.connect(self.on_stream_finished)
        self.stream_worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_stream(self) -> None:
        self.latest_waving_counts = {cloth: 0 for cloth in CLOTH_ORDER}
        self.latest_shirt_counts = {cloth: 0 for cloth in CLOTH_ORDER}
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
        self.latest_waving_counts = {cloth: 0 for cloth in CLOTH_ORDER}
        self.latest_shirt_counts = {cloth: 0 for cloth in CLOTH_ORDER}
        self.detected_text.setPlainText("No detections")
        self.status_label.setText(
            f"Model: {MODEL_PATH.name} | Device: {self.detector.device_name} | "
            f"Confidence: {self.detector.confidence:.2f}"
        )
        if self.video_label.pixmap() is None or self.video_label.pixmap().isNull():
            self._set_idle_frame()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_stream()
        self.sacn_sender.close()
        super().closeEvent(event)
