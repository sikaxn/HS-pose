import time
from urllib.parse import urlparse

from PyQt5 import QtCore, QtGui, QtWidgets

from hs_pose.auto_tracker import AutoTracker
from hs_pose.config import load_config, save_config
from hs_pose.constants import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RTSP_TRANSPORT,
    DEFAULT_VISCA_PORT,
    MODEL_PATH,
)
from hs_pose.detector import YoloV5Detector
from hs_pose.energy_game import CLOTH_ORDER, EnergyGameEngine, GameParams
from hs_pose.led_test_patterns import TEST_PALETTES, build_test_pixels
from hs_pose.sacn_sender import SacnSender
from hs_pose.stream_worker import StreamWorker
from hs_pose.ui.led_strip_simulator import LedStripSimulatorWidget
from hs_pose.visca_over_ip import ViscaOverIpClient


class VideoLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(QtCore.QPoint)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(event.pos())
            event.accept()
            return
        super().mousePressEvent(event)


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
        self._visca_status_timer = QtCore.QTimer(self)
        self._visca_status_timer.timeout.connect(self._refresh_visca_status_auto)
        self._visca_status_busy = False
        self._auto_track_timer = QtCore.QTimer(self)
        self._auto_track_timer.timeout.connect(self._auto_track_tick)
        self._last_game_tick = time.monotonic()
        self._started_at = time.monotonic()
        self._last_visca_status_refresh = 0.0
        self.auto_tracker = AutoTracker(
            detector=self.detector,
            visca_client_factory=lambda timeout: self._visca_client(
                warn_if_missing=False,
                timeout_seconds=timeout,
            ),
        )

        self._build_ui()
        self._set_idle_frame()
        self._apply_game_params(save=False)
        self._apply_output_params(save=False)
        self._game_timer.start()
        self._visca_status_timer.start(100)
        self._auto_track_timer.start(180)
        self.status_label.setText(
            f"Model: {MODEL_PATH.name} | Device: {self.detector.device_name} | "
            f"Confidence: {self.detector.confidence:.2f}"
        )

    def _build_ui(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        controls_layout = QtWidgets.QHBoxLayout()
        visca_layout = QtWidgets.QHBoxLayout()

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

        visca_cfg = self.config.get("visca", {})
        self.visca_address_input = QtWidgets.QLineEdit(str(visca_cfg.get("address", "")))
        self.visca_address_input.setPlaceholderText("VISCA-over-IP camera address")
        self.use_camera_button = QtWidgets.QPushButton("Use Camera")
        self.visca_port_input = QtWidgets.QSpinBox()
        self.visca_port_input.setRange(1, 65535)
        self.visca_port_input.setValue(int(visca_cfg.get("port", DEFAULT_VISCA_PORT)))

        self.ptz_up_button = QtWidgets.QPushButton("Up")
        self.ptz_down_button = QtWidgets.QPushButton("Down")
        self.ptz_left_button = QtWidgets.QPushButton("Left")
        self.ptz_right_button = QtWidgets.QPushButton("Right")
        self.zoom_in_button = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_button = QtWidgets.QPushButton("Zoom -")
        self.focus_in_button = QtWidgets.QPushButton("Focus +")
        self.focus_out_button = QtWidgets.QPushButton("Focus -")
        self.autofocus_button = QtWidgets.QPushButton("AF")
        self.home_button = QtWidgets.QPushButton("Home")
        self.ptz_speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ptz_speed_slider.setRange(1, 24)
        self.ptz_speed_slider.setValue(int(visca_cfg.get("ptz_speed", 8)))
        self.ptz_speed_value = QtWidgets.QLabel(str(self.ptz_speed_slider.value()))
        self.zoom_speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoom_speed_slider.setRange(0, 7)
        self.zoom_speed_slider.setValue(int(visca_cfg.get("zoom_speed", 2)))
        self.zoom_speed_value = QtWidgets.QLabel(str(self.zoom_speed_slider.value()))
        auto_track_cfg = visca_cfg.get("auto_track", {})
        if not isinstance(auto_track_cfg, dict):
            auto_track_cfg = {}

        self.ptz_pad_widget = QtWidgets.QWidget()
        ptz_pad_layout = QtWidgets.QGridLayout(self.ptz_pad_widget)
        ptz_pad_layout.setContentsMargins(0, 0, 0, 0)
        ptz_pad_layout.setHorizontalSpacing(4)
        ptz_pad_layout.setVerticalSpacing(4)
        ptz_pad_layout.addWidget(self.zoom_out_button, 0, 0)
        ptz_pad_layout.addWidget(self.ptz_up_button, 0, 1)
        ptz_pad_layout.addWidget(self.zoom_in_button, 0, 2)
        ptz_pad_layout.addWidget(self.ptz_left_button, 1, 0)
        ptz_pad_layout.addWidget(self.home_button, 1, 1)
        ptz_pad_layout.addWidget(self.ptz_right_button, 1, 2)
        ptz_pad_layout.addWidget(self.focus_out_button, 2, 0)
        ptz_pad_layout.addWidget(self.ptz_down_button, 2, 1)
        ptz_pad_layout.addWidget(self.focus_in_button, 2, 2)
        ptz_pad_layout.addWidget(self.autofocus_button, 3, 1)

        speed_widget = QtWidgets.QWidget()
        speed_layout = QtWidgets.QHBoxLayout(speed_widget)
        speed_layout.setContentsMargins(0, 4, 0, 0)
        speed_layout.setSpacing(8)

        ptz_speed_box = QtWidgets.QVBoxLayout()
        ptz_speed_title = QtWidgets.QLabel("PTZ Speed")
        ptz_speed_title.setAlignment(QtCore.Qt.AlignCenter)
        ptz_speed_box.addWidget(ptz_speed_title)
        ptz_speed_row = QtWidgets.QHBoxLayout()
        ptz_speed_row.addWidget(self.ptz_speed_slider, 1)
        ptz_speed_row.addWidget(self.ptz_speed_value)
        ptz_speed_box.addLayout(ptz_speed_row)

        zoom_speed_box = QtWidgets.QVBoxLayout()
        zoom_speed_title = QtWidgets.QLabel("Zoom Speed")
        zoom_speed_title.setAlignment(QtCore.Qt.AlignCenter)
        zoom_speed_box.addWidget(zoom_speed_title)
        zoom_speed_row = QtWidgets.QHBoxLayout()
        zoom_speed_row.addWidget(self.zoom_speed_slider, 1)
        zoom_speed_row.addWidget(self.zoom_speed_value)
        zoom_speed_box.addLayout(zoom_speed_row)

        speed_layout.addLayout(ptz_speed_box, 1)
        speed_layout.addLayout(zoom_speed_box, 1)

        visca_status_group = QtWidgets.QGroupBox("VISCA Status")
        visca_status_layout = QtWidgets.QVBoxLayout(visca_status_group)
        visca_status_layout.setContentsMargins(8, 8, 8, 8)
        visca_status_controls = QtWidgets.QHBoxLayout()
        self.refresh_visca_status_button = QtWidgets.QPushButton("Refresh")
        self.visca_status_time_label = QtWidgets.QLabel("Not updated")
        visca_status_controls.addWidget(self.refresh_visca_status_button)
        visca_status_controls.addWidget(self.visca_status_time_label)
        visca_status_controls.addStretch(1)
        self.visca_status_text = QtWidgets.QPlainTextEdit()
        self.visca_status_text.setReadOnly(True)
        self.visca_status_text.setMaximumHeight(92)
        self.visca_status_text.setPlainText("No status yet")
        visca_status_layout.addLayout(visca_status_controls)
        visca_status_layout.addWidget(self.visca_status_text)

        auto_track_group = QtWidgets.QGroupBox("Auto Track")
        auto_track_layout = QtWidgets.QVBoxLayout(auto_track_group)
        auto_track_layout.setContentsMargins(8, 8, 8, 8)
        auto_track_controls = QtWidgets.QHBoxLayout()
        self.auto_track_toggle_button = QtWidgets.QPushButton("Off")
        self.auto_track_toggle_button.setCheckable(True)
        self.auto_track_toggle_button.setChecked(
            bool(auto_track_cfg.get("enabled", False))
        )
        self.auto_track_use_zoom_input = QtWidgets.QCheckBox("Use Zoom")
        self.auto_track_use_zoom_input.setChecked(
            bool(auto_track_cfg.get("use_zoom", False))
        )
        auto_track_controls.addWidget(QtWidgets.QLabel("Track"))
        auto_track_controls.addWidget(self.auto_track_toggle_button)
        auto_track_controls.addWidget(self.auto_track_use_zoom_input)
        auto_track_controls.addStretch(1)

        sensitivity_row = QtWidgets.QHBoxLayout()
        self.auto_track_sensitivity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.auto_track_sensitivity_slider.setRange(1, 100)
        self.auto_track_sensitivity_slider.setValue(
            int(auto_track_cfg.get("sensitivity", 50))
        )
        self.auto_track_sensitivity_value = QtWidgets.QLabel(
            str(self.auto_track_sensitivity_slider.value())
        )
        sensitivity_row.addWidget(QtWidgets.QLabel("Sensitivity"))
        sensitivity_row.addWidget(self.auto_track_sensitivity_slider, 1)
        sensitivity_row.addWidget(self.auto_track_sensitivity_value)

        self.auto_track_status_label = QtWidgets.QLabel("Click a pose to select target.")
        self.auto_track_status_label.setStyleSheet("color: #000;")
        self.auto_track_status_label.setWordWrap(True)
        auto_track_layout.addLayout(auto_track_controls)
        auto_track_layout.addLayout(sensitivity_row)
        auto_track_layout.addWidget(self.auto_track_status_label)

        visca_layout.addWidget(QtWidgets.QLabel("VISCA IP"))
        visca_layout.addWidget(self.visca_address_input, 1)
        visca_layout.addWidget(self.use_camera_button)
        visca_layout.addWidget(QtWidgets.QLabel("Port"))
        visca_layout.addWidget(self.visca_port_input)
        visca_layout.addStretch(1)

        self.video_label = VideoLabel()
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
        main_layout.addLayout(visca_layout)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.addWidget(self.video_label, 1)
        detected_layout = QtWidgets.QVBoxLayout()
        detected_layout.addWidget(self.ptz_pad_widget)
        detected_layout.addWidget(speed_widget)
        detected_layout.addWidget(visca_status_group)
        detected_layout.addWidget(auto_track_group)
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
        self.use_camera_button.clicked.connect(self._use_camera_for_visca)
        self.home_button.clicked.connect(self._visca_home)

        self.ptz_up_button.pressed.connect(lambda: self._manual_move_pressed("up"))
        self.ptz_up_button.released.connect(self._manual_pan_tilt_released)
        self.ptz_down_button.pressed.connect(lambda: self._manual_move_pressed("down"))
        self.ptz_down_button.released.connect(self._manual_pan_tilt_released)
        self.ptz_left_button.pressed.connect(lambda: self._manual_move_pressed("left"))
        self.ptz_left_button.released.connect(self._manual_pan_tilt_released)
        self.ptz_right_button.pressed.connect(lambda: self._manual_move_pressed("right"))
        self.ptz_right_button.released.connect(self._manual_pan_tilt_released)
        self.zoom_in_button.pressed.connect(lambda: self._manual_move_pressed("zoom_in"))
        self.zoom_in_button.released.connect(self._manual_zoom_released)
        self.zoom_out_button.pressed.connect(lambda: self._manual_move_pressed("zoom_out"))
        self.zoom_out_button.released.connect(self._manual_zoom_released)
        self.focus_in_button.pressed.connect(lambda: self._manual_move_pressed("focus_in"))
        self.focus_in_button.released.connect(self._manual_focus_released)
        self.focus_out_button.pressed.connect(lambda: self._manual_move_pressed("focus_out"))
        self.focus_out_button.released.connect(self._manual_focus_released)
        self.autofocus_button.clicked.connect(self._visca_autofocus)
        self.ptz_speed_slider.valueChanged.connect(
            lambda value: self.ptz_speed_value.setText(str(value))
        )
        self.zoom_speed_slider.valueChanged.connect(
            lambda value: self.zoom_speed_value.setText(str(value))
        )
        self.ptz_speed_slider.sliderReleased.connect(
            lambda: self._apply_visca_params(save=True)
        )
        self.zoom_speed_slider.sliderReleased.connect(
            lambda: self._apply_visca_params(save=True)
        )
        self.refresh_visca_status_button.clicked.connect(
            lambda: self._refresh_visca_status(warn_if_missing=True, save_settings=True)
        )
        self.video_label.clicked.connect(self._on_video_clicked)
        self.auto_track_toggle_button.toggled.connect(self._on_auto_track_toggled)
        self.auto_track_use_zoom_input.toggled.connect(
            lambda _checked: self._apply_visca_params(save=True)
        )
        self.auto_track_sensitivity_slider.valueChanged.connect(
            self._on_auto_track_sensitivity_changed
        )
        self.auto_track_sensitivity_slider.sliderReleased.connect(
            lambda: self._apply_visca_params(save=True)
        )
        self.auto_track_toggle_button.setText(
            "On" if self.auto_track_toggle_button.isChecked() else "Off"
        )
        self._update_auto_track_status()

    def _on_auto_track_sensitivity_changed(self, value: int) -> None:
        self.auto_track_sensitivity_value.setText(str(int(value)))
        self._apply_visca_params(save=False)

    def _manual_move_pressed(self, direction: str) -> None:
        if self.auto_track_toggle_button.isChecked():
            self.auto_tracker.begin_manual_override()
            self._update_auto_track_status()
        self._visca_send_move(direction)

    def _manual_pan_tilt_released(self) -> None:
        self._visca_stop_pan_tilt()
        if self.auto_track_toggle_button.isChecked():
            self.auto_tracker.end_manual_override()
            self._update_auto_track_status()

    def _manual_zoom_released(self) -> None:
        self._visca_stop_zoom()
        if self.auto_track_toggle_button.isChecked():
            self.auto_tracker.end_manual_override()
            self._update_auto_track_status()

    def _manual_focus_released(self) -> None:
        self._visca_stop_focus()
        if self.auto_track_toggle_button.isChecked():
            self.auto_tracker.end_manual_override()
            self._update_auto_track_status()

    def _on_auto_track_toggled(self, enabled: bool) -> None:
        self.auto_track_toggle_button.setText("On" if enabled else "Off")
        if not enabled:
            self.auto_tracker.stop_motion(ptz_speed=int(self.ptz_speed_slider.value()))
        self._apply_visca_params(save=True)
        self._update_auto_track_status()

    def _on_pose_data_changed(self, pose_data: object) -> None:
        self.auto_tracker.update_pose_data(pose_data)
        self._update_auto_track_status()

    def _on_video_clicked(self, point: QtCore.QPoint) -> None:
        status_text = self.auto_tracker.select_from_click(point)
        if status_text:
            self.auto_track_status_label.setText(status_text)
        else:
            self._update_auto_track_status()

    def _auto_track_tick(self) -> None:
        status_text = self.auto_tracker.tick(
            enabled=self.auto_track_toggle_button.isChecked(),
            use_zoom=self.auto_track_use_zoom_input.isChecked(),
            sensitivity=int(self.auto_track_sensitivity_slider.value()),
            ptz_speed=int(self.ptz_speed_slider.value()),
            zoom_speed_limit=int(self.zoom_speed_slider.value()),
        )
        if status_text:
            self.auto_track_status_label.setText(status_text)

    def _update_auto_track_status(self, extra_text: str | None = None) -> None:
        if extra_text:
            self.auto_track_status_label.setText(extra_text)
            return
        self.auto_track_status_label.setText(
            self.auto_tracker.build_status_text(
                enabled=self.auto_track_toggle_button.isChecked()
            )
        )

    def _set_idle_frame(self) -> None:
        self.video_label.setText("No video")
        self.video_label.setPixmap(QtGui.QPixmap())
        self.auto_tracker.set_frame_mapping(QtCore.QSize(), QtCore.QRect())

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

    def _apply_visca_params(self, save: bool) -> None:
        sensitivity = int(self.auto_track_sensitivity_slider.value())
        self.config["visca"] = {
            "address": self.visca_address_input.text().strip(),
            "port": int(self.visca_port_input.value()),
            "ptz_speed": int(self.ptz_speed_slider.value()),
            "zoom_speed": int(self.zoom_speed_slider.value()),
            "auto_track": {
                "enabled": self.auto_track_toggle_button.isChecked(),
                "use_zoom": self.auto_track_use_zoom_input.isChecked(),
                "sensitivity": sensitivity,
            },
        }
        if save:
            save_config(self.config)

    def _apply_all_settings(self, save: bool) -> None:
        self._apply_game_params(save=False)
        self._apply_output_params(save=False)
        self._apply_visca_params(save=False)
        if save:
            save_config(self.config)

    def _extract_rtsp_host(self, rtsp_url: str) -> str:
        if not rtsp_url:
            return ""
        parsed = urlparse(rtsp_url)
        if parsed.hostname:
            return parsed.hostname
        parsed = urlparse(f"rtsp://{rtsp_url}")
        if parsed.hostname:
            return parsed.hostname
        return ""

    def _use_camera_for_visca(self) -> None:
        host = self._extract_rtsp_host(self.rtsp_input.text().strip())
        if not host:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid RTSP URL",
                "Could not extract a camera host from the RTSP URL.",
            )
            return
        self.visca_address_input.setText(host)
        self._apply_visca_params(save=True)

    def _visca_client(
        self, warn_if_missing: bool = True, timeout_seconds: float = 0.5
    ) -> ViscaOverIpClient | None:
        address = self.visca_address_input.text().strip()
        if not address:
            if warn_if_missing:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing VISCA Address",
                    "Enter the VISCA-over-IP camera address first.",
                )
            return None
        return ViscaOverIpClient(
            host=address,
            port=int(self.visca_port_input.value()),
            timeout=timeout_seconds,
        )

    def _refresh_visca_status(
        self, warn_if_missing: bool = True, save_settings: bool = False, timeout_seconds: float = 0.2
    ) -> None:
        self._apply_visca_params(save=save_settings)
        try:
            client = self._visca_client(
                warn_if_missing=warn_if_missing,
                timeout_seconds=timeout_seconds,
            )
            if client is None:
                return
            status = client.read_status()
            power = status.get("power")
            focus_mode = status.get("focus_mode")
            zoom_position = status.get("zoom_position")
            power_text = "On" if power is True else ("Off" if power is False else "Unknown")
            focus_text = str(focus_mode) if focus_mode is not None else "Unknown"
            zoom_text = str(zoom_position) if zoom_position is not None else "Unknown"
            self.visca_status_text.setPlainText(
                f"Power: {power_text}\nFocus: {focus_text}\nZoom Position: {zoom_text}"
            )
            self._last_visca_status_refresh = time.monotonic()
            self.visca_status_time_label.setText("Updated just now")
        except (OSError, TimeoutError, ValueError) as exc:
            self.visca_status_text.setPlainText(f"Status read failed:\n{exc}")

    def _refresh_visca_status_auto(self) -> None:
        if self._visca_status_busy:
            return
        self._visca_status_busy = True
        try:
            self._refresh_visca_status(
                warn_if_missing=False,
                save_settings=False,
                timeout_seconds=0.08,
            )
        finally:
            self._visca_status_busy = False

    def _visca_send_move(self, direction: str) -> None:
        self._apply_visca_params(save=True)
        try:
            client = self._visca_client()
            if client is None:
                return
            ptz_speed = int(self.ptz_speed_slider.value())
            zoom_speed = int(self.zoom_speed_slider.value())
            if direction == "left":
                client.pan_left(pan_speed=ptz_speed, tilt_speed=ptz_speed)
            elif direction == "right":
                client.pan_right(pan_speed=ptz_speed, tilt_speed=ptz_speed)
            elif direction == "up":
                client.tilt_up(pan_speed=ptz_speed, tilt_speed=ptz_speed)
            elif direction == "down":
                client.tilt_down(pan_speed=ptz_speed, tilt_speed=ptz_speed)
            elif direction == "zoom_in":
                client.zoom_in(speed=zoom_speed)
            elif direction == "zoom_out":
                client.zoom_out(speed=zoom_speed)
            elif direction == "focus_in":
                client.focus_far()
            elif direction == "focus_out":
                client.focus_near()
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "VISCA Error", f"VISCA send failed: {exc}")

    def _visca_stop_pan_tilt(self) -> None:
        try:
            client = self._visca_client(warn_if_missing=False)
            if client is None:
                return
            ptz_speed = int(self.ptz_speed_slider.value())
            client.pan_tilt_stop(pan_speed=ptz_speed, tilt_speed=ptz_speed)
        except OSError:
            pass

    def _visca_stop_zoom(self) -> None:
        try:
            client = self._visca_client(warn_if_missing=False)
            if client is None:
                return
            client.zoom_stop()
        except OSError:
            pass

    def _visca_stop_focus(self) -> None:
        try:
            client = self._visca_client(warn_if_missing=False)
            if client is None:
                return
            client.focus_stop()
        except OSError:
            pass

    def _visca_autofocus(self) -> None:
        self._apply_visca_params(save=True)
        try:
            client = self._visca_client()
            if client is None:
                return
            client.autofocus_on()
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "VISCA Error", f"VISCA send failed: {exc}")

    def _visca_home(self) -> None:
        self._apply_visca_params(save=True)
        try:
            client = self._visca_client()
            if client is None:
                return
            client.home()
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "VISCA Error", f"VISCA send failed: {exc}")

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
        self.stream_worker.pose_data_changed.connect(self._on_pose_data_changed)
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
        self.auto_tracker.clear_selection()
        self._update_auto_track_status()
        self.auto_tracker.stop_motion(ptz_speed=int(self.ptz_speed_slider.value()))
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
        content = self.video_label.contentsRect()
        x_offset = content.x() + max(0, (content.width() - scaled.width()) // 2)
        y_offset = content.y() + max(0, (content.height() - scaled.height()) // 2)
        self.auto_tracker.set_frame_mapping(
            image.size(),
            QtCore.QRect(
                x_offset,
                y_offset,
                scaled.width(),
                scaled.height(),
            ),
        )

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
        self.auto_tracker.clear_selection()
        self._update_auto_track_status()
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
