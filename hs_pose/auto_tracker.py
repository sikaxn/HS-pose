from __future__ import annotations

from collections.abc import Callable

from PyQt5 import QtCore


class AutoTracker:
    def __init__(self, detector, visca_client_factory: Callable[[float], object | None]) -> None:
        self._detector = detector
        self._visca_client_factory = visca_client_factory
        self._latest_pose_data = []
        self._selected_pose_track_id = None
        self._selected_pose_anchor = None
        self._last_source_frame_size = QtCore.QSize()
        self._last_display_rect = QtCore.QRect()
        self._last_pan_tilt = None
        self._last_zoom = None
        self._status_override = None
        self._manual_override = False

    def set_frame_mapping(self, source_size: QtCore.QSize, display_rect: QtCore.QRect) -> None:
        self._last_source_frame_size = QtCore.QSize(source_size)
        self._last_display_rect = QtCore.QRect(display_rect)

    def clear_selection(self) -> None:
        self._latest_pose_data = []
        self._selected_pose_track_id = None
        self._selected_pose_anchor = None
        self._detector.set_selected_pose_track_id(None)
        self._status_override = None
        self._manual_override = False

    def update_pose_data(self, pose_data: object) -> None:
        if isinstance(pose_data, list):
            self._latest_pose_data = pose_data
        else:
            self._latest_pose_data = []

        if self._selected_pose_track_id is None:
            self._status_override = None
            return

        active_ids = {
            int(pose.get("track_id", -1))
            for pose in self._latest_pose_data
            if isinstance(pose, dict) and "track_id" in pose
        }
        if self._selected_pose_track_id not in active_ids:
            self._status_override = "Selected pose lost."
        else:
            self._status_override = None

    def select_from_click(self, point: QtCore.QPoint) -> str | None:
        if not self._latest_pose_data:
            return "No poses available to select."
        if self._last_display_rect.isNull() or self._last_source_frame_size.isEmpty():
            return None
        if not self._last_display_rect.contains(point):
            return None

        frame_x = (
            (point.x() - self._last_display_rect.x())
            * self._last_source_frame_size.width()
            / max(1, self._last_display_rect.width())
        )
        frame_y = (
            (point.y() - self._last_display_rect.y())
            * self._last_source_frame_size.height()
            / max(1, self._last_display_rect.height())
        )
        pose = self._pick_pose_at_point(frame_x, frame_y)
        if pose is None:
            return "No pose near click point."
        self._select_pose(pose)
        return f"Selected Pose {self._selected_pose_track_id}."

    def build_status_text(self, enabled: bool) -> str:
        if self._manual_override:
            return "Manual PTZ override active. Auto-track paused."
        if self._status_override:
            return self._status_override
        if self._selected_pose_track_id is None:
            return "Auto-track is ON. Click a pose." if enabled else "Click a pose to select target."
        mode = "ON" if enabled else "OFF"
        return f"Pose {self._selected_pose_track_id} selected. Auto-track {mode}."

    def stop_motion(self, ptz_speed: int) -> None:
        self._send_pan_tilt(0x03, 0x03, int(ptz_speed), int(ptz_speed))
        self._send_zoom_stop()

    def tick(
        self,
        enabled: bool,
        use_zoom: bool,
        sensitivity: int,
        ptz_speed: int,
        zoom_speed_limit: int,
    ) -> str | None:
        if not enabled:
            return None
        if self._manual_override:
            return None
        if self._selected_pose_track_id is None or self._selected_pose_anchor is None:
            return "Auto-track is ON. Click a pose."

        selected_pose = None
        for pose in self._latest_pose_data:
            if not isinstance(pose, dict):
                continue
            if int(pose.get("track_id", -1)) == self._selected_pose_track_id:
                selected_pose = pose
                break
        if selected_pose is None:
            self.stop_motion(ptz_speed=ptz_speed)
            return "Selected pose is not visible."

        center = selected_pose.get("center")
        if not isinstance(center, list) or len(center) != 2:
            return None
        frame_w = max(1, self._last_source_frame_size.width())
        frame_h = max(1, self._last_source_frame_size.height())
        current_x = min(max(float(center[0]) / frame_w, 0.0), 1.0)
        current_y = min(max(float(center[1]) / frame_h, 0.0), 1.0)
        current_area = float(selected_pose.get("area", 0.0))
        current_norm_area = min(max(current_area / float(frame_w * frame_h), 0.0), 1.0)

        anchor_x, anchor_y, anchor_area = self._selected_pose_anchor
        error_x = current_x - anchor_x
        error_y = current_y - anchor_y
        area_error = current_norm_area - anchor_area
        sensitivity_01 = min(1.0, max(0.01, float(sensitivity) / 100.0))
        deadband = max(0.012, 0.11 - (0.095 * sensitivity_01))
        pan_dir, pan_speed = self._axis_to_ptz(error_x, deadband, sensitivity_01, ptz_speed)
        tilt_dir, tilt_speed = self._axis_to_ptz(error_y, deadband, sensitivity_01, ptz_speed)
        self._send_pan_tilt(pan_dir, tilt_dir, pan_speed, tilt_speed)
        if use_zoom:
            self._send_zoom(area_error, deadband * 0.75, sensitivity_01, zoom_speed_limit)
        else:
            self._send_zoom_stop()
        return (
            f"Tracking Pose {self._selected_pose_track_id} | "
            f"dx={error_x:+.3f}, dy={error_y:+.3f}"
        )

    def begin_manual_override(self) -> None:
        self._manual_override = True
        self._last_pan_tilt = None
        self._last_zoom = None

    def end_manual_override(self) -> None:
        self._manual_override = False
        self._rebase_anchor_to_current_pose()
        self._last_pan_tilt = None
        self._last_zoom = None

    def _pick_pose_at_point(self, frame_x: float, frame_y: float) -> dict | None:
        containing = []
        nearest = None
        nearest_dist_sq = None
        for pose in self._latest_pose_data:
            if not isinstance(pose, dict):
                continue
            bbox = pose.get("bbox")
            center = pose.get("center")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if not isinstance(center, list) or len(center) != 2:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx, cy = float(center[0]), float(center[1])
            if x1 <= frame_x <= x2 and y1 <= frame_y <= y2:
                containing.append((max(1.0, (x2 - x1) * (y2 - y1)), pose))
            dx = cx - frame_x
            dy = cy - frame_y
            dist_sq = dx * dx + dy * dy
            if nearest_dist_sq is None or dist_sq < nearest_dist_sq:
                nearest_dist_sq = dist_sq
                nearest = pose

        if containing:
            containing.sort(key=lambda item: item[0])
            return containing[0][1]
        if nearest is None or nearest_dist_sq is None:
            return None
        max_distance = max(80.0, 0.08 * max(self._last_source_frame_size.width(), 1))
        if nearest_dist_sq > max_distance * max_distance:
            return None
        return nearest

    def _select_pose(self, pose: dict) -> None:
        track_id = int(pose.get("track_id", -1))
        if track_id < 0:
            return
        center = pose.get("center")
        area = float(pose.get("area", 0.0))
        if not isinstance(center, list) or len(center) != 2:
            return
        frame_w = max(1, self._last_source_frame_size.width())
        frame_h = max(1, self._last_source_frame_size.height())
        anchor_x = min(max(float(center[0]) / frame_w, 0.0), 1.0)
        anchor_y = min(max(float(center[1]) / frame_h, 0.0), 1.0)
        norm_area = min(max(area / float(frame_w * frame_h), 0.0), 1.0)
        self._selected_pose_track_id = track_id
        self._selected_pose_anchor = (anchor_x, anchor_y, norm_area)
        self._detector.set_selected_pose_track_id(track_id)
        self._last_pan_tilt = None
        self._last_zoom = None
        self._status_override = None

    def _rebase_anchor_to_current_pose(self) -> None:
        if self._selected_pose_track_id is None:
            return
        selected_pose = None
        for pose in self._latest_pose_data:
            if not isinstance(pose, dict):
                continue
            if int(pose.get("track_id", -1)) == self._selected_pose_track_id:
                selected_pose = pose
                break
        if selected_pose is None:
            return
        center = selected_pose.get("center")
        area = float(selected_pose.get("area", 0.0))
        if not isinstance(center, list) or len(center) != 2:
            return
        frame_w = max(1, self._last_source_frame_size.width())
        frame_h = max(1, self._last_source_frame_size.height())
        anchor_x = min(max(float(center[0]) / frame_w, 0.0), 1.0)
        anchor_y = min(max(float(center[1]) / frame_h, 0.0), 1.0)
        norm_area = min(max(area / float(frame_w * frame_h), 0.0), 1.0)
        self._selected_pose_anchor = (anchor_x, anchor_y, norm_area)

    def _axis_to_ptz(
        self, error: float, deadband: float, sensitivity_01: float, ptz_speed: int
    ) -> tuple[int, int]:
        if abs(error) <= deadband:
            return 0x03, max(1, int(ptz_speed))
        max_speed = max(1, int(ptz_speed))
        span = max(0.0001, 0.5 - deadband)
        normalized = min(1.0, (abs(error) - deadband) / span)
        scaled = normalized ** (0.85 - (0.45 * sensitivity_01))
        speed = max(1, min(24, int(round(1 + scaled * (max_speed - 1)))))
        direction = 0x01 if error < 0 else 0x02
        return direction, speed

    def _send_pan_tilt(
        self, pan_dir: int, tilt_dir: int, pan_speed: int, tilt_speed: int
    ) -> None:
        state = (int(pan_dir), int(tilt_dir), int(pan_speed), int(tilt_speed))
        if state == self._last_pan_tilt:
            return
        try:
            client = self._visca_client_factory(0.08)
            if client is None:
                return
            client.pan_tilt(
                pan_speed=max(1, int(pan_speed)),
                tilt_speed=max(1, int(tilt_speed)),
                pan_dir=int(pan_dir),
                tilt_dir=int(tilt_dir),
            )
            self._last_pan_tilt = state
        except OSError:
            pass

    def _send_zoom(
        self,
        area_error: float,
        deadband: float,
        sensitivity_01: float,
        zoom_speed_limit: int,
    ) -> None:
        zoom_speed_limit = max(0, int(zoom_speed_limit))
        if zoom_speed_limit <= 0 or abs(area_error) <= deadband:
            self._send_zoom_stop()
            return
        normalized = min(
            1.0, (abs(area_error) - deadband) / max(0.0001, 0.35 - deadband)
        )
        speed = max(
            1,
            min(
                7,
                int(
                    round(
                        1
                        + normalized
                        * zoom_speed_limit
                        * (0.5 + 0.5 * sensitivity_01)
                    )
                ),
            ),
        )
        direction = "out" if area_error > 0 else "in"
        state = (direction, speed)
        if state == self._last_zoom:
            return
        try:
            client = self._visca_client_factory(0.08)
            if client is None:
                return
            if direction == "in":
                client.zoom_in(speed=speed)
            else:
                client.zoom_out(speed=speed)
            self._last_zoom = state
        except OSError:
            pass

    def _send_zoom_stop(self) -> None:
        if self._last_zoom == ("stop", 0):
            return
        try:
            client = self._visca_client_factory(0.08)
            if client is None:
                return
            client.zoom_stop()
            self._last_zoom = ("stop", 0)
        except OSError:
            pass
