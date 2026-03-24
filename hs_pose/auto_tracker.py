from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np
from PyQt5 import QtCore


class AutoTracker:
    def __init__(
        self,
        detector,
        visca_client_factory: Callable[[float], object | None],
        face_recognizer=None,
    ) -> None:
        self._detector = detector
        self._visca_client_factory = visca_client_factory
        self._face_recognizer = face_recognizer
        self._latest_pose_data = []
        self._latest_frame_bgr = None
        self._selected_pose_track_id = None
        self._selected_pose_anchor = None
        self._last_source_frame_size = QtCore.QSize()
        self._last_display_rect = QtCore.QRect()
        self._last_pan_tilt = None
        self._last_zoom = None
        self._status_override = None
        self._manual_override = False
        self._anchor_mode = "full_body"
        self._use_face_recognition = False
        self._face_target_embedding = None
        self._face_target_crop = None
        self._last_face_learn_attempt = 0.0
        self._face_status_message = "Face recognition is off."

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
        self._face_target_embedding = None
        self._face_target_crop = None
        self._last_face_learn_attempt = 0.0
        if self._use_face_recognition:
            self._face_status_message = "Waiting to learn face..."
        else:
            self._face_status_message = "Face recognition is off."

    def reset_selection_to_zero(self) -> None:
        self.clear_selection()
        reset_fn = getattr(self._detector, "reset_pose_tracking", None)
        if callable(reset_fn):
            reset_fn(start_track_id=0)

    def set_anchor_mode(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"head", "half_body", "full_body"}:
            normalized = "full_body"
        self._anchor_mode = normalized
        set_mode_fn = getattr(self._detector, "set_selected_anchor_mode", None)
        if callable(set_mode_fn):
            set_mode_fn(normalized)
        self._rebase_anchor_to_current_pose()

    def set_use_face_recognition(self, enabled: bool) -> None:
        self._use_face_recognition = bool(enabled)
        if not self._use_face_recognition:
            self._face_target_embedding = None
            self._face_target_crop = None
            self._face_status_message = "Face recognition is off."
        else:
            self._face_status_message = "Waiting to learn face..."
        self._last_face_learn_attempt = 0.0

    def set_latest_frame(self, frame_bgr) -> None:
        self._latest_frame_bgr = frame_bgr

    def get_learned_face_crop(self):
        return self._face_target_crop

    def is_face_learned(self) -> bool:
        return self._face_target_embedding is not None

    def get_face_status_text(self) -> str:
        return self._face_status_message

    def reset_face_data(self) -> None:
        self._face_target_embedding = None
        self._face_target_crop = None
        self._last_face_learn_attempt = 0.0
        if self._use_face_recognition:
            self._face_status_message = "Waiting to learn face..."
        else:
            self._face_status_message = "Face recognition is off."

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
            if self._use_face_recognition and self._face_target_embedding is None:
                selected_pose = self._find_selected_pose()
                if selected_pose is not None:
                    self._learn_face_from_pose(selected_pose, force=False)

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
        return (
            f"Pose {self._selected_pose_track_id} selected. "
            f"Auto-track {mode}. Anchor: {self._anchor_mode.replace('_', ' ')}."
        )

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
            reacquired_pose = self._try_reacquire_with_face()
            if reacquired_pose is not None:
                self._reacquire_pose_keep_anchor(reacquired_pose)
                return f"Face reacquired Pose {self._selected_pose_track_id}."
            self.stop_motion(ptz_speed=ptz_speed)
            return "Selected pose is not visible."
        if self._use_face_recognition and self._face_target_embedding is None:
            self._learn_face_from_pose(selected_pose, force=False)

        anchor = self._anchor_from_pose(selected_pose)
        if anchor is None:
            return None
        current_x, current_y, current_norm_area = anchor

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
        anchor = self._anchor_from_pose(pose)
        if anchor is None:
            return
        anchor_x, anchor_y, norm_area = anchor
        self._selected_pose_track_id = track_id
        self._selected_pose_anchor = (anchor_x, anchor_y, norm_area)
        self._detector.set_selected_pose_track_id(track_id)
        self._last_pan_tilt = None
        self._last_zoom = None
        self._status_override = None
        self._learn_face_from_pose(pose, force=True)

    def _reacquire_pose_keep_anchor(self, pose: dict) -> None:
        track_id = int(pose.get("track_id", -1))
        if track_id < 0:
            return
        # Keep existing anchor so the camera returns to the same relative composition.
        self._selected_pose_track_id = track_id
        self._detector.set_selected_pose_track_id(track_id)
        self._last_pan_tilt = None
        self._last_zoom = None
        self._status_override = "Face reacquired. Restoring previous framing."
        if self._use_face_recognition:
            # Refine learned face sample on reacquire using current implementation path.
            self._learn_face_from_pose(pose, force=False)

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
        anchor = self._anchor_from_pose(selected_pose)
        if anchor is None:
            return
        anchor_x, anchor_y, norm_area = anchor
        self._selected_pose_anchor = (anchor_x, anchor_y, norm_area)

    def _anchor_from_pose(self, pose: dict) -> tuple[float, float, float] | None:
        bbox = pose.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        ax1, ay1, ax2, ay2 = x1, y1, x2, y2

        if self._anchor_mode == "head":
            ax1 = x1 + (0.2 * width)
            ax2 = x2 - (0.2 * width)
            ay2 = y1 + (0.35 * height)
        elif self._anchor_mode == "half_body":
            ax1 = x1 + (0.1 * width)
            ax2 = x2 - (0.1 * width)
            ay2 = y1 + (0.6 * height)

        if ax2 <= ax1:
            ax1, ax2 = x1, x2
        if ay2 <= ay1:
            ay1, ay2 = y1, y2

        frame_w = max(1, self._last_source_frame_size.width())
        frame_h = max(1, self._last_source_frame_size.height())
        center_x = (ax1 + ax2) / 2.0
        center_y = (ay1 + ay2) / 2.0
        area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        norm_x = min(max(center_x / frame_w, 0.0), 1.0)
        norm_y = min(max(center_y / frame_h, 0.0), 1.0)
        norm_area = min(max(area / float(frame_w * frame_h), 0.0), 1.0)
        return norm_x, norm_y, norm_area

    def _learn_face_from_pose(self, pose: dict, force: bool = False) -> None:
        if not self._use_face_recognition:
            return
        now = time.monotonic()
        if not force and now - self._last_face_learn_attempt < 0.4:
            return
        self._last_face_learn_attempt = now
        if self._face_recognizer is None or self._latest_frame_bgr is None:
            self._status_override = "Learning face... waiting for frame."
            self._face_status_message = "Learning face... waiting for frame."
            return
        bbox = pose.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return
        embedding, crop = self._face_recognizer.learn_from_pose_bbox(
            self._latest_frame_bgr,
            bbox,
        )
        if embedding is None:
            self._status_override = "Learning face... no face found yet."
            self._face_target_embedding = None
            self._face_target_crop = None
            self._face_status_message = "Learning face... no face found yet."
            return
        self._face_target_embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self._face_target_crop = crop
        self._status_override = "Face learned for selected pose."
        self._face_status_message = "Face learned."

    def _find_selected_pose(self) -> dict | None:
        if self._selected_pose_track_id is None:
            return None
        for pose in self._latest_pose_data:
            if not isinstance(pose, dict):
                continue
            if int(pose.get("track_id", -1)) == self._selected_pose_track_id:
                return pose
        return None

    def _try_reacquire_with_face(self) -> dict | None:
        if not self._use_face_recognition:
            return None
        if self._face_recognizer is None:
            return None
        if self._latest_frame_bgr is None:
            return None
        if self._face_target_embedding is None:
            return None
        return self._face_recognizer.find_pose_for_identity(
            self._latest_frame_bgr,
            self._latest_pose_data,
            self._face_target_embedding,
        )

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
