import time
from collections import deque
from pathlib import Path

import cv2
import torch
import yolov5
from ultralytics import YOLO

from hs_pose.constants import DEFAULT_CONFIDENCE, DEFAULT_IOU, POSE_MODEL_PATH


def patch_torch_load_for_yolov5() -> None:
    if getattr(torch.load, "_hs_pose_compat", False):
        return

    original_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    compat_load._hs_pose_compat = True
    torch.load = compat_load


class YoloV5Detector:
    _POINT_CONFIDENCE = 0.5
    _TRACK_MAX_AGE_SECONDS = 2.0
    _WAVE_WINDOW_SECONDS = 0.8
    _WAVE_MIN_SAMPLES = 4
    _POSE_SKELETON = (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    )

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        patch_torch_load_for_yolov5()
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.model = yolov5.load(str(model_path), device=self.device)
        self.pose_model = YOLO(str(POSE_MODEL_PATH))
        self.model.conf = DEFAULT_CONFIDENCE
        self.model.iou = DEFAULT_IOU
        self.model.agnostic = False
        self.model.multi_label = False
        self.model.max_det = 100
        self.names = self.model.names
        self.device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
        self.confidence = DEFAULT_CONFIDENCE
        self._tracks = {}
        self._next_track_id = 1

    def set_confidence(self, confidence: float) -> None:
        confidence = min(max(float(confidence), 0.0), 1.0)
        self.model.conf = confidence
        self.confidence = confidence

    def infer(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame, size=640)
        detections = results.xyxy[0].detach().cpu().numpy()
        annotated = frame.copy()
        shirt_detections = []

        for x1, y1, x2, y2, confidence, class_id in detections:
            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            class_index = int(class_id)
            class_name = self.names.get(class_index, str(class_index))
            label = f"{class_name} {confidence:.2f}"
            color = self._color_for_class(class_index)
            shirt_detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "color": color,
                }
            )

            cv2.rectangle(annotated, (x1_i, y1_i), (x2_i, y2_i), color, 2)
            text_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            text_top = max(y1_i - text_size[1] - baseline - 6, 0)
            text_bottom = text_top + text_size[1] + baseline + 6
            text_right = x1_i + text_size[0] + 8

            cv2.rectangle(
                annotated,
                (x1_i, text_top),
                (text_right, text_bottom),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1_i + 4, text_bottom - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        pose_count, waving_count = self._draw_pose_and_actions(
            annotated, rgb_frame, shirt_detections
        )
        return annotated, len(detections), pose_count, waving_count

    def _draw_pose_and_actions(self, annotated, rgb_frame, shirt_detections) -> tuple[int, int]:
        pose_results = self.pose_model.predict(
            source=rgb_frame,
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )
        if not pose_results:
            return 0

        result = pose_results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None:
            return 0

        xy = keypoints.xy.detach().cpu().numpy()
        conf = None
        if getattr(keypoints, "conf", None) is not None:
            conf = keypoints.conf.detach().cpu().numpy()
        boxes = []
        if getattr(result, "boxes", None) is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()

        people = []
        person_count = 0
        for idx, person_points in enumerate(xy):
            point_conf = conf[idx] if conf is not None else None
            if self._draw_single_pose(annotated, person_points, point_conf):
                person_count += 1
                people.append(
                    {
                        "points": person_points,
                        "point_conf": point_conf,
                        "bbox": tuple(boxes[idx]) if idx < len(boxes) else self._points_bbox(person_points, point_conf),
                    }
                )

        waving_count = self._annotate_person_waving(annotated, shirt_detections, people)
        return person_count, waving_count

    def _draw_single_pose(self, annotated, points, point_conf) -> bool:
        visible_points = {}
        for index, point in enumerate(points):
            x_coord, y_coord = map(int, point[:2])
            confidence = point_conf[index] if point_conf is not None else 1.0
            if confidence < self._POINT_CONFIDENCE:
                continue

            visible_points[index] = (x_coord, y_coord)
            cv2.circle(annotated, (x_coord, y_coord), 4, (0, 255, 255), -1)

        if not visible_points:
            return False

        for start_idx, end_idx in self._POSE_SKELETON:
            if start_idx not in visible_points or end_idx not in visible_points:
                continue
            cv2.line(
                annotated,
                visible_points[start_idx],
                visible_points[end_idx],
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return True

    def _annotate_person_waving(self, annotated, shirt_detections, people) -> int:
        self._prune_tracks()
        waving_count = 0

        for person in people:
            track = self._update_track(person)
            left_waving, right_waving = self._update_wave_state(track, person)
            if not left_waving and not right_waving:
                continue

            waving_count += 1
            hand_text = self._wave_label(left_waving, right_waving)
            shirt_detection = self._match_shirt_to_person(person, shirt_detections)
            person_box = person["bbox"]
            x1, y1, x2, _y2 = map(int, person_box)
            if shirt_detection is None:
                banner = f"Waving {hand_text}"
                banner_color = (0, 200, 255)
            else:
                banner = f"{shirt_detection['class_name']} waving {hand_text}"
                banner_color = shirt_detection["color"]
            cv2.rectangle(
                annotated,
                (x1, max(y1 - 28, 0)),
                (x2, y1),
                banner_color,
                -1,
            )
            cv2.putText(
                annotated,
                banner,
                (x1 + 4, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        return waving_count

    def _match_shirt_to_person(self, person, shirt_detections):
        best_detection = None
        best_score = 0.0
        person_bbox = person["bbox"]

        for detection in shirt_detections:
            shirt_bbox = detection["bbox"]
            shirt_center = self._bbox_center(shirt_bbox)
            iou = self._bbox_iou(shirt_bbox, person_bbox)
            center_inside = self._point_in_bbox(shirt_center, person_bbox)
            score = iou + (1.0 if center_inside else 0.0)
            if score > best_score:
                best_score = score
                best_detection = detection

        return best_detection

    def _update_track(self, person) -> dict:
        center_x, center_y = self._bbox_center(person["bbox"])
        box_width = max(person["bbox"][2] - person["bbox"][0], 1.0)
        box_height = max(person["bbox"][3] - person["bbox"][1], 1.0)
        max_distance = max(80.0, 0.35 * max(box_width, box_height))

        best_track_id = None
        best_distance = None
        for track_id, track in self._tracks.items():
            distance = ((track["center_x"] - center_x) ** 2 + (track["center_y"] - center_y) ** 2) ** 0.5
            if distance > max_distance:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[best_track_id] = {
                "left_history": deque(),
                "right_history": deque(),
            }

        track = self._tracks[best_track_id]
        track["center_x"] = center_x
        track["center_y"] = center_y
        track["last_seen"] = time.monotonic()
        track["person_bbox"] = person["bbox"]
        return track

    def _update_wave_state(self, track, person) -> tuple[bool, bool]:
        points = person["points"]
        point_conf = person["point_conf"]
        person_bbox = person["bbox"]
        now = time.monotonic()

        left_waving = self._update_hand_wave(
            track["left_history"],
            now,
            points,
            point_conf,
            wrist_idx=9,
            elbow_idx=7,
            shoulder_idx=5,
            person_bbox=person_bbox,
        )
        right_waving = self._update_hand_wave(
            track["right_history"],
            now,
            points,
            point_conf,
            wrist_idx=10,
            elbow_idx=8,
            shoulder_idx=6,
            person_bbox=person_bbox,
        )
        return left_waving, right_waving

    def _update_hand_wave(
        self,
        history: deque,
        now: float,
        points,
        point_conf,
        wrist_idx: int,
        elbow_idx: int,
        shoulder_idx: int,
        person_bbox,
    ) -> bool:
        wrist = self._visible_point(points, point_conf, wrist_idx)
        elbow = self._visible_point(points, point_conf, elbow_idx)
        shoulder = self._visible_point(points, point_conf, shoulder_idx)
        if wrist is None or elbow is None or shoulder is None:
            history.clear()
            return False

        raised = wrist[1] < shoulder[1] and wrist[1] < elbow[1]
        history.append((now, wrist[0], wrist[1], raised))
        while history and now - history[0][0] > self._WAVE_WINDOW_SECONDS:
            history.popleft()

        raised_samples = [sample for sample in history if sample[3]]
        if len(raised_samples) < self._WAVE_MIN_SAMPLES:
            return False

        xs = [sample[1] for sample in raised_samples]
        box_width = max(person_bbox[2] - person_bbox[0], 1.0)
        movement_threshold = max(25.0, box_width * 0.08)
        span = max(xs) - min(xs)
        if span < movement_threshold:
            return False

        significant_deltas = []
        min_delta = max(8.0, box_width * 0.02)
        for first, second in zip(xs, xs[1:]):
            delta = second - first
            if abs(delta) >= min_delta:
                significant_deltas.append(delta)

        sign_changes = 0
        for first, second in zip(significant_deltas, significant_deltas[1:]):
            if first == 0 or second == 0:
                continue
            if first * second < 0:
                sign_changes += 1

        return sign_changes >= 1

    def _visible_point(self, points, point_conf, index: int):
        if index >= len(points):
            return None
        confidence = point_conf[index] if point_conf is not None else 1.0
        if confidence < self._POINT_CONFIDENCE:
            return None
        x_coord, y_coord = points[index][:2]
        return float(x_coord), float(y_coord)

    def _prune_tracks(self) -> None:
        now = time.monotonic()
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.get("last_seen", 0.0) > self._TRACK_MAX_AGE_SECONDS
        ]
        for track_id in expired:
            del self._tracks[track_id]

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _bbox_iou(first_bbox, second_bbox) -> float:
        x1 = max(first_bbox[0], second_bbox[0])
        y1 = max(first_bbox[1], second_bbox[1])
        x2 = min(first_bbox[2], second_bbox[2])
        y2 = min(first_bbox[3], second_bbox[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if intersection <= 0.0:
            return 0.0

        first_area = max(0.0, first_bbox[2] - first_bbox[0]) * max(
            0.0, first_bbox[3] - first_bbox[1]
        )
        second_area = max(0.0, second_bbox[2] - second_bbox[0]) * max(
            0.0, second_bbox[3] - second_bbox[1]
        )
        union = first_area + second_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    @staticmethod
    def _point_in_bbox(point, bbox) -> bool:
        x_coord, y_coord = point
        return bbox[0] <= x_coord <= bbox[2] and bbox[1] <= y_coord <= bbox[3]

    @classmethod
    def _points_bbox(cls, points, point_conf):
        visible_points = [
            tuple(point[:2])
            for index, point in enumerate(points)
            if point_conf is None or point_conf[index] >= cls._POINT_CONFIDENCE
        ]
        if not visible_points:
            return (0.0, 0.0, 0.0, 0.0)

        xs = [point[0] for point in visible_points]
        ys = [point[1] for point in visible_points]
        return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

    @staticmethod
    def _wave_label(left_waving: bool, right_waving: bool) -> str:
        if left_waving and right_waving:
            return "both hands"
        if left_waving:
            return "left hand"
        return "right hand"

    @staticmethod
    def _color_for_class(class_index: int):
        palette = [
            (70, 179, 230),
            (92, 212, 120),
            (84, 87, 255),
            (255, 191, 0),
            (255, 105, 180),
            (180, 105, 255),
        ]
        return palette[class_index % len(palette)]
