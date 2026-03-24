from __future__ import annotations

import math

import numpy as np


class InsightFaceRecognizer:
    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = float(threshold)
        self._app = None
        self._load_error = None

    @property
    def available(self) -> bool:
        self._ensure_app()
        return self._app is not None

    @property
    def load_error(self) -> str | None:
        self._ensure_app()
        return self._load_error

    def learn_from_pose_bbox(
        self,
        frame_bgr,
        pose_bbox: list[float] | tuple[float, float, float, float],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        face = self._best_face_for_pose(frame_bgr, pose_bbox)
        if face is None:
            return None, None
        embedding = self._face_embedding(face)
        crop = self._face_crop(frame_bgr, face)
        return embedding, crop

    def find_pose_for_identity(
        self,
        frame_bgr,
        pose_data: list[dict],
        target_embedding: np.ndarray,
    ) -> dict | None:
        faces = self._detect_faces(frame_bgr)
        if not faces:
            return None

        best_pose = None
        best_score = -1.0
        for face in faces:
            embedding = self._face_embedding(face)
            if embedding is None:
                continue
            score = self._cosine_similarity(embedding, target_embedding)
            if score < self.threshold:
                continue
            pose = self._pose_for_face(pose_data, face)
            if pose is None:
                continue
            if score > best_score:
                best_score = score
                best_pose = pose
        return best_pose

    def _best_face_for_pose(self, frame_bgr, pose_bbox) -> object | None:
        faces = self._detect_faces(frame_bgr)
        if not faces:
            return None
        px1, py1, px2, py2 = [float(v) for v in pose_bbox]
        pcx = (px1 + px2) * 0.5
        pcy = (py1 + py2) * 0.5
        pdiag = max(1.0, ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5)
        best = None
        best_score = -1e9
        for face in faces:
            bbox = self._face_bbox(face)
            if bbox is None:
                continue
            fx1, fy1, fx2, fy2 = bbox
            fcx = (fx1 + fx2) * 0.5
            fcy = (fy1 + fy2) * 0.5
            iou = self._bbox_iou((fx1, fy1, fx2, fy2), (px1, py1, px2, py2))
            dist_norm = (((fcx - pcx) ** 2 + (fcy - pcy) ** 2) ** 0.5) / pdiag
            area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
            # Prefer overlap, then closeness to pose center, then larger faces.
            score = (iou * 3.0) - dist_norm + (area * 1e-6)
            cx = (fx1 + fx2) * 0.5
            cy = (fy1 + fy2) * 0.5
            if iou <= 0.0 and not (px1 <= cx <= px2 and py1 <= cy <= py2):
                # Allow close faces near the pose box edge.
                if dist_norm > 0.9:
                    continue
            if score > best_score:
                best = face
                best_score = score
        return best

    def _pose_for_face(self, pose_data: list[dict], face: object) -> dict | None:
        bbox = self._face_bbox(face)
        if bbox is None:
            return None
        fx1, fy1, fx2, fy2 = bbox
        cx = (fx1 + fx2) * 0.5
        cy = (fy1 + fy2) * 0.5
        for pose in pose_data:
            if not isinstance(pose, dict):
                continue
            pbbox = pose.get("bbox")
            if not isinstance(pbbox, list) or len(pbbox) != 4:
                continue
            px1, py1, px2, py2 = [float(v) for v in pbbox]
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                return pose
        return None

    def _face_crop(self, frame_bgr, face: object):
        bbox = self._face_bbox(face)
        if bbox is None:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x1 = min(w - 1, x1)
        x2 = min(w, max(x1 + 1, x2))
        y1 = min(h - 1, y1)
        y2 = min(h, max(y1 + 1, y2))
        return frame_bgr[y1:y2, x1:x2].copy()

    def _face_embedding(self, face: object) -> np.ndarray | None:
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            return None
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-8:
            return None
        return arr / norm

    def _face_bbox(self, face: object) -> tuple[float, float, float, float] | None:
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            return None
        arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if arr.size < 4:
            return None
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

    def _detect_faces(self, frame_bgr) -> list:
        self._ensure_app()
        if self._app is None:
            return []
        try:
            faces = self._app.get(frame_bgr)
        except Exception:
            return []
        if not isinstance(faces, list):
            return []
        return faces

    def _ensure_app(self) -> None:
        if self._app is not None or self._load_error is not None:
            return
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app
        except Exception as exc:
            self._load_error = str(exc)

    @staticmethod
    def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
        if first.size == 0 or second.size == 0:
            return -1.0
        if first.shape != second.shape:
            return -1.0
        dot = float(np.dot(first, second))
        if not math.isfinite(dot):
            return -1.0
        return dot

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
