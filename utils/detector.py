from __future__ import annotations

import numpy as np
from ultralytics import YOLO
from config import Config


class PersonDetector:
    """
    YOLOv8 + ByteTrack-based person detector with persistent IDs.
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or Config()

        print(f"[PersonDetector] Loading model : {self._cfg.MODEL_PATH}")
        print(f"[PersonDetector] Tracker       : {self._cfg.TRACKER}")

        self.model = YOLO(self._cfg.MODEL_PATH)
        self.conf  = self._cfg.CONFIDENCE_THRESHOLD
        self.iou   = self._cfg.NMS_IOU_THRESHOLD

        print("[PersonDetector] Model ready.")

    # ─────────────────────────────────────────────
    # 🔍 DETECTION + TRACKING
    # ─────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns tracked detections with IDs.
        """

        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[self._cfg.PERSON_CLASS_ID],
            tracker=self._cfg.TRACKER,
            persist=True,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:

                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Confidence
                conf = float(box.conf[0])

                # Track ID (critical)
                track_id = int(box.id[0]) if box.id is not None else None

                # Ignore untracked objects
                if track_id is None:
                    continue

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "conf": conf,
                    "track_id": track_id
                })

        return detections

    # ─────────────────────────────────────────────
    # 📊 COUNTING
    # ─────────────────────────────────────────────
    def count(self, frame: np.ndarray) -> tuple[int, list[dict]]:
        detections = self.detect(frame)
        count = len(detections)  # already filtered by valid track_id
        return count, detections

    # ─────────────────────────────────────────────
    # 🧠 UTILITY (OPTIONAL)
    # ─────────────────────────────────────────────
    def get_boxes_array(self, detections: list[dict]) -> np.ndarray:
        """
        Convert detections → numpy array (useful for analytics/ML)
        """
        if not detections:
            return np.empty((0, 4))

        return np.array([d["bbox"] for d in detections])