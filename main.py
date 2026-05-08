from __future__ import annotations

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch

from config import Config
from config.zones import get_zones   # ✅ FIXED

# EXISTING
from utils import PersonDetector

# NEW
from utils.zone_counter import assign_zone
from utils.flow_estimator import FlowEstimator
from utils.crowd_classifier import classify
from utils.logger import DataLogger
from utils.predictor import Predictor
from forecasting.lstm_model import LSTMModel

# FLOW PIPELINE
from utils.optical_flow_pipeline import OpticalFlowPipeline   # ✅ FIXED


# ───────────────── CLI ─────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--algo", choices=["yolo", "flow"], default="yolo")
    parser.add_argument("--delay", type=int, default=1)
    return parser.parse_args()


def parse_source(source):
    """Allow camera indexes to be passed from the CLI as --source 0."""
    if source is None:
        return None
    try:
        return int(source)
    except (TypeError, ValueError):
        return source


def load_predictor(model_path: str):
    if not model_path:
        return None

    path = Path(model_path)
    if not path.exists():
        print(f"[Forecast] Model not found, skipping prediction: {path}")
        return None

    try:
        model = LSTMModel()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print(f"[Forecast] Loaded model: {path}")
        return Predictor(model)
    except Exception as exc:
        print(f"[Forecast] Could not load model, skipping prediction: {exc}")
        return None


# ───────────────── MAIN LOOP ─────────────────

def run(cfg, processor, algo):

    cap = cv2.VideoCapture(cfg.SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {cfg.SOURCE!r}")

    flow_estimator = FlowEstimator()
    logger = DataLogger()

    # Forecast model
    predictor = load_predictor(getattr(cfg, "FORECAST_MODEL", None))

    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        # ✅ Dynamic zones
        zones = get_zones(w, h)

        timestamp = time.time()

        # ============================
        # YOLO MODE
        # ============================
        if algo == "yolo":

            detections = processor.detect(frame)

            zone_counts = {z: 0 for z in zones}

            # 🔥 Draw zones (debug + important)
            for name, poly in zones.items():
                pts = np.array(poly, np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
                cv2.putText(frame, name, tuple(pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cx, cy = d["center"]
                tid = d["track_id"]

                # Zone assignment
                zone = assign_zone((x1, y1, x2, y2), zones)
                if zone in zone_counts:
                    zone_counts[zone] += 1

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            total_count = len(detections)

            # Flow estimation
            flow_in, flow_out = flow_estimator.update(detections)

            # Crowd classification
            crowd_level = classify(total_count, cfg)

            # ✅ FIXED prediction input
            prediction = None
            if predictor:
                prediction = predictor.update([
                    total_count,
                    zone_counts.get("door_left", 0),
                    zone_counts.get("door_right", 0),
                    zone_counts.get("center", 0),
                    flow_in,
                    flow_out
                ])

            # Logging
            logger.log([
                timestamp,
                total_count,
                zone_counts.get("door_left", 0),
                zone_counts.get("door_right", 0),
                zone_counts.get("center", 0),
                flow_in,
                flow_out
            ])

            # FPS
            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            # HUD
            cv2.putText(frame, f"Count: {total_count}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Level: {crowd_level}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"In:{flow_in} Out:{flow_out}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if prediction:
                cv2.putText(frame, f"Pred: {int(prediction)}", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ============================
        # FLOW MODE
        # ============================
        else:
            active_flows, density_score, frame, estimated_people = processor.process(frame)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            cv2.putText(frame, f"Flow Count: {estimated_people}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Metro Crowd System", frame)

        if cv2.waitKey(cfg.WAIT_DELAY) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ───────────────── ENTRY ─────────────────

def main():
    args = parse_args()

    cfg = Config()

    if args.source is not None:
        cfg.SOURCE = parse_source(args.source)
    if args.model:
        cfg.MODEL_PATH = args.model
    if args.conf is not None:
        cfg.CONFIDENCE_THRESHOLD = args.conf

    cfg.WAIT_DELAY = args.delay

    if args.algo == "yolo":
        processor = PersonDetector(cfg)
    else:
        processor = OpticalFlowPipeline(cfg)

    run(cfg, processor, args.algo)


if __name__ == "__main__":
    main()
