"""
batch_processor.py
------------------
Automatically process all videos in the dataset directory.
Simulates a metro train with multiple coaches by treating each video as a coach.
Aggregates crowd density statistics across all coaches.

Usage
-----
# Process all videos with default settings
    python batch_processor.py

# Process with flow algorithm (experimental)
    python batch_processor.py --algo flow

# Process only first 5 videos
    python batch_processor.py --limit 5

# Set output directory for results
    python batch_processor.py --output results/

# Enable automatic restart (loop continuously)
    python batch_processor.py --loop
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from config import Config
from utils import (
    PersonDetector, 
    OpticalFlowPipeline,
)
from utils.forecaster import CrowdAnalyzer
from utils.notifier import Notifier
from utils.dashboard import DashboardReport
from utils.learning import AutoLearner


# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_DIR = Path("data/dataset")
OUTPUT_DIR = Path("outputs")


# ── Batch Statistics Tracker ───────────────────────────────────────────────────

class CoachStatistics:
    """Tracks crowd statistics for a single coach (video)."""
    
    def __init__(self, coach_id: int, video_name: str):
        self.coach_id = coach_id
        self.video_name = video_name
        self.frame_count = 0
        self.total_people = 0
        self.max_people = 0
        self.avg_people = 0.0
        self.processing_time = 0.0
        self.start_time = None
        
    def record(self, count: int):
        """Record a frame's person count."""
        self.frame_count += 1
        self.total_people += count
        self.max_people = max(self.max_people, count)
        
    def finalize(self, elapsed: float):
        """Finalize statistics."""
        self.processing_time = elapsed
        if self.frame_count > 0:
            self.avg_people = self.total_people / self.frame_count
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            'coach_id': self.coach_id,
            'video': self.video_name,
            'frames': self.frame_count,
            'total_people': self.total_people,
            'avg_people': f"{self.avg_people:.1f}",
            'max_people': self.max_people,
            'processing_time_sec': f"{self.processing_time:.2f}",
            'fps': f"{self.frame_count / max(self.processing_time, 0.1):.1f}",
        }


class BatchSession:
    """Manages batch processing session and statistics."""
    
    def __init__(self, output_dir: Path, cfg: Config | None = None, keep_last_reports: int = 5):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coaches: list[CoachStatistics] = []
        self.session_start = datetime.now()
        self.cfg = cfg or Config()
        self.analyzer = CrowdAnalyzer(cfg=self.cfg)
        self.notifier = Notifier()
        self.keep_last_reports = keep_last_reports
        
    def add_coach(self, stats: CoachStatistics):
        """Add coach statistics to session."""
        self.coaches.append(stats)
    
    def cleanup_old_reports(self):
        """
        Remove legacy timestamped report files and keep only the current fixed report files.
        Old timestamped files are no longer needed because the system now writes
        to a single set of report files.
        """
        patterns = [
            'batch_report_*.csv',
            'batch_summary_*.txt',
            'report_*.json',
            'alerts_*.csv'
        ]

        for pattern in patterns:
            for old_file in self.output_dir.glob(pattern):
                try:
                    old_file.unlink()
                    print(f"  🗑️ Removed legacy file: {old_file.name}")
                except Exception as e:
                    print(f"  ⚠️ Could not delete {old_file.name}: {e}")

    
    def save_report(self):
        """Save statistics report to CSV and text."""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Clean up any legacy timestamped files before writing fixed output names.
        self.cleanup_old_reports()

        # CSV report
        csv_path = self.output_dir / "batch_report.csv"
        existing_rows = {}

        if csv_path.exists():
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get('coach_id'), row.get('video'))
                    existing_rows[key] = row

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if self.coaches:
                fieldnames = self.coaches[0].to_dict().keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for coach in self.coaches:
                    row = coach.to_dict()
                    key = (str(row['coach_id']), row['video'])
                    existing_rows[key] = row
                for row in existing_rows.values():
                    writer.writerow(row)

        # Analyze with forecaster
        for i, coach in enumerate(self.coaches, start=1):
            analysis = self.analyzer.analyze_coach(i, coach.to_dict())

        # Text summary with predictions
        summary_path = self.output_dir / "batch_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("METRO TRAIN BATCH PROCESSING REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Coaches Processed: {len(self.coaches)}\n")
            f.write("\n" + "-" * 70 + "\n")
            f.write("COACH STATISTICS:\n")
            f.write("-" * 70 + "\n")
            
            total_frames = 0
            total_people = 0
            total_time = 0.0
            
            for coach in self.coaches:
                f.write(f"\nCoach {coach.coach_id}: {coach.video_name}\n")
                f.write(f"  Frames Processed:  {coach.frame_count}\n")
                f.write(f"  Total People:      {coach.total_people}\n")
                f.write(f"  Average Density:   {coach.avg_people:.1f} people/frame\n")
                f.write(f"  Peak Crowding:     {coach.max_people} people\n")
                f.write(f"  Processing Time:   {coach.processing_time:.2f}s\n")
                f.write(f"  Speed:             {coach.frame_count / max(coach.processing_time, 0.1):.1f} FPS\n")
                
                total_frames += coach.frame_count
                total_people += coach.total_people
                total_time += coach.processing_time
            
            # Aggregate stats
            f.write("\n" + "=" * 70 + "\n")
            f.write("AGGREGATE STATISTICS:\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total Frames:      {total_frames}\n")
            f.write(f"Total People Detected: {total_people}\n")
            f.write(f"Average Density:   {total_people / max(total_frames, 1):.1f} people/frame\n")
            f.write(f"Total Processing Time: {total_time:.2f}s\n")
            f.write(f"Overall Speed:     {total_frames / max(total_time, 0.1):.1f} FPS\n")
            
            # Predictions section
            f.write("\n" + "=" * 70 + "\n")
            f.write("CROWD PREDICTIONS & ALERTS:\n")
            f.write("=" * 70 + "\n")
            
            for coach_id in sorted(self.analyzer.coach_data.keys()):
                analysis = self.analyzer.coach_data[coach_id]
                f.write(f"\nCoach {coach_id}: {analysis['video']}\n")
                f.write(f"  Current Average:  {analysis['current_avg']:.1f} people\n")
                f.write(f"  Current Peak:     {analysis['current_max']} people\n")
                f.write(f"  Risk Level:       {analysis['risk_level']}\n")
                f.write(f"  Next 5 Frames:    {[f'{p:.0f}' for p in analysis['forecast'][:5]]}\n")
                f.write(f"  Recommendation:   {analysis['recommendation']}\n")
        
        print(f"\n✓ Reports saved:")
        print(f"  CSV: {csv_path}")
        print(f"  Summary: {summary_path}")
        
        # Print prediction report to console
        self.analyzer.print_report()
        
        # Save alert log
        alert_log_path = self.output_dir / "alerts.csv"
        self.analyzer.alerts.save_alert_log(str(alert_log_path), append=True)
        
        # Generate dashboard
        dashboard = DashboardReport(self.output_dir)
        html_path = dashboard.generate_html(
            coaches_data=list(self.analyzer.coach_data.values()),
            alerts_data=self.analyzer.alerts.alert_log,
            timestamp=timestamp
        )
        json_path = dashboard.generate_json_report(
            coaches_data=list(self.analyzer.coach_data.values()),
            alerts_data=self.analyzer.alerts.alert_log,
            timestamp=timestamp
        )
        
        print(f"  HTML Dashboard: {html_path}")
        print(f"  JSON Report: {json_path}")
        print(f"  (Dashboard updates in place - open 'dashboard.html' to view all coaches)")

        
        # Cleanup legacy timestamped reports to save storage
        print(f"\n🧹 Removing legacy timestamped reports and keeping only the fixed report files:")
        self.cleanup_old_reports()


# ── Processing Functions ──────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    coach_id: int,
    cfg: Config,
    processor,
    algo: str,
    headless: bool = False
) -> CoachStatistics:
    """
    Process a single video and return statistics.
    
    Parameters
    ----------
    video_path : Path
        Path to video file.
    coach_id : int
        Coach identifier (1, 2, 3, ...).
    cfg : Config
        Application config.
    processor : PersonDetector | OpticalFlowPipeline
        Detector instance.
    algo : str
        Algorithm ("yolo" or "flow").
    headless : bool
        If True, don't display video (faster).
    
    Returns
    -------
    CoachStatistics
        Statistics for this coach.
    """
    stats = CoachStatistics(coach_id, video_path.name)
    stats.start_time = time.perf_counter()
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"✗ Cannot open {video_path.name}")
        return stats
    
    print(f"\n[Coach {coach_id}] Processing {video_path.name}...")
    
    frame_idx = 0
    prev_time = time.perf_counter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Process frame
            if algo == "yolo":
                count, _ = processor.count(frame)
            else:  # flow
                _, _, _, count = processor.process(frame, draw_labels=False)
            
            stats.record(count)
            
            # Print progress every 30 frames
            if frame_idx % 30 == 0:
                now = time.perf_counter()
                fps = 30 / (now - prev_time)
                print(f"  Frame {frame_idx}: {count} people | {fps:.1f} FPS", end='\r')
                prev_time = now
            
            # Display (if not headless)
            if not headless:
                cv2.imshow(f"Coach {coach_id}: {video_path.name}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"\n[Coach {coach_id}] Quit requested.")
                    break
    
    finally:
        elapsed = time.perf_counter() - stats.start_time
        stats.finalize(elapsed)
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
    
    print(f"  ✓ Coach {coach_id} complete: {stats.frame_count} frames, "
          f"avg {stats.avg_people:.1f} people, "
          f"{stats.frame_count / max(elapsed, 0.1):.1f} FPS")
    
    return stats


def natural_sort_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def discover_videos(dataset_dir: Path, limit: int | None = None) -> list[Path]:
    """
    Discover all MP4 videos in dataset directory (sorted numerically).
    
    Parameters
    ----------
    dataset_dir : Path
        Directory containing videos.
    limit : int | None
        If set, only return first N videos.
    
    Returns
    -------
    list[Path]
        Sorted list of video paths.
    """
    videos = sorted(dataset_dir.glob("*.mp4"), key=natural_sort_key)
    if limit:
        videos = videos[:limit]
    return videos


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch process all metro videos to simulate multi-coach train.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        choices=["yolo", "flow"],
        default="yolo",
        help="Algorithm to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLOv8 model (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (0–1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N videos (for testing).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Don't display video (faster).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop continuously (process again after all videos).",
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="Retrain the LSTM forecast model after processing.",
    )
    parser.add_argument(
        "--alerts",
        choices=["console", "email", "sms", "slack"],
        action="append",
        default=[],
        help="Alert channels to enable (can be used multiple times).",
    )
    parser.add_argument(
        "--keep-reports",
        type=int,
        default=5,
        help="Number of latest report batches to keep (older ones are deleted).",
    )
    
    args = parser.parse_args()
    
    # Default alerts to console if not specified
    if not args.alerts:
        args.alerts = ['console']
    
    # Build config
    cfg = Config()
    if args.model:
        cfg.MODEL_PATH = args.model
    if args.conf is not None:
        cfg.CONFIDENCE_THRESHOLD = args.conf
    
    # Initialize processor
    print(f"[Batch] Initializing {args.algo.upper()} processor...")
    if args.algo == "yolo":
        processor = PersonDetector(cfg)
    else:
        processor = OpticalFlowPipeline(cfg)
    
    # Discover videos
    videos = discover_videos(DATASET_DIR, limit=args.limit)
    if not videos:
        print(f"✗ No MP4 videos found in {DATASET_DIR}")
        return
    
    print(f"[Batch] Found {len(videos)} video(s) to process.\n")
    
    # Main loop
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")
        
        session = BatchSession(args.output, cfg=cfg, keep_last_reports=args.keep_reports)
        
        for coach_id, video_path in enumerate(videos, start=1):
            stats = process_video(
                video_path=video_path,
                coach_id=coach_id,
                cfg=cfg,
                processor=processor,
                algo=args.algo,
                headless=args.headless
            )
            session.add_coach(stats)
        
        # Save reports
        session.save_report()
        
        # Retraining is opt-in because it mutates the saved forecast model.
        if args.learn:
            learner = AutoLearner()
            learner.learn()
        
        # Loop control
        if not args.loop:
            break
        
        print(f"\nCompleted iteration {iteration}. Restarting in 5 seconds...")
        print("(Press Ctrl+C to exit)")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

