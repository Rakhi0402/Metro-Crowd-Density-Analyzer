"""
pipeline.py
-----------
Integrated Optical Flow Pipeline combining preprocessing, feature extraction,
tracking, and density estimation for crowd counting using the IEEE 7275777 methodology.
"""

from __future__ import annotations

import numpy as np

from config import Config
from .preprocessing import VideoPreprocessor
from .feature_extractor import FASTFeatureExtractor
from .optical_flow import DenseFarnebackTracker
from .density_estimator import DensityEstimator


class OpticalFlowPipeline:
    """
    Wires together the IEEE 7275777 methodology modules into one clean pass.
    
    Combines:
    1. Preprocessing (blur, motion detection)
    2. Feature extraction (FAST corners)
    3. Optical flow tracking (Dense Farneback)
    4. Density estimation and grid analysis
    """
    
    def __init__(self, cfg: Config):
        """
        Initialize the optical flow pipeline with configuration.
        
        Parameters
        ----------
        cfg : Config
            Application configuration object.
        """
        self.config = cfg
        self.preprocessor = VideoPreprocessor(
            blur_kernel=cfg.OF_BLUR_KERNEL, 
            threshold=cfg.OF_DIFF_THRESHOLD
        )
        self.feature_extractor = FASTFeatureExtractor(
            threshold=cfg.OF_FAST_THRESHOLD, 
            max_points=cfg.OF_MAX_POINTS
        )
        self.tracker = DenseFarnebackTracker(
            pyr_scale=cfg.OF_DENSE_PYRAMID_SCALE,
            levels=cfg.OF_DENSE_LEVELS,
            winsize=cfg.OF_DENSE_WINSIZE,
            iterations=cfg.OF_DENSE_ITERATIONS,
            poly_n=cfg.OF_DENSE_POLY_N,
            poly_sigma=cfg.OF_DENSE_POLY_SIGMA,
            min_motion=cfg.OF_MIN_MOTION,
            sample_step=cfg.OF_DENSE_SAMPLE_STEP
        )
        self.density_estimator = DensityEstimator(
            grid_rows=cfg.OF_GRID_ROWS,
            grid_cols=cfg.OF_GRID_COLS,
            threshold_high=cfg.OF_GRID_HIGH_THRESH,
            threshold_medium=cfg.OF_GRID_MED_THRESH
        )
        self.count_history = []
        self.smooth_window = cfg.WAIT_DELAY * 15  # Adaptive window size
        if self.smooth_window < 5: 
            self.smooth_window = 15

    def process(self, frame: np.ndarray, draw_labels: bool = True) -> tuple[int, int, np.ndarray, int]:
        """
        Process a frame through the optical flow pipeline.
        
        Parameters
        ----------
        frame : np.ndarray
            Input video frame (BGR).
        draw_labels : bool
            Whether to draw labels on output frame.
        
        Returns
        -------
        tuple[int, int, np.ndarray, int]
            - active_flow_count: Number of active motion vectors
            - global_density_score: Density metric (0-100)
            - annotated_frame: Output frame with annotations
            - smoothed_count: Temporally smoothed person count estimate
        """
        # 1. Preprocessing (Isolate Motion ROI)
        gray_blurred, motion_roi_mask = self.preprocessor.process(frame)
        
        # 2. Extract Features (Inside ROI only)
        new_keypoints = self.feature_extractor.extract(gray_blurred, mask=motion_roi_mask)
        
        # 3. Track optical flow vectors
        annotated_frame, motion_vectors, active_flow_count = self.tracker.track(
            current_gray=gray_blurred, 
            _ = None,  # Placeholder for backward compatibility
            frame=frame
        )
        
        # 4. Cluster vectors and draw density grid
        grid_annotated_frame, global_density_score = self.density_estimator.estimate_and_draw(
            frame=annotated_frame, 
            motion_vectors=motion_vectors,
            draw_labels=draw_labels
        )
        
        # 4.1. Add Premium Heatmap Overlay
        grid_annotated_frame = self.density_estimator.draw_heatmap(
            frame=grid_annotated_frame,
            motion_vectors=motion_vectors
        )

        # 5. Temporal Smoothing (Rolling Average)
        raw_count = int(active_flow_count / self.config.OF_VECTORS_PER_PERSON)
        self.count_history.append(raw_count)
        if len(self.count_history) > self.smooth_window:
            self.count_history.pop(0)
        smoothed_count = int(sum(self.count_history) / len(self.count_history))
        
        return active_flow_count, global_density_score, grid_annotated_frame, smoothed_count
