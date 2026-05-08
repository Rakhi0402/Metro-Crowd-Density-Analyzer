from .optical_flow import (
    LucasKanadeTracker,
    DenseFarnebackTracker
)
from .preprocessing import VideoPreprocessor
from .feature_extractor import FASTFeatureExtractor
from .density_estimator import DensityEstimator


class OpticalFlowPipeline:
    """
    Wrapper pipeline combining preprocessing + flow + density
    """

    def __init__(self, cfg):
        self.config = cfg

        self.preprocessor = VideoPreprocessor(
            blur_kernel=cfg.OF_BLUR_KERNEL,
            threshold=cfg.OF_DIFF_THRESHOLD
        )

        self.feature_extractor = FASTFeatureExtractor(
            threshold=cfg.OF_FAST_THRESHOLD,
            max_points=cfg.OF_MAX_POINTS
        )

        # 🔥 You can switch tracker here
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

    # ─────────────────────────────
    # MAIN PROCESS FUNCTION
    # ─────────────────────────────
    def process(self, frame, draw_labels=True):

        # Step 1: preprocess
        gray_blurred, motion_mask = self.preprocessor.process(frame)

        # Step 2: extract keypoints
        keypoints = self.feature_extractor.extract(
            gray_blurred,
            mask=motion_mask
        )

        # Step 3: track motion
        annotated_frame, motion_vectors, flow_count = self.tracker.track(
            current_gray=gray_blurred,
            _=keypoints,
            frame=frame
        )

        # Step 4: density estimation
        output_frame, density_score = self.density_estimator.estimate_and_draw(
            frame=annotated_frame,
            motion_vectors=motion_vectors,
            draw_labels=draw_labels
        )

        # Step 5: estimate people
        estimated_people = int(flow_count / 8)

        return flow_count, density_score, output_frame, estimated_people