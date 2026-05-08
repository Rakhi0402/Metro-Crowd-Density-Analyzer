"""
feature_extractor.py
--------------------
Module for Step 2 of the IEEE 7275777 methodology.
Extracts FAST (Features from Accelerated Segment Test) corners from the 
preprocessed ROI frames to mathematically identify distinct high-contrast 
points (such as heads/shoulders) within a dense crowd.
"""

import cv2
import numpy as np


class FASTFeatureExtractor:
    """
    Identifies high-contrast corners within dense crowds using FAST.
    Designed to feed directly into Lucas-Kanade/Horn-Schunck optical flow.
    """

    def __init__(self, threshold: int = 15, nonmax_suppression: bool = True, max_points: int = 400):
        """
        Parameters
        ----------
        threshold : int
            FAST difference threshold. Lower picks up more subtle corners (denser crowds).
        nonmax_suppression : bool
            Whether to apply non-maximum suppression to prevent clumped clusters.
        max_points : int
            Maximum number of features to extract, prioritising the highest response score.
        """
        self.fast = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=nonmax_suppression
        )
        self.max_points = max_points

    def extract(self, gray_frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        """
        Extracts mathematical corner points from a frame.

        Parameters
        ----------
        gray_frame : np.ndarray
            The preprocessed grayscale frame.
        mask : np.ndarray, optional
            Binary ROI mask (e.g. from frame-differencing) to restrict search area.

        Returns
        -------
        np.ndarray | None
            Array of shape (N, 1, 2) of float32 coordinates, or None if no points.
        """
        keypoints = self.fast.detect(gray_frame, mask=mask)
        
        if not keypoints:
            return None

        # Sort by response score descending to get strongest crowd features
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        
        # Limit the track count for computational efficiency in dense fields
        keypoints = keypoints[:self.max_points]

        # Convert to numpy array format [(X, Y)] expected by Optical Flow algorithms
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        return pts.reshape(-1, 1, 2)

    def draw_keypoints(self, frame: np.ndarray, points: np.ndarray | None) -> np.ndarray:
        """
        Visually overlay extracted keypoints on the given frame.

        Parameters
        ----------
        frame : np.ndarray
            The original BGR frame.
        points : np.ndarray | None
            Points extracted via `extract`.

        Returns
        -------
        np.ndarray
            The annotated frame overlayed with FAST points.
        """
        annotated = frame.copy()
        if points is not None:
            for pt in points:
                x, y = pt[0]
                # Draw a distinct bright red circle (matches the provided IEEE figure)
                cv2.circle(annotated, (int(x), int(y)), 4, (0, 0, 255), -1)
        return annotated
