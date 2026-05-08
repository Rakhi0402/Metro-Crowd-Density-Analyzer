"""
preprocessing.py
----------------
Module for Step 1 of the IEEE 7275777 methodology.
Handles converting raw BGR frames to denoised grayscale, and computes
a frame-differencing motion mask to isolate regions of interest (ROI)
where crowd movement is actively occurring.
"""

import cv2
import numpy as np


class VideoPreprocessor:
    """
    Stateful preprocessor that isolates active crowd motion.
    Maintains the previous frame state to perform absolute differencing.
    """

    def __init__(self, blur_kernel: tuple[int, int] = (5, 5), threshold: int = 25):
        """
        Parameters
        ----------
        blur_kernel : tuple[int, int]
            Kernel size for Gaussian blur to suppress camera noise.
        threshold : int
            Pixel-intensity threshold for the frame difference. 
            Lower catches more subtle movement, higher ignores noise.
        """
        self.blur_kernel = blur_kernel
        self.threshold = threshold
        self.prev_frame_gray: np.ndarray | None = None

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes a raw BGR frame into a denoised grayscale frame and a motion mask.

        Parameters
        ----------
        frame : np.ndarray
            Current raw BGR frame from OpenCV.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (blurred_gray_frame, motion_roi_mask)
            - blurred_gray_frame: Grayscale, blurred image for feature detection.
            - motion_roi_mask: Binary mask (0 or 255) where motion was detected.
        """
        # Step 1.1: Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 1.2: Gaussian Blur (noise reduction so noise isn't counted as motion)
        blurred_gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # Step 1.3: Initialize state on first frame
        if self.prev_frame_gray is None:
            self.prev_frame_gray = blurred_gray.copy()
            # No motion on the very first frame
            return blurred_gray, np.zeros_like(blurred_gray)

        # Step 1.4: Frame Differencing
        # Compute absolute difference between current and previous frame
        frame_diff = cv2.absdiff(self.prev_frame_gray, blurred_gray)

        # Step 1.5: Binary Thresholding to create ROI mask
        _, motion_mask = cv2.threshold(
            frame_diff, self.threshold, 255, cv2.THRESH_BINARY
        )

        # Dilation can optionally be applied here to bridge gaps in the motion mask
        # motion_mask = cv2.dilate(motion_mask, None, iterations=2)

        # Update state for next frame
        self.prev_frame_gray = blurred_gray.copy()

        return blurred_gray, motion_mask
