

import cv2
import numpy as np


class VideoPreprocessor:


    def __init__(self, blur_kernel: tuple[int, int] = (5, 5), threshold: int = 25):
      
        self.blur_kernel = blur_kernel
        self.threshold = threshold
        self.prev_frame_gray: np.ndarray | None = None

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      
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
