"""
optical_flow.py
---------------
Module for Step 3 of the IEEE 7275777 methodology.
Tracks the spatial displacement of FAST keypoints between frame T and T+1
using Pyramidal Lucas-Kanade (cv2.calcOpticalFlowPyrLK). 
It calculates the magnitude and angle of the resulting motion vectors, 
filtering out stationary noise, and maintaining temporal tracking state.
"""

import cv2
import numpy as np


class LucasKanadeTracker:
    """
    Stateful tracker that applies Lucas-Kanade optical flow to mathematical corners.
    Filters raw vectors by magnitude to isolate true crowd movement.
    """

    def __init__(self, win_size: tuple[int, int] = (15, 15), max_level: int = 2, min_motion: float = 1.0, max_motion: float = 50.0):
        """
        Parameters
        ----------
        win_size : tuple[int, int]
            Size of the search window at each pyramid level.
        max_level : int
            0-based maximal pyramid level number.
        min_motion : float
            Minimum magnitude (pixels) to be considered valid crowd movement.
        max_motion : float
            Maximum magnitude (pixels) to reject impossible jumps/noise.
        """
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.min_motion = min_motion
        self.max_motion = max_motion

        # Tracking state T
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.trail_mask: np.ndarray | None = None

    def track(self, current_gray: np.ndarray, new_keypoints: np.ndarray | None, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Executes the Optical Flow pipeline on the current frame.

        Parameters
        ----------
        current_gray : np.ndarray
            The grayscale frame at T+1.
        new_keypoints : np.ndarray | None
            Fresh FAST keypoints. If previous points exist, we track those instead,
            but we can use new_keypoints to replenish lost tracks.
        frame : np.ndarray
            The raw BGR frame used strictly as a canvas to draw visualisations.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            (annotated_frame, valid_motion_vectors, active_flow_count)
        """
        annotated_frame = frame.copy()
        valid_vectors = np.empty((0, 4), dtype=np.float32)  # Will store (x1, y1, x2, y2)

        # Initialize state on the first frame
        if self.prev_gray is None or self.trail_mask is None:
            self.prev_gray = current_gray.copy()
            self.prev_points = new_keypoints
            self.trail_mask = np.zeros_like(frame)
            return annotated_frame, valid_vectors, 0

        # If we have no points to track, try to adopt the new ones
        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_gray = current_gray.copy()
            self.prev_points = new_keypoints
            return annotated_frame, valid_vectors, 0

        # Calculate Pyramidal Lucas-Kanade Optical Flow from T to T+1
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, current_gray, self.prev_points, None, **self.lk_params
        )

        if p1 is not None and st is not None:
            # Select good points that Lucas-Kanade successfully matched
            good_new = p1[st == 1]
            good_old = self.prev_points[st == 1]

            vectors_list = []

            # Calculate mathematical magnitude and angle
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                # Mathematical displacement vector (dx, dy)
                dx = a - c
                dy = b - d

                # Calculate magnitude (spatial displacement)
                magnitude = np.sqrt(dx**2 + dy**2)

                # Filter out stationary noise and extreme jitters
                if self.min_motion < magnitude < self.max_motion:
                    vectors_list.append([c, d, a, b])
                    # Draw actual motion trail (green)
                    self.trail_mask = cv2.line(self.trail_mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    # Draw current position head (red)
                    annotated_frame = cv2.circle(annotated_frame, (int(a), int(b)), 4, (0, 0, 255), -1)

            if vectors_list:
                valid_vectors = np.array(vectors_list, dtype=np.float32)

            # Replenish points: if our tracked population drops, we mix in the incoming FAST points
            # In a full implementation, you'd calculate distance to avoid overlapping points,
            # but concatenating and letting FAST non-max suppression handle it mostly works.
            if len(good_new) < 50 and new_keypoints is not None:
                self.prev_points = new_keypoints
            else:
                self.prev_points = good_new.reshape(-1, 1, 2)
        else:
            self.prev_points = new_keypoints

        # Composite the trail mask onto the frame
        annotated_frame = cv2.add(annotated_frame, self.trail_mask)
        
        # Temporal fade out of the trail mask (so lines don't stay on screen forever)
        self.trail_mask = cv2.addWeighted(self.trail_mask, 0.85, np.zeros_like(self.trail_mask), 0.15, 0)

        # Update historical frame
        self.prev_gray = current_gray.copy()

        return annotated_frame, valid_vectors, len(valid_vectors)


class DenseFarnebackTracker:
    """
    Computes Dense Optical Flow across the entire frame.
    Provides a motion vector for every pixel, allowing for extremely 
    high-resolution density estimation in massive crowds.
    """

    def __init__(self, 
                 pyr_scale=0.5, levels=3, winsize=15, 
                 iterations=3, poly_n=5, poly_sigma=1.2,
                 min_motion=0.5, sample_step=16):
        self.params = {
            'pyr_scale': pyr_scale,
            'levels': levels,
            'winsize': winsize,
            'iterations': iterations,
            'poly_n': poly_n,
            'poly_sigma': poly_sigma,
            'flags': 0
        }
        self.min_motion = min_motion
        self.sample_step = sample_step
        self.prev_gray = None

    def track(self, current_gray: np.ndarray, _, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Computes dense flow and samples it for visualization.
        """
        annotated_frame = frame.copy()
        h, w = current_gray.shape
        valid_vectors = []

        if self.prev_gray is None:
            self.prev_gray = current_gray
            return annotated_frame, np.empty((0, 4)), 0

        # Compute Dense Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, current_gray, None, **self.params)

        # Sample the flow for visualization and counting
        # We step through the image in a grid
        for y in range(0, h, self.sample_step):
            for x in range(0, w, self.sample_step):
                dx, dy = flow[y, x]
                magnitude = np.sqrt(dx**2 + dy**2)

                if magnitude > self.min_motion:
                    # Angle-based color encoding (Directional flow)
                    # Map [-pi, pi] to [0, 179] for HSV hue
                    angle = np.arctan2(dy, dx)
                    hue = int((angle + np.pi) * (179 / (2 * np.pi)))
                    
                    # Use HSV for a smooth professional color wheel
                    color_bgr = cv2.applyColorMap(np.array([[hue]], dtype=np.uint8), cv2.COLORMAP_HSV)[0, 0]
                    color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

                    x2, y2 = int(x + dx), int(y + dy)
                    valid_vectors.append([float(x), float(y), float(x2), float(y2)])
                    
                    # Draw current vector with its unique directional color
                    cv2.line(annotated_frame, (x, y), (x2, y2), color, 1, cv2.LINE_AA)
                    cv2.circle(annotated_frame, (x2, y2), 2, color, -1)

        valid_vectors = np.array(valid_vectors, dtype=np.float32) if valid_vectors else np.empty((0, 4), dtype=np.float32)
        
        self.prev_gray = current_gray
        return annotated_frame, valid_vectors, len(valid_vectors)
