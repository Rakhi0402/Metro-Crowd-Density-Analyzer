"""
density_estimator.py
--------------------
Module for Step 4 of the IEEE 7275777 methodology.
Takes the active optical flow motion vectors and clusters them spatially.
Calculates a 'Crowd Density Score' by mapping flow density in regional 
patches into 'Low', 'Medium', and 'High' thresholds, overlaying a colored 
grid that mathematically visualises crowd concentration.
"""

import cv2
import numpy as np


class DensityEstimator:
    """
    Subdivides the screen into a grid and clusters Optical Flow vectors internally 
    to gauge regional crowd density.
    """

    def __init__(self, grid_rows: int = 4, grid_cols: int = 3, threshold_high: int = 15, threshold_medium: int = 5):
        """
        Parameters
        ----------
        grid_rows : int
            Number of horizontal grid divisions.
        grid_cols : int
            Number of vertical grid divisions.
        threshold_high : int
            Number of active vectors required in a cell to mark it 'High' density.
        threshold_medium : int
            Number of active vectors required in a cell to mark it 'Medium' density.
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium

    def estimate_and_draw(self, frame: np.ndarray, motion_vectors: np.ndarray, draw_labels: bool = True) -> tuple[np.ndarray, int]:
        """
        Partitions the frame into a spatial grid, counts vectors within each cell, 
        assigns a density class (High/Medium/Low), and overlays the visual heat-grid.

        Parameters
        ----------
        frame : np.ndarray
            The underlying BGR canvas.
        motion_vectors : np.ndarray
            Array of valid optical flow vectors shape (N, 4) -> (X1, Y1, X2, Y2).
        draw_labels : bool, optional
            Whether to draw the colored grid and text labels (default: True).

        Returns
        -------
        tuple[np.ndarray, int]
            (grid_annotated_frame, total_global_density_score)
        """
        annotated_frame = frame.copy()
        
        height, width = frame.shape[:2]
        cell_w = width // self.grid_cols
        cell_h = height // self.grid_rows

        # Initialize an empty count grid
        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

        # Cluster vectors into corresponding grid cells
        for vector in motion_vectors:
            c, d, a, b = vector
            col = min(int(a) // cell_w, self.grid_cols - 1)
            row = min(int(b) // cell_h, self.grid_rows - 1)
            grid_counts[row, col] += 1

        total_density_score = np.sum(grid_counts)

        if not draw_labels:
            # We skip the heavy overlays but still might want thin grid lines
            for r in range(self.grid_rows + 1):
                y = min(r * cell_h, height - 1)
                cv2.line(annotated_frame, (0, y), (width, y), (100, 100, 100), 1)
            for c in range(self.grid_cols + 1):
                x = min(c * cell_w, width - 1)
                cv2.line(annotated_frame, (x, 0), (x, height), (100, 100, 100), 1)
            return annotated_frame, total_density_score

        overlay = frame.copy()  # For transparent grid drawing
        
        # Draw the spatial heat-grid based on localized counts
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                count = grid_counts[r, c]
                
                # Determine rect boundaries
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = (c + 1) * cell_w
                y2 = (r + 1) * cell_h

                # Classify Density
                if count >= self.threshold_high:
                    color = (0, 0, 255)       # Red: High Density
                    label = "High"
                elif count >= self.threshold_medium:
                    color = (0, 165, 255)     # Orange: Medium Density
                    label = "Med"
                else:
                    color = (255, 0, 0)       # Blue: Low Density
                    label = "Low"

                # Draw the colored cell fill on the overlay
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
                
                # Add a 'Glow' effect for High Density
                if label == "High":
                    glow = np.zeros_like(annotated_frame)
                    cv2.rectangle(glow, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    glow = cv2.GaussianBlur(glow, (15, 15), 0)
                    annotated_frame = cv2.addWeighted(annotated_frame, 1.0, glow, 0.5, 0)

                # Draw the cell border
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

                # Overlay the Density Label in the corner of the cell
                cv2.putText(
                    annotated_frame, 
                    f"{label} ({count}v)", 
                    (x1 + 5, y1 + 20), 
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.LINE_AA
                )

        # Alpha composite the heat-grid overlay atop the frame (25% opacity)
        annotated_frame = cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0)

        return annotated_frame, total_density_score

    def draw_heatmap(self, frame: np.ndarray, motion_vectors: np.ndarray) -> np.ndarray:
        """
        Creates a high-end Gaussian-smoothed heatmap overlay.
        """
        if len(motion_vectors) == 0:
            return frame
            
        h, w = frame.shape[:2]
        # Create a small accumulator for performance, then upscale
        acc_h, acc_w = h // 16, w // 16
        accumulator = np.zeros((acc_h, acc_w), dtype=np.float32)
        
        for vx1, vy1, vx2, vy2 in motion_vectors:
            ax, ay = int(vx1 * acc_w / w), int(vy1 * acc_h / h)
            if 0 <= ax < acc_w and 0 <= ay < acc_h:
                accumulator[ay, ax] += 1.0
                
        # Smooth the accumulator
        accumulator = cv2.GaussianBlur(accumulator, (15, 15), 0)
        
        # Normalize and colorize
        if np.max(accumulator) > 0:
            cv2.normalize(accumulator, accumulator, 0, 255, cv2.NORM_MINMAX)
        
        accum_8u = accumulator.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(accum_8u, cv2.COLORMAP_JET)
        
        # Upscale and blend
        heatmap_big = cv2.resize(heatmap_color, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = 0.4
        return cv2.addWeighted(frame, 1.0 - alpha, heatmap_big, alpha, 0)
