
import cv2
import numpy as np
from collections import deque


class TemporalVisualizer:
    """
    Maintains a rolling buffer of 4 frames and stacks them into a 2x2 grid.
    """

    def __init__(self, buffer_size: int = 4):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, frame: np.ndarray) -> np.ndarray:
      
        self.buffer.append(frame.copy())
        
        if len(self.buffer) < 4:
            return frame

        # Get the 4 frames (f1 is oldest, f4 is newest)
        f1, f2, f3, f4 = list(self.buffer)

        # Scale down for 2x2 fit (so the final window isn't massive)
        # We target a reasonable total width (e.g. 1280px or same as original)
        h, w = frame.shape[:2]
        small_w, small_h = w // 2, h // 2
        
        sf1 = cv2.resize(f1, (small_w, small_h))
        sf2 = cv2.resize(f2, (small_w, small_h))
        sf3 = cv2.resize(f3, (small_w, small_h))
        sf4 = cv2.resize(f4, (small_w, small_h))

        # Horizontal stacks
        top_row = np.hstack((sf1, sf2))
        bottom_row = np.hstack((sf3, sf4))

        # Vertical stack
        grid_2x2 = np.vstack((top_row, bottom_row))

        # Optional: labels on each sub-panel
        for i, (x, y, text) in enumerate([
            (10, 20, "T-3"),
            (small_w + 10, 20, "T-2"),
            (10, small_h + 20, "T-1"),
            (small_w + 10, small_h + 20, "Current (T)")
        ]):
            cv2.putText(grid_2x2, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return grid_2x2
