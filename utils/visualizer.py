"""
visualizer.py
-------------
All OpenCV drawing helpers live here so that main.py stays clean.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import Config


# ── Constants ──────────────────────────────────────────────────────────────────
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_FONT   = cv2.FONT_HERSHEY_DUPLEX
_HUD_PAD    = 12   # pixels of padding inside the HUD background rectangle


# ── Public helpers ─────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    count: int,
    fps: float,
    cfg: Config | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes, per-box confidence labels, and a HUD overlay.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to draw on (modified **in-place** for performance).
    detections : list[dict]
        List of detection dicts as returned by ``PersonDetector.detect()``.
        Each dict must contain ``"bbox"`` and ``"conf"``.
    count : int
        Total number of people detected (drawn in the HUD).
    fps : float
        Current processing FPS (drawn in the HUD).
    cfg : Config, optional
        Configuration object.  Uses default Config() if not supplied.

    Returns
    -------
    np.ndarray
        The annotated frame (same object that was passed in).
    """
    cfg = cfg or Config()

    _draw_boxes(frame, detections, cfg)
    _draw_hud(frame, count, fps, cfg)
    return frame


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_id_color(track_id: int | None) -> tuple:
    """
    Return a visually distinct BGR colour for a given track ID.
    IDs cycle through 12 hues so they are easy to tell apart on screen.
    Falls back to white for unconfirmed tracks (track_id is None).
    """
    if track_id is None:
        return (200, 200, 200)
    # 12 evenly-spaced hues in HSV → convert one to BGR
    hues = [0, 15, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    hue = hues[track_id % len(hues)]
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # OpenCV is BGR


def _draw_boxes(frame: np.ndarray, detections: list[dict], cfg: Config) -> None:
    """Draw bounding box + track-ID pill + confidence label per detection."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        track_id = det.get("track_id")

        box_color = _get_id_color(track_id)

        # ── Bounding box ──────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, cfg.BOX_THICKNESS)

        # ── Track-ID pill (above the box) ─────────────────────────────
        id_label = f"ID {track_id}" if track_id is not None else "?"
        id_size, id_base = cv2.getTextSize(id_label, _LABEL_FONT, 0.6, 2)
        iw, ih = id_size

        pill_y1 = max(y1 - ih - id_base - 6, 0)
        pill_y2 = y1
        cv2.rectangle(frame, (x1, pill_y1), (x1 + iw + 6, pill_y2), box_color, cv2.FILLED)
        cv2.putText(
            frame, id_label,
            (x1 + 3, pill_y2 - id_base - 1),
            _LABEL_FONT, 0.6,
            (0, 0, 0),          # black text on coloured pill
            2, cv2.LINE_AA,
        )

        # ── Confidence score (bottom-left corner of box) ──────────────
        conf_label = f"{conf:.2f}"
        cv2.putText(
            frame, conf_label,
            (x1 + 4, y2 - 6),
            _LABEL_FONT, 0.45,
            box_color, 1, cv2.LINE_AA,
        )


def _draw_hud(frame: np.ndarray, count: int, fps: float, cfg: Config) -> None:
    """Draw the heads-up display in the top-left corner."""
    lines = [
        f"People : {count}",
        f"FPS    : {fps:.1f}",
        f"Press Q to quit",
    ]

    font_scale = cfg.HUD_FONT_SCALE
    thickness  = cfg.HUD_FONT_THICKNESS
    color      = cfg.HUD_COLOR

    # Compute the bounding box for the HUD background
    line_sizes = [
        cv2.getTextSize(line, _HUD_FONT, font_scale, thickness)[0]
        for line in lines
    ]
    max_w  = max(w for w, _ in line_sizes)
    line_h = max(h for _, h in line_sizes)
    total_h = len(lines) * (line_h + _HUD_PAD) + _HUD_PAD

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (max_w + _HUD_PAD * 2, total_h),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Text lines
    for i, line in enumerate(lines):
        y = _HUD_PAD + (i + 1) * (line_h + _HUD_PAD)
        cv2.putText(
            frame, line,
            (_HUD_PAD, y),
            _HUD_FONT, font_scale,
            color, thickness, cv2.LINE_AA,
        )
