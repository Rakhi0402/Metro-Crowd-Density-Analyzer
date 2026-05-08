"""
zones.py
--------
Defines spatial regions inside the metro coach.

IMPORTANT:
- Coordinates are normalized (0 → 1)
- Automatically scale to frame size
"""

def get_zones(frame_width, frame_height):
    """
    Returns zones scaled to frame size.

    Layout assumption:
    | door_left | center | door_right |
    """

    return {
        "door_left": [
            (0.0 * frame_width, 0.0 * frame_height),
            (0.33 * frame_width, 0.0 * frame_height),
            (0.33 * frame_width, 1.0 * frame_height),
            (0.0 * frame_width, 1.0 * frame_height),
        ],

        "center": [
            (0.33 * frame_width, 0.0 * frame_height),
            (0.66 * frame_width, 0.0 * frame_height),
            (0.66 * frame_width, 1.0 * frame_height),
            (0.33 * frame_width, 1.0 * frame_height),
        ],

        "door_right": [
            (0.66 * frame_width, 0.0 * frame_height),
            (1.0 * frame_width, 0.0 * frame_height),
            (1.0 * frame_width, 1.0 * frame_height),
            (0.66 * frame_width, 1.0 * frame_height),
        ]
    }