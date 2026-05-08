import cv2
import numpy as np


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon
    """
    polygon = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def assign_zone(bbox, zones):
    """
    Assign a detection to a zone based on its center.

    Args:
        bbox: (x1, y1, x2, y2)
        zones: dict of polygons

    Returns:
        zone_name (str)
    """

    x1, y1, x2, y2 = bbox

    # center of person
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    for zone_name, polygon in zones.items():
        if point_in_polygon((cx, cy), polygon):
            return zone_name

    return "unknown"