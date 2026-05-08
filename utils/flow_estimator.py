class FlowEstimator:
    """
    Estimates crowd flow (entry/exit) using tracked person movement.

    Uses vertical movement (cy) to detect:
    - entering (moving downward)
    - exiting (moving upward)
    """

    def __init__(self, movement_threshold=8):
        """
        Args:
            movement_threshold: Minimum pixel movement to count as real motion
        """
        self.prev_positions = {}   # track_id -> previous cy
        self.movement_threshold = movement_threshold

    def update(self, detections):
        """
        Args:
            detections: list of dicts from detector
                        {bbox, center, track_id}

        Returns:
            flow_in, flow_out
        """

        flow_in = 0
        flow_out = 0

        current_ids = set()

        for d in detections:
            tid = d["track_id"]
            cx, cy = d["center"]

            current_ids.add(tid)

            if tid in self.prev_positions:
                prev_cy = self.prev_positions[tid]

                diff = cy - prev_cy

                # Ignore tiny noise
                if abs(diff) > self.movement_threshold:

                    if diff > 0:
                        flow_in += 1     # moving down
                    else:
                        flow_out += 1    # moving up

            # update position
            self.prev_positions[tid] = cy

        # 🔥 CLEANUP: remove disappeared IDs
        self.prev_positions = {
            tid: self.prev_positions[tid]
            for tid in current_ids
        }

        return flow_in, flow_out