import csv
import os
from datetime import datetime


class DataLogger:
    """
    Logs crowd analytics data into CSV file.

    Automatically:
    - creates folder
    - creates file
    - appends data safely
    """

    def __init__(self, file_path="outputs/logs.csv"):
        self.file_path = file_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Create file with header if not exists
        if not os.path.exists(file_path):
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "datetime",
                    "timestamp",
                    "total_count",
                    "door_left",
                    "door_right",
                    "center",
                    "flow_in",
                    "flow_out"
                ])

    # ─────────────────────────────
    # MAIN LOG FUNCTION
    # ─────────────────────────────
    def log(self, row):
        """
        row format:
        [
            timestamp,
            total_count,
            door_left,
            door_right,
            center,
            flow_in,
            flow_out
        ]
        """

        timestamp = row[0]

        # Convert to readable datetime
        dt = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        final_row = [dt] + row

        try:
            with open(self.file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(final_row)

        except Exception as e:
            print(f"[Logger Error] {e}")