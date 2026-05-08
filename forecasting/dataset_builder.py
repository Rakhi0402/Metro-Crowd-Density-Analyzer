import csv
import numpy as np


class DatasetBuilder:
    """
    Converts logs.csv into LSTM-ready sequences
    """

    def __init__(self, seq_length=20):
        self.seq_length = seq_length

    # ─────────────────────────────
    # LOAD CSV
    # ─────────────────────────────
    def load_data(self, file_path):
        columns = [
            "total_count",
            "door_left",
            "door_right",
            "center",
            "flow_in",
            "flow_out"
        ]

        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [
                [float(row[col]) for col in columns]
                for row in reader
            ]

        return np.array(data, dtype=np.float32)

    # ─────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────
    def normalize(self, data):
        """
        Min-Max scaling
        """
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

        return (data - self.min) / (self.max - self.min + 1e-8)

    # ─────────────────────────────
    # CREATE SEQUENCES
    # ─────────────────────────────
    def create_sequences(self, data):
        X, y = [], []

        values = data

        for i in range(len(values) - self.seq_length):
            X.append(values[i:i+self.seq_length])
            y.append(values[i+self.seq_length][0])  # predict total_count

        return np.array(X), np.array(y)

    # ─────────────────────────────
    # FULL PIPELINE
    # ─────────────────────────────
    def build(self, file_path):
        df = self.load_data(file_path)

        normalized = self.normalize(df)

        X, y = self.create_sequences(normalized)

        print(f"[DatasetBuilder] X shape: {X.shape}")
        print(f"[DatasetBuilder] y shape: {y.shape}")

        return X, y