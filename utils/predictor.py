import torch
import numpy as np


class Predictor:
    """
    Real-time predictor using trained LSTM model
    """

    def __init__(self, model, seq_length=20):
        self.model = model
        self.seq_length = seq_length
        self.history = []

    # ─────────────────────────────
    # UPDATE + PREDICT
    # ─────────────────────────────
    def update(self, features):
        """
        features = [
            total_count,
            door_left,
            door_right,
            center,
            flow_in,
            flow_out
        ]
        """

        self.history.append(features)

        # Not enough data yet
        if len(self.history) < self.seq_length:
            return None

        # Keep last N
        seq = self.history[-self.seq_length:]

        # Convert to tensor
        seq = np.array(seq)
        seq = torch.tensor(seq).float()

        # Shape: (seq_len, batch=1, features)
        seq = seq.unsqueeze(1)

        with torch.no_grad():
            pred = self.model(seq)

        return pred.item()