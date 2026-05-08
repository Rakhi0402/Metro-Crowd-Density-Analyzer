import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM model for crowd forecasting
    Input shape: (seq_len, batch, features)
    Output: next crowd count
    """

    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (seq_len, batch, features)

        out, _ = self.lstm(x)

        # Take last timestep output
        out = out[-1]

        out = self.fc(out)

        return out