from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from forecasting.dataset_builder import DatasetBuilder
from forecasting.lstm_model import LSTMModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the LSTM crowd forecasting model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--logs", default="outputs/logs.csv", help="Input analytics CSV.")
    parser.add_argument("--output", default="forecasting/lstm_model.pth", help="Model output path.")
    parser.add_argument("--seq-length", type=int, default=20, help="Sequence length.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    return parser.parse_args()


def train(logs_path: str, output_path: str, seq_length: int, epochs: int, lr: float) -> None:
    builder = DatasetBuilder(seq_length=seq_length)
    X, y = builder.build(logs_path)

    if len(X) == 0:
        raise ValueError(
            f"Not enough rows in {logs_path!r}; need more than {seq_length} logged frames."
        )

    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    X = X.permute(1, 0, 2)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\nTraining started...\n")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"\nModel saved at: {output}")


def main():
    args = parse_args()
    train(
        logs_path=args.logs,
        output_path=args.output,
        seq_length=args.seq_length,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
