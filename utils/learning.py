"""
learning.py
-----------
Automatic learning module for the crowd tracking system.
Retrains the LSTM model on new data and manages data redundancy.
"""

from __future__ import annotations

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime, timedelta

from forecasting.dataset_builder import DatasetBuilder
from forecasting.lstm_model import LSTMModel


class AutoLearner:
    """
    Handles automatic model retraining and data management.
    """

    def __init__(self, logs_path: str = "outputs/logs.csv",
                 model_path: str = "forecasting/lstm_model.pth",
                 seq_length: int = 20, epochs: int = 10, lr: float = 0.001,
                 max_data_age_days: int = 30):
        """
        Initialize the auto-learner.

        Parameters
        ----------
        logs_path : str
            Path to the logs CSV file.
        model_path : str
            Path to save the trained model.
        seq_length : int
            Sequence length for LSTM.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        max_data_age_days : int
            Maximum age of data to keep (in days).
        """
        self.logs_path = Path(logs_path)
        self.model_path = Path(model_path)
        self.seq_length = seq_length
        self.epochs = epochs
        self.lr = lr
        self.max_data_age_days = max_data_age_days

    def should_retrain(self) -> bool:
        """
        Check if retraining is needed (e.g., new data available).
        """
        if not self.logs_path.exists():
            return False

        # Check if logs have new data since last training
        # For simplicity, always retrain if data exists
        return True

    def clean_redundant_data(self):
        """
        Remove redundant/old data from logs.
        """
        if not self.logs_path.exists():
            return

        # Load data
        data = []
        with open(self.logs_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        if not data:
            return

        # Filter by date
        cutoff = datetime.now() - timedelta(days=self.max_data_age_days)
        filtered_data = []
        for row in data:
            try:
                row_date = datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S')
                if row_date >= cutoff:
                    filtered_data.append(row)
            except ValueError:
                # Keep if can't parse
                filtered_data.append(row)

        # Remove duplicates based on timestamp
        seen_timestamps = set()
        deduped_data = []
        for row in filtered_data:
            ts = row.get('timestamp')
            if ts and ts not in seen_timestamps:
                seen_timestamps.add(ts)
                deduped_data.append(row)

        # Save cleaned data
        with open(self.logs_path, 'w', newline='', encoding='utf-8') as f:
            if deduped_data:
                writer = csv.DictWriter(f, fieldnames=deduped_data[0].keys())
                writer.writeheader()
                writer.writerows(deduped_data)

        print(f"✓ Cleaned logs: kept {len(deduped_data)} entries (removed {len(data) - len(deduped_data)})")

    def retrain_model(self):
        """
        Retrain the LSTM model on current data.
        """
        if not self.logs_path.exists():
            print("⚠️ No logs data available for training")
            return

        print("\n🚀 Starting model retraining...")

        # Load and prepare data
        builder = DatasetBuilder(seq_length=self.seq_length)
        try:
            X, y = builder.build(str(self.logs_path))
        except Exception as e:
            print(f"❌ Failed to build dataset: {e}")
            return

        if len(X) == 0:
            print("⚠️ Insufficient data for training")
            return

        # Convert to tensors
        X = torch.tensor(X).float()      # (samples, seq, features)
        y = torch.tensor(y).float()      # (samples,)

        # LSTM expects (seq, batch, features)
        X = X.permute(1, 0, 2)

        # Initialize model
        model = LSTMModel()

        # Load existing model if available
        if self.model_path.exists():
            try:
                model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                print("✓ Loaded existing model for fine-tuning")
            except Exception as e:
                print(f"⚠️ Could not load existing model: {e}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Training loop
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}] Loss: {loss.item():.4f}")

        # Save updated model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_path)
        print(f"✅ Model updated and saved: {self.model_path}")

    def learn(self):
        """
        Perform automatic learning: clean data and retrain model.
        """
        if not self.should_retrain():
            print("ℹ️ No retraining needed")
            return

        # Clean redundant data
        self.clean_redundant_data()

        # Retrain model
        self.retrain_model()

        print("🎓 Learning complete - system updated with new knowledge")
