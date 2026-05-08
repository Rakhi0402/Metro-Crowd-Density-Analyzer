from __future__ import annotations

import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import Config


# ─────────────────────────────────────────────
# CROWD FORECASTER
# ─────────────────────────────────────────────

class CrowdForecaster:

    def __init__(
        self,
        window_size: int = 30,
        alert_threshold: float = 15.0,
        moderate_threshold: float = 12.0,
        high_threshold: float = 18.0,
        critical_threshold: float = 25.0,
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.history = deque(maxlen=window_size)

        self.alpha = 0.3
        self.smoothed_value = None
        self.trend = 0.0
        self.predictions = []

    def add_observation(self, crowd_count: int) -> dict:

        self.history.append(crowd_count)

        # Exponential smoothing
        if self.smoothed_value is None:
            self.smoothed_value = crowd_count
        else:
            old = self.smoothed_value
            self.smoothed_value = (
                self.alpha * crowd_count +
                (1 - self.alpha) * self.smoothed_value
            )

            # Trend update
            self.trend = 0.1 * (self.smoothed_value - old) + 0.9 * self.trend

        predictions = self._forecast()
        alerts = self._check_alerts(crowd_count, predictions)

        return {
            "current": crowd_count,
            "smoothed": self.smoothed_value,
            "trend": self.trend,
            "predictions": predictions,
            "alerts": alerts
        }

    def _forecast(self, steps: int = 5) -> list[float]:

        forecasts = []

        if len(self.history) >= 2:
            for step in range(1, steps + 1):
                forecast = self.smoothed_value + (self.trend * step * 0.8)
                forecast = max(0, min(forecast, 50))  # safer upper bound
                forecasts.append(float(forecast))
        else:
            forecasts = [self.smoothed_value or 0] * steps

        self.predictions = forecasts
        return forecasts

    def _check_alerts(self, current: int, predictions: list[float]) -> list[str]:

        alerts = []

        if current >= self.alert_threshold:
            alerts.append(f"HIGH CROWDING: {current}")

        if predictions:
            if predictions[-1] > current + 3:
                alerts.append(f"Increasing trend -> {predictions[-1]:.0f}")

            if any(p >= self.alert_threshold for p in predictions):
                alerts.append(f"Forecast overcrowding -> {max(predictions):.0f}")

        if len(self.history) >= 3:
            recent_avg = np.mean(list(self.history)[-3:])
            if current > recent_avg * 1.5:
                alerts.append("Sudden spike detected")

        return alerts

    def get_recommendation(self, current: int, predictions: list[float]) -> str:

        if not predictions:
            return "NORMAL"

        max_pred = max([current] + predictions)

        if max_pred >= self.critical_threshold:
            return "CRITICAL: Limit entry immediately"
        elif max_pred >= self.high_threshold:
            return "HIGH: Prepare control measures"
        elif max_pred >= self.moderate_threshold:
            return "MODERATE: Monitor closely"
        return "NORMAL"


# ─────────────────────────────────────────────
# ALERT SYSTEM
# ─────────────────────────────────────────────

class AlertSystem:

    def __init__(self):
        self.alert_log = []
        self.count = 0

    def send_alert(self, msg, level="warning", coach_id=None):

        self.count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        print(f"[{timestamp}] {msg}")

        self.alert_log.append({
            "time": timestamp,
            "timestamp": timestamp,
            "level": level,
            "msg": msg,
            "message": msg,
            "coach": coach_id
        })

    def save_alert_log(self, filepath: str, append: bool = False):
        """Save the alert log to a CSV file."""
        import csv

        if not self.alert_log:
            return

        mode = 'a' if append else 'w'
        write_header = not append or not Path(filepath).exists()

        with open(filepath, mode, encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.alert_log[0].keys())
            if write_header:
                writer.writeheader()
            for entry in self.alert_log:
                writer.writerow(entry)

        print(f"✓ Alert log saved: {filepath}")


# ─────────────────────────────────────────────
# CROWD ANALYZER (FIXED)
# ─────────────────────────────────────────────

class CrowdAnalyzer:

    def __init__(self, cfg: Config | None = None, alert_threshold: float | None = None):
        if cfg is None:
            cfg = Config()

        if alert_threshold is None:
            alert_threshold = cfg.ALERT_THRESHOLD

        self.forecaster = CrowdForecaster(
            alert_threshold=alert_threshold,
            moderate_threshold=cfg.RISK_MODERATE_THRESHOLD,
            high_threshold=cfg.RISK_HIGH_THRESHOLD,
            critical_threshold=cfg.RISK_CRITICAL_THRESHOLD,
        )
        self.alerts = AlertSystem()
        self.coach_data = {}

    def analyze_coach(self, coach_id: int, coach_stats: dict) -> dict:

        # 🔥 FIXED SAFE CONVERSION
        avg_raw = coach_stats.get("avg_people", 0)
        max_raw = coach_stats.get("max_people", 0)

        try:
            avg = round(float(avg_raw))
        except:
            avg = 0

        try:
            peak = int(float(max_raw))
        except:
            peak = 0

        # Forecast
        forecast_result = self.forecaster.add_observation(avg)

        # Recommendation
        recommendation = self.forecaster.get_recommendation(
            avg,
            forecast_result["predictions"]
        )

        # Alerts
        if peak >= self.forecaster.alert_threshold:
            self.alerts.send_alert(
                f"Coach {coach_id} overcrowded ({peak})",
                level="critical",
                coach_id=coach_id
            )

        risk_level = "CRITICAL" if "CRITICAL" in recommendation else (
            "HIGH" if "HIGH" in recommendation else (
                "MODERATE" if "MODERATE" in recommendation else "LOW"
            )
        )

        result = {
            "coach_id": coach_id,
            "video": coach_stats.get("video", "unknown"),
            "current_avg": avg,
            "current_max": peak,
            "forecast": forecast_result["predictions"],
            "recommendation": recommendation,
            "risk_level": risk_level
        }

        self.coach_data[coach_id] = result
        return result

    def print_report(self):

        print("\n=== CROWD REPORT ===")

        for cid, data in self.coach_data.items():
            print(f"\nCoach {cid}")
            print(f"Avg: {data.get('current_avg', data.get('avg', 0))}")
            print(f"Peak: {data.get('current_max', data.get('peak', 0))}")
            print(f"Forecast: {[round(x) for x in data['forecast']]}")
            print(f"Advice: {data['recommendation']}")
