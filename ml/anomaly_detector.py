"""
Experimental anomaly detection for SafeSewer telemetry.

The model is trained on the recent valid telemetry window each time the dashboard
refreshes. It is unsupervised: it identifies unusual patterns, not confirmed
floods, gas hazards, or sensor failures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SENSOR_COLUMNS = ["mq135", "mq7", "distance"]
MIN_TRAINING_ROWS = 30


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create sensor-level and short-term change features from chronological data."""
    features = frame[SENSOR_COLUMNS].copy()

    for column in SENSOR_COLUMNS:
        # A one-reading change can highlight abrupt spikes/drops.
        features[f"{column}_delta"] = features[column].diff()
        # Local context helps detect values unusual relative to recent behavior.
        features[f"{column}_rolling_mean"] = (
            features[column].rolling(window=5, min_periods=2).mean()
        )
        features[f"{column}_rolling_std"] = (
            features[column].rolling(window=5, min_periods=2).std()
        )

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    return features.fillna(0.0)


def detect_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return valid readings in chronological order with:
      - is_anomaly: model flag
      - ml_status: human-readable label
      - anomaly_score: Isolation Forest decision_function (lower is more unusual)

    Requires timestamp, mq135, mq7, and distance columns.
    """
    required = ["timestamp", *SENSOR_COLUMNS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required telemetry columns: {', '.join(missing)}")

    frame = data[required].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["timestamp", *SENSOR_COLUMNS])
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    if len(frame) < MIN_TRAINING_ROWS:
        return pd.DataFrame()

    features = _build_features(frame)
    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42,
    )
    predictions = model.fit_predict(features)
    scores = model.decision_function(features)

    frame["is_anomaly"] = predictions == -1
    frame["ml_status"] = np.where(frame["is_anomaly"], "Anomaly detected", "Normal pattern")
    frame["anomaly_score"] = scores.astype(float)
    return frame
