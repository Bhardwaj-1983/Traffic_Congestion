"""
Shared pytest fixtures.

Adds the project root to sys.path so that ``import src.*`` resolves
when tests are run from any directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def sample_cleaned_df() -> pd.DataFrame:
    """
    Produce a small but realistic "cleaned trips" DataFrame with a mix of
    valid rows and edge cases. Tests can further manipulate this fixture
    to exercise specific filters.
    """
    rng = np.random.default_rng(42)
    n = 200
    pickup = pd.date_range("2023-01-02 00:00:00", periods=n, freq="3min")
    duration_s = rng.integers(120, 1800, size=n).astype("float32")  # 2-30 min
    distance = rng.uniform(0.5, 10.0, size=n).astype("float32")      # 0.5-10 mi
    zone = rng.integers(1, 264, size=n)

    df = pd.DataFrame({
        "pickup_datetime": pickup,
        "dropoff_datetime": pickup + pd.to_timedelta(duration_s, unit="s"),
        "trip_distance": distance,
        "trip_duration": duration_s,
        "PULocationID": pd.array(zone, dtype="Int32"),
        "DOLocationID": pd.array(zone, dtype="Int32"),
    })
    return df


@pytest.fixture
def sample_aggregated_df() -> pd.DataFrame:
    """
    A synthetic "aggregated_zone_hour" DataFrame with clear cluster
    signals so tests can confirm K-Means recovers sensible congestion levels.

    Includes the *full* clustering-feature set required by the upgraded
    pipeline: hour_sin/hour_cos cyclic encoding, is_weekend, speed_deviation,
    and log_trip_density.
    """
    rng = np.random.default_rng(7)
    rows = []
    # Low congestion: high speed, low density, positive speed_deviation
    for _ in range(100):
        rows.append({
            "zone_id": rng.integers(1, 264),
            "hour_of_day": rng.integers(0, 24),
            "day_of_week": rng.integers(0, 7),
            "is_weekend": False,
            "avg_speed_mph": rng.normal(28.0, 2.5),
            "trip_density": rng.integers(5, 25),
            "speed_deviation": rng.normal(4.0, 1.0),
        })
    # Medium congestion
    for _ in range(100):
        rows.append({
            "zone_id": rng.integers(1, 264),
            "hour_of_day": rng.integers(0, 24),
            "day_of_week": rng.integers(0, 7),
            "is_weekend": False,
            "avg_speed_mph": rng.normal(15.0, 2.0),
            "trip_density": rng.integers(40, 100),
            "speed_deviation": rng.normal(0.0, 1.0),
        })
    # High congestion
    for _ in range(100):
        rows.append({
            "zone_id": rng.integers(1, 264),
            "hour_of_day": rng.integers(0, 24),
            "day_of_week": rng.integers(0, 7),
            "is_weekend": True,
            "avg_speed_mph": rng.normal(6.0, 1.5),
            "trip_density": rng.integers(120, 220),
            "speed_deviation": rng.normal(-5.0, 1.0),
        })

    df = pd.DataFrame(rows)
    df["avg_speed_mph"] = df["avg_speed_mph"].astype("float32")
    df["trip_density"] = df["trip_density"].astype("int32")
    df["speed_deviation"] = df["speed_deviation"].astype("float32")

    # Add the engineered features the upgraded clusterer requires
    angle = 2.0 * np.pi * df["hour_of_day"].astype("float32") / 24.0
    df["hour_sin"] = np.sin(angle).astype("float32")
    df["hour_cos"] = np.cos(angle).astype("float32")
    df["log_trip_density"] = np.log1p(df["trip_density"].astype("float32")).astype("float32")
    return df
