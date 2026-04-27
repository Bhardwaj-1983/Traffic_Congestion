"""Tests for src/feature_engineering.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import feature_engineering as fe


def test_add_time_features_produces_expected_columns(sample_cleaned_df):
    df = fe.add_time_features(sample_cleaned_df)
    for col in ("hour_of_day", "day_of_week", "is_weekend", "month"):
        assert col in df.columns
    assert df["hour_of_day"].between(0, 23).all()
    assert df["day_of_week"].between(0, 6).all()
    assert df["is_weekend"].dtype == bool
    # is_weekend must match day_of_week semantics
    assert ((df["is_weekend"]) == (df["day_of_week"] >= 5)).all()


def test_add_speed_feature_matches_distance_over_duration(sample_cleaned_df):
    df = fe.add_speed_feature(sample_cleaned_df)
    expected = sample_cleaned_df["trip_distance"] / (sample_cleaned_df["trip_duration"] / 3600)
    # Compare element-wise with tolerance
    np.testing.assert_allclose(
        df["speed_mph"].astype("float64").values,
        expected.astype("float64").values,
        rtol=1e-5,
    )


def test_aggregate_zone_hour_reduces_granularity(sample_cleaned_df):
    df = fe.add_time_features(sample_cleaned_df)
    df = fe.add_speed_feature(df)
    df = fe.add_zone_feature(df)
    df = df.dropna(subset=["speed_mph"])

    agg = fe.aggregate_zone_hour(df)

    assert {"zone_id", "hour_of_day", "day_of_week",
            "is_weekend", "avg_speed_mph", "trip_density"} <= set(agg.columns)

    # The aggregated row count must be ≤ the raw row count
    assert len(agg) <= len(df)
    # All trip densities must be ≥ 1
    assert (agg["trip_density"] >= 1).all()


def test_speed_deviation_is_zero_on_average_within_zone_hour():
    """
    Since the baseline is the mean speed per (zone_id, hour_of_day), the
    average speed_deviation **within each (zone, hour)** must be zero.
    """
    # Two zones × two hours × two days → 8 rows
    rows = []
    for zone in (1, 2):
        for hour in (8, 17):
            for dow, speed in [(1, 20.0), (3, 30.0)]:
                rows.append({
                    "zone_id": zone,
                    "hour_of_day": hour,
                    "day_of_week": dow,
                    "is_weekend": dow >= 5,
                    "avg_speed_mph": speed,
                    "trip_density": 100,
                    "avg_trip_duration": 600.0,
                })
    agg = pd.DataFrame(rows)
    agg = fe.compute_speed_deviation(agg)
    # Grouped mean of speed_deviation should be ≈ 0
    grouped = agg.groupby(["zone_id", "hour_of_day"])["speed_deviation"].mean()
    np.testing.assert_allclose(grouped.values, np.zeros(len(grouped)), atol=1e-5)


def test_speed_deviation_negative_for_below_baseline_row():
    """A zone/hour that's running slower than baseline must have negative deviation."""
    agg = pd.DataFrame([
        {"zone_id": 1, "hour_of_day": 9, "day_of_week": 1,
         "is_weekend": False, "avg_speed_mph": 10.0,
         "trip_density": 200, "avg_trip_duration": 700.0},
        {"zone_id": 1, "hour_of_day": 9, "day_of_week": 3,
         "is_weekend": False, "avg_speed_mph": 30.0,
         "trip_density": 50, "avg_trip_duration": 300.0},
    ])
    agg = fe.compute_speed_deviation(agg)
    slow_row = agg[agg["avg_speed_mph"] == 10.0].iloc[0]
    fast_row = agg[agg["avg_speed_mph"] == 30.0].iloc[0]
    assert slow_row["speed_deviation"] < 0
    assert fast_row["speed_deviation"] > 0
