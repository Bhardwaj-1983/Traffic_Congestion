"""Tests for src/preprocessing.py cleaning rules."""

from __future__ import annotations

import pandas as pd

from src import preprocessing
from src.config import (
    MAX_DISTANCE_MILES,
    MAX_DURATION_SECONDS,
    MIN_DISTANCE_MILES,
    MIN_DURATION_SECONDS,
    MIN_ZONE_ID,
)


def test_remove_invalid_timestamps_drops_wrong_order(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    # Swap timestamps in the first 5 rows to simulate bad records
    swap = df.iloc[:5].copy()
    swap["pickup_datetime"], swap["dropoff_datetime"] = (
        df.iloc[:5]["dropoff_datetime"].values,
        df.iloc[:5]["pickup_datetime"].values,
    )
    df.iloc[:5] = swap

    df_clean = preprocessing.remove_invalid_timestamps(df)
    assert len(df_clean) == len(df) - 5
    assert (df_clean["pickup_datetime"] < df_clean["dropoff_datetime"]).all()


def test_remove_invalid_distance_clamps_both_ends(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    df.loc[df.index[0], "trip_distance"] = 0.0           # below min
    df.loc[df.index[1], "trip_distance"] = MAX_DISTANCE_MILES + 1  # above max

    df_clean = preprocessing.remove_invalid_distance(df)
    assert df_clean["trip_distance"].between(
        MIN_DISTANCE_MILES, MAX_DISTANCE_MILES
    ).all()
    assert len(df_clean) == len(df) - 2


def test_remove_invalid_duration(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    df.loc[df.index[0], "trip_duration"] = 0
    df.loc[df.index[1], "trip_duration"] = MAX_DURATION_SECONDS + 1

    df_clean = preprocessing.remove_invalid_duration(df)
    assert df_clean["trip_duration"].between(
        MIN_DURATION_SECONDS, MAX_DURATION_SECONDS
    ).all()
    assert len(df_clean) == len(df) - 2


def test_remove_speed_outliers_filters_unrealistic_speeds(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    # Craft a row with speed > 100 mph (e.g. 200 miles / 600 s = 1200 mph)
    df.loc[df.index[0], ["trip_distance", "trip_duration"]] = [200.0, 600.0]
    # And one with speed < 1 mph (0.001 miles / 1000 s ≈ 0.0036 mph)
    df.loc[df.index[1], ["trip_distance", "trip_duration"]] = [0.001, 1000.0]

    df_clean = preprocessing.remove_speed_outliers(df)
    # Both outliers must be dropped
    assert len(df_clean) == len(df) - 2


def test_remove_invalid_zones_drops_out_of_range(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    df.loc[df.index[0], "PULocationID"] = 0      # below min
    df.loc[df.index[1], "PULocationID"] = 999    # above max

    df_clean = preprocessing.remove_invalid_zones(df)
    assert df_clean["PULocationID"].min() >= MIN_ZONE_ID
    assert len(df_clean) == len(df) - 2


def test_clean_dataframe_end_to_end_retains_valid_rows(sample_cleaned_df):
    """Fixture rows are all valid, so cleaning should not drop anything."""
    df_clean = preprocessing.clean_dataframe(sample_cleaned_df, month_label="2023-01")
    assert len(df_clean) == len(sample_cleaned_df)


def test_remove_nulls_drops_nan_rows(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    df.loc[df.index[0], "trip_distance"] = pd.NA
    df.loc[df.index[1], "trip_duration"] = pd.NA

    df_clean = preprocessing.remove_nulls(df)
    assert len(df_clean) == len(df) - 2


# ── New-filter coverage ─────────────────────────────────────────────────────

def test_remove_duplicates_drops_exact_rows(sample_cleaned_df):
    # Duplicate the first 3 rows
    df = pd.concat([sample_cleaned_df, sample_cleaned_df.iloc[:3]],
                   ignore_index=True)
    assert len(df) == len(sample_cleaned_df) + 3
    df_clean = preprocessing.remove_duplicates(df)
    assert len(df_clean) == len(sample_cleaned_df)


def test_remove_wrong_month_drops_stragglers(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    # Place the first 4 rows in Feb 2023 (they were built in Jan 2023)
    df.loc[df.index[:4], "pickup_datetime"] = pd.Timestamp("2023-02-15 10:00:00")

    df_clean = preprocessing.remove_wrong_month(df, expected_ym="2023-01")
    assert len(df_clean) == len(df) - 4
    assert (df_clean["pickup_datetime"].dt.month == 1).all()


def test_remove_wrong_month_noop_on_bad_stamp(sample_cleaned_df):
    # An unparseable YM should short-circuit gracefully
    df_clean = preprocessing.remove_wrong_month(sample_cleaned_df, expected_ym="garbage")
    assert len(df_clean) == len(sample_cleaned_df)


def test_remove_trivial_same_zone_trips_drops_pu_equals_do(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    # Rows 0-4: same zone AND tiny distance → should drop
    df.loc[df.index[:5], "DOLocationID"] = df.loc[df.index[:5], "PULocationID"]
    df.loc[df.index[:5], "trip_distance"] = 0.03  # below 0.10 mi threshold
    # Row 5: same zone BUT legitimate distance → should keep
    df.loc[df.index[5], "DOLocationID"] = df.loc[df.index[5], "PULocationID"]
    df.loc[df.index[5], "trip_distance"] = 2.5

    df_clean = preprocessing.remove_trivial_same_zone_trips(df)
    assert len(df_clean) == len(df) - 5
    # The long same-zone trip survived
    survivor_idx = df.index[5]
    assert survivor_idx in df_clean.index


def test_remove_iqr_speed_outliers_handles_small_zones(sample_cleaned_df):
    """Zones with fewer than min_trips_per_zone rows pass through untouched."""
    df_clean = preprocessing.remove_iqr_speed_outliers_by_zone(
        sample_cleaned_df, min_trips_per_zone=10_000,
    )
    assert len(df_clean) == len(sample_cleaned_df)


def test_remove_iqr_speed_outliers_filters_extreme_speed():
    """Inject a clear outlier into a single zone and verify it's removed."""
    import numpy as np
    # 300 rows all in zone 1, speed ~ 20 mph, plus 1 outlier at 90 mph
    n = 300
    pickup = pd.date_range("2023-01-02", periods=n + 1, freq="1min")
    duration = np.full(n + 1, 600.0, dtype="float32")      # 10 min
    distance = np.full(n + 1, 20.0 * 600 / 3600, dtype="float32")  # 20 mph
    # Outlier: 90 mph
    distance[-1] = 90.0 * 600 / 3600
    df = pd.DataFrame({
        "pickup_datetime": pickup,
        "dropoff_datetime": pickup + pd.to_timedelta(duration, unit="s"),
        "trip_distance": distance,
        "trip_duration": duration,
        "PULocationID": pd.array(np.full(n + 1, 1), dtype="Int32"),
        "DOLocationID": pd.array(np.full(n + 1, 1), dtype="Int32"),
    })
    df_clean = preprocessing.remove_iqr_speed_outliers_by_zone(
        df, k=3.0, min_trips_per_zone=100,
    )
    # Exactly one row should be removed
    assert len(df_clean) == len(df) - 1


def test_remove_invalid_quality_signals_drops_negative_fare(sample_cleaned_df):
    df = sample_cleaned_df.copy()
    df["fare_amount"] = 10.0
    df["passenger_count"] = pd.array([1] * len(df), dtype="Int16")
    df.loc[df.index[0], "fare_amount"] = -5.0      # negative fare
    df.loc[df.index[1], "passenger_count"] = 0     # zero passengers
    df_clean = preprocessing.remove_invalid_quality_signals(df)
    assert len(df_clean) == len(df) - 2


def test_remove_invalid_quality_signals_noop_when_cols_absent(sample_cleaned_df):
    df_clean = preprocessing.remove_invalid_quality_signals(sample_cleaned_df)
    assert len(df_clean) == len(sample_cleaned_df)
