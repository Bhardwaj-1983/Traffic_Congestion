"""
feature_engineering.py
-----------------------
Derives time, spatial, and congestion features — including the project's
*core innovation*, ``speed_deviation`` — from cleaned trip data and then
aggregates to the (zone_id, hour_of_day, day_of_week) granularity used
by the clustering model.

Implements PRD §8.3–§8.4 and §12, with post-review upgrades:

  · Cyclic hour encoding (hour_sin, hour_cos) so that 23:00 and 00:00 are
    correctly adjacent under Euclidean distance.
  · log1p(trip_density) so heavy-tailed zone-hour counts don't pull the
    scaler and distort K-Means.
  · Post-hoc zone attribution (is_airport, is_cbd) for dashboard colour
    coding — NOT used as clustering features.
  · Rush-hour flag (is_rush_hour) as carried metadata for dashboard drill-downs.

Stages:
  1. Per-trip feature derivation (time, speed, zone_id)
  2. Aggregate by (zone_id, hour_of_day, day_of_week, is_weekend)
  3. Compute speed_deviation, log_trip_density, cyclic hour,
     is_airport, is_cbd, is_rush_hour

Usage (from project root):
    python -m src.feature_engineering
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AGGREGATED_FILE,
    AIRPORT_ZONES,
    CBD_ZONES,
    CLEANED_TRIPS_GLOB,
    PROCESSED_DIR,
    RUSH_HOUR_EVENING,
    RUSH_HOUR_MORNING,
)
from src.utils import get_logger, read_parquet_safe, timeit, write_parquet

logger = get_logger(__name__)


# ── Per-trip feature derivation ───────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day (0–23), day_of_week (0=Mon), is_weekend, month."""
    dt = df["pickup_datetime"]
    df = df.copy()
    df["hour_of_day"] = dt.dt.hour.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("bool")
    df["month"] = dt.dt.month.astype("int8")
    return df


def add_speed_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trip speed (mph). NaN for zero-duration rows."""
    df = df.copy()
    hours = df["trip_duration"] / 3600.0
    # Protect against divide-by-zero → NaN (dropped downstream)
    hours = hours.replace(0, np.nan)
    df["speed_mph"] = (df["trip_distance"] / hours).astype("float32")
    return df


def add_zone_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Use PULocationID as the zone identifier (per PRD §8.3)."""
    df = df.copy()
    df["zone_id"] = df["PULocationID"].astype("int32")
    return df


@timeit("engineer_features_monthly")
def engineer_features_monthly(df: pd.DataFrame, ym: str) -> pd.DataFrame:
    """Apply every per-trip feature derivation step to a single month."""
    logger.info(f"Feature engineering {ym}: rows={len(df):,}")
    df = add_time_features(df)
    df = add_speed_feature(df)
    df = add_zone_feature(df)

    # Drop any rows with NaN speed (likely zero-duration records)
    before = len(df)
    df = df.dropna(subset=["speed_mph"])
    dropped = before - len(df)
    if dropped:
        logger.debug(f"  dropped {dropped:,} rows with NaN speed")
    return df


# ── Aggregation to zone-hour ──────────────────────────────────────────────────

def aggregate_zone_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (zone_id, hour_of_day, day_of_week, is_weekend) and compute:
        avg_speed_mph     — mean speed
        trip_density      — trip count
        avg_trip_duration — mean duration (seconds)
    """
    agg = (
        df.groupby(
            ["zone_id", "hour_of_day", "day_of_week", "is_weekend"],
            observed=True,
        )
        .agg(
            avg_speed_mph=("speed_mph", "mean"),
            trip_density=("speed_mph", "count"),
            avg_trip_duration=("trip_duration", "mean"),
        )
        .reset_index()
    )
    agg["avg_speed_mph"] = agg["avg_speed_mph"].astype("float32")
    agg["trip_density"] = agg["trip_density"].astype("int32")
    agg["avg_trip_duration"] = agg["avg_trip_duration"].astype("float32")
    return agg


def compute_speed_deviation(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Core innovation (PRD §12):

        speed_deviation = avg_speed_mph - baseline_speed(zone_id, hour_of_day)

    The baseline is the zone–hour grand mean across all day types. A
    negative deviation indicates the zone is currently slower than its
    historical norm — a genuine congestion signal, distinct from a zone
    that is *naturally* slow (e.g. a school zone).
    """
    logger.info("Computing speed_deviation (core innovation)…")

    baseline = (
        agg.groupby(["zone_id", "hour_of_day"])["avg_speed_mph"]
        .mean()
        .rename("baseline_speed")
        .reset_index()
    )

    agg = agg.merge(baseline, on=["zone_id", "hour_of_day"], how="left")
    agg["speed_deviation"] = (
        agg["avg_speed_mph"] - agg["baseline_speed"]
    ).astype("float32")
    agg = agg.drop(columns=["baseline_speed"])
    return agg


def add_cyclic_hour(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour_of_day as (sin, cos) so 23:00 ↔ 00:00 are adjacent.
    This is essential for any distance-based model on circular time.
    """
    angle = 2.0 * np.pi * agg["hour_of_day"].astype("float32") / 24.0
    agg = agg.copy()
    agg["hour_sin"] = np.sin(angle).astype("float32")
    agg["hour_cos"] = np.cos(angle).astype("float32")
    return agg


def add_log_density(agg: pd.DataFrame) -> pd.DataFrame:
    """Log-transform the heavy-tailed trip density count."""
    agg = agg.copy()
    agg["log_trip_density"] = np.log1p(agg["trip_density"].astype("float32")).astype("float32")
    return agg


def add_zone_attributes(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Attach post-hoc zone attributes used by the dashboard but NOT by the
    clustering model:

    · is_airport   — zone is EWR/JFK/LGA (distinct congestion signature)
    · is_cbd       — zone falls inside Manhattan CBD tolling region
    · is_rush_hour — pickup hour is in morning or evening rush window
    """
    agg = agg.copy()
    agg["is_airport"] = agg["zone_id"].isin(AIRPORT_ZONES)
    agg["is_cbd"] = agg["zone_id"].isin(CBD_ZONES)
    rush = RUSH_HOUR_MORNING | RUSH_HOUR_EVENING
    agg["is_rush_hour"] = agg["hour_of_day"].isin(rush)
    return agg


# ── Full pipeline ─────────────────────────────────────────────────────────────

@timeit("run_feature_engineering")
def run_feature_engineering(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """
    End-to-end feature-engineering stage. Writes per-month feature files
    and the consolidated aggregated zone-hour parquet.

    Returns the aggregated DataFrame for convenience/testing.
    """
    files = sorted(glob.glob(str(processed_dir / CLEANED_TRIPS_GLOB)))
    if not files:
        raise FileNotFoundError(
            f"No {CLEANED_TRIPS_GLOB} files found in {processed_dir}. "
            "Run `python -m src.data_loader` and `python -m src.preprocessing` first."
        )

    all_slices: list[pd.DataFrame] = []
    cols_needed = [
        "zone_id", "hour_of_day", "day_of_week", "is_weekend",
        "speed_mph", "trip_duration", "month",
    ]

    for f in files:
        path = Path(f)
        ym = path.stem.replace("cleaned_trips_", "")
        df = read_parquet_safe(path, label="preprocessing")
        df = engineer_features_monthly(df, ym)

        # Persist per-month feature file (for reproducibility / EDA)
        out_path = processed_dir / f"features_trips_{ym}.parquet"
        write_parquet(df, out_path)
        logger.info(f"  saved {out_path.name}")

        # Keep only columns needed for aggregation to save memory
        all_slices.append(df[cols_needed])

    logger.info("Concatenating monthly slices for aggregation…")
    combined = pd.concat(all_slices, ignore_index=True)
    logger.info(f"Combined dataset rows: {len(combined):,}")

    logger.info("Aggregating → zone-hour granularity…")
    agg = aggregate_zone_hour(combined)
    logger.info(f"Aggregated shape: {agg.shape}")

    # Core innovation + downstream derivations
    agg = compute_speed_deviation(agg)
    agg = add_cyclic_hour(agg)
    agg = add_log_density(agg)
    agg = add_zone_attributes(agg)

    out_path = processed_dir / AGGREGATED_FILE
    write_parquet(agg, out_path)
    logger.info(f"Saved: {out_path.name}")

    # Quick summary for the log
    logger.info("Aggregated summary:")
    logger.info(f"  unique zones      = {agg['zone_id'].nunique()}")
    logger.info(f"  hours covered     = {agg['hour_of_day'].nunique()}")
    logger.info(f"  rows              = {len(agg):,}")
    logger.info(f"  speed_dev   mean  = {agg['speed_deviation'].mean():+.4f} mph")
    logger.info(f"  speed_dev   std   = {agg['speed_deviation'].std():.4f} mph")
    logger.info(f"  log_density mean  = {agg['log_trip_density'].mean():.3f}")
    logger.info(f"  airport rows      = {agg['is_airport'].sum():,}")
    logger.info(f"  CBD rows          = {agg['is_cbd'].sum():,}")
    logger.info(f"  rush-hour rows    = {agg['is_rush_hour'].sum():,}")
    return agg


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Stage 3 :: Feature Engineering ===")
    run_feature_engineering()
    logger.info("Done.")


if __name__ == "__main__":
    main()
