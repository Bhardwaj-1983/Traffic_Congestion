"""
preprocessing.py
----------------
Applies data-quality filters to the monthly DataFrames produced by
``data_loader``. Reads from ``data/processed/cleaned_trips_*.parquet``
and overwrites each file in place after cleaning.

Cleaning rules (post-review upgrade):
  1. Cast columns to canonical dtypes
  2. Drop exact duplicate rows (TLC data has duplicates)
  3. Drop rows with nulls in critical columns
  4. Drop rows with invalid timestamps (pickup >= dropoff)
  5. Drop rows whose pickup_datetime falls outside the file's nominal month
     (TLC data occasionally contains stragglers from adjacent months)
  6. Drop trips with non-positive or unrealistically large distances
  7. Drop trips with non-positive or unrealistically long durations
  8. Drop trips whose computed speed lies outside [MIN_SPEED, MAX_SPEED] mph
  9. Drop trips where BOTH pickup and dropoff zone IDs are outside the valid
     TLC range (previous versions only validated PULocationID)
 10. Drop "trivial same-zone" trips (PU == DO with near-zero distance):
     parking-lot / cancelled-trip artifacts that corrupt the low end of the
     speed distribution even though they clear the MIN_DISTANCE gate
 11. (optional) Drop zone-level speed outliers using an IQR fence per zone:
     a more principled data-driven filter than the hard MIN/MAX_SPEED caps.
     Disabled by default; enable via ``USE_IQR_SPEED_FILTER`` or
     ``--iqr`` on the CLI.
 12. (optional) Drop obviously invalid quality-signal rows (negative fares,
     zero-passenger records) when ``fare_amount`` / ``passenger_count``
     are retained by the loader.

Usage (from project root):
    python -m src.preprocessing
    python -m src.preprocessing --iqr           # enable per-zone IQR filter
    python -m src.preprocessing --no-quality    # skip fare/passenger QC gate
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_TRIPS_GLOB,
    IQR_FENCE_K,
    IQR_MIN_TRIPS_PER_ZONE,
    KEEP_QUALITY_COLUMNS,
    MAX_DISTANCE_MILES,
    MAX_DURATION_SECONDS,
    MAX_SPEED_MPH,
    MAX_ZONE_ID,
    MIN_DISTANCE_MILES,
    MIN_DURATION_SECONDS,
    MIN_SPEED_MPH,
    MIN_ZONE_ID,
    PROCESSED_DIR,
    QUALITY_COLUMNS,
    TRIVIAL_SAME_ZONE_DISTANCE_MILES,
    USE_IQR_SPEED_FILTER,
)
from src.utils import get_logger, read_parquet_safe, timeit, write_parquet

logger = get_logger(__name__)


# ── Dtype casting ─────────────────────────────────────────────────────────────

def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce").astype("float32")
    df["trip_duration"] = pd.to_numeric(df["trip_duration"], errors="coerce").astype("float32")
    df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce").astype("Int32")
    df["DOLocationID"] = pd.to_numeric(df["DOLocationID"], errors="coerce").astype("Int32")
    # Optional quality-signal columns — cast only if present
    if "fare_amount" in df.columns:
        df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce").astype("float32")
    if "passenger_count" in df.columns:
        df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce").astype("Int16")
    return df


# ── Filters ───────────────────────────────────────────────────────────────────

def _filter(df: pd.DataFrame, mask: pd.Series, rule_name: str) -> pd.DataFrame:
    """Apply a boolean mask and log the number of rows removed."""
    before = len(df)
    df = df[mask]
    dropped = before - len(df)
    if dropped:
        logger.debug(f"  {rule_name}: dropped {dropped:,} rows")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows on the natural key. TLC parquets contain
    occasional duplicate records (same PU/DO timestamps, distance, zones).
    """
    before = len(df)
    subset = [
        "pickup_datetime", "dropoff_datetime",
        "PULocationID", "DOLocationID", "trip_distance",
    ]
    df = df.drop_duplicates(subset=subset)
    dropped = before - len(df)
    if dropped:
        logger.debug(f"  remove_duplicates: dropped {dropped:,} rows")
    return df


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    critical = [
        "pickup_datetime", "dropoff_datetime",
        "trip_distance", "trip_duration",
        "PULocationID", "DOLocationID",
    ]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)
    if dropped:
        logger.debug(f"  remove_nulls: dropped {dropped:,} rows")
    return df


def remove_invalid_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    return _filter(
        df,
        df["pickup_datetime"] < df["dropoff_datetime"],
        "remove_invalid_timestamps",
    )


def remove_wrong_month(df: pd.DataFrame, expected_ym: str) -> pd.DataFrame:
    """
    Drop trips whose pickup_datetime falls outside the file's nominal month.

    TLC files occasionally contain rows from adjacent months (late arrivals
    or data-engineering bugs); leaving them in corrupts the monthly bucket
    used by the elbow/silhouette sweep.
    """
    try:
        expected = pd.Period(expected_ym, freq="M")
    except (ValueError, TypeError):
        logger.warning(f"  remove_wrong_month: unparseable YM '{expected_ym}' — skipping filter")
        return df
    mask = df["pickup_datetime"].dt.to_period("M") == expected
    return _filter(df, mask, f"remove_wrong_month (expected={expected_ym})")


def remove_invalid_distance(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["trip_distance"].between(MIN_DISTANCE_MILES, MAX_DISTANCE_MILES)
    return _filter(df, mask, "remove_invalid_distance")


def remove_invalid_duration(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["trip_duration"].between(MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
    return _filter(df, mask, "remove_invalid_duration")


def remove_speed_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Speed = distance (miles) / duration (hours); drop outside [MIN, MAX] mph."""
    duration_hours = (df["trip_duration"] / 3600.0).replace(0, np.nan)
    speed = df["trip_distance"] / duration_hours
    mask = speed.between(MIN_SPEED_MPH, MAX_SPEED_MPH)
    return _filter(df, mask, "remove_speed_outliers")


def remove_invalid_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Require BOTH PU and DO zone IDs to be inside the valid TLC range."""
    mask = (
        df["PULocationID"].between(MIN_ZONE_ID, MAX_ZONE_ID)
        & df["DOLocationID"].between(MIN_ZONE_ID, MAX_ZONE_ID)
    )
    return _filter(df, mask, "remove_invalid_zones")


def remove_trivial_same_zone_trips(
    df: pd.DataFrame,
    max_distance: float = TRIVIAL_SAME_ZONE_DISTANCE_MILES,
) -> pd.DataFrame:
    """
    Drop trips where pickup and dropoff zones are identical AND the
    distance is below ``max_distance`` (default 0.10 mi). These are
    overwhelmingly cancelled / rebooked / parking-lot artifacts that
    pass the distance gate but produce pathological speeds at the low end
    and inflate trip-density counts in airport / rideshare lots.
    """
    same_zone = df["PULocationID"] == df["DOLocationID"]
    near_zero = df["trip_distance"] <= max_distance
    keep = ~(same_zone & near_zero)
    return _filter(df, keep, f"remove_trivial_same_zone_trips (≤{max_distance} mi)")


def remove_iqr_speed_outliers_by_zone(
    df: pd.DataFrame,
    k: float = IQR_FENCE_K,
    min_trips_per_zone: int = IQR_MIN_TRIPS_PER_ZONE,
) -> pd.DataFrame:
    """
    Drop trips whose speed lies outside [Q1 - k·IQR, Q3 + k·IQR] computed
    *within each pickup zone*. A more principled outlier filter than the
    global MIN/MAX_SPEED caps because it preserves naturally fast/slow
    zones (FDR vs. school zones).

    Zones with fewer than ``min_trips_per_zone`` trips are passed through
    unfiltered to avoid noisy quantile estimates on small samples.
    """
    duration_hours = (df["trip_duration"] / 3600.0).replace(0, np.nan)
    speed = df["trip_distance"] / duration_hours

    # Compute per-zone Q1/Q3 vectorised, then merge fences back onto rows.
    grp = pd.DataFrame({
        "PULocationID": df["PULocationID"],
        "_speed": speed,
    }).dropna(subset=["_speed"])

    counts = grp.groupby("PULocationID")["_speed"].size()
    eligible = counts[counts >= min_trips_per_zone].index

    if len(eligible) == 0:
        logger.debug("  remove_iqr_speed_outliers_by_zone: no zones eligible — skipping")
        return df

    sub = grp[grp["PULocationID"].isin(eligible)]
    fences = (
        sub.groupby("PULocationID")["_speed"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.75: "q3"})
    )
    fences["iqr"] = fences["q3"] - fences["q1"]
    fences["lower"] = fences["q1"] - k * fences["iqr"]
    fences["upper"] = fences["q3"] + k * fences["iqr"]

    # Build a per-row lower/upper, defaulting to ±inf for non-eligible zones
    bounds = (
        df[["PULocationID"]]
        .merge(fences[["lower", "upper"]], left_on="PULocationID",
               right_index=True, how="left")
    )
    lower = bounds["lower"].fillna(-np.inf).values
    upper = bounds["upper"].fillna(np.inf).values

    keep = (speed.values >= lower) & (speed.values <= upper)
    keep = keep | speed.isna().values   # NaN speed handled by other filters
    return _filter(df, pd.Series(keep, index=df.index),
                   f"remove_iqr_speed_outliers_by_zone (k={k}, min_n={min_trips_per_zone})")


def remove_invalid_quality_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    If ``fare_amount`` / ``passenger_count`` are present, drop obviously
    invalid records: negative fares (refunds & data errors) and rides
    with zero passengers. These columns are only used here as quality
    signals — they are NOT retained in the clustering feature set.
    """
    masks = []
    if "fare_amount" in df.columns:
        masks.append(df["fare_amount"] >= 0)
    if "passenger_count" in df.columns:
        # passenger_count may be NaN for legacy rows; only drop explicit zeros
        pc = df["passenger_count"]
        masks.append(pc.isna() | (pc.astype("Int64") > 0))
    if not masks:
        return df
    combined = masks[0]
    for m in masks[1:]:
        combined = combined & m
    return _filter(df, combined, "remove_invalid_quality_signals")


# ── Composite pipeline ────────────────────────────────────────────────────────

@timeit("clean_dataframe")
def clean_dataframe(
    df: pd.DataFrame,
    month_label: str = "",
    expected_ym: str | None = None,
    use_iqr_filter: bool = USE_IQR_SPEED_FILTER,
    use_quality_gate: bool = KEEP_QUALITY_COLUMNS,
) -> pd.DataFrame:
    """
    Apply every cleaning rule in sequence; return the filtered DataFrame.

    Parameters
    ----------
    df                : raw/partially-cleaned monthly DataFrame
    month_label       : free-form label used for logging only
    expected_ym       : ``YYYY-MM`` stamp used to enforce the
                        ``remove_wrong_month`` filter. If None, that filter
                        is skipped (useful for synthetic test data).
    use_iqr_filter    : when True, apply per-zone IQR-fence filtering on
                        speed (more principled than the global MIN/MAX caps).
    use_quality_gate  : when True AND fare_amount/passenger_count are
                        present, drop rows with invalid quality signals.
    """
    initial = len(df)
    logger.info(f"Cleaning {month_label}: start_rows={initial:,}")

    df = cast_dtypes(df)
    df = remove_duplicates(df)
    df = remove_nulls(df)
    df = remove_invalid_timestamps(df)
    if expected_ym is not None:
        df = remove_wrong_month(df, expected_ym)
    df = remove_invalid_distance(df)
    df = remove_invalid_duration(df)
    df = remove_speed_outliers(df)
    df = remove_invalid_zones(df)
    df = remove_trivial_same_zone_trips(df)
    if use_quality_gate:
        df = remove_invalid_quality_signals(df)
    if use_iqr_filter:
        df = remove_iqr_speed_outliers_by_zone(df)

    final = len(df)
    pct = 100.0 * final / initial if initial else 0.0
    logger.info(f"  {month_label} cleaning done: kept {final:,}/{initial:,} ({pct:.1f}%)")
    return df


def process_all_monthly_files(
    processed_dir: Path = PROCESSED_DIR,
    use_iqr_filter: bool = USE_IQR_SPEED_FILTER,
    use_quality_gate: bool = KEEP_QUALITY_COLUMNS,
) -> None:
    """Load → clean → overwrite every cleaned_trips_*.parquet in ``processed_dir``."""
    files = sorted(glob.glob(str(processed_dir / CLEANED_TRIPS_GLOB)))
    if not files:
        raise FileNotFoundError(
            f"No {CLEANED_TRIPS_GLOB} files found in {processed_dir}. "
            "Run `python -m src.data_loader` first."
        )

    for f in files:
        path = Path(f)
        ym = path.stem.replace("cleaned_trips_", "")
        df = read_parquet_safe(path, label="data_loader")
        df = clean_dataframe(
            df,
            month_label=ym,
            expected_ym=ym,
            use_iqr_filter=use_iqr_filter,
            use_quality_gate=use_quality_gate,
        )
        write_parquet(df, path)
        logger.info(f"  overwritten {path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the preprocessing stage")
    parser.add_argument(
        "--iqr", action="store_true",
        help="Enable per-zone IQR-fence speed filter (opt-in)."
    )
    parser.add_argument(
        "--no-quality", action="store_true",
        help="Skip the fare/passenger quality-signal gate."
    )
    args = parser.parse_args()

    logger.info("=== Stage 2 :: Preprocessing ===")
    process_all_monthly_files(
        use_iqr_filter=args.iqr or USE_IQR_SPEED_FILTER,
        use_quality_gate=not args.no_quality and KEEP_QUALITY_COLUMNS,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
