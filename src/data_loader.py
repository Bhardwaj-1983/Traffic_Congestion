"""
data_loader.py
--------------
Loads raw NYC Taxi Trip Record parquet files from ``data/raw/``, selects
only the columns required for downstream processing, standardises column
names across dataset vintages, and writes cleaned monthly parquet files
to ``data/processed/cleaned_trips_YYYY-MM.parquet``.

Implements PRD §8.1 ("Data Loading").

Usage (from project root):
    python -m src.data_loader
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    COLUMN_ALIASES,
    KEEP_QUALITY_COLUMNS,
    OPTIONAL_COLUMNS,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_directories,
)
from src.utils import get_logger, parse_year_month, timeit, write_parquet

logger = get_logger(__name__)


# ── Column standardisation ────────────────────────────────────────────────────

def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Return the first alias present in ``df.columns`` — else ``None``."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw columns to canonical names.

    Required columns that cannot be resolved raise ``ValueError``;
    columns listed in ``OPTIONAL_COLUMNS`` are silently skipped when
    absent — they are loaded opportunistically for quality-signal use
    and should not break older datasets that don't include them.
    """
    rename_map: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _resolve_column(df, aliases)
        if found is None:
            if canonical in OPTIONAL_COLUMNS:
                logger.debug(f"  optional column '{canonical}' not present — skipping")
                continue
            raise ValueError(
                f"Cannot find a column for '{canonical}'. "
                f"Tried aliases: {aliases}. "
                f"Available columns: {list(df.columns)}"
            )
        if found != canonical:
            rename_map[found] = canonical
    return df.rename(columns=rename_map)


# ── File-level load ───────────────────────────────────────────────────────────

def load_raw_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single raw parquet file, returning a DataFrame that contains only
    the canonical columns needed downstream.
    """
    logger.info(f"Loading: {filepath.name}")
    table = pq.read_table(filepath)
    df = table.to_pandas()
    logger.info(f"  rows={len(df):,}  raw_cols={len(df.columns)}")

    df = _standardize_columns(df)

    # Always keep the required columns; keep optional quality columns only
    # when (a) the config flag is on AND (b) the column exists post-rename.
    required = [c for c in COLUMN_ALIASES.keys() if c not in OPTIONAL_COLUMNS]
    keep = list(required)
    if KEEP_QUALITY_COLUMNS:
        for c in OPTIONAL_COLUMNS:
            if c in df.columns:
                keep.append(c)
    df = df[keep].copy()
    return df


def compute_trip_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``trip_duration`` (seconds) from pickup/dropoff timestamps."""
    df["trip_duration"] = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds().astype("float32")
    return df


# ── Bulk load ─────────────────────────────────────────────────────────────────

@timeit("load_all_months")
def load_all_months(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Load every ``*.parquet`` file found in ``raw_dir``.

    Returns a mapping ``{YYYY-MM: DataFrame}``. YYYY-MM is parsed from
    the filename; if parsing fails the mode month of ``pickup_datetime``
    is used as a fallback.
    """
    files = sorted(glob.glob(str(raw_dir / "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No .parquet files found in {raw_dir}. Download NYC Taxi "
            "Trip Records from the TLC site or run `scripts/download_data.py`."
        )

    monthly: dict[str, pd.DataFrame] = {}
    for f in files:
        path = Path(f)
        df = load_raw_file(path)
        df = compute_trip_duration(df)

        try:
            ym = parse_year_month(path)
        except ValueError:
            # Fallback: use the modal pickup month
            ym = (
                df["pickup_datetime"]
                .dt.to_period("M")
                .mode()[0]
                .strftime("%Y-%m")
            )
        monthly[ym] = df
        logger.info(f"  parsed month {ym}: rows={len(df):,}")

    return monthly


# ── Persistence ───────────────────────────────────────────────────────────────

def save_monthly_data(
    monthly_data: dict[str, pd.DataFrame],
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Write each month's DataFrame to ``cleaned_trips_YYYY-MM.parquet``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ym, df in monthly_data.items():
        out_path = output_dir / f"cleaned_trips_{ym}.parquet"
        write_parquet(df, out_path)
        logger.info(f"  saved {out_path.name} rows={len(df):,}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ensure_directories()
    logger.info("=== Stage 1 :: Data Loading ===")
    monthly = load_all_months()
    save_monthly_data(monthly)
    total = sum(len(v) for v in monthly.values())
    logger.info(f"Done. Months={len(monthly)}  total_rows={total:,}")


if __name__ == "__main__":
    main()
