"""
config.py
---------
Centralised configuration for the Traffic Congestion pipeline.

All filesystem paths are resolved relative to the project root (the
directory containing this file's grandparent). Importing modules should
**never** hard-code paths — always reference the constants here.

All thresholds and algorithm hyper-parameters live in this module so
that tuning and experimentation are one edit away from reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ── Path constants ────────────────────────────────────────────────────────────

# Project root = <repo>/ (the parent of src/)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"

MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

# ── Intermediate / output file names ──────────────────────────────────────────

CLEANED_TRIPS_GLOB: Final[str] = "cleaned_trips_*.parquet"
FEATURES_TRIPS_GLOB: Final[str] = "features_trips_*.parquet"
AGGREGATED_FILE: Final[str] = "aggregated_zone_hour.parquet"
SCALED_FILE: Final[str] = "scaled_features.parquet"
CLUSTER_LABELS_FILE: Final[str] = "cluster_labels.parquet"
VIZ_DATA_FILE: Final[str] = "viz_data_precomputed.parquet"
ZONE_CENTROIDS_FILE: Final[str] = "taxi_zone_centroids.csv"

MODEL_FILE: Final[str] = "model.pkl"
SCALER_FILE: Final[str] = "scaler.pkl"
METRICS_FILE: Final[str] = "metrics.json"

# ── Cleaning thresholds (per PRD §8.2) ────────────────────────────────────────

# Instantaneous speed bounds (miles per hour)
MIN_SPEED_MPH: Final[float] = 1.0
MAX_SPEED_MPH: Final[float] = 100.0

# Trip distance bounds (miles)
MIN_DISTANCE_MILES: Final[float] = 0.01
MAX_DISTANCE_MILES: Final[float] = 200.0

# Trip duration bounds (seconds)
MIN_DURATION_SECONDS: Final[int] = 30
MAX_DURATION_SECONDS: Final[int] = 4 * 3600  # 4 hours

# Valid NYC TLC taxi zone IDs
MIN_ZONE_ID: Final[int] = 1
MAX_ZONE_ID: Final[int] = 263

# Trivial same-zone trips (PU == DO with near-zero distance) are
# cancelled-trip / parking-lot artifacts that clear the MIN_DISTANCE
# gate but corrupt the low end of the speed distribution. Anything
# <= this distance AND same-zone is dropped.
TRIVIAL_SAME_ZONE_DISTANCE_MILES: Final[float] = 0.10

# ── Data-driven outlier removal (optional) ───────────────────────────────────
#
# Hard-coded MIN/MAX speeds are reasonable but arbitrary: a 40 mph trip
# in Midtown is an outlier while the same speed on the FDR is normal.
# Enabling per-zone IQR-fence filtering removes speeds that fall outside
# [Q1 - k·IQR, Q3 + k·IQR] *within their own zone*. Only zones with
# enough trips for the quantile estimate to be stable are filtered.
USE_IQR_SPEED_FILTER: Final[bool] = False   # opt-in via CLI / Makefile
IQR_FENCE_K: Final[float] = 3.0             # conventional outlier multiplier
IQR_MIN_TRIPS_PER_ZONE: Final[int] = 200    # skip zones with too few trips

# ── Quality-signal columns (optional carry-through) ──────────────────────────
#
# When enabled, preprocessing retains fare_amount and passenger_count as
# quality signals (they aren't used as clustering features) so that
# negative fares, zero-passenger records, and similar anomalies can be
# audited downstream.
KEEP_QUALITY_COLUMNS: Final[bool] = True
QUALITY_COLUMNS: Final[list[str]] = ["fare_amount", "passenger_count"]

# ── Feature sets ──────────────────────────────────────────────────────────────

# Columns carried from raw parquet → cleaned parquet
RAW_KEEP_COLUMNS: Final[list[str]] = [
    "pickup_datetime",
    "dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
]

# Column aliases across NYC TLC dataset vintages.
# Quality-signal columns (``fare_amount``, ``passenger_count``) are optional:
# they are retained when ``KEEP_QUALITY_COLUMNS`` is True and used as QC
# gates in preprocessing, but never fed to the clusterer.
COLUMN_ALIASES: Final[dict[str, list[str]]] = {
    "pickup_datetime": [
        "tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime",
    ],
    "dropoff_datetime": [
        "tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime",
    ],
    "trip_distance": ["trip_distance"],
    "PULocationID": ["PULocationID"],
    "DOLocationID": ["DOLocationID"],
    "fare_amount": ["fare_amount"],
    "passenger_count": ["passenger_count"],
}

# Columns considered *optional* — load if present, silently skip if absent.
OPTIONAL_COLUMNS: Final[set[str]] = {"fare_amount", "passenger_count"}

# Final feature set consumed by the clusterer.
#
# Design rationale (post-review upgrade):
#   · zone_id REMOVED — treating a categorical ID as a continuous feature
#     was smuggling geographic nonsense into Euclidean distance.
#   · hour_of_day REPLACED with cyclic (hour_sin, hour_cos) so that
#     23:00 and 00:00 are correctly adjacent.
#   · avg_speed_mph REMOVED — it is linearly redundant with
#     speed_deviation (the baseline is just a per-zone-hour offset),
#     which was double-weighting the speed dimension.
#   · trip_density REPLACED with log1p(trip_density) to tame the
#     heavy-tailed power-law across zone-hours.
#   · is_weekend ADDED to separate weekday vs. weekend regimes.
CLUSTER_FEATURES: Final[list[str]] = [
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "speed_deviation",
    "log_trip_density",
]

# Columns preserved on the cluster-labels parquet for downstream viz
CLUSTER_LABEL_CARRY_COLUMNS: Final[list[str]] = [
    "zone_id", "hour_of_day", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos",
    "avg_speed_mph", "trip_density", "log_trip_density",
    "speed_deviation", "is_airport",
    "congestion_level",
]

# ── KPI targets (per PRD §4) ──────────────────────────────────────────────────

TARGET_SILHOUETTE: Final[float] = 0.5       # > 0.5
TARGET_CVINTRA: Final[float] = 0.20         # < 20 %
TARGET_LATENCY_MS: Final[int] = 500         # p95

# ── K-Means hyper-parameters ──────────────────────────────────────────────────

K_RANGE: Final[tuple[int, int]] = (2, 11)   # inclusive-exclusive upper bound
DEFAULT_K: Final[int] = 6                   # fallback if elbow inconclusive
KMEANS_RANDOM_STATE: Final[int] = 42
KMEANS_N_INIT: Final[int] = 10
KMEANS_MAX_ITER: Final[int] = 300

# Cap silhouette-evaluation sample size for speed on huge datasets
SILHOUETTE_SAMPLE_SIZE: Final[int] = 50_000

# Stability analysis — ARI across multiple random seeds
STABILITY_N_SEEDS: Final[int] = 5
STABILITY_ARI_MIN: Final[float] = 0.70   # warning threshold

# ── Scaler selection ─────────────────────────────────────────────────────────
# "standard" — classic z-score (mean=0, std=1). Pulled around by tails.
# "robust"   — uses median + IQR; resistant to heavy-tailed features such
#              as trip_density (even after log1p).
SCALER_TYPE: Final[str] = "standard"        # "standard" | "robust"

# ── Elbow detection ──────────────────────────────────────────────────────────
# If the optional ``kneed`` package is installed, use it for principled
# elbow detection. Otherwise fall back to the 2nd-derivative heuristic.
USE_KNEED_ELBOW: Final[bool] = True

# ── DBSCAN characterization ──────────────────────────────────────────────────
# Maximum fraction of points classified as noise before we flag the run.
DBSCAN_NOISE_WARN_FRACTION: Final[float] = 0.30

# ── DBSCAN hyper-parameters (optional secondary algorithm) ───────────────────

DBSCAN_EPS: Final[float] = 0.5
DBSCAN_MIN_SAMPLES: Final[int] = 10

# ── Semantic congestion labels ────────────────────────────────────────────────
# 0 = Low (fastest cluster), …, k-1 = High (slowest cluster)
# Human-readable names for k=3; for other k, labels fall back to the integer id.

CONGESTION_LABELS: Final[dict[int, str]] = {0: "Low", 1: "Medium", 2: "High"}
CONGESTION_HEX_COLORS: Final[dict[int, str]] = {
    0: "#2ecc71",  # Low    — green
    1: "#f39c12",  # Medium — amber
    2: "#e74c3c",  # High   — red
}
# RGBA (0–255) for pydeck ScatterplotLayer
CONGESTION_RGBA_COLORS: Final[dict[int, list[int]]] = {
    0: [46, 204, 113, 180],
    1: [243, 156, 18, 180],
    2: [231, 76, 60, 200],
}

DAY_NAMES: Final[list[str]] = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
DAY_NAMES_SHORT: Final[list[str]] = [
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
]

# ── NYC reference coordinates (initial map view) ─────────────────────────────

NYC_CENTER_LAT: Final[float] = 40.7128
NYC_CENTER_LON: Final[float] = -74.0060

# ── Zone metadata (post-hoc labels, NOT used as clustering features) ─────────
# Kept in cluster_labels.parquet so the dashboard can highlight airport and
# CBD zones, but deliberately excluded from CLUSTER_FEATURES to avoid biasing
# the unsupervised model.
AIRPORT_ZONES: Final[set[int]] = {1, 132, 138}       # EWR, JFK, LGA

# Manhattan Central Business District Tolling zones (south of 60th Street).
# Conservative set of well-known CBD zone IDs from the NYC TLC zone lookup.
# Used post-hoc to contextualise congestion clusters — not as features.
CBD_ZONES: Final[set[int]] = {
    4, 12, 13, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114,
    125, 137, 140, 141, 142, 143, 144, 148, 151, 158, 161, 162,
    163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233,
    234, 246, 249, 261, 262, 263,
}

RUSH_HOUR_MORNING: Final[set[int]] = {7, 8, 9}
RUSH_HOUR_EVENING: Final[set[int]] = {16, 17, 18, 19}


def ensure_directories() -> None:
    """Create all standard directories if they do not yet exist."""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
