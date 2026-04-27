# Traffic Congestion Pattern Discovery and Visualization

Unsupervised-learning system that discovers, analyses, and visualises traffic
congestion patterns from NYC Taxi Trip Records using **K-Means** (primary) and
**DBSCAN** (optional) clustering, then exposes the insights through an
interactive **Streamlit** dashboard.

> Reference: see `Product Requirements Document_ Traffic Congestion Pattern Discovery and Visualization.md`.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Data Acquisition](#data-acquisition)
6. [Running the Pipeline](#running-the-pipeline)
7. [Running the Web App](#running-the-web-app)
8. [Testing](#testing)
9. [Docker](#docker)
10. [KPIs and Evaluation](#kpis-and-evaluation)
11. [Key Design Decisions](#key-design-decisions)
12. [Limitations](#limitations)

---

## Project Overview

This project mines six months of NYC Yellow Taxi trip records to:

- Discover hidden traffic-congestion patterns via unsupervised clustering.
- Analyse spatial and temporal variations across the 263 NYC TLC taxi zones.
- Surface the insights through an interactive Streamlit dashboard.

**Core innovation** — the `speed_deviation` feature measures the gap between
a zone/hour's *current* average speed and its *historical baseline*,
enabling detection of genuine congestion versus naturally slow zones (e.g.
school zones with low speed limits).

---

## Architecture

```
       ┌──────────────────┐
       │ Raw .parquet     │  data/raw/*.parquet
       │ (NYC TLC)        │
       └──────────┬───────┘
                  ▼
   ┌─────────────────────────────┐
   │ Stage 1 · Data Loading      │  src/data_loader.py
   │   → cleaned_trips_*.parquet │
   └──────────────┬──────────────┘
                  ▼
   ┌─────────────────────────────┐
   │ Stage 2 · Preprocessing     │  src/preprocessing.py
   │   (quality filters)         │
   └──────────────┬──────────────┘
                  ▼
   ┌─────────────────────────────┐
   │ Stage 3 · Feature Eng.      │  src/feature_engineering.py
   │   · time features           │
   │   · speed_mph               │
   │   · zone aggregation        │
   │   · speed_deviation ★       │
   │   → aggregated_zone_hour    │
   └──────────────┬──────────────┘
                  ▼
   ┌─────────────────────────────┐
   │ Stage 4 · Clustering        │  src/clustering.py
   │   · StandardScaler          │
   │   · Elbow → k*              │
   │   · K-Means                 │
   │   · DBSCAN (optional)       │
   │   · Evaluation (Silhouette, │
   │     CVintra)                │
   │   → cluster_labels.parquet  │
   │   → model.pkl, scaler.pkl   │
   └──────────────┬──────────────┘
                  ▼
   ┌─────────────────────────────┐
   │ Stage 5 · Visualization     │  src/visualization.py
   │   · viz_data_precomputed    │
   │   · static PNG charts       │
   └──────────────┬──────────────┘
                  ▼
   ┌─────────────────────────────┐
   │ Streamlit Dashboard         │  app/app.py
   │   · Map · Charts · Filters  │
   │   · Insights panel          │
   └─────────────────────────────┘
```

---

## Project Structure

```
Traffic_congestion/
├── src/                            # Offline pipeline (ML + processing)
│   ├── __init__.py
│   ├── config.py                   # Centralised paths, thresholds, hyper-params
│   ├── utils.py                    # Shared logger, parquet/JSON I/O, timing
│   ├── data_loader.py              # Stage 1
│   ├── preprocessing.py            # Stage 2
│   ├── feature_engineering.py      # Stage 3
│   ├── clustering.py               # Stage 4
│   └── visualization.py            # Stage 5
│
├── app/                            # Streamlit web app (online serving)
│   ├── __init__.py
│   └── app.py
│
├── scripts/                        # Standalone CLI helpers
│   ├── download_data.py            # Fetch NYC TLC parquet files
│   └── make_centroids.py           # Build taxi_zone_centroids.csv
│
├── notebooks/
│   └── 01_eda.py                   # Jupytext-paired EDA notebook
│
├── tests/                          # pytest suite
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_clustering.py
│   └── test_utils.py
│
├── data/
│   ├── raw/                        # (gitignored) raw TLC parquets + centroids
│   └── processed/                  # (gitignored) intermediate parquets
│
├── models/                         # (gitignored) model.pkl, scaler.pkl
├── outputs/                        # (gitignored) static charts + metrics.json
│
├── main.py                         # Pipeline orchestrator
├── Dockerfile                      # Container build
├── docker-compose.yml              # One-shot compose deployment
├── Makefile                        # Developer shortcuts
├── pyproject.toml                  # Packaging + pytest config
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
```

---

## Setup and Installation

**Requirements**: Python 3.9+

```bash
pip install -r requirements.txt
```

Or use the Makefile:

```bash
make install
```

---

## Data Acquisition

### 1 · Trip records

Download six months of NYC Yellow Taxi trip records (default: January–June 2023):

```bash
python scripts/download_data.py
# or: make download
```

Custom range:

```bash
python scripts/download_data.py --year 2023 --start 1 --end 6 --cab yellow
```

Files land in `data/raw/`.

### 2 · Zone centroids (for the map view)

The Streamlit app places zones using an `(x, y)` centroid per TLC zone. Build
the file from the official TLC shapefile:

```bash
python scripts/make_centroids.py
# or: make centroids
```

This creates `data/raw/taxi_zone_centroids.csv`.

---

## Running the Pipeline

**Full pipeline (recommended):**

```bash
python main.py
# or: make pipeline
```

**Stage-by-stage:**

```bash
python -m src.data_loader         # Stage 1
python -m src.preprocessing       # Stage 2
python -m src.feature_engineering # Stage 3
python -m src.clustering          # Stage 4 (use --k N to skip elbow)
python -m src.visualization       # Stage 5
```

**Useful flags:**

```bash
python main.py --k 4              # force k=4, skip elbow
python main.py --dbscan           # also run DBSCAN for hotspots
python main.py --only-viz         # regenerate viz data only
python main.py --skip-load --skip-preprocess
```

**Stage-specific upgrades (post-review):**

```bash
# Preprocessing — enable per-zone IQR-fence speed filter
python -m src.preprocessing --iqr

# Preprocessing — skip the fare/passenger quality-signal gate
python -m src.preprocessing --no-quality

# Clustering — switch to RobustScaler (better for heavy-tailed features)
python -m src.clustering --scaler robust
```

All artefacts are cached on disk, so you can re-run later stages without
repeating the expensive early ones.

---

## Running the Web App

```bash
streamlit run app/app.py
# or: make app
```

Open `http://localhost:8501`. The dashboard exposes:

- Hour-of-day slider and day-type filter (All / Weekdays / Weekends).
- Pydeck map of zones coloured by congestion level.
- Hourly trend (avg speed + trip density).
- Cluster distribution for the current filter.
- Top 15 most-congested zones.
- Insights panel (peak hours, top zones, current-filter KPIs, model quality).
- Expandable raw-data table.

---

## Testing

```bash
python -m pytest tests/ -v
# or: make test
```

Coverage focus:

- Preprocessing rule correctness (timestamps, distance, duration, speed, zones).
- Feature-engineering correctness (`speed_mph`, `speed_deviation`).
- Clustering: feature prep, scaling, semantic label monotonicity, evaluation.
- Utils: parquet/JSON I/O round-trip, filename parsing.

---

## Docker

```bash
docker build -t traffic-congestion .
docker run --rm -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/outputs:/app/outputs \
    traffic-congestion
```

Or use Docker Compose:

```bash
docker compose up --build
```

To run the pipeline inside the container:

```bash
docker compose run --rm app python main.py
```

---

## KPIs and Evaluation

| Metric | Target | Definition |
|---|---|---|
| **Silhouette Score** | > 0.5 | Cluster cohesion and separation |
| **CVintra (speed)** | < 20 % | Intra-cluster coefficient of variation for speed |
| **Inference Latency** | < 500 ms (p95) | Streamlit filter → render |
| **Data Freshness** | < 24 hours | Raw → processed pipeline lag |

After `python -m src.clustering`, metrics are persisted at
`outputs/metrics.json` and surfaced in the dashboard footer.

---

## Key Design Decisions

- **K-Means over Hierarchical Clustering** — O(n·k·i) vs O(n²·log n);
  the latter is infeasible at >1 M records.
- **`speed_deviation` as the core feature** — separates genuine congestion
  events from permanently slow zones.
- **Zone-hour aggregation** — collapses millions of trips into thousands of
  observations for fast Streamlit rendering.
- **Streamlit `@st.cache_data` + `@st.cache_resource`** — avoids reloading
  parquet files and the model on every filter change.
- **Centralised `src/config.py`** — every magic number lives in one place for
  reproducibility and tuning.

### Post-review upgrades (this branch)

Feature-engineering hardening that directly lifts cluster quality:

- **`zone_id` dropped from the cluster feature set** — treating a categorical
  TLC ID as continuous Euclidean distance was smuggling geographic nonsense
  into the model. Zones are still inherited by the labelled output for
  dashboard colouring, but do not influence the unsupervised fit.
- **Cyclic hour encoding** — `hour_of_day` is replaced by `(hour_sin, hour_cos)`
  so 23:00 and 00:00 are adjacent under Euclidean distance.
- **Single speed signal** — `avg_speed_mph` removed in favour of
  `speed_deviation` to avoid double-weighting the speed dimension.
- **`log1p(trip_density)`** — tames the heavy-tailed zone-hour count
  distribution so `StandardScaler` is well-behaved.
- **Optional `RobustScaler`** — median/IQR scaling for the most skewed runs
  (`python -m src.clustering --scaler robust`).

Preprocessing hardening for 2025-vintage data:

- Exact-duplicate removal on the natural key.
- Wrong-month straggler filter (drops rows whose `pickup_datetime` falls
  outside the filename-stamped month).
- Trivial same-zone-trip filter (`PU == DO` with distance ≤ 0.10 mi).
- Both `PULocationID` and `DOLocationID` validated against the TLC zone range.
- Optional per-zone IQR-fence speed filter (`--iqr`) — a more principled
  data-driven outlier rule than the global MIN/MAX speed caps.
- Quality-signal gate (`fare_amount ≥ 0`, `passenger_count > 0`) when those
  columns are available in the raw parquet.

Cluster-quality diagnostics:

- Davies–Bouldin and Calinski–Harabasz reported alongside Silhouette.
- Per-cluster Silhouette (not just the scalar average).
- Seed-stability analysis via pairwise Adjusted Rand Index across
  multiple random starts.
- `kneed`-library elbow detection (with deterministic 2nd-derivative fallback
  if `kneed` is not installed).
- Joint elbow + silhouette sweep persisted to `outputs/model_selection.png`.
- DBSCAN now returns noise fraction, noise-warn flag, and hotspot-size
  distribution (min / median / max).

Visual diagnostics (static PNGs under `outputs/`):

- `silhouette_plot.png`, `pca_projection.png`, `cluster_profile.png`,
  `monthly_trend.png`, `zone_hour_heatmap.png`,
  `speed_deviation_histogram.png`, `feature_distributions.png`,
  `model_selection.png` — on top of the original elbow / top-zones /
  density-deviation / day-hour-heatmap set.

---

## Limitations

- K-Means assumes spherical clusters; real congestion patterns may be
  irregularly shaped.
- Zone-hour aggregation may obscure short, localised congestion events.
- Unsupervised learning identifies *where* and *when*, not *why* — no causal
  inference is attempted.
- The synthetic centroid fallback is for demo only — use
  `scripts/make_centroids.py` for real maps.
