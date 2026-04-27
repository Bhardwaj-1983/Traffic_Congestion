"""
main.py
-------
Offline-pipeline orchestrator for the Traffic Congestion project.

Runs the full batch pipeline in order:

    1. data_loader          load + standardise raw parquet files
    2. preprocessing        clean + filter records
    3. feature_engineering  derive features + aggregate to zone-hour
    4. clustering           StandardScaler + K-Means (+ optional DBSCAN)
    5. visualization        pre-compute viz parquet + static charts

Examples
--------
    # Run everything (recommended on a fresh checkout)
    python main.py

    # Skip stages you've already cached
    python main.py --skip-load --skip-preprocess

    # Force a specific k and also run DBSCAN for hotspot detection
    python main.py --k 4 --dbscan

    # Run just the app-facing stages (useful after model tweaking)
    python main.py --only-viz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make this script importable from repo root regardless of CWD
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_directories  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("main")


# ── Stage runners ─────────────────────────────────────────────────────────────

def _run_stage(name: str, fn, *args, **kwargs) -> None:
    logger.info("")
    logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"  ▶ {name}")
    logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    logger.info(f"  ✓ {name} done in {dt:.2f}s")


def run_pipeline(
    skip_load: bool = False,
    skip_preprocess: bool = False,
    skip_features: bool = False,
    skip_cluster: bool = False,
    skip_viz: bool = False,
    only_viz: bool = False,
    k: int | None = None,
    use_dbscan: bool = False,
) -> None:
    """Execute the pipeline, respecting skip flags."""
    ensure_directories()

    if only_viz:
        skip_load = skip_preprocess = skip_features = skip_cluster = True
        skip_viz = False

    # Lazy imports keep `python main.py --help` snappy.
    if not skip_load:
        from src import data_loader
        _run_stage("Stage 1 · Data Loading", data_loader.main)

    if not skip_preprocess:
        from src import preprocessing
        _run_stage("Stage 2 · Preprocessing", preprocessing.main)

    if not skip_features:
        from src import feature_engineering
        _run_stage("Stage 3 · Feature Engineering", feature_engineering.main)

    if not skip_cluster:
        from src import clustering
        _run_stage(
            "Stage 4 · Clustering",
            clustering.run_clustering_pipeline,
            k=k, use_dbscan=use_dbscan,
        )

    if not skip_viz:
        from src import visualization
        _run_stage("Stage 5 · Visualization", visualization.main)

    logger.info("")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  ✓ Pipeline complete.")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  Next: run the dashboard with:")
    logger.info("      streamlit run app/app.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the offline batch pipeline for Traffic Congestion "
            "Pattern Discovery and Visualization."
        )
    )
    p.add_argument("--skip-load", action="store_true",
                   help="Skip Stage 1 (data loading).")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip Stage 2 (cleaning).")
    p.add_argument("--skip-features", action="store_true",
                   help="Skip Stage 3 (feature engineering).")
    p.add_argument("--skip-cluster", action="store_true",
                   help="Skip Stage 4 (clustering).")
    p.add_argument("--skip-viz", action="store_true",
                   help="Skip Stage 5 (viz precomputation).")
    p.add_argument("--only-viz", action="store_true",
                   help="Run only Stage 5 (skip everything else).")
    p.add_argument("--k", type=int, default=None,
                   help="Override K-Means k (skips the elbow method).")
    p.add_argument("--dbscan", action="store_true",
                   help="Also run DBSCAN for hotspot detection.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        run_pipeline(
            skip_load=args.skip_load,
            skip_preprocess=args.skip_preprocess,
            skip_features=args.skip_features,
            skip_cluster=args.skip_cluster,
            skip_viz=args.skip_viz,
            only_viz=args.only_viz,
            k=args.k,
            use_dbscan=args.dbscan,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception:
        logger.exception("Pipeline crashed unexpectedly")
        sys.exit(2)


if __name__ == "__main__":
    main()
