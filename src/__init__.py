"""
Traffic Congestion Pattern Discovery & Visualization — source package.

Modules:
    config              — centralized constants, paths, and thresholds.
    utils               — shared logger, decorators, and path helpers.
    data_loader         — load raw NYC TLC parquet files.
    preprocessing       — data-quality filtering and cleaning.
    feature_engineering — derive time/spatial/congestion features and aggregate.
    clustering          — StandardScaler + K-Means (+ optional DBSCAN) + evaluation.
    visualization       — precompute viz data and render static charts.
"""

__version__ = "1.0.0"
