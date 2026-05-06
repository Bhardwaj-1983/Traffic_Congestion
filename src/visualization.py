"""
visualization.py
----------------
Pre-computes the visualization-ready parquet consumed by the Streamlit
app and renders a rich set of static "offline" charts that live under
``outputs/`` for inclusion in reports.

Inputs:
    data/processed/cluster_labels.parquet   (output of clustering.py)
    data/processed/scaled_features.parquet  (output of clustering.py)
    data/processed/features_trips_*.parquet (output of feature_engineering.py)

Outputs (PNG in ``outputs/``):
    1. speed_density_by_hour.png        — legacy dual-axis hourly trend
    2. cluster_distribution.png         — legacy bar of cluster sizes
    3. congestion_heatmap.png           — day × hour mean congestion
    4. top_congested_zones.png          — top-N zone bar chart
    5. speed_deviation_scatter.png      — density vs. deviation scatter
    6. silhouette_plot.png              — NEW: per-sample silhouette
    7. pca_projection.png               — NEW: 2D PCA of clusters
    8. cluster_profile.png              — NEW: parallel coords of centroids
    9. monthly_trend.png                — NEW: speed + density by month
   10. zone_hour_heatmap.png            — NEW: zone × hour congestion
   11. speed_deviation_histogram.png    — NEW: distribution of the core signal
   12. feature_distributions.png        — NEW: histograms of each cluster feature

Usage:
    python -m src.visualization
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from src.config import (
    CLUSTER_FEATURES,
    CLUSTER_LABELS_FILE,
    CONGESTION_HEX_COLORS,
    CONGESTION_LABELS,
    DAY_NAMES_SHORT,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    SCALED_FILE,
    SILHOUETTE_SAMPLE_SIZE,
    VIZ_DATA_FILE,
)
from src.utils import get_logger, read_parquet_safe, timeit, write_parquet

logger = get_logger(__name__)

_DEFAULT_HEX = "#7f8c8d"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cluster_labels(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = processed_dir / CLUSTER_LABELS_FILE
    df = read_parquet_safe(path, label="clustering")
    logger.info(f"Loaded cluster labels: {df.shape}")
    return df


def load_scaled_features(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame | None:
    path = processed_dir / SCALED_FILE
    if not path.exists():
        logger.warning(f"Scaled features not found at {path} — skipping PCA/silhouette plots")
        return None
    return pd.read_parquet(path)


# ── Pre-computed viz parquet ──────────────────────────────────────────────────

def build_viz_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cluster_labels.parquet to one row per
    (zone_id, hour_of_day, day_of_week, is_weekend).

    Adds congestion_label, color_hex, day_name for convenience, plus
    pass-through of is_airport / is_cbd / is_rush_hour when present.
    """
    group_cols = ["zone_id", "hour_of_day", "day_of_week", "is_weekend"]

    if "is_weekend" not in df.columns and "day_of_week" in df.columns:
        df = df.copy()
        df["is_weekend"] = df["day_of_week"] >= 5

    for col in group_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' missing from cluster labels")

    def _mode_int(s: pd.Series) -> int:
        """Robust majority-vote that works even with a single value."""
        m = s.mode()
        return int(m.iloc[0]) if not m.empty else int(s.iloc[0])

    # Build the base aggregation
    agg_map: dict[str, tuple] = {
        "congestion_level": ("congestion_level", _mode_int),
        "avg_speed_mph": ("avg_speed_mph", "mean"),
        "trip_density": ("trip_density", "sum"),
        "speed_deviation": ("speed_deviation", "mean"),
    }
    for col in ("is_airport", "is_cbd", "is_rush_hour"):
        if col in df.columns:
            agg_map[col] = (col, "first")

    viz = (
        df.groupby(group_cols, observed=True)
        .agg(**agg_map)
        .reset_index()
    )

    viz["congestion_label"] = viz["congestion_level"].map(
        lambda v: CONGESTION_LABELS.get(int(v), f"L{int(v)}")
    )
    viz["color_hex"] = viz["congestion_level"].map(
        lambda v: CONGESTION_HEX_COLORS.get(int(v), _DEFAULT_HEX)
    )
    viz["day_name"] = viz["day_of_week"].map(
        lambda d: DAY_NAMES_SHORT[int(d)] if 0 <= int(d) < 7 else str(d)
    )

    logger.info(f"Viz data shape: {viz.shape}")
    return viz


def save_viz_data(viz: pd.DataFrame, processed_dir: Path = PROCESSED_DIR) -> None:
    write_parquet(viz, processed_dir / VIZ_DATA_FILE)
    logger.info(f"Saved: {VIZ_DATA_FILE}")


# ── Shared style ──────────────────────────────────────────────────────────────

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    })


# ── Legacy charts (kept) ──────────────────────────────────────────────────────

def plot_speed_density_by_hour(viz: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    hourly = (
        viz.groupby("hour_of_day")
        .agg(avg_speed=("avg_speed_mph", "mean"),
             avg_density=("trip_density", "mean"))
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    c_speed = "#2980b9"
    ax1.plot(hourly["hour_of_day"], hourly["avg_speed"],
             color=c_speed, linewidth=2.5, marker="o", markersize=4)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Avg Speed (mph)", color=c_speed)
    ax1.tick_params(axis="y", labelcolor=c_speed)
    ax1.set_xticks(range(24))

    ax2 = ax1.twinx()
    c_density = "#e67e22"
    ax2.fill_between(hourly["hour_of_day"], hourly["avg_density"],
                     alpha=0.25, color=c_density)
    ax2.set_ylabel("Avg Trip Density", color=c_density)
    ax2.tick_params(axis="y", labelcolor=c_density)

    plt.title("Average Speed & Trip Density by Hour of Day",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = outputs_dir / "speed_density_by_hour.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_cluster_distribution(viz: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    order = sorted(viz["congestion_level"].unique())
    labels = [CONGESTION_LABELS.get(i, f"L{i}") for i in order]
    colors = [CONGESTION_HEX_COLORS.get(i, _DEFAULT_HEX) for i in order]
    counts = viz["congestion_level"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.01,
                f"{int(val):,}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Congestion Level")
    ax.set_ylabel("Zone-Hour Observations")
    ax.set_title("Distribution of Congestion Clusters",
                 fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    out = outputs_dir / "cluster_distribution.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_congestion_heatmap(viz: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    pivot = (
        viz.groupby(["day_of_week", "hour_of_day"])["congestion_level"]
        .mean()
        .unstack(level="hour_of_day")
    )
    pivot.index = [DAY_NAMES_SHORT[int(i)] for i in pivot.index]

    k_levels = int(viz["congestion_level"].max())
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot,
        cmap="RdYlGn_r",
        vmin=0, vmax=max(2, k_levels),
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Mean Congestion Level"},
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title("Congestion Heatmap: Day × Hour",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = outputs_dir / "congestion_heatmap.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_top_congested_zones(
    viz: pd.DataFrame, top_n: int = 20, outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    zone_cong = (
        viz.groupby("zone_id")["congestion_level"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    colors = [
        CONGESTION_HEX_COLORS.get(min(2, int(round(v))), _DEFAULT_HEX)
        for v in zone_cong["congestion_level"]
    ]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(zone_cong["zone_id"].astype(str),
            zone_cong["congestion_level"],
            color=colors, edgecolor="white")
    ax.set_xlabel("Mean Congestion Level")
    ax.set_ylabel("Zone ID")
    ax.set_title(f"Top {top_n} Most Congested Zones",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    out = outputs_dir / "top_congested_zones.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_speed_deviation_scatter(
    viz: pd.DataFrame, outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """Scatter: trip_density vs speed_deviation, coloured by congestion level."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    fig, ax = plt.subplots(figsize=(9, 6))
    for level in sorted(viz["congestion_level"].unique()):
        sub = viz[viz["congestion_level"] == level]
        color = CONGESTION_HEX_COLORS.get(int(level), _DEFAULT_HEX)
        label = CONGESTION_LABELS.get(int(level), f"L{level}")
        ax.scatter(
            sub["trip_density"].clip(0, np.quantile(viz["trip_density"], 0.99)),
            sub["speed_deviation"],
            alpha=0.35, s=10, color=color, label=label,
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Trip Density (clipped at p99)")
    ax.set_ylabel("Speed Deviation (mph)")
    ax.set_title(
        "Core Congestion Signal: Density vs Speed Deviation",
        fontsize=14, fontweight="bold",
    )
    ax.legend(title="Congestion")
    plt.tight_layout()
    out = outputs_dir / "speed_deviation_scatter.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


# ── NEW CHARTS ────────────────────────────────────────────────────────────────

def plot_silhouette(
    scaled_df: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """
    Per-sample silhouette plot. Each cluster is a colored band whose
    horizontal extent equals the sample's silhouette value; wider is
    better, values below zero indicate misclassified samples.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    feature_cols = [c for c in CLUSTER_FEATURES if c in scaled_df.columns]
    X = scaled_df[feature_cols].values
    labels = scaled_df["congestion_level"].values

    # Sub-sample for performance
    n = len(X)
    if n > SILHOUETTE_SAMPLE_SIZE:
        idx = np.random.default_rng(42).choice(n, size=SILHOUETTE_SAMPLE_SIZE, replace=False)
        X = X[idx]
        labels = labels[idx]

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning("Silhouette plot skipped — only one cluster present")
        return

    sample_sils = silhouette_samples(X, labels)
    avg_sil = float(sample_sils.mean())

    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10
    for level in sorted(unique_labels):
        vals = np.sort(sample_sils[labels == level])
        size = len(vals)
        y_upper = y_lower + size
        color = CONGESTION_HEX_COLORS.get(int(level), _DEFAULT_HEX)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=color, edgecolor=color, alpha=0.85)
        name = CONGESTION_LABELS.get(int(level), f"L{level}")
        ax.text(-0.05, y_lower + 0.5 * size, f"{name} ({level})",
                ha="right", va="center", fontsize=10)
        y_lower = y_upper + 10

    ax.axvline(avg_sil, color="red", linestyle="--",
               label=f"Mean silhouette = {avg_sil:+.3f}")
    ax.set_xlabel("Silhouette value")
    ax.set_yticks([])
    ax.set_title("Per-Sample Silhouette Plot",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = outputs_dir / "silhouette_plot.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_pca_projection(
    scaled_df: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """2D PCA projection of the scaled feature matrix, coloured by cluster."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    feature_cols = [c for c in CLUSTER_FEATURES if c in scaled_df.columns]
    X = scaled_df[feature_cols].values
    labels = scaled_df["congestion_level"].values

    # Sub-sample for visual clarity
    n = len(X)
    if n > 15_000:
        idx = np.random.default_rng(0).choice(n, size=15_000, replace=False)
        X, labels = X[idx], labels[idx]

    pca = PCA(n_components=2, random_state=0)
    pts = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for level in sorted(np.unique(labels)):
        mask = labels == level
        color = CONGESTION_HEX_COLORS.get(int(level), _DEFAULT_HEX)
        name = CONGESTION_LABELS.get(int(level), f"L{level}")
        ax.scatter(pts[mask, 0], pts[mask, 1], s=8, alpha=0.45,
                   color=color, label=f"{name} ({level})")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title("PCA Projection of Clusters",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Congestion", loc="best")
    plt.tight_layout()
    out = outputs_dir / "pca_projection.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}  (explained={explained.round(3).tolist()})")


def plot_cluster_profile(
    cluster_labels_df: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """
    Parallel-coordinates plot of each cluster's mean feature values
    (z-normalised for visual comparability). Shows what distinguishes
    one cluster from another in a single picture.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    profile_features = [
        f for f in ["avg_speed_mph", "speed_deviation", "log_trip_density",
                    "trip_density", "hour_of_day"]
        if f in cluster_labels_df.columns
    ]
    if not profile_features:
        logger.warning("cluster_profile: no features available")
        return

    means = (
        cluster_labels_df
        .groupby("congestion_level")[profile_features]
        .mean()
    )
    # z-normalise across clusters so each feature lives on the same axis
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(profile_features))
    for level, row in z.iterrows():
        color = CONGESTION_HEX_COLORS.get(int(level), _DEFAULT_HEX)
        name = CONGESTION_LABELS.get(int(level), f"L{level}")
        ax.plot(x, row.values, marker="o", linewidth=2.5,
                color=color, label=f"{name} ({level})")
    ax.set_xticks(x)
    ax.set_xticklabels(profile_features, rotation=15, ha="right")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("z-normalised mean")
    ax.set_title("Cluster Profile — Feature Signatures",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Congestion", loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = outputs_dir / "cluster_profile.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_monthly_trend(
    processed_dir: Path = PROCESSED_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """
    Avg speed and trip volume per month, computed from the per-month
    feature files. Shows seasonality and data-collection anomalies.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    files = sorted(glob.glob(str(processed_dir / "features_trips_*.parquet")))
    if not files:
        logger.warning("monthly_trend: no features_trips_*.parquet found — skipping")
        return

    rows: list[dict] = []
    for f in files:
        ym = Path(f).stem.replace("features_trips_", "")
        df = pd.read_parquet(f, columns=["speed_mph"])
        rows.append({
            "month": ym,
            "avg_speed": float(df["speed_mph"].mean()),
            "median_speed": float(df["speed_mph"].median()),
            "n_trips": int(len(df)),
        })
    trend = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    c1 = "#2980b9"
    ax1.plot(trend["month"], trend["avg_speed"], "o-",
             color=c1, linewidth=2.5, markersize=7, label="Avg speed")
    ax1.plot(trend["month"], trend["median_speed"], "s--",
             color=c1, linewidth=1.5, markersize=6, alpha=0.6, label="Median speed")
    ax1.set_ylabel("Speed (mph)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xlabel("Month")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    c2 = "#e67e22"
    ax2.bar(trend["month"], trend["n_trips"], alpha=0.25,
            color=c2, label="Trips")
    ax2.set_ylabel("Trips (count)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.title("Monthly Speed & Trip Volume",
              fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = outputs_dir / "monthly_trend.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_zone_hour_heatmap(
    viz: pd.DataFrame,
    top_n_zones: int = 40,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """
    Zone × hour heatmap of mean congestion level for the most-trafficked
    zones. Reveals which zone-hour cells drive the congested clusters.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # Keep only the top-N zones by total density to stay readable
    top_zones = (
        viz.groupby("zone_id")["trip_density"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_zones)
        .index
    )
    sub = viz[viz["zone_id"].isin(top_zones)]
    pivot = (
        sub.groupby(["zone_id", "hour_of_day"])["congestion_level"]
        .mean()
        .unstack(level="hour_of_day")
        .reindex(top_zones)
    )

    k_levels = int(viz["congestion_level"].max())
    fig, ax = plt.subplots(figsize=(14, max(6, top_n_zones * 0.22)))
    sns.heatmap(
        pivot, cmap="RdYlGn_r",
        vmin=0, vmax=max(2, k_levels),
        linewidths=0.2, ax=ax,
        cbar_kws={"label": "Mean Congestion Level"},
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Zone ID")
    ax.set_title(f"Zone × Hour Congestion — Top {top_n_zones} Busiest Zones",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = outputs_dir / "zone_hour_heatmap.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_speed_deviation_histogram(
    viz: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """Histogram of speed_deviation, coloured by congestion level."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(
        viz["speed_deviation"].quantile(0.005),
        viz["speed_deviation"].quantile(0.995),
        60,
    )
    for level in sorted(viz["congestion_level"].unique()):
        sub = viz[viz["congestion_level"] == level]
        color = CONGESTION_HEX_COLORS.get(int(level), _DEFAULT_HEX)
        name = CONGESTION_LABELS.get(int(level), f"L{level}")
        ax.hist(sub["speed_deviation"], bins=bins, alpha=0.55,
                color=color, label=f"{name} ({level})", edgecolor="white")

    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Speed Deviation (mph)")
    ax.set_ylabel("Zone-Hour Count")
    ax.set_title("Distribution of Speed Deviation — Core Congestion Signal",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Congestion")
    plt.tight_layout()
    out = outputs_dir / "speed_deviation_histogram.png"
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"Saved chart: {out.name}")


def plot_feature_distributions(
    cluster_labels_df: pd.DataFrame,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    """Grid of feature histograms — one per clustering feature."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    feats = [f for f in CLUSTER_FEATURES if f in cluster_labels_df.columns]
    if not feats:
        logger.warning("feature_distributions: no cluster features present — skipping")
        return

    n = len(feats)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(feats):
        ax = axes[i]
        for level in sorted(cluster_labels_df["congestion_level"].unique()):
            sub = cluster_labels_df[cluster_labels_df["congestion_level"] == level]
            color = CONGESTION_HEX_COLORS.get(int(level), _DEFAULT_HEX)
            vals = sub[feat].dropna().astype(float)
            if vals.empty:
                continue
            ax.hist(vals, bins=40, alpha=0.55,
                    color=color, edgecolor="white", density=True)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions by Cluster",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = outputs_dir / "feature_distributions.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved chart: {out.name}")


# ── Pipeline entry point ──────────────────────────────────────────────────────

@timeit("run_visualization_pipeline")
def run_visualization_pipeline() -> pd.DataFrame:
    df = load_cluster_labels()
    viz = build_viz_data(df)
    save_viz_data(viz)

    logger.info("Generating static charts…")

    # Legacy / core charts
    plot_speed_density_by_hour(viz)
    plot_cluster_distribution(viz)
    plot_congestion_heatmap(viz)
    plot_top_congested_zones(viz)
    plot_speed_deviation_scatter(viz)

    # New charts
    plot_cluster_profile(df)
    plot_monthly_trend()
    plot_zone_hour_heatmap(viz)
    plot_speed_deviation_histogram(viz)
    plot_feature_distributions(df)

    # PCA and silhouette plots need the scaled features matrix
    scaled = load_scaled_features()
    if scaled is not None:
        plot_silhouette(scaled)
        plot_pca_projection(scaled)

    logger.info("All static charts saved.")
    return viz


def main() -> None:
    logger.info("=== Stage 5 :: Visualization Precomputation ===")
    run_visualization_pipeline()
    logger.info("Done.")


if __name__ == "__main__":
    main()
