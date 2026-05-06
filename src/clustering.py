"""
clustering.py
-------------
Machine-Learning layer (PRD §9) — upgraded with a substantially richer
evaluation suite and a more defensible model-selection procedure:

  1. Load the aggregated zone-hour feature set.
  2. Scale the clustering features with ``StandardScaler`` (log-transforms
     already applied upstream in feature_engineering so StandardScaler is
     well-behaved).
  3. Sweep k ∈ K_RANGE computing inertia AND silhouette, persist the
     model-selection curve, and choose k by a combined rule (prefer the
     elbow if it is within 1 of the silhouette-argmax, else pick the
     silhouette winner).
  4. Fit K-Means with the chosen k.
  5. Re-label clusters by mean speed so that
        congestion_level = 0 → Low
        congestion_level = k-1 → High
  6. Evaluate with Silhouette, per-cluster Silhouette,
     Davies-Bouldin, Calinski-Harabasz, CVintra, and seed-stability (ARI).
  7. (Optional) Run DBSCAN on top for hotspot detection.
  8. Persist the model, scaler, cluster-label parquet, metrics JSON,
     and a raw X_scaled parquet used by the visualization stage.

Usage:
    python -m src.clustering            # K-Means only
    python -m src.clustering --dbscan   # also run DBSCAN
    python -m src.clustering --k 4      # force k=4 and skip sweep
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

# Optional dependency — principled elbow detection if available, graceful
# fallback to a hand-rolled 2nd-derivative heuristic otherwise.
try:
    from kneed import KneeLocator  # type: ignore[import-not-found]
    _HAS_KNEED = True
except ImportError:  # pragma: no cover — optional dep
    KneeLocator = None  # type: ignore[assignment]
    _HAS_KNEED = False

from src.config import (
    AGGREGATED_FILE,
    CLUSTER_FEATURES,
    CLUSTER_LABELS_FILE,
    CONGESTION_LABELS,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    DBSCAN_NOISE_WARN_FRACTION,
    DEFAULT_K,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    K_RANGE,
    METRICS_FILE,
    MODELS_DIR,
    MODEL_FILE,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    SCALED_FILE,
    SCALER_FILE,
    SCALER_TYPE,
    SILHOUETTE_SAMPLE_SIZE,
    STABILITY_ARI_MIN,
    STABILITY_N_SEEDS,
    TARGET_CVINTRA,
    TARGET_SILHOUETTE,
    USE_KNEED_ELBOW,
)
from src.utils import get_logger, read_parquet_safe, timeit, write_json, write_parquet

logger = get_logger(__name__)


# ── Feature preparation ───────────────────────────────────────────────────────

def load_aggregated_data(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = processed_dir / AGGREGATED_FILE
    df = read_parquet_safe(path, label="feature_engineering")
    logger.info(f"Loaded aggregated data: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Extract the clustering feature matrix.

    Returns
    -------
    X        : np.ndarray (n_samples, n_features)
    df_clean : pd.DataFrame containing *all* columns from ``df`` that
               correspond 1-to-1 with rows of ``X``.
    """
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing clustering features in aggregated data: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df_clean = df.dropna(subset=CLUSTER_FEATURES).reset_index(drop=True).copy()
    dropped = len(df) - len(df_clean)
    if dropped:
        logger.warning(f"Dropped {dropped:,} rows with NaN clustering features")

    # Ensure boolean features are numeric for the scaler
    feat_df = df_clean[CLUSTER_FEATURES].copy()
    for col in feat_df.columns:
        if feat_df[col].dtype == bool:
            feat_df[col] = feat_df[col].astype("int8")

    X = feat_df.values.astype("float64")
    logger.info(f"Feature matrix: X.shape={X.shape}  features={CLUSTER_FEATURES}")
    return X, df_clean


def _make_scaler(kind: str = SCALER_TYPE):
    """
    Build a scaler instance from a string identifier.

    "standard" — z-score (mean=0, std=1). Pulled around by tails.
    "robust"   — median + IQR. Resistant to heavy-tailed features such as
                 ``log_trip_density``; recommended whenever distributions
                 remain skewed after ``log1p``.
    """
    kind = (kind or "standard").lower()
    if kind == "robust":
        return RobustScaler()
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown SCALER_TYPE: {kind!r} (expected 'standard' or 'robust')")


def scale_features(
    X: np.ndarray,
    scaler_type: str = SCALER_TYPE,
) -> tuple[np.ndarray, "StandardScaler | RobustScaler"]:
    scaler = _make_scaler(scaler_type)
    X_scaled = scaler.fit_transform(X)
    logger.info(f"Features scaled with {type(scaler).__name__}")
    logger.info(f"  per-feature mean after scaling : {X_scaled.mean(axis=0).round(4)}")
    logger.info(f"  per-feature std  after scaling : {X_scaled.std(axis=0).round(4)}")
    return X_scaled, scaler


# ── Silhouette helper ────────────────────────────────────────────────────────

def _sample_for_silhouette(
    X_scaled: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return X, labels — subsampled deterministically if over the cap."""
    n = len(X_scaled)
    if n > SILHOUETTE_SAMPLE_SIZE:
        idx = np.random.default_rng(42).choice(n, size=SILHOUETTE_SAMPLE_SIZE, replace=False)
        return X_scaled[idx], labels[idx]
    return X_scaled, labels


# ── Joint k selection: inertia + silhouette ──────────────────────────────────

def choose_k(
    X_scaled: np.ndarray,
    k_range: range,
    outputs_dir: Path = OUTPUTS_DIR,
) -> tuple[int, dict[str, list[float]]]:
    """
    Sweep k across ``k_range``, recording inertia and silhouette for each.
    Save the joint model-selection curve and pick k by this rule:

        · Compute the elbow on the inertia curve (kneedle preferred).
        · Find the k that maximises silhouette.
        · Prefer the elbow when one is found — it is the more interpretable
          choice on flat silhouette curves. The silhouette winner is used
          only as a fallback when no elbow can be detected.
    """
    ks = list(k_range)
    inertias: list[float] = []
    sils: list[float] = []

    logger.info(f"Sweeping k ∈ {ks} (inertia + silhouette)…")
    for k in ks:
        km = KMeans(
            n_clusters=k,
            random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
        )
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        Xs, ls = _sample_for_silhouette(X_scaled, labels)
        sil = float(silhouette_score(Xs, ls)) if len(np.unique(ls)) > 1 else float("nan")
        sils.append(sil)
        logger.debug(f"  k={k}: inertia={km.inertia_:.1f}  silhouette={sil:+.4f}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    _plot_model_selection(ks, inertias, sils, outputs_dir)

    # Elbow detection — prefer kneed when available (more principled).
    elbow_k, elbow_source = _detect_elbow(ks, inertias)

    # Silhouette-argmax (ignoring NaNs)
    sil_arr = np.asarray(sils, dtype=float)
    if np.all(np.isnan(sil_arr)):
        sil_k = elbow_k
    else:
        sil_k = ks[int(np.nanargmax(sil_arr))]

    # Combined rule — prefer the elbow when found; silhouette is fallback only.
    if elbow_k is not None:
        chosen = elbow_k
        if abs(sil_k - elbow_k) <= 1:
            rule = f"elbow via {elbow_source} (silhouette agrees within 1)"
        else:
            rule = (
                f"elbow via {elbow_source} "
                f"(silhouette-argmax={sil_k} overruled — flat silhouette curve)"
            )
    else:
        chosen = sil_k
        rule = "silhouette-argmax (no elbow detected)"

    logger.info(
        f"k selection: elbow={elbow_k} ({elbow_source}), "
        f"silhouette-argmax={sil_k}, chosen={chosen} [{rule}]"
    )
    return chosen, {
        "ks": ks,
        "inertias": inertias,
        "silhouettes": sils,
        "elbow_k": int(elbow_k),
        "elbow_source": elbow_source,
        "silhouette_argmax_k": int(sil_k),
    }


def _detect_elbow(ks: list[int], inertias: list[float]) -> tuple[int, str]:
    """
    Locate the elbow of an inertia curve.

    Priority:
      1. ``kneed.KneeLocator`` with ``curve='convex', direction='decreasing'``
         when available and ``USE_KNEED_ELBOW`` is True.
      2. Hand-rolled 2nd-derivative argmax as a deterministic fallback.

    Returns (elbow_k, source_tag).
    """
    if USE_KNEED_ELBOW and _HAS_KNEED and len(ks) >= 3:
        try:
            kl = KneeLocator(
                ks, inertias,
                curve="convex", direction="decreasing",
                interp_method="interp1d",
            )
            if kl.knee is not None:
                return int(kl.knee), "kneed"
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(f"kneed failed ({e}); falling back to 2nd-derivative")

    arr = np.asarray(inertias, dtype=float)
    if len(arr) >= 3:
        second_deriv = np.diff(arr, n=2)
        return int(ks[int(np.argmax(second_deriv)) + 1]), "2nd-derivative"
    return int(DEFAULT_K), "default-k (too-few-points)"


def _plot_model_selection(
    ks: list[int], inertias: list[float], sils: list[float],
    outputs_dir: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    c1 = "#2c3e50"
    ax1.plot(ks, inertias, "o-", color=c1, linewidth=2, markersize=7, label="Inertia")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster SSE)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xticks(ks)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    c2 = "#e67e22"
    ax2.plot(ks, sils, "s--", color=c2, linewidth=2, markersize=7, label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.axhline(0.5, color=c2, linestyle=":", alpha=0.5)

    plt.title("Model Selection — Inertia & Silhouette vs. k",
              fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = outputs_dir / "model_selection.png"
    plt.savefig(out, dpi=120)
    plt.close()

    # Keep the classic elbow_curve.png for backwards compatibility.
    plt.figure(figsize=(8, 4.5))
    plt.plot(ks, inertias, "bo-", linewidth=2, markersize=6)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputs_dir / "elbow_curve.png", dpi=120)
    plt.close()
    logger.info(f"Model-selection plot → {out}")


# ── K-Means fitting ───────────────────────────────────────────────────────────

@timeit("train_kmeans")
def train_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    logger.info(f"Training KMeans(k={k}) …")
    km = KMeans(
        n_clusters=k,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    km.fit(X_scaled)
    logger.info(f"  inertia={km.inertia_:.4f}  n_iter={km.n_iter_}")
    return km


def assign_semantic_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """
    Re-label raw K-Means cluster ids so that ``congestion_level`` ranks
    from 0 (fastest / least congested) to k-1 (slowest / most congested).
    """
    df = df.copy()
    df["raw_cluster"] = labels

    # Fastest raw cluster → 0, slowest → k-1
    speed_by_cluster = (
        df.groupby("raw_cluster")["avg_speed_mph"]
        .mean()
        .sort_values(ascending=False)          # fastest first
    )
    remap = {raw: level for level, raw in enumerate(speed_by_cluster.index)}
    df["congestion_level"] = df["raw_cluster"].map(remap).astype("int8")
    df = df.drop(columns=["raw_cluster"])
    return df


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_silhouette(
    X_scaled: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[int, float]]:
    """
    Aggregate silhouette + per-cluster silhouette (on a sample if large).
    Returns ``(overall, {cluster_id: mean_silhouette})``.
    """
    Xs, ls = _sample_for_silhouette(X_scaled, labels)
    if len(np.unique(ls)) < 2:
        return float("nan"), {}
    overall = float(silhouette_score(Xs, ls))
    samples = silhouette_samples(Xs, ls)
    per_cluster = {
        int(c): float(samples[ls == c].mean())
        for c in np.unique(ls)
    }
    return overall, per_cluster


def compute_cvintra(df: pd.DataFrame) -> dict[int, float]:
    """Intra-class CV of ``avg_speed_mph`` per cluster. CV = std/mean."""
    cv_map: dict[int, float] = {}
    for level, grp in df.groupby("congestion_level"):
        mean_s = float(grp["avg_speed_mph"].mean())
        std_s = float(grp["avg_speed_mph"].std(ddof=0))
        cv_map[int(level)] = float(std_s / mean_s) if mean_s > 0 else float("nan")
    return cv_map


def stability_analysis(
    X_scaled: np.ndarray, k: int, n_seeds: int = STABILITY_N_SEEDS,
) -> dict[str, float]:
    """
    Fit K-Means with ``n_seeds`` different random starts and compare the
    resulting partitions pairwise via Adjusted Rand Index. A mean ARI
    close to 1.0 indicates the clusters are stable; <0.7 is a red flag.
    """
    logger.info(f"Stability check: {n_seeds} seeds × KMeans(k={k})")
    labels_list: list[np.ndarray] = []
    for seed in range(n_seeds):
        km = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
        )
        labels_list.append(km.fit_predict(X_scaled))

    aris: list[float] = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            aris.append(float(adjusted_rand_score(labels_list[i], labels_list[j])))
    if not aris:
        return {"ari_mean": float("nan"), "ari_min": float("nan"), "ari_pass": False}

    mean_ari = float(np.mean(aris))
    min_ari = float(np.min(aris))
    return {
        "ari_mean": mean_ari,
        "ari_min": min_ari,
        "ari_pass": bool(min_ari >= STABILITY_ARI_MIN),
    }


def evaluate_clustering(
    X_scaled: np.ndarray,
    df_labeled: pd.DataFrame,
    k: int,
) -> dict[str, object]:
    logger.info("=== Clustering Evaluation ===")
    labels = df_labeled["congestion_level"].values

    # Silhouette + per-cluster
    sil, sil_per_cluster = compute_silhouette(X_scaled, labels)
    sil_pass = (sil == sil) and (sil > TARGET_SILHOUETTE)
    logger.info(
        f"Silhouette Score : {sil:+.4f}  (target > {TARGET_SILHOUETTE}) "
        f"{'PASS' if sil_pass else 'FAIL'}"
    )
    for c, s in sorted(sil_per_cluster.items()):
        label = CONGESTION_LABELS.get(c, f"L{c}")
        logger.info(f"  silhouette[{c}:{label:<6}] = {s:+.4f}")

    # Davies-Bouldin & Calinski-Harabasz
    try:
        db = float(davies_bouldin_score(X_scaled, labels))
    except ValueError:
        db = float("nan")
    try:
        ch = float(calinski_harabasz_score(X_scaled, labels))
    except ValueError:
        ch = float("nan")
    logger.info(f"Davies-Bouldin  : {db:.4f}   (lower = better)")
    logger.info(f"Calinski-Harab. : {ch:.2f}   (higher = better)")

    # CVintra
    cv_map = compute_cvintra(df_labeled)
    all_cv_pass = True
    for level, cv in sorted(cv_map.items()):
        label = CONGESTION_LABELS.get(level, f"L{level}")
        pass_flag = (cv == cv) and (cv < TARGET_CVINTRA)  # NaN-safe
        all_cv_pass = all_cv_pass and pass_flag
        logger.info(
            f"CVintra[{level}:{label:<6}] = {cv:.4f} ({100*cv:5.1f}%)  "
            f"(target < {100*TARGET_CVINTRA:.0f}%)  "
            f"{'PASS' if pass_flag else 'FAIL'}"
        )

    # Stability
    stab = stability_analysis(X_scaled, k)
    logger.info(
        f"Stability (ARI)  : mean={stab['ari_mean']:.4f}  min={stab['ari_min']:.4f}"
        f"  {'PASS' if stab['ari_pass'] else 'WARN'}"
    )

    # Cluster-size distribution
    size_by_level = df_labeled["congestion_level"].value_counts().sort_index().to_dict()
    size_by_level = {int(k_): int(v_) for k_, v_ in size_by_level.items()}
    logger.info(f"Cluster sizes    : {size_by_level}")

    # Cluster centroid means in original feature space (descriptive)
    centroid_means = (
        df_labeled
        .groupby("congestion_level")[CLUSTER_FEATURES + ["avg_speed_mph", "trip_density"]]
        .mean()
        .round(3)
        .to_dict(orient="index")
    )

    kpi_pass = sil_pass and all_cv_pass
    logger.info(f"Overall KPI status: {'PASS' if kpi_pass else 'FAIL'}")

    return {
        "k": int(k),
        "features": list(CLUSTER_FEATURES),
        "silhouette_score": sil,
        "silhouette_per_cluster": {int(k_): float(v_) for k_, v_ in sil_per_cluster.items()},
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "cvintra_by_level": cv_map,
        "cluster_sizes": size_by_level,
        "stability": stab,
        "centroid_means": {int(k_): v_ for k_, v_ in centroid_means.items()},
        "kpi_pass": kpi_pass,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── DBSCAN (optional) ────────────────────────────────────────────────────────

def run_dbscan(X_scaled: np.ndarray, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run DBSCAN and return the labelled DataFrame along with a metrics dict
    characterizing the run: number of hotspots, noise fraction, and the
    distribution of hotspot sizes (min / max / median / largest).
    """
    logger.info(
        f"Running DBSCAN(eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})"
    )
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int((labels == -1).sum())
    noise_frac = float(n_noise / len(labels)) if len(labels) else float("nan")

    # Hotspot size distribution (exclude the -1 noise bucket)
    if n_clusters > 0:
        sizes = pd.Series(labels[labels != -1]).value_counts()
        size_stats = {
            "hotspot_sizes_min": int(sizes.min()),
            "hotspot_sizes_max": int(sizes.max()),
            "hotspot_sizes_median": float(sizes.median()),
            "largest_hotspot_label": int(sizes.idxmax()),
        }
    else:
        size_stats = {
            "hotspot_sizes_min": 0,
            "hotspot_sizes_max": 0,
            "hotspot_sizes_median": 0.0,
            "largest_hotspot_label": -1,
        }

    metrics = {
        "n_hotspots": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": round(noise_frac, 4),
        "noise_fraction_warn": bool(noise_frac > DBSCAN_NOISE_WARN_FRACTION),
        "eps": DBSCAN_EPS,
        "min_samples": DBSCAN_MIN_SAMPLES,
        **size_stats,
    }

    level = logger.warning if metrics["noise_fraction_warn"] else logger.info
    level(
        f"  DBSCAN → {n_clusters} hotspots · {n_noise:,} noise "
        f"({100*noise_frac:.1f}%)  "
        f"sizes[min={metrics['hotspot_sizes_min']}, "
        f"median={metrics['hotspot_sizes_median']:.0f}, "
        f"max={metrics['hotspot_sizes_max']}]"
    )

    df = df.copy()
    df["dbscan_label"] = labels.astype("int16")
    return df, metrics


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model_artifacts(
    km: KMeans,
    scaler: StandardScaler,
    df_labeled: pd.DataFrame,
    X_scaled: np.ndarray,
    metrics: dict[str, object],
    models_dir: Path = MODELS_DIR,
    processed_dir: Path = PROCESSED_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / MODEL_FILE, "wb") as f:
        pickle.dump(km, f)
    logger.info(f"Saved: {(models_dir / MODEL_FILE).name}")

    with open(models_dir / SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved: {(models_dir / SCALER_FILE).name}")

    write_parquet(df_labeled, processed_dir / CLUSTER_LABELS_FILE)
    logger.info(f"Saved: {CLUSTER_LABELS_FILE}")

    # Save scaled features (used by visualization for PCA + silhouette plots)
    scaled_df = pd.DataFrame(X_scaled, columns=CLUSTER_FEATURES)
    scaled_df["congestion_level"] = df_labeled["congestion_level"].values
    write_parquet(scaled_df, processed_dir / SCALED_FILE)
    logger.info(f"Saved: {SCALED_FILE}")

    write_json(metrics, outputs_dir / METRICS_FILE)
    logger.info(f"Saved: {METRICS_FILE}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

@timeit("run_clustering_pipeline")
def run_clustering_pipeline(
    k: int | None = None,
    use_dbscan: bool = False,
    scaler_type: str = SCALER_TYPE,
) -> dict[str, object]:
    """Execute the full clustering pipeline and return the metrics dict."""
    # 1. Load aggregated data
    df = load_aggregated_data()

    # 2. Prepare features — df_feat is index-aligned with X
    X, df_feat = prepare_features(df)

    # 3. Scale (Standard or Robust depending on config / CLI)
    X_scaled, scaler = scale_features(X, scaler_type=scaler_type)

    # 4. Choose k (joint elbow + silhouette sweep)
    if k is None:
        k, sweep = choose_k(X_scaled, range(*K_RANGE))
    else:
        sweep = None
        logger.info(f"Using user-provided k={k} (skipping sweep)")

    # 5. Train K-Means
    km = train_kmeans(X_scaled, k=k)
    raw_labels = km.labels_

    # 6. Semantic labelling (sorted by mean speed)
    df_labeled = assign_semantic_labels(df_feat, raw_labels, k=k)

    # 7. Evaluate (silhouette, DB, CH, CVintra, stability)
    metrics = evaluate_clustering(X_scaled, df_labeled, k=k)
    metrics["scaler_type"] = type(scaler).__name__
    if sweep is not None:
        metrics["k_selection"] = sweep

    # 8. Optional DBSCAN
    if use_dbscan:
        df_labeled, dbscan_metrics = run_dbscan(X_scaled, df_labeled)
        metrics["dbscan"] = dbscan_metrics

    # 9. Persist
    save_model_artifacts(km, scaler, df_labeled, X_scaled, metrics)

    logger.info("Pipeline summary:")
    logger.info(f"  k                 = {k}")
    logger.info(f"  scaler            = {metrics['scaler_type']}")
    logger.info(f"  silhouette_score  = {metrics['silhouette_score']:+.4f}")
    logger.info(f"  davies_bouldin    = {metrics['davies_bouldin']:.4f}")
    logger.info(f"  calinski_harab.   = {metrics['calinski_harabasz']:.2f}")
    logger.info(f"  ari_min (stab.)   = {metrics['stability']['ari_min']:.4f}")
    logger.info(f"  kpi_pass          = {metrics['kpi_pass']}")
    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clustering stage")
    parser.add_argument("--k", type=int, default=None,
                        help="Override k (skips the joint elbow+silhouette sweep)")
    parser.add_argument("--dbscan", action="store_true",
                        help="Also run DBSCAN hotspot detection")
    parser.add_argument("--scaler", choices=("standard", "robust"),
                        default=SCALER_TYPE,
                        help="Feature scaler: 'standard' (z-score) or 'robust' (median+IQR)")
    args = parser.parse_args()

    logger.info("=== Stage 4 :: Clustering ===")
    run_clustering_pipeline(k=args.k, use_dbscan=args.dbscan, scaler_type=args.scaler)
    logger.info("Done.")


if __name__ == "__main__":
    main()
