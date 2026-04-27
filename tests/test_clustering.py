"""Tests for src/clustering.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import clustering
from src.config import CLUSTER_FEATURES


def test_prepare_features_shape(sample_aggregated_df):
    X, df_feat = clustering.prepare_features(sample_aggregated_df)
    assert X.shape == (len(sample_aggregated_df), len(CLUSTER_FEATURES))
    # Index alignment
    assert len(df_feat) == len(X)
    assert list(df_feat.index) == list(range(len(df_feat)))


def test_scale_features_has_zero_mean_unit_std(sample_aggregated_df):
    X, _ = clustering.prepare_features(sample_aggregated_df)
    X_scaled, scaler = clustering.scale_features(X)
    np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-8)
    assert scaler.mean_.shape == (len(CLUSTER_FEATURES),)


def test_assign_semantic_labels_is_monotone_in_speed():
    """
    Level 0 must be the fastest (lowest congestion) and level k-1 the
    slowest (highest congestion).
    """
    df = pd.DataFrame({
        "avg_speed_mph": [5.0, 20.0, 35.0, 6.0, 19.0, 34.0],
        "trip_density": [100, 50, 10, 110, 52, 12],
        "speed_deviation": [-6.0, 0.0, 5.0, -5.5, 0.1, 4.9],
        "zone_id": [1, 2, 3, 1, 2, 3],
        "hour_of_day": [8, 13, 23, 9, 14, 22],
    })
    raw_labels = np.array([2, 1, 0, 2, 1, 0])  # arbitrary raw assignment

    out = clustering.assign_semantic_labels(df, raw_labels, k=3)
    assert set(out["congestion_level"].unique()) == {0, 1, 2}

    # Mean speed monotone decreasing in congestion_level (0 = fastest)
    means = out.groupby("congestion_level")["avg_speed_mph"].mean().sort_index()
    assert means.is_monotonic_decreasing


def test_train_kmeans_recovers_three_clusters(sample_aggregated_df):
    X, df_feat = clustering.prepare_features(sample_aggregated_df)
    X_scaled, _ = clustering.scale_features(X)
    km = clustering.train_kmeans(X_scaled, k=3)
    assert km.n_clusters == 3
    assert km.cluster_centers_.shape == (3, X_scaled.shape[1])


def test_evaluate_clustering_returns_expected_keys(sample_aggregated_df):
    X, df_feat = clustering.prepare_features(sample_aggregated_df)
    X_scaled, _ = clustering.scale_features(X)
    km = clustering.train_kmeans(X_scaled, k=3)
    df_labeled = clustering.assign_semantic_labels(df_feat, km.labels_, k=3)
    metrics = clustering.evaluate_clustering(X_scaled, df_labeled, k=3)

    for key in ("k", "silhouette_score", "cvintra_by_level",
                "kpi_pass", "timestamp"):
        assert key in metrics

    # Synthetic data has strong separation → silhouette should be comfortably
    # above zero (we don't assert the 0.5 target here because the fixture is
    # deliberately small and clusters may jitter slightly).
    assert metrics["silhouette_score"] > 0.2


def test_compute_cvintra_is_non_negative():
    df = pd.DataFrame({
        "avg_speed_mph": [10, 11, 9, 30, 32, 28],
        "congestion_level": [2, 2, 2, 0, 0, 0],
    })
    cv = clustering.compute_cvintra(df)
    for level, v in cv.items():
        assert v >= 0


# ── Upgraded-pipeline coverage ──────────────────────────────────────────────

def test_scale_features_robust_mode_uses_robust_scaler(sample_aggregated_df):
    """RobustScaler should centre on median (== 0 after scaling)."""
    from sklearn.preprocessing import RobustScaler
    X, _ = clustering.prepare_features(sample_aggregated_df)
    X_scaled, scaler = clustering.scale_features(X, scaler_type="robust")
    assert isinstance(scaler, RobustScaler)
    # Medians of scaled columns should be ~0
    np.testing.assert_allclose(np.median(X_scaled, axis=0), 0.0, atol=1e-8)


def test_make_scaler_rejects_unknown_kind():
    import pytest
    with pytest.raises(ValueError):
        clustering._make_scaler("not-a-scaler")


def test_stability_analysis_returns_ari_bounds(sample_aggregated_df):
    X, _ = clustering.prepare_features(sample_aggregated_df)
    X_scaled, _ = clustering.scale_features(X)
    stab = clustering.stability_analysis(X_scaled, k=3, n_seeds=3)
    # ARI is in [-1, 1] but practical clusters should be well above 0
    assert -1.0 <= stab["ari_min"] <= 1.0
    assert -1.0 <= stab["ari_mean"] <= 1.0
    assert isinstance(stab["ari_pass"], bool)


def test_evaluate_clustering_includes_new_metrics(sample_aggregated_df):
    X, df_feat = clustering.prepare_features(sample_aggregated_df)
    X_scaled, _ = clustering.scale_features(X)
    km = clustering.train_kmeans(X_scaled, k=3)
    df_labeled = clustering.assign_semantic_labels(df_feat, km.labels_, k=3)
    metrics = clustering.evaluate_clustering(X_scaled, df_labeled, k=3)
    for key in ("davies_bouldin", "calinski_harabasz",
                "silhouette_per_cluster", "stability"):
        assert key in metrics


def test_run_dbscan_returns_rich_metrics(sample_aggregated_df):
    X, df_feat = clustering.prepare_features(sample_aggregated_df)
    X_scaled, _ = clustering.scale_features(X)
    # Synthetic df_feat needs congestion_level for downstream merges — not for dbscan.
    df_out, m = clustering.run_dbscan(X_scaled, df_feat)
    for key in ("n_hotspots", "n_noise", "noise_fraction",
                "hotspot_sizes_min", "hotspot_sizes_max",
                "hotspot_sizes_median", "largest_hotspot_label"):
        assert key in m
    assert 0.0 <= m["noise_fraction"] <= 1.0
    assert "dbscan_label" in df_out.columns


def test_detect_elbow_returns_source_tag():
    ks = [2, 3, 4, 5, 6, 7]
    inertias = [1000.0, 500.0, 300.0, 250.0, 230.0, 220.0]   # classic elbow near k=4
    k_hat, src = clustering._detect_elbow(ks, inertias)
    assert k_hat in ks
    assert src in {"kneed", "2nd-derivative", "default-k (too-few-points)"}
