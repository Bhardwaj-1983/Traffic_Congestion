"""
app.py — NYC Traffic Congestion Pattern Explorer (main page)
-------------------------------------------------------------
Streamlit dashboard. The Explorer page is the live, filterable
view of the clustered zone-hour grid. Companion pages live under
``app/pages/``:

    1_📖_How_It_Works.py     — methodology + visual walkthrough
    2_📊_Model_Diagnostics.py — cluster-quality metrics + figures
    3_🌆_Cluster_Stories.py   — drill into one cluster at a time

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CLUSTER_LABELS_FILE,
    CONGESTION_HEX_COLORS,
    CONGESTION_LABELS,
    CONGESTION_RGBA_COLORS,
    DAY_NAMES,
    MODELS_DIR,
    MODEL_FILE,
    NYC_CENTER_LAT,
    NYC_CENTER_LON,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    VIZ_DATA_FILE,
    ZONE_CENTROIDS_FILE,
)
from src.utils import read_json  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

VIZ_DATA_PATH = PROCESSED_DIR / VIZ_DATA_FILE
MODEL_PATH = MODELS_DIR / MODEL_FILE
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
CENTROIDS_PATH = RAW_DIR / ZONE_CENTROIDS_FILE
MAPBOX_TOKEN = os.environ.get("MAPBOX_API_KEY")

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NYC Traffic Congestion Explorer",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — card styling, KPI metrics, headers
st.markdown(
    """
    <style>
      /* Tighten top padding */
      .main > div { padding-top: 1rem; }

      /* Hero header */
      .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border-left: 4px solid #f1c40f;
        margin-bottom: 1.25rem;
      }
      .hero-title {
        font-size: 2.0rem; font-weight: 700; color: #f1c40f;
        margin: 0 0 0.25rem 0; letter-spacing: -0.02em;
      }
      .hero-subtitle {
        color: #a0aec0; font-size: 0.95rem; margin: 0;
      }

      /* KPI cards — equal-width, equal-height, numeric values clamped */
      .kpi {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        height: 100%;
        min-height: 110px;
        display: flex; flex-direction: column; justify-content: center;
        overflow: hidden;
      }
      .kpi-label {
        font-size: 0.78rem; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .kpi-value {
        font-size: 1.7rem; font-weight: 700; color: #f1f5f9;
        line-height: 1.1; font-variant-numeric: tabular-nums;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .kpi-trend {
        font-size: 0.82rem; color: #64748b; margin-top: 0.3rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      .kpi-accent-green { border-left: 3px solid #2ecc71; }
      .kpi-accent-red   { border-left: 3px solid #e74c3c; }
      .kpi-accent-blue  { border-left: 3px solid #3498db; }
      .kpi-accent-amber { border-left: 3px solid #f39c12; }

      /* Section headings */
      .section-h {
        font-size: 1.15rem; font-weight: 600; color: #e2e8f0;
        margin: 1rem 0 0.5rem 0;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 0.4rem;
      }

      /* Pill badges */
      .pill {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; margin-right: 4px;
        background: #1e293b; color: #cbd5e1; border: 1px solid #334155;
      }
      .pill-pass { background: #064e3b; color: #6ee7b7; border-color: #065f46; }
      .pill-fail { background: #7f1d1d; color: #fecaca; border-color: #991b1b; }

      /* Insight cards */
      .insight-card {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #2d3748;
        margin-bottom: 0.6rem;
      }
      .insight-title {
        font-size: 0.78rem; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
      }
      .insight-body { font-size: 0.95rem; color: #e2e8f0; }

      /* Chart containers */
      .stPlotlyChart, .stDeckGlJsonChart {
        background: #1a1f2e; border-radius: 10px;
        border: 1px solid #2d3748; padding: 6px;
      }

      /* Cluster summary card — used inside st.columns grid */
      .cluster-card {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 0.7rem 0.9rem 0.8rem 1.0rem;
        position: relative; overflow: hidden; height: 100%;
        transition: border-color 0.15s ease;
      }
      .cluster-card:hover { border-color: #475569; }
      .cluster-card .swatch {
        position: absolute; top: 0; left: 0; bottom: 0; width: 4px;
        border-top-left-radius: 10px; border-bottom-left-radius: 10px;
      }
      .cluster-card .cc-label {
        font-size: 0.72rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.1rem;
      }
      .cluster-card .cc-name {
        font-size: 0.98rem; font-weight: 700; color: #f1f5f9;
        margin-bottom: 0.5rem;
      }
      .cluster-card .cc-stats {
        font-size: 0.82rem; color: #cbd5e1;
        font-variant-numeric: tabular-nums; line-height: 1.45;
      }
      .cluster-card .cc-stats b { color: #f1f5f9; }

      /* Story panel — softer, less shouty */
      .story-panel {
        background: #161a26;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 0.85rem 1.15rem;
        margin: 0 0 0.9rem 0;
        color: #cbd5e1;
        font-size: 0.94rem;
        line-height: 1.6;
      }
      .story-panel b { color: #e2e8f0; }
      .story-panel .icon { color: #64748b; margin-right: 0.4rem; }

      /* Section dividers */
      .soft-divider {
        height: 1px; background: linear-gradient(90deg, transparent, #2d3748, transparent);
        margin: 1.4rem 0 0.4rem; border: 0;
      }

      /* Cluster chip selector */
      .stButton > button {
        font-weight: 600 !important;
      }

      /* Rename the auto-generated "app" entry in the sidebar nav.
         Streamlit derives that label from the entrypoint filename and offers
         no public hook for it, so we hide the text and render our own. */
      [data-testid="stSidebarNav"] ul li:first-child a span:not([role]) ,
      [data-testid="stSidebarNav"] ul li:first-child a p {
        display: none !important;
      }
      [data-testid="stSidebarNav"] ul li:first-child a::after {
        content: "🏠  Dashboard";
        color: inherit;
        font-size: 0.95rem;
        font-weight: 500;
        margin-left: 0.15rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading visualization data…")
def load_viz_data() -> pd.DataFrame:
    if not VIZ_DATA_PATH.exists():
        st.error(
            f"Visualization data not found at `{VIZ_DATA_PATH}`.\n\n"
            "Run the pipeline first: `python main.py` (or `make pipeline`)."
        )
        st.stop()
    return pd.read_parquet(VIZ_DATA_PATH)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        try:
            return read_json(METRICS_PATH)
        except Exception:
            return None
    return None


def _centroids_cache_key() -> float:
    return CENTROIDS_PATH.stat().st_mtime if CENTROIDS_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_centroids_cached(_mtime: float) -> pd.DataFrame:
    if CENTROIDS_PATH.exists():
        df = pd.read_csv(CENTROIDS_PATH)
        df = df.rename(columns={
            "LocationID": "zone_id", "lat": "latitude", "lon": "longitude",
            "Latitude": "latitude", "Longitude": "longitude",
        })
        return df[["zone_id", "latitude", "longitude"]]
    rng = np.random.default_rng(0)
    zone_ids = np.arange(1, 264)
    lats = rng.uniform(40.50, 40.92, size=len(zone_ids))
    lons = rng.uniform(-74.25, -73.70, size=len(zone_ids))
    return pd.DataFrame({"zone_id": zone_ids, "latitude": lats, "longitude": lons})


def get_zone_centroids() -> pd.DataFrame:
    mtime = _centroids_cache_key()
    df = _load_centroids_cached(mtime)
    if mtime == 0.0:
        st.warning(
            f"Zone centroid file not found at `{CENTROIDS_PATH}`. "
            "Using approximate positions — run `python scripts/make_centroids.py` "
            "for real TLC centroids."
        )
    return df


# ── Dynamic palette (works for any k) ───────────────────────────────────────

# High-contrast, severity-ordered palette. Each colour is far enough from its
# neighbours in hue/lightness that adjacent clusters are visually distinct,
# while the overall ramp still reads cool→warm = uncongested→congested.
DISTINCT_SEVERITY_PALETTE: list[str] = [
    "#2ecc71",   # 0  emerald          — free-flow
    "#16a085",   # 1  teal             — light
    "#3498db",   # 2  bright blue      — light–mod
    "#9b59b6",   # 3  purple           — moderate
    "#f1c40f",   # 4  yellow           — mod–heavy
    "#f39c12",   # 5  amber            — heavy
    "#e67e22",   # 6  orange           — heavy–severe
    "#e74c3c",   # 7  red              — severe
    "#c0392b",   # 8  dark red         — gridlock
    "#7f1d1d",   # 9  oxblood          — extreme
    "#581c87",   # 10 deep violet      — extreme overflow
    "#1f2937",   # 11 near-black       — extreme overflow
]


def _hex_to_rgba(c: str, alpha: int = 200) -> list[int]:
    s = c.lstrip("#")
    if len(s) == 6:
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), alpha]
    return [128, 128, 128, alpha]


def build_palette(levels: list[int]) -> tuple[dict[int, str], dict[int, list[int]], dict[int, str]]:
    """
    Generate label, hex-color, and RGBA-color dicts for an arbitrary set of
    congestion levels. Uses the canonical 3-level palette when k=3, a 4-stop
    ramp when k=4, and a hand-curated high-contrast palette for k≥5 so
    adjacent clusters remain visually distinct.

    Returns (label_map, rgba_map, hex_map).
    """
    levels = sorted(set(levels))
    n = len(levels)

    # Use the configured palette for the canonical k=3 case
    if n == 3 and set(levels) == {0, 1, 2}:
        labels = dict(CONGESTION_LABELS)
        hexes = dict(CONGESTION_HEX_COLORS)
        rgbas = dict(CONGESTION_RGBA_COLORS)
        return labels, rgbas, hexes

    # k=4: clean four-stop green→red ramp
    if n == 4:
        cols = ["#2ecc71", "#f1c40f", "#e67e22", "#c0392b"]
    elif n <= len(DISTINCT_SEVERITY_PALETTE):
        # Take the first n colours from the curated palette in severity order
        cols = DISTINCT_SEVERITY_PALETTE[:n]
    else:
        # Beyond 12 — fall back to plotly's high-contrast Dark24
        cols = px.colors.qualitative.Dark24[:n]
        if len(cols) < n:
            cols = list(cols) + ["#7f8c8d"] * (n - len(cols))

    labels: dict[int, str] = {}
    hexes: dict[int, str] = {}
    rgbas: dict[int, list[int]] = {}
    descriptors = ["Free-flow", "Light", "Light–Mod", "Moderate", "Mod–Heavy",
                   "Heavy", "Heavy–Severe", "Severe", "Gridlock",
                   "Extreme", "Critical", "Off-scale"]
    for i, lvl in enumerate(levels):
        if n == 3:
            labels[lvl] = ["Low", "Medium", "High"][i]
        elif n == 4:
            labels[lvl] = ["Low", "Moderate", "High", "Severe"][i]
        elif i < len(descriptors):
            labels[lvl] = descriptors[i]
        else:
            labels[lvl] = f"Level {lvl}"
        hexes[lvl] = cols[i]
        rgbas[lvl] = _hex_to_rgba(cols[i], alpha=210)
    return labels, rgbas, hexes


# ── Filtering ────────────────────────────────────────────────────────────────

def apply_filters(df, hour, day_filter, selected_days):
    f = df[df["hour_of_day"] == hour].copy()
    if day_filter == "Weekdays Only":
        f = f[~f["is_weekend"]]
    elif day_filter == "Weekends Only":
        f = f[f["is_weekend"]]
    if selected_days:
        f = f[f["day_of_week"].isin(selected_days)]
    return f


# ── Map ──────────────────────────────────────────────────────────────────────

def build_map_dataframe(filtered, centroids, rgba_map):
    merged = filtered.merge(centroids, on="zone_id", how="left")
    merged = merged.dropna(subset=["latitude", "longitude"]).copy()

    def _rgba(v):
        return rgba_map.get(int(v), [127, 140, 141, 180])

    merged["color"] = merged["congestion_level"].map(_rgba)
    if "congestion_label_dynamic" not in merged.columns:
        merged["congestion_label_dynamic"] = merged["congestion_level"].astype(str)

    max_density = merged["trip_density"].max()
    if max_density and max_density > 0:
        merged["radius"] = (merged["trip_density"] / max_density * 350 + 140).clip(140, 580)
    else:
        merged["radius"] = 180.0
    return merged


def render_map(map_df):
    if map_df.empty:
        st.info("No data available for the selected filters.")
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.85,
        stroked=True,
        line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(
        latitude=NYC_CENTER_LAT,
        longitude=NYC_CENTER_LON,
        zoom=10.5, pitch=35, bearing=0,
    )
    tooltip = {
        "html": (
            "<b>Zone {zone_id}</b><br/>"
            "Congestion: <b>{congestion_label_dynamic}</b><br/>"
            "Avg Speed: <b>{avg_speed_mph:.1f} mph</b><br/>"
            "Trip Density: <b>{trip_density}</b><br/>"
            "Speed Deviation: <b>{speed_deviation:+.2f} mph</b>"
        ),
        "style": {
            "backgroundColor": "#0f172a", "color": "#e2e8f0",
            "fontSize": "13px", "padding": "10px",
            "border": "1px solid #334155", "borderRadius": "6px",
        },
    }

    # Carto basemap — no Mapbox token required, works everywhere
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider="carto",
        map_style="dark",
    )
    if MAPBOX_TOKEN:
        deck.api_keys = {"mapbox": MAPBOX_TOKEN}

    st.pydeck_chart(deck, use_container_width=True)


# ── Charts ───────────────────────────────────────────────────────────────────

PLOT_BG = "#1a1f2e"
PAPER_BG = "#1a1f2e"
GRID = "#2d3748"
TEXT = "#cbd5e1"


def render_hourly_chart(df, selected_hour):
    hourly = (
        df.groupby("hour_of_day")
        .agg(avg_speed=("avg_speed_mph", "mean"),
             density=("trip_density", "mean"),
             deviation=("speed_deviation", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour_of_day"], y=hourly["density"],
        name="Trip Density",
        marker_color="rgba(241,196,15,0.30)",
        marker_line_color="rgba(241,196,15,0.6)", marker_line_width=1,
        yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour_of_day"], y=hourly["avg_speed"],
        mode="lines+markers", name="Avg Speed (mph)",
        line=dict(color="#3498db", width=3),
        marker=dict(size=7, color="#3498db", line=dict(color="#fff", width=1)),
        yaxis="y1",
    ))
    fig.add_vline(
        x=selected_hour, line_width=2, line_dash="dash", line_color="#f1c40f",
        annotation_text=f" {selected_hour:02d}:00",
        annotation_position="top right",
        annotation_font=dict(color="#f1c40f", size=12),
    )
    fig.update_layout(
        title=dict(text="Average Speed & Trip Density by Hour", font=dict(size=15)),
        xaxis=dict(title="Hour of Day", tickmode="linear", dtick=1,
                   gridcolor=GRID, color=TEXT),
        yaxis=dict(title="Avg Speed (mph)", side="left",
                   showgrid=True, gridcolor=GRID, color=TEXT),
        yaxis2=dict(title="Trip Density", side="right", overlaying="y",
                    showgrid=False, color=TEXT),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=dict(color=TEXT),
        height=340, margin=dict(l=50, r=50, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cluster_distribution(filtered, label_map, hex_map):
    """Bar chart that adapts to whatever k the model produced."""
    levels = sorted(filtered["congestion_level"].unique().tolist())
    counts = (
        filtered["congestion_level"].value_counts()
        .reindex(levels, fill_value=0).reset_index()
    )
    counts.columns = ["level", "count"]
    counts["label"] = counts["level"].map(label_map).fillna(
        counts["level"].apply(lambda v: f"L{int(v)}"))
    counts["color"] = counts["level"].map(hex_map).fillna("#7f8c8d")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["label"], y=counts["count"],
        marker_color=counts["color"],
        marker_line_color="rgba(255,255,255,0.15)", marker_line_width=1,
        text=counts["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        textfont=dict(color=TEXT, size=11),
    ))
    fig.update_layout(
        title=dict(text="Cluster Distribution (Current Filter)", font=dict(size=14)),
        xaxis=dict(title="Congestion Level", color=TEXT),
        yaxis=dict(title="Zone-Hour Observations",
                   gridcolor=GRID, color=TEXT),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=dict(color=TEXT),
        showlegend=False, height=320, margin=dict(l=50, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_zones_chart(filtered, label_map, hex_map):
    top = (
        filtered.groupby("zone_id")
        .agg(avg_level=("congestion_level", "mean"),
             avg_speed=("avg_speed_mph", "mean"),
             density=("trip_density", "sum"))
        .sort_values("avg_level", ascending=False)
        .head(15)
        .reset_index()
    )
    if top.empty:
        st.info("No zones match the selected filters.")
        return

    # Color each bar by the zone's modal congestion level (rounded)
    top["lvl_int"] = top["avg_level"].round().astype(int).clip(
        min(label_map.keys()), max(label_map.keys())
    )
    top["color"] = top["lvl_int"].map(hex_map).fillna("#e74c3c")
    top["label"] = "Zone " + top["zone_id"].astype(int).astype(str)

    # Reverse for top-down bar order
    top = top.iloc[::-1].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["avg_level"], y=top["label"],
        orientation="h",
        marker=dict(color=top["color"],
                    line=dict(color="rgba(255,255,255,0.18)", width=1)),
        text=top["avg_level"].apply(lambda v: f"{v:.2f}"),
        textposition="outside", textfont=dict(color=TEXT, size=11),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg level: %{x:.2f}<br>"
            "Avg speed: %{customdata[0]:.1f} mph<br>"
            "Total trips: %{customdata[1]:,}<extra></extra>"
        ),
        customdata=top[["avg_speed", "density"]].values,
    ))
    fig.update_layout(
        title=dict(text="Top 15 Most Congested Zones", font=dict(size=14)),
        xaxis=dict(title="Avg Congestion Level",
                   gridcolor=GRID, color=TEXT, zerolinecolor=GRID),
        yaxis=dict(title="", color=TEXT, automargin=True,
                   tickfont=dict(size=11)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=dict(color=TEXT),
        height=440, margin=dict(l=110, r=40, t=50, b=40), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_zone_hour_heatmap(df):
    """Heatmap: top zones × hour-of-day, coloured by avg congestion level."""
    top_n = 25
    top_zones = (
        df.groupby("zone_id")["congestion_level"].mean()
        .sort_values(ascending=False).head(top_n).index.tolist()
    )
    sub = df[df["zone_id"].isin(top_zones)]
    pivot = (
        sub.groupby(["zone_id", "hour_of_day"])["congestion_level"]
        .mean().unstack(fill_value=np.nan)
    )
    pivot = pivot.reindex(top_zones)  # preserve top-zone order

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}" for h in pivot.columns],
        y=[f"Zone {int(z)}" for z in pivot.index],
        colorscale=[[0, "#2ecc71"], [0.4, "#f1c40f"], [0.7, "#e67e22"], [1, "#c0392b"]],
        colorbar=dict(title=dict(text="Cong.<br>Level", font=dict(color=TEXT, size=10)),
                      tickfont=dict(color=TEXT, size=10)),
        hovertemplate="<b>%{y}</b><br>Hour: %{x}<br>Avg level: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Top {top_n} Zones × Hour-of-Day (avg congestion)", font=dict(size=14)),
        xaxis=dict(title="Hour of Day", color=TEXT, side="bottom"),
        yaxis=dict(title="", color=TEXT, autorange="reversed", tickfont=dict(size=10)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=dict(color=TEXT),
        height=520, margin=dict(l=80, r=50, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Cluster summary strip ────────────────────────────────────────────────────

def cluster_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster with descriptive stats, ordered by congestion level
    (ascending = least → most congested, matching the semantic-label convention
    enforced by src.clustering.assign_semantic_labels)."""
    return (
        df.groupby("congestion_level")
        .agg(n_obs=("zone_id", "size"),
             n_zones=("zone_id", "nunique"),
             avg_speed=("avg_speed_mph", "mean"),
             avg_dev=("speed_deviation", "mean"),
             avg_density=("trip_density", "mean"),
             wknd_share=("is_weekend", "mean"))
        .sort_index()
        .reset_index()
    )


def render_cluster_strip(df: pd.DataFrame, label_map: dict, hex_map: dict,
                         per_row: int = 4):
    """Render one card per cluster in a proper grid (st.columns), ordered by
    cluster ID. Cards wrap to new rows after `per_row` cards."""
    summary = cluster_summary_table(df)
    rows = list(summary.itertuples(index=False))
    n = len(rows)

    # Choose a grid width: tighten if we have a lot of clusters
    if n <= 4:
        per_row = n
    elif n <= 6:
        per_row = 3
    elif n <= 8:
        per_row = 4
    else:
        per_row = 5  # 9 clusters → 5 + 4

    for r_idx, start in enumerate(range(0, n, per_row)):
        # Breathing room between rows (skip before the first row)
        if r_idx > 0:
            st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)
        slice_ = rows[start:start + per_row]
        cols = st.columns(per_row, gap="small")
        for i, row in enumerate(slice_):
            lvl = int(row.congestion_level)
            name = label_map.get(lvl, f"Level {lvl}")
            col = hex_map.get(lvl, "#7f8c8d")
            n_obs = int(row.n_obs)
            n_zones = int(row.n_zones)
            avg_dev = row.avg_dev
            avg_speed = row.avg_speed
            dev_color = "#2ecc71" if avg_dev >= 0 else "#e74c3c"
            with cols[i]:
                st.markdown(
                    f"""
                    <div class='cluster-card'>
                      <div class='swatch' style='background:{col}'></div>
                      <div class='cc-label'>Cluster {lvl}</div>
                      <div class='cc-name' style='color:{col}'>{name}</div>
                      <div class='cc-stats'>
                        <b>{n_obs:,}</b> obs · <b>{n_zones}</b> zones<br/>
                        avg speed <b>{avg_speed:.1f}</b> mph<br/>
                        Δ baseline <b style='color:{dev_color}'>{avg_dev:+.2f}</b>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        # Fill any unused trailing cells in the last partial row with a placeholder
        # to keep card heights uniform
        for j in range(len(slice_), per_row):
            with cols[j]:
                st.markdown(
                    "<div style='height:1px'></div>", unsafe_allow_html=True
                )


def render_centroid_radar(metrics: dict | None, label_map: dict, hex_map: dict):
    """Radar chart of cluster centroids in standardised feature space."""
    centroids = metrics.get("centroid_means", {}) if metrics else {}
    if not centroids:
        st.info("Centroid means not available in metrics. Re-run `make pipeline`.")
        return

    # Use only the standardised cluster features (skip avg_speed/trip_density derived)
    feat_keys = ["hour_sin", "hour_cos", "is_weekend",
                 "speed_deviation", "log_trip_density"]

    fig = go.Figure()
    for lvl_str, feat_dict in centroids.items():
        lvl = int(lvl_str)
        name = label_map.get(lvl, f"L{lvl}")
        col = hex_map.get(lvl, "#7f8c8d")
        vals = [float(feat_dict.get(k, 0)) for k in feat_keys]
        # Close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=feat_keys + [feat_keys[0]],
            name=f"L{lvl} · {name}",
            fill="toself", opacity=0.4,
            line=dict(color=col, width=2),
        ))
    fig.update_layout(
        title=dict(text="Cluster Centroids — Feature Space (standardised)", font=dict(size=14)),
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(visible=True, gridcolor=GRID, color=TEXT,
                            angle=90, tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=GRID, color=TEXT, tickfont=dict(size=10)),
        ),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font=dict(color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=10),
                    orientation="v", x=1.02, y=0.5),
        height=380, margin=dict(l=20, r=120, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def auto_story(filtered: pd.DataFrame, df: pd.DataFrame,
               hour: int, day_filter: str, label_map: dict) -> str:
    """Compose a one-paragraph narrative summarising the current view."""
    if filtered.empty:
        return "No data matches this filter."
    n_zones = filtered["zone_id"].nunique()
    avg_speed = float(filtered["avg_speed_mph"].mean())
    avg_speed_all = float(df["avg_speed_mph"].mean())
    delta = avg_speed - avg_speed_all
    avg_dev = float(filtered["speed_deviation"].mean())
    dom_level = int(filtered["congestion_level"].mode().iloc[0])
    dom_label = label_map.get(dom_level, f"Level {dom_level}")
    pct_high = 100 * (filtered["congestion_level"] >= max(label_map.keys()) - 1).sum() / len(filtered)

    if delta > 1:
        speed_phrase = f"<b>{abs(delta):.1f} mph faster</b> than the all-day average"
    elif delta < -1:
        speed_phrase = f"<b>{abs(delta):.1f} mph slower</b> than the all-day average"
    else:
        speed_phrase = "close to the all-day average"

    if avg_dev > 1:
        dev_phrase = "zones are <b>moving above their typical baseline</b> — uncongested"
    elif avg_dev < -1:
        dev_phrase = "zones are <b>moving below their typical baseline</b> — congested"
    else:
        dev_phrase = "zones are at their typical baseline speed"

    return (
        f"At <b>{hour:02d}:00</b> ({day_filter.lower()}), {n_zones} zones are active. "
        f"Average speed is <b>{avg_speed:.1f} mph</b>, {speed_phrase}, and {dev_phrase}. "
        f"The dominant cluster is <b>{dom_label}</b>, and <b>{pct_high:.0f}%</b> of "
        f"observations sit in the highest one-or-two congestion levels."
    )


# ── Insights ─────────────────────────────────────────────────────────────────

def compute_insights(df):
    hourly_cong = (
        df.groupby("hour_of_day")["congestion_level"].mean()
        .sort_values(ascending=False)
    )
    peak_hours = hourly_cong.head(3).index.tolist()
    zone_cong = (
        df.groupby("zone_id")["congestion_level"].mean()
        .sort_values(ascending=False)
    )
    top_zones = zone_cong.head(5).index.tolist()
    level_pct = (df["congestion_level"].value_counts(normalize=True) * 100).to_dict()
    return {
        "peak_hours": peak_hours,
        "top_zones": top_zones,
        "level_pct": level_pct,
        "overall_avg_speed": float(df["avg_speed_mph"].mean()),
        "total_density": int(df["trip_density"].sum()),
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(label_map, hex_map):
    st.sidebar.markdown("### 🚕 NYC Congestion Explorer")
    st.sidebar.markdown(
        "<span style='color:#94a3b8;font-size:0.9rem'>"
        "Discover hidden congestion patterns in NYC taxi data using unsupervised learning. "
        "Use the pages on the left to learn how the model works and view diagnostics."
        "</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    hour = st.sidebar.slider(
        "Hour of Day", min_value=0, max_value=23, value=9, format="%d:00",
        help="Filter map and charts to this hour.",
    )
    day_filter = st.sidebar.selectbox(
        "Day Type",
        options=["All Days", "Weekdays Only", "Weekends Only"], index=0,
    )

    # Cluster focus — show only zones in a chosen cluster (or all)
    focus_options = [("All clusters", -1)] + [
        (f"Only · {label_map[lvl]}", int(lvl)) for lvl in sorted(label_map.keys())
    ]
    focus_choice = st.sidebar.selectbox(
        "Cluster focus",
        options=focus_options, index=0,
        format_func=lambda t: t[0],
        help="Show only zones currently assigned to one cluster.",
    )
    focus_level = focus_choice[1]

    with st.sidebar.expander("Advanced · Day of Week", expanded=False):
        day_options = list(enumerate(DAY_NAMES))
        chosen = st.multiselect(
            "Restrict to specific days", options=day_options, default=[],
            format_func=lambda t: t[1],
            help="Optional — leave empty to keep all days selected.",
        )
        selected_days = [i for i, _ in chosen]

    st.sidebar.divider()
    st.sidebar.markdown("**Legend**")
    legend_html = "<div style='line-height:1.7'>"
    # Order legend by congestion level (low -> high), matching the cluster strip
    for lvl in sorted(label_map.keys()):
        col = hex_map.get(lvl, "#7f8c8d")
        legend_html += (
            f"<span style='display:inline-block;width:11px;height:11px;"
            f"background:{col};border-radius:50%;margin-right:6px;"
            f"vertical-align:middle;border:1px solid rgba(255,255,255,0.2)'></span>"
            f"<span style='color:#cbd5e1;font-size:0.88rem'>{label_map[lvl]}</span><br/>"
        )
    legend_html += "</div>"
    st.sidebar.markdown(legend_html, unsafe_allow_html=True)
    st.sidebar.caption("Data: NYC TLC Yellow-Taxi Trip Records 2025")
    return hour, day_filter, selected_days, focus_level


# ── KPI cards ────────────────────────────────────────────────────────────────

def render_kpi_row(filtered, df, insights, metrics, label_map):
    """Top-row KPI cards."""
    n_obs = len(filtered)
    n_zones = filtered["zone_id"].nunique()
    avg_speed_filtered = float(filtered["avg_speed_mph"].mean()) if n_obs else 0.0
    avg_speed_overall = float(df["avg_speed_mph"].mean())
    delta_speed = avg_speed_filtered - avg_speed_overall

    # High-congestion threshold = top third of unique levels
    levels_sorted = sorted(label_map.keys())
    high_threshold = levels_sorted[int(len(levels_sorted) * 0.66)] if levels_sorted else 2
    pct_high_filtered = (
        100 * (filtered["congestion_level"] >= high_threshold).sum() / max(1, n_obs)
    )

    total_trips = int(filtered["trip_density"].sum())

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f"""<div class='kpi kpi-accent-blue'>
                  <div class='kpi-label'>Avg Speed (filter)</div>
                  <div class='kpi-value'>{avg_speed_filtered:.1f} <span style='font-size:1rem;color:#94a3b8'>mph</span></div>
                  <div class='kpi-trend'>{'+' if delta_speed>=0 else ''}{delta_speed:.2f} vs all-day mean</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""<div class='kpi kpi-accent-amber'>
                  <div class='kpi-label'>Trips (filter)</div>
                  <div class='kpi-value'>{total_trips:,}</div>
                  <div class='kpi-trend'>across {n_zones} active zones</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"""<div class='kpi kpi-accent-red'>
                  <div class='kpi-label'>High-Congestion Share</div>
                  <div class='kpi-value'>{pct_high_filtered:.1f}<span style='font-size:1rem;color:#94a3b8'>%</span></div>
                  <div class='kpi-trend'>level ≥ {high_threshold}</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with cols[3]:
        if metrics:
            sil = float(metrics.get("silhouette_score", 0))
            k = metrics.get("k", "?")
            kpi_pass = bool(metrics.get("kpi_pass", False))
            pill = ("<span class='pill pill-pass'>PASS</span>" if kpi_pass
                    else "<span class='pill pill-fail'>FAIL</span>")
            colour = "kpi-accent-green" if kpi_pass else "kpi-accent-red"
            sil_str = f"{sil:+.3f}".lstrip("+") if sil >= 0 else f"{sil:.3f}"
            st.markdown(
                f"""<div class='kpi {colour}'>
                      <div class='kpi-label'>Silhouette · k = {k}</div>
                      <div class='kpi-value'>{sil_str}</div>
                      <div class='kpi-trend'>vs target ≥ 0.50  ·  {pill}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class='kpi'>
                     <div class='kpi-label'>Model Quality</div>
                     <div class='kpi-value'>—</div>
                     <div class='kpi-trend'>metrics unavailable</div>
                   </div>""",
                unsafe_allow_html=True,
            )


# ── Insights cards ───────────────────────────────────────────────────────────

def render_insight_cards(insights, metrics):
    peak_str = " · ".join([f"<span class='pill'>{h:02d}:00</span>" for h in insights["peak_hours"]])
    zones_str = " · ".join([f"<span class='pill'>Zone {int(z)}</span>" for z in insights["top_zones"]])

    st.markdown(
        f"""<div class='insight-card'>
              <div class='insight-title'>Peak Congestion Hours (overall)</div>
              <div class='insight-body'>{peak_str}</div>
            </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class='insight-card'>
              <div class='insight-title'>Most Congested Zones</div>
              <div class='insight-body'>{zones_str}</div>
            </div>""",
        unsafe_allow_html=True,
    )

    if metrics:
        sil = float(metrics.get("silhouette_score", 0))
        db = metrics.get("davies_bouldin", None)
        ch = metrics.get("calinski_harabasz", None)
        stab = metrics.get("stability", {}) or {}
        ari_min = stab.get("ari_min", None)
        body = f"silhouette = <b>{sil:+.3f}</b>"
        if db is not None:
            body += f"   ·   DB = <b>{float(db):.3f}</b>"
        if ch is not None:
            body += f"   ·   CH = <b>{float(ch):,.0f}</b>"
        if ari_min is not None:
            body += f"   ·   ARI<sub>min</sub> = <b>{float(ari_min):.2f}</b>"
        st.markdown(
            f"""<div class='insight-card'>
                  <div class='insight-title'>Model Diagnostics</div>
                  <div class='insight-body'>{body}</div>
                </div>""",
            unsafe_allow_html=True,
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_viz_data()
    centroids = get_zone_centroids()
    metrics = load_metrics()

    # Build a palette that adapts to whatever k the model used
    levels = sorted(df["congestion_level"].unique().tolist())
    label_map, rgba_map, hex_map = build_palette(levels)

    # Rebuild the dynamic-label column on the df for tooltip use
    df = df.copy()
    df["congestion_label_dynamic"] = df["congestion_level"].map(label_map).astype(str)

    hour, day_filter, selected_days, focus_level = render_sidebar(label_map, hex_map)

    # Hero header — richer subline
    focus_text = (
        f"  ·  Cluster focus: <b>{label_map.get(focus_level, '—')}</b>"
        if focus_level >= 0 else "  ·  All clusters"
    )
    st.markdown(
        f"""<div class='hero'>
              <div class='hero-title'>🚕 NYC Traffic Congestion Pattern Explorer</div>
              <div class='hero-subtitle'>
                Hour <b>{hour:02d}:00</b>  ·  <b>{day_filter}</b>{focus_text}  ·
                <b>{len(df):,}</b> zone-hour observations across <b>{df['zone_id'].nunique()}</b> NYC taxi zones  ·
                model: <b>k = {metrics.get('k', '?') if metrics else '?'}</b>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )

    filtered = apply_filters(df, hour, day_filter, selected_days)
    if focus_level >= 0:
        filtered = filtered[filtered["congestion_level"] == focus_level]

    if filtered.empty:
        st.warning("No data matches the selected filters. Try a different hour, day, or cluster focus.")
        return

    insights = compute_insights(df)
    map_df = build_map_dataframe(filtered, centroids, rgba_map)

    # Row 1 — KPI cards (top-of-page summary of the *current filter*)
    st.markdown("<div class='section-h'>📈 Snapshot of the current view</div>",
                unsafe_allow_html=True)
    render_kpi_row(filtered, df, insights, metrics, label_map)

    # Row 2 — auto-narrated story for the current view
    st.markdown(
        f"<div class='story-panel'><span class='icon'>📖</span>"
        f"{auto_story(filtered, df, hour, day_filter, label_map)}</div>",
        unsafe_allow_html=True,
    )

    # Soft divider between filter-scoped KPIs and dataset-wide cluster legend
    st.markdown("<hr class='soft-divider'/>", unsafe_allow_html=True)

    # Row 3 — Cluster legend strip (one card per cluster, dataset-wide stats,
    # ordered by cluster ID = severity ascending)
    st.markdown("<div class='section-h'>🏷️ Cluster legend &amp; profile</div>",
                unsafe_allow_html=True)
    render_cluster_strip(df, label_map, hex_map)

    # Row 2 — Map (left, large) + Insights/Cluster distribution (right)
    st.markdown("<div class='section-h'>🗺️ Spatial View</div>", unsafe_allow_html=True)
    col_map, col_right = st.columns([7, 4], gap="medium")
    with col_map:
        render_map(map_df)
    with col_right:
        render_insight_cards(insights, metrics)
        render_cluster_distribution(filtered, label_map, hex_map)

    # Row 3 — Hourly trend + Centroid radar
    st.markdown("<div class='section-h'>⏱️ Temporal &amp; Feature-Space View</div>",
                unsafe_allow_html=True)
    col_temp, col_radar = st.columns([3, 2], gap="medium")
    with col_temp:
        render_hourly_chart(df, selected_hour=hour)
    with col_radar:
        render_centroid_radar(metrics, label_map, hex_map)

    # Row 4 — Heatmap + Top zones
    st.markdown("<div class='section-h'>🔥 Hotspots</div>", unsafe_allow_html=True)
    col_heat, col_top = st.columns([3, 2], gap="medium")
    with col_heat:
        render_zone_hour_heatmap(df)
    with col_top:
        render_top_zones_chart(filtered, label_map, hex_map)

    # Raw data
    with st.expander("📋 View raw filtered data"):
        display_cols = [
            "zone_id", "hour_of_day", "day_name", "congestion_label_dynamic",
            "avg_speed_mph", "trip_density", "speed_deviation",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        sort_col = "congestion_level" if "congestion_level" in filtered.columns else display_cols[0]
        st.dataframe(
            filtered.sort_values(sort_col, ascending=False)[display_cols],
            use_container_width=True, hide_index=True,
        )


if __name__ == "__main__":
    main()
