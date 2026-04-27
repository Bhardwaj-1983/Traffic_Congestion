"""
3_🌆_Cluster_Stories.py — Drill into one cluster at a time. See its
zones, its temporal fingerprint, and its position in feature space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DAY_NAMES, NYC_CENTER_LAT, NYC_CENTER_LON,
    PROCESSED_DIR, RAW_DIR, VIZ_DATA_FILE, ZONE_CENTROIDS_FILE,
    OUTPUTS_DIR,
)
from src.utils import read_json  # noqa: E402

VIZ_DATA_PATH = PROCESSED_DIR / VIZ_DATA_FILE
CENTROIDS_PATH = RAW_DIR / ZONE_CENTROIDS_FILE
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

st.set_page_config(
    page_title="Cluster Stories · NYC Congestion",
    page_icon="🌆",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main > div { padding-top: 1rem; }
      .stories-hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        padding: 1.6rem 2rem; border-radius: 12px; border-left: 4px solid #e74c3c;
        margin-bottom: 1.25rem;
      }
      .stories-hero h1 { color: #e74c3c; margin: 0; font-size: 2rem; letter-spacing: -0.02em; }
      .stories-hero p { color: #a0aec0; margin: 0.25rem 0 0; }
      .cluster-pick {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.7rem;
      }
      .stat-tile {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 0.9rem 1.2rem; text-align: center;
      }
      .stat-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;
                    letter-spacing: 0.06em; }
      .stat-value { color: #f1f5f9; font-size: 1.6rem; font-weight: 700;
                    font-variant-numeric: tabular-nums; margin-top: 0.2rem; }
      .narrative {
        background: rgba(241,196,15,0.06); border-left: 3px solid #f1c40f;
        padding: 0.8rem 1.1rem; border-radius: 6px; color: #e2e8f0;
        font-size: 0.95rem; line-height: 1.6; margin: 0.5rem 0 1rem;
      }

      /* Rename the bare "app" entry in the sidebar nav to something proper. */
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

PLOT_BG = "#1a1f2e"; GRID = "#2d3748"; TEXT = "#cbd5e1"


def _color_to_rgb(c: str) -> tuple[int, int, int]:
    """Parse a hex (#rrggbb) or 'rgb(r, g, b)' / 'rgba(...)' string into a (r,g,b) tuple."""
    s = c.strip()
    if s.startswith("#"):
        s = s.lstrip("#")
        if len(s) == 6:
            return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        if len(s) == 3:
            return int(s[0]*2, 16), int(s[1]*2, 16), int(s[2]*2, 16)
    if s.startswith("rgb"):
        # 'rgb(231, 41, 138)' or 'rgba(231, 41, 138, 0.8)'
        inner = s[s.index("(") + 1: s.index(")")]
        parts = [p.strip() for p in inner.split(",")][:3]
        return int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
    return 128, 128, 128


@st.cache_data(show_spinner="Loading visualisation data…")
def _load_data() -> pd.DataFrame | None:
    if VIZ_DATA_PATH.exists():
        return pd.read_parquet(VIZ_DATA_PATH)
    return None


def _centroids_mtime() -> float:
    return CENTROIDS_PATH.stat().st_mtime if CENTROIDS_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_centroids(_mtime: float) -> pd.DataFrame:
    if CENTROIDS_PATH.exists():
        c = pd.read_csv(CENTROIDS_PATH).rename(columns={
            "LocationID": "zone_id", "lat": "latitude", "lon": "longitude",
            "Latitude": "latitude", "Longitude": "longitude",
        })
        return c[["zone_id", "latitude", "longitude"]]
    rng = np.random.default_rng(0)
    zone_ids = np.arange(1, 264)
    return pd.DataFrame({
        "zone_id": zone_ids,
        "latitude": rng.uniform(40.50, 40.92, size=len(zone_ids)),
        "longitude": rng.uniform(-74.25, -73.70, size=len(zone_ids)),
    })


@st.cache_data(show_spinner=False)
def _load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        try:
            return read_json(METRICS_PATH)
        except Exception:
            return None
    return None


df = _load_data()
metrics = _load_metrics()
centroids = _load_centroids(_centroids_mtime())

if df is None:
    st.error("No precomputed data. Run `make pipeline` first.")
    st.stop()

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='stories-hero'>
      <h1>🌆 Cluster Stories</h1>
      <p>Pick a cluster — see which zones live in it, when it activates, and what makes it different
      from its neighbours in feature space.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Cluster picker ──────────────────────────────────────────────────────────
levels = sorted(df["congestion_level"].unique().tolist())
n = len(levels)

# Build automatic narrative descriptors per cluster from the data
cluster_summary = (
    df.groupby("congestion_level")
      .agg(n_obs=("zone_id", "size"),
           n_zones=("zone_id", "nunique"),
           avg_speed=("avg_speed_mph", "mean"),
           avg_dev=("speed_deviation", "mean"),
           avg_density=("trip_density", "mean"),
           wknd_share=("is_weekend", "mean"))
      .sort_values("avg_dev", ascending=False)
      .reset_index()
)
# Ranking by descending speed_deviation (positive = uncongested) gives a natural ordering
cluster_summary["rank"] = range(len(cluster_summary))
order_lookup = dict(zip(cluster_summary["congestion_level"], cluster_summary["rank"]))


def _label_for(lvl: int) -> str:
    rank = order_lookup.get(lvl, lvl)
    if n <= 4:
        names = ["Low", "Moderate", "High", "Severe"]
        return names[rank] if rank < len(names) else f"L{lvl}"
    descriptors = ["Free-flow", "Light", "Light–Mod", "Moderate", "Mod–Heavy",
                   "Heavy", "Heavy–Severe", "Severe", "Gridlock"]
    return descriptors[rank] if rank < len(descriptors) else f"L{lvl}"


labels_by_level = {int(lv): _label_for(int(lv)) for lv in levels}

# Cluster picker as visual chip row
chip_cols = st.columns(min(n, 9))
sel_key = "selected_cluster_v2"
if sel_key not in st.session_state:
    # Pick the most-congested cluster as default (lowest deviation)
    st.session_state[sel_key] = int(cluster_summary.iloc[-1]["congestion_level"])

# High-contrast severity-ordered palette (must match app.py)
DISTINCT_SEVERITY_PALETTE = [
    "#2ecc71", "#16a085", "#3498db", "#9b59b6", "#f1c40f",
    "#f39c12", "#e67e22", "#e74c3c", "#c0392b", "#7f1d1d",
    "#581c87", "#1f2937",
]
if n <= len(DISTINCT_SEVERITY_PALETTE):
    palette = DISTINCT_SEVERITY_PALETTE[:n]
else:
    palette = px.colors.qualitative.Dark24[:n]

hex_by_level = {}
for i, lv in enumerate(sorted(levels, key=lambda x: order_lookup.get(x, 0))):
    hex_by_level[int(lv)] = palette[i]

for i, lv in enumerate(sorted(levels)):
    lv_int = int(lv)
    col = chip_cols[i % len(chip_cols)]
    selected = (st.session_state[sel_key] == lv_int)
    border = "3px solid #f1c40f" if selected else "1px solid #334155"
    bg = hex_by_level.get(lv_int, "#7f8c8d")
    label = labels_by_level.get(lv_int, f"L{lv_int}")
    if col.button(f"L{lv_int} · {label}", key=f"chip_{lv_int}",
                  use_container_width=True):
        st.session_state[sel_key] = lv_int

selected = int(st.session_state[sel_key])
sub = df[df["congestion_level"] == selected].copy()
sel_summary = cluster_summary[cluster_summary["congestion_level"] == selected].iloc[0]
sel_label = labels_by_level[selected]
sel_color = hex_by_level.get(selected, "#3498db")

# ── Cluster header card ─────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style='background:#1a1f2e;border-radius:12px;padding:1.2rem 1.6rem;
                border-left:6px solid {sel_color};margin:0.8rem 0 1rem;'>
      <div style='color:#94a3b8;font-size:0.9rem'>You are viewing</div>
      <div style='color:{sel_color};font-size:1.7rem;font-weight:700;letter-spacing:-0.01em;
                  margin:0.1rem 0'>Cluster {selected} — {sel_label}</div>
      <div style='color:#cbd5e1;font-size:0.95rem'>
        <b>{int(sel_summary['n_obs']):,}</b> zone-hour observations spanning
        <b>{int(sel_summary['n_zones'])}</b> distinct zones
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Quick stats row ─────────────────────────────────────────────────────────
sc = st.columns(4)
with sc[0]:
    st.markdown(
        f"""<div class='stat-tile'>
              <div class='stat-label'>Avg Speed</div>
              <div class='stat-value'>{sel_summary['avg_speed']:.1f} <span style='font-size:1rem;color:#94a3b8'>mph</span></div>
            </div>""", unsafe_allow_html=True)
with sc[1]:
    color = "#2ecc71" if sel_summary['avg_dev'] >= 0 else "#e74c3c"
    st.markdown(
        f"""<div class='stat-tile'>
              <div class='stat-label'>Avg Speed Deviation</div>
              <div class='stat-value' style='color:{color}'>{sel_summary['avg_dev']:+.2f}</div>
            </div>""", unsafe_allow_html=True)
with sc[2]:
    st.markdown(
        f"""<div class='stat-tile'>
              <div class='stat-label'>Avg Trip Density</div>
              <div class='stat-value'>{sel_summary['avg_density']:.1f}</div>
            </div>""", unsafe_allow_html=True)
with sc[3]:
    st.markdown(
        f"""<div class='stat-tile'>
              <div class='stat-label'>Weekend Share</div>
              <div class='stat-value'>{sel_summary['wknd_share']*100:.0f}%</div>
            </div>""", unsafe_allow_html=True)

# ── Auto-generated narrative ────────────────────────────────────────────────
def _narrative(s) -> str:
    parts = []
    if s["avg_dev"] >= 2:
        parts.append("speeds noticeably <b>above</b> typical zone baseline — free-flowing conditions")
    elif s["avg_dev"] >= -1:
        parts.append("speeds <b>near</b> typical zone baseline — neither fast nor slow")
    elif s["avg_dev"] >= -4:
        parts.append("speeds <b>moderately below</b> baseline — congestion building")
    else:
        parts.append("speeds <b>well below</b> baseline — heavy congestion")

    if s["avg_density"] >= 100:
        parts.append("very high trip volume")
    elif s["avg_density"] >= 30:
        parts.append("elevated trip volume")
    else:
        parts.append("modest trip volume")

    if s["wknd_share"] > 0.45:
        parts.append("more often realised on <b>weekends</b>")
    elif s["wknd_share"] < 0.20:
        parts.append("strongly skewed toward <b>weekdays</b>")
    return " · ".join(parts)


st.markdown(
    f"<div class='narrative'>This cluster shows {_narrative(sel_summary)}.</div>",
    unsafe_allow_html=True,
)

# ── Map + temporal ──────────────────────────────────────────────────────────
left, right = st.columns([5, 4], gap="medium")

with left:
    st.markdown(f"<div style='color:#e2e8f0;font-weight:600;margin-bottom:0.4rem'>"
                f"Where this cluster shows up</div>", unsafe_allow_html=True)
    map_df = sub.merge(centroids, on="zone_id", how="left").dropna(
        subset=["latitude", "longitude"])
    if not map_df.empty:
        # Aggregate to one row per zone (mean over hours that fall in this cluster)
        agg = (map_df.groupby("zone_id")
               .agg(latitude=("latitude", "first"),
                    longitude=("longitude", "first"),
                    n_hours=("hour_of_day", "size"),
                    avg_dev=("speed_deviation", "mean"),
                    avg_speed=("avg_speed_mph", "mean"),
                    density=("trip_density", "sum"))
               .reset_index())
        r, g, b = _color_to_rgb(sel_color)
        c_int = [r, g, b, 200]
        agg["color"] = [c_int] * len(agg)
        max_n = max(1, agg["n_hours"].max())
        agg["radius"] = (agg["n_hours"] / max_n * 350 + 140).clip(140, 580)
        layer = pdk.Layer(
            "ScatterplotLayer", data=agg,
            get_position=["longitude", "latitude"],
            get_color="color", get_radius="radius",
            pickable=True, opacity=0.85, stroked=True,
            line_width_min_pixels=1,
        )
        view_state = pdk.ViewState(
            latitude=NYC_CENTER_LAT, longitude=NYC_CENTER_LON,
            zoom=10.5, pitch=35, bearing=0,
        )
        tooltip = {
            "html": (
                "<b>Zone {zone_id}</b><br/>"
                "Hours in this cluster: <b>{n_hours}</b><br/>"
                "Avg speed (in cluster): <b>{avg_speed:.1f} mph</b><br/>"
                "Avg deviation: <b>{avg_dev:+.2f} mph</b>"
            ),
            "style": {"backgroundColor": "#0f172a", "color": "#e2e8f0",
                      "fontSize": "13px", "padding": "10px",
                      "border": "1px solid #334155", "borderRadius": "6px"},
        }
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                        tooltip=tooltip, map_provider="carto", map_style="dark")
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("Circle radius scales with the number of hours each zone spends in this cluster.")

with right:
    st.markdown(f"<div style='color:#e2e8f0;font-weight:600;margin-bottom:0.4rem'>"
                f"When this cluster activates</div>", unsafe_allow_html=True)
    by_hour = sub.groupby("hour_of_day").size().reindex(range(24), fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(24)), y=by_hour.values,
        marker_color=sel_color,
        marker_line_color="rgba(255,255,255,0.18)", marker_line_width=1,
        text=by_hour.values, textposition="outside",
        textfont=dict(color=TEXT, size=9),
    ))
    fig.update_layout(
        xaxis=dict(title="Hour of day", tickmode="linear", dtick=2,
                   gridcolor=GRID, color=TEXT),
        yaxis=dict(title="Zone-hour count in cluster",
                   gridcolor=GRID, color=TEXT),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=dict(color=TEXT),
        height=240, showlegend=False, margin=dict(l=50, r=20, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    by_dow = sub.groupby("day_of_week").size().reindex(range(7), fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=DAY_NAMES, y=by_dow.values,
        marker_color=sel_color, opacity=0.85,
        marker_line_color="rgba(255,255,255,0.18)", marker_line_width=1,
        text=by_dow.values, textposition="outside",
        textfont=dict(color=TEXT, size=10),
    ))
    fig.update_layout(
        xaxis=dict(title="Day of week", color=TEXT),
        yaxis=dict(title="Zone-hour count", gridcolor=GRID, color=TEXT),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=dict(color=TEXT),
        height=240, showlegend=False, margin=dict(l=50, r=20, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Top zones in cluster ────────────────────────────────────────────────────
st.markdown("<div style='color:#e2e8f0;font-weight:600;margin:1rem 0 0.4rem'>"
            "Top 15 zones spending the most time in this cluster</div>",
            unsafe_allow_html=True)
top = (sub.groupby("zone_id")
        .agg(n_hours=("hour_of_day", "size"),
             avg_speed=("avg_speed_mph", "mean"),
             avg_dev=("speed_deviation", "mean"))
        .sort_values("n_hours", ascending=False).head(15).reset_index())
if not top.empty:
    top = top.iloc[::-1].reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["n_hours"], y=[f"Zone {int(z)}" for z in top["zone_id"]],
        orientation="h", marker=dict(color=sel_color,
                                     line=dict(color="rgba(255,255,255,0.18)", width=1)),
        text=top["n_hours"], textposition="outside",
        textfont=dict(color=TEXT, size=11),
        hovertemplate=("<b>Zone %{y}</b><br>"
                       "Hours in cluster: %{x}<br>"
                       "Avg speed: %{customdata[0]:.1f} mph<br>"
                       "Avg deviation: %{customdata[1]:+.2f}<extra></extra>"),
        customdata=top[["avg_speed", "avg_dev"]].values,
    ))
    fig.update_layout(
        xaxis=dict(title="Hours in cluster", gridcolor=GRID, color=TEXT),
        yaxis=dict(title="", color=TEXT, automargin=True,
                   tickfont=dict(size=11)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=dict(color=TEXT),
        height=440, showlegend=False, margin=dict(l=110, r=40, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Compare against whole dataset ───────────────────────────────────────────
st.markdown("<div style='color:#e2e8f0;font-weight:600;margin:1rem 0 0.4rem'>"
            "How this cluster compares to the rest</div>",
            unsafe_allow_html=True)
comparison = pd.DataFrame({
    "Metric": ["Avg speed (mph)", "Avg deviation (mph)",
               "Avg trip density", "Weekend share"],
    "This cluster": [
        f"{sel_summary['avg_speed']:.2f}",
        f"{sel_summary['avg_dev']:+.2f}",
        f"{sel_summary['avg_density']:.1f}",
        f"{sel_summary['wknd_share']*100:.0f}%",
    ],
    "All clusters": [
        f"{df['avg_speed_mph'].mean():.2f}",
        f"{df['speed_deviation'].mean():+.2f}",
        f"{df['trip_density'].mean():.1f}",
        f"{df['is_weekend'].mean()*100:.0f}%",
    ],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.divider()
st.caption("Use the chips at the top to switch clusters. The map and charts refresh live.")
