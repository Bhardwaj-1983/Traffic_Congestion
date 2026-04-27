"""
1_📖_How_It_Works.py — Methodology & visual walkthrough
--------------------------------------------------------
Explains the project: data, the speed_deviation feature, cyclic time
encoding, scaling decisions, k-selection, and the clustering output —
with live-computed plots and embedded static figures from outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    OUTPUTS_DIR, PROCESSED_DIR, VIZ_DATA_FILE,
)
from src.utils import read_json  # noqa: E402

VIZ_DATA_PATH = PROCESSED_DIR / VIZ_DATA_FILE
METRICS_PATH = OUTPUTS_DIR / "metrics.json"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="How It Works · NYC Congestion",
    page_icon="📖",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main > div { padding-top: 1rem; }
      .doc-hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        padding: 1.6rem 2rem; border-radius: 12px; border-left: 4px solid #3498db;
        margin-bottom: 1.25rem;
      }
      .doc-hero h1 { color: #3498db; margin: 0; font-size: 2rem; letter-spacing: -0.02em; }
      .doc-hero p  { color: #a0aec0; margin: 0.25rem 0 0; }
      .step-card {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 1.1rem 1.4rem; margin-bottom: 0.8rem;
      }
      .step-num {
        display: inline-block; width: 28px; height: 28px; line-height: 28px;
        text-align: center; border-radius: 50%; background: #f1c40f;
        color: #1a1f2e; font-weight: 800; margin-right: 0.6rem; font-size: 0.95rem;
      }
      .step-title { color: #f1f5f9; font-weight: 700; font-size: 1.05rem;
                    display: inline-block; vertical-align: middle; }
      .step-body  { color: #cbd5e1; font-size: 0.95rem; margin-top: 0.5rem;
                    line-height: 1.55; }
      .formula-box {
        background: #0f172a; border: 1px solid #1e293b; border-left: 3px solid #f1c40f;
        padding: 0.7rem 1rem; border-radius: 6px; margin: 0.5rem 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.92rem; color: #f1f5f9;
      }
      .feature-pill {
        display: inline-block; padding: 5px 12px; border-radius: 20px;
        font-family: ui-monospace, monospace; font-size: 0.82rem;
        background: #1e293b; color: #cbd5e1; border: 1px solid #334155;
        margin: 3px;
      }
      .axis-table {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px;
        padding: 0.4rem;
      }
      .callout {
        border-left: 3px solid #2ecc71; background: rgba(46,204,113,0.06);
        padding: 0.7rem 1rem; border-radius: 6px; margin: 0.5rem 0;
        color: #e2e8f0; font-size: 0.94rem;
      }
      .warn {
        border-left: 3px solid #f39c12; background: rgba(243,156,18,0.06);
        padding: 0.7rem 1rem; border-radius: 6px; margin: 0.5rem 0;
        color: #e2e8f0; font-size: 0.94rem;
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


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame | None:
    if VIZ_DATA_PATH.exists():
        return pd.read_parquet(VIZ_DATA_PATH)
    return None


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

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='doc-hero'>
      <h1>📖 How It Works</h1>
      <p>From 18 million raw taxi trips to a clustered view of New York City's congestion fingerprint —
      the design choices, the maths, and what every chart axis actually represents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Step 1: The problem ──────────────────────────────────────────────────────
st.markdown(
    """
    <div class='step-card'>
      <span class='step-num'>1</span>
      <span class='step-title'>The problem with raw average speed</span>
      <div class='step-body'>
        If you cluster taxi zones by their average speed, the slowest zones turn out to include
        both Times Square at 6 PM and a residential stretch in Queens with a 25-mph speed limit.
        Same number, very different stories. A model that can't tell those apart is useless for routing,
        planning, or operations. We need a feature that captures <i>congestion as event</i>, not <i>speed as geometry</i>.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Step 2: speed_deviation ─────────────────────────────────────────────────
st.markdown(
    """
    <div class='step-card'>
      <span class='step-num'>2</span>
      <span class='step-title'>The speed_deviation feature — the core idea</span>
      <div class='step-body'>
        For every <b>(zone, hour-of-day)</b> pair we compute the gap between the zone-hour's average speed
        and that zone's own historical baseline:
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class='formula-box'>
       Δ(z, h)  =  s(z, h)  −  baseline(z)<br>
       baseline(z)  =  mean over h of  s(z, h)
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class='callout'>
      A 25-mph school zone has a near-zero deviation all day — not flagged.
      Times Square at peak has Δ ≈ −7 mph — clearly flagged. The same partition would have called
      both \"slow\" if we'd used absolute speed.
    </div>
    """,
    unsafe_allow_html=True,
)

# Live histogram of speed_deviation if data is available
if df is not None and "speed_deviation" in df.columns:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["speed_deviation"],
        nbinsx=70,
        marker_color="#3498db",
        marker_line_color="rgba(255,255,255,0.15)",
        marker_line_width=1,
        opacity=0.85,
    ))
    fig.add_vline(x=0, line_color="#f1c40f", line_dash="dash", line_width=2,
                  annotation_text="zero deviation",
                  annotation_position="top right",
                  annotation_font=dict(color="#f1c40f"))
    fig.update_layout(
        title=dict(text="Live: distribution of speed_deviation across zone-hours",
                   font=dict(size=14)),
        xaxis=dict(title="speed_deviation (mph)  ·  negative = slower than baseline",
                   gridcolor=GRID, color=TEXT, zerolinecolor=GRID),
        yaxis=dict(title="Count of zone-hour observations",
                   gridcolor=GRID, color=TEXT),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=dict(color=TEXT),
        height=320, margin=dict(l=50, r=30, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Roughly symmetric, centred near zero — confirms the feature is well-conditioned. "
        "The negative tail is the congestion signal."
    )

# ── Step 3: feature engineering decisions ────────────────────────────────────
st.markdown(
    """
    <div class='step-card'>
      <span class='step-num'>3</span>
      <span class='step-title'>Feature engineering decisions that matter</span>
      <div class='step-body'>
        The clustering operates on <b>five</b> features per (zone, hour) row. Three of them are not obvious.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 1], gap="medium")
with c1:
    st.markdown(
        """
        <div class='step-card' style='margin:0;'>
          <div style='color:#f1c40f;font-weight:700;margin-bottom:0.4rem'>Cyclic time encoding</div>
          <div class='step-body' style='margin:0;'>
            Raw <code>hour_of_day</code> puts 23:00 and 00:00 as 23 hours apart under Euclidean distance.
            Replacing it with <code>(sin, cos)</code> on a unit circle makes them adjacent.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Visual: sin/cos circle
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    xs = np.cos(angles); ys = np.sin(angles)
    hours = list(range(24))
    fig = go.Figure()
    # connect adjacent hours
    fig.add_trace(go.Scatter(
        x=list(xs) + [xs[0]], y=list(ys) + [ys[0]],
        mode="lines", line=dict(color="#475569", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=14, color=hours, colorscale="Turbo",
                    line=dict(color="#fff", width=1)),
        text=[f"{h:02d}" for h in hours], textposition="top center",
        textfont=dict(color="#cbd5e1", size=10), hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Hour-of-day on the unit circle", font=dict(size=13)),
        xaxis=dict(title="hour_cos = cos(2πh/24)", range=[-1.4, 1.4],
                   zeroline=True, zerolinecolor=GRID, gridcolor=GRID, color=TEXT),
        yaxis=dict(title="hour_sin = sin(2πh/24)", range=[-1.4, 1.4],
                   zeroline=True, zerolinecolor=GRID, gridcolor=GRID, color=TEXT,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=dict(color=TEXT),
        height=320, margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown(
        """
        <div class='step-card' style='margin:0;'>
          <div style='color:#f1c40f;font-weight:700;margin-bottom:0.4rem'>log1p on trip_density</div>
          <div class='step-body' style='margin:0;'>
            Trip count per zone-hour is heavy-tailed — a few Manhattan zones get thousands per hour,
            most get single digits. A direct <code>log</code> would blow up at zero;
            <code>log1p(x) = log(1 + x)</code> handles it gracefully.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if df is not None and "trip_density" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["trip_density"], nbinsx=60, name="raw",
            marker_color="#e74c3c", opacity=0.6,
        ))
        fig.add_trace(go.Histogram(
            x=np.log1p(df["trip_density"]), nbinsx=60, name="log1p",
            marker_color="#2ecc71", opacity=0.7,
        ))
        fig.update_layout(
            title=dict(text="Raw trip_density vs log1p(trip_density)", font=dict(size=13)),
            xaxis=dict(title="value", gridcolor=GRID, color=TEXT),
            yaxis=dict(title="count", gridcolor=GRID, color=TEXT),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, font=dict(color=TEXT),
            barmode="overlay",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
            height=320, margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
    <div class='step-card' style='margin-top:0.6rem;'>
      <div style='color:#f1c40f;font-weight:700;margin-bottom:0.4rem'>Two features we deliberately drop</div>
      <div class='step-body'>
        <b>zone_id</b> — TLC location IDs are nominal categories. Treating them as continuous Euclidean
        coordinates would smuggle geographic nonsense into the distance metric. Zone 132 is not "between"
        zones 131 and 133 in any meaningful way.<br/><br/>
        <b>avg_speed_mph</b> — already implicit in <code>speed_deviation</code>. Including both means the
        scaler weights the speed dimension twice — multicollinearity by another name.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("**Final feature vector passed to K-Means:**", unsafe_allow_html=False)
st.markdown(
    "<span class='feature-pill'>hour_sin</span>"
    "<span class='feature-pill'>hour_cos</span>"
    "<span class='feature-pill'>is_weekend</span>"
    "<span class='feature-pill'>speed_deviation</span>"
    "<span class='feature-pill'>log_trip_density</span>",
    unsafe_allow_html=True,
)

# ── Step 4: choosing k ──────────────────────────────────────────────────────
st.markdown(
    """
    <div class='step-card'>
      <span class='step-num'>4</span>
      <span class='step-title'>Choosing k — four indicators, not one</span>
      <div class='step-body'>
        We sweep k ∈ {2…10}, refit K-Means at each value, and consult four independent criteria.
        Eyeballing the elbow is unreproducible; we use the <b>kneedle algorithm</b> (Satopää et al. 2011)
        with a deterministic second-derivative fallback.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
elbow_path = OUTPUTS_DIR / "elbow_curve.png"
modelsel_path = OUTPUTS_DIR / "model_selection.png"
cols = st.columns(2)
with cols[0]:
    if elbow_path.exists():
        st.image(str(elbow_path), caption="Elbow plot — x: k, y: inertia (within-cluster sum of squares).",
                 use_container_width=True)
    else:
        st.info("Elbow plot not yet generated. Run `make pipeline` first.")
with cols[1]:
    if modelsel_path.exists():
        st.image(str(modelsel_path), caption="Joint model-selection — silhouette and inertia versus k.",
                 use_container_width=True)
    else:
        st.info("Model-selection plot not yet generated.")

if metrics:
    chosen_k = metrics.get("k", "—")
    st.markdown(
        f"<div class='callout'>This run chose <b>k = {chosen_k}</b> as the operating point.</div>",
        unsafe_allow_html=True,
    )

# ── Step 5: stability ───────────────────────────────────────────────────────
st.markdown(
    """
    <div class='step-card'>
      <span class='step-num'>5</span>
      <span class='step-title'>Stability — the partition shouldn't depend on a lucky seed</span>
      <div class='step-body'>
        K-Means is sensitive to initialisation. We refit under five distinct random seeds and compute
        the <b>Adjusted Rand Index</b> pairwise. Plain Rand has a non-zero baseline; Adjusted Rand
        subtracts the chance agreement, so random labellings give ≈ 0 and a perfect match gives 1.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if metrics and "stability" in metrics:
    stab = metrics["stability"]
    c1, c2, c3 = st.columns(3)
    c1.metric("ARI · minimum", f"{stab.get('ari_min', 0):.3f}",
              delta="≥ 0.80 target" if stab.get("ari_min", 0) >= 0.80 else "below 0.80")
    c2.metric("ARI · mean", f"{stab.get('ari_mean', 0):.3f}",
              delta="≥ 0.90 target" if stab.get("ari_mean", 0) >= 0.90 else "below 0.90")
    c3.metric("Stability gate", "PASS" if stab.get("ari_pass") else "FAIL")

# ── Step 6: the pipeline ────────────────────────────────────────────────────
st.markdown(
    """
    <div class='step-card'>
      <span class='step-num'>6</span>
      <span class='step-title'>The five pipeline stages</span>
      <div class='step-body'>
        Each stage writes Parquet to disk, so re-running a later stage doesn't repeat the expensive
        early ones.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
stages = [
    ("Stage 1 · Data loader",
     "Reads raw TLC parquet, normalises column names across schema versions."),
    ("Stage 2 · Preprocessing",
     "Eight sequential filters: timestamp ordering, distance/duration/speed bounds, "
     "exact-duplicate removal, wrong-month stragglers, same-zone-near-zero trips, "
     "and zone-ID range validation. Optional per-zone IQR fence on speed."),
    ("Stage 3 · Feature engineering",
     "Computes speed_mph per trip, aggregates trip-level data to (zone, hour-of-day), "
     "derives speed_deviation, cyclic time, log_trip_density."),
    ("Stage 4 · Clustering",
     "Sweeps k, picks via kneedle, fits K-Means, reports silhouette + Davies-Bouldin + "
     "Calinski-Harabasz + ARI stability. Optional DBSCAN for hotspot detection."),
    ("Stage 5 · Visualisation precompute",
     "Produces the parquet the dashboard reads, plus 12 static PNG diagnostics."),
]
for title, body in stages:
    st.markdown(
        f"""
        <div class='step-card' style='padding: 0.7rem 1rem; margin-bottom: 0.4rem;'>
          <div style='color:#f1c40f;font-weight:700'>{title}</div>
          <div style='color:#cbd5e1;font-size:0.9rem;margin-top:0.25rem'>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Step 7: chart axes reference ────────────────────────────────────────────
st.markdown(
    """
    <div class='step-card'>
      <span class='step-num'>7</span>
      <span class='step-title'>What every chart axis means — viva quick-reference</span>
      <div class='step-body'>
        If a question is "what's on the x-axis of …", these are the answers.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
axes = pd.DataFrame([
    ["Elbow plot", "k (number of clusters)", "Inertia (within-cluster sum of squares)"],
    ["Silhouette plot", "Silhouette coefficient ∈ [−1, 1]", "Observations sorted by cluster, then by silhouette"],
    ["Model selection (joint)", "k", "Inertia (left axis) · Silhouette (right axis)"],
    ["PCA projection", "PC1 (first principal component)", "PC2 (second principal component)"],
    ["Cluster profile", "Feature name (categorical)", "Standardised feature value (z-score)"],
    ["Hourly trend", "Hour of day (0–23)", "Avg speed (mph) · Trip density (right axis)"],
    ["Zone × hour heatmap", "Hour of day (0–23)", "Zone ID (top-N by activity); colour = avg level"],
    ["Day × hour heatmap", "Hour of day (0–23)", "Day of week (Mon–Sun); colour = avg level"],
    ["Speed deviation hist.", "speed_deviation (mph)", "Count of zone-hour observations"],
    ["Top-N zones bar", "Avg congestion level", "Zone ID (sorted descending)"],
    ["Cluster distribution", "Congestion level (cluster index)", "Count of zone-hour observations"],
    ["Map view", "Longitude (−74.25 to −73.70)", "Latitude (40.50 to 40.92)"],
], columns=["Chart", "x-axis", "y-axis"])
st.dataframe(axes, use_container_width=True, hide_index=True)

# ── Step 8: limitations ─────────────────────────────────────────────────────
st.markdown(
    """
    <div class='step-card'>
      <span class='step-num'>8</span>
      <span class='step-title'>Limitations we don't pretend away</span>
      <div class='step-body'>
        Yellow-cab data is a sample, not the population — outer boroughs underrepresented because
        Uber/Lyft and private cars dominate there. K-Means assumes spherical clusters; some of ours
        aren't quite. The 263-zone TLC partition averages over varying conditions inside each zone.
        And the historical baseline is computed on the same window we cluster — a mild form of
        in-sample leakage that's numerically negligible at this sample size but worth flagging.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='warn'>This pipeline is descriptive. It identifies <b>where</b> and <b>when</b> "
    "congestion happens. <b>Why</b> requires causal inference and is out of scope.</div>",
    unsafe_allow_html=True,
)

st.divider()
st.caption(
    "References: Lloyd 1982 (K-Means) · Rousseeuw 1987 (silhouette) · "
    "Davies & Bouldin 1979 · Caliński & Harabasz 1974 · Hubert & Arabie 1985 (ARI) · "
    "Satopää et al. 2011 (kneedle) · Ester et al. 1996 (DBSCAN). "
    "Data: NYC TLC Yellow-Taxi Trip Records 2025."
)
