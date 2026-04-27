# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Exploratory Data Analysis: NYC Taxi Traffic Congestion
#
# **Objective**: Understand the raw data distribution, validate cleaning rules,
# and explore potential clustering features before running the ML pipeline.
#
# **Dataset**: NYC Yellow Taxi Trip Records (6 months, Jan–Jun)
#
# **Sections**:
# 1. Environment Setup
# 2. Data Loading
# 3. Univariate Analysis
# 4. Temporal Patterns
# 5. Spatial Patterns (Zone-level)
# 6. Speed & Congestion Features
# 7. Feature Correlations
# 8. Preliminary Clustering Signals

# %% [markdown]
# ## 1. Environment Setup

# %%
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Plotting style
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
sns.set_palette("tab10")

print("Python:", sys.version)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)

# %%
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# %% [markdown]
# ## 2. Data Loading
#
# Load all cleaned monthly files (output of `data_loader.py` + `preprocessing.py`).

# %%
files = sorted(glob.glob(str(PROCESSED_DIR / "cleaned_trips_*.parquet")))
print(f"Found {len(files)} monthly files")

dfs = []
for f in files:
    ym = Path(f).stem.replace("cleaned_trips_", "")
    df = pd.read_parquet(f)
    df["month_label"] = ym
    dfs.append(df)
    print(f"  {ym}: {len(df):,} rows")

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows: {len(df_all):,}")
df_all.info()

# %%
df_all.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

# %% [markdown]
# ## 3. Univariate Analysis

# %% [markdown]
# ### 3.1 Trip Distance Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df_all["trip_distance"].clip(0, 30), bins=80, color="#3498db", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("Trip Distance (miles)")
axes[0].set_ylabel("Count")
axes[0].set_title("Trip Distance Distribution (clipped at 30 mi)")

axes[1].hist(np.log1p(df_all["trip_distance"]), bins=80, color="#e67e22", edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("log(1 + Trip Distance)")
axes[1].set_title("Trip Distance (log scale)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 Trip Duration Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

duration_min = df_all["trip_duration"] / 60
axes[0].hist(duration_min.clip(0, 90), bins=80, color="#9b59b6", edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("Trip Duration (minutes)")
axes[0].set_ylabel("Count")
axes[0].set_title("Trip Duration Distribution (clipped at 90 min)")

axes[1].boxplot(duration_min[duration_min < 120], vert=True, patch_artist=True,
                boxprops=dict(facecolor="#9b59b6", alpha=0.6))
axes[1].set_ylabel("Duration (minutes)")
axes[1].set_title("Duration Boxplot (< 120 min)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Computed Speed Distribution

# %%
speed = df_all["trip_distance"] / (df_all["trip_duration"] / 3600)
speed = speed.clip(0, 100)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(speed, bins=100, color="#1abc9c", edgecolor="white", linewidth=0.3)
ax.set_xlabel("Speed (mph)")
ax.set_ylabel("Count")
ax.set_title("Computed Trip Speed Distribution")
ax.axvline(speed.median(), color="red", linestyle="--", label=f"Median: {speed.median():.1f} mph")
ax.axvline(speed.mean(), color="orange", linestyle="--", label=f"Mean: {speed.mean():.1f} mph")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Speed stats:\n{speed.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99])}")

# %% [markdown]
# ## 4. Temporal Patterns

# %%
# Parse time features if not already present
if "hour_of_day" not in df_all.columns:
    df_all["pickup_datetime"] = pd.to_datetime(df_all["pickup_datetime"])
    df_all["hour_of_day"] = df_all["pickup_datetime"].dt.hour
    df_all["day_of_week"] = df_all["pickup_datetime"].dt.dayofweek
    df_all["is_weekend"] = df_all["day_of_week"] >= 5

# %% [markdown]
# ### 4.1 Trip Volume by Hour of Day

# %%
hourly_counts = df_all.groupby("hour_of_day").size().reset_index(name="trip_count")

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(hourly_counts["hour_of_day"], hourly_counts["trip_count"] / 1000,
       color="#3498db", edgecolor="white")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Trip Count (thousands)")
ax.set_title("Trip Volume by Hour of Day")
ax.set_xticks(range(24))
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Average Speed by Hour of Day (Weekday vs Weekend)

# %%
df_all["speed_mph"] = (df_all["trip_distance"] / (df_all["trip_duration"] / 3600)).clip(1, 100)

hourly_speed = df_all.groupby(["hour_of_day", "is_weekend"])["speed_mph"].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
for is_wknd, label, color in [(False, "Weekday", "#3498db"), (True, "Weekend", "#e74c3c")]:
    sub = hourly_speed[hourly_speed["is_weekend"] == is_wknd]
    ax.plot(sub["hour_of_day"], sub["speed_mph"], label=label, color=color,
            linewidth=2.5, marker="o", markersize=5)

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Speed (mph)")
ax.set_title("Average Speed by Hour: Weekday vs Weekend")
ax.set_xticks(range(24))
ax.legend()
ax.axhspan(0, 10, alpha=0.05, color="red", label="Congestion Zone")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 Congestion Heatmap: Day × Hour

# %%
day_hour_speed = df_all.groupby(["day_of_week", "hour_of_day"])["speed_mph"].mean().unstack()
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day_hour_speed.index = [day_names[i] for i in day_hour_speed.index]

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(day_hour_speed, cmap="RdYlGn", ax=ax, linewidths=0.2,
            cbar_kws={"label": "Avg Speed (mph)"})
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day of Week")
ax.set_title("Speed Heatmap: Day × Hour (Green=Fast, Red=Slow)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Spatial Patterns (Zone-level)

# %%
# ── Top 20 zones by trip volume ──────────────────────────────────────────────
zone_trips = df_all.groupby("PULocationID").size().sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
zone_trips.plot(kind="barh", ax=ax, color="#3498db")
ax.set_xlabel("Trip Count")
ax.set_ylabel("Zone ID")
ax.set_title("Top 20 Pickup Zones by Trip Volume")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
plt.tight_layout()
plt.show()

# %%
# ── Top 20 zones by average speed ────────────────────────────────────────────
zone_speed = df_all.groupby("PULocationID")["speed_mph"].mean().sort_values(ascending=True).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(zone_speed)))
zone_speed.plot(kind="barh", ax=ax, color=colors[::-1])
ax.set_xlabel("Avg Speed (mph)")
ax.set_ylabel("Zone ID")
ax.set_title("20 Slowest Zones by Average Speed (Congestion Candidates)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Speed Deviation Feature Analysis
#
# The `speed_deviation` is the **core innovation** of this project.
# Here we verify its construction and distribution.

# %%
# Load aggregated zone-hour data (if feature_engineering.py has been run)
agg_path = PROCESSED_DIR / "aggregated_zone_hour.parquet"

if agg_path.exists():
    agg = pd.read_parquet(agg_path)
    print(f"Aggregated data shape: {agg.shape}")
    print(agg[["avg_speed_mph", "trip_density", "speed_deviation"]].describe())
else:
    print("aggregated_zone_hour.parquet not found. Run feature_engineering.py first.")
    print("Using on-the-fly computation for illustration…")

    # On-the-fly computation for illustration
    df_all["zone_id"] = df_all["PULocationID"]
    agg = (
        df_all.groupby(["zone_id", "hour_of_day", "day_of_week"])
        .agg(avg_speed_mph=("speed_mph", "mean"), trip_density=("speed_mph", "count"))
        .reset_index()
    )
    baseline = agg.groupby(["zone_id", "hour_of_day"])["avg_speed_mph"].mean().rename("baseline")
    agg = agg.merge(baseline.reset_index(), on=["zone_id", "hour_of_day"], how="left")
    agg["speed_deviation"] = agg["avg_speed_mph"] - agg["baseline"]

# %%
# Distribution of speed_deviation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(agg["speed_deviation"].clip(-20, 20), bins=80, color="#9b59b6", edgecolor="white", linewidth=0.3)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Baseline (0)")
axes[0].set_xlabel("Speed Deviation (mph)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Speed Deviation")
axes[0].legend()

axes[1].scatter(agg["trip_density"].clip(0, 500), agg["speed_deviation"].clip(-20, 20),
                alpha=0.15, s=5, color="#e74c3c")
axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Trip Density")
axes[1].set_ylabel("Speed Deviation (mph)")
axes[1].set_title("Trip Density vs Speed Deviation\n(Core Congestion Signal)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretation
#
# - **Negative** `speed_deviation` = the zone is moving **slower than its historical baseline** → genuine congestion signal
# - **Positive** `speed_deviation` = the zone is moving **faster than usual** → light traffic
# - High trip density + high negative deviation = **classic rush-hour congestion**

# %% [markdown]
# ## 7. Feature Correlations

# %%
feature_cols = ["avg_speed_mph", "trip_density", "speed_deviation", "hour_of_day"]
available = [c for c in feature_cols if c in agg.columns]

corr = agg[available].corr()

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Preliminary Clustering Signals
#
# Quick K-Means with k=3 on a sample to visualize cluster separability
# before running the full pipeline.

# %%
FEATURES = ["avg_speed_mph", "trip_density", "speed_deviation"]
sample = agg[FEATURES].dropna().sample(min(20_000, len(agg)), random_state=42)

scaler = StandardScaler()
X_sample = scaler.fit_transform(sample)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X_sample)
sil = silhouette_score(X_sample, labels, sample_size=5000)
print(f"Preliminary Silhouette Score (k=3, sample): {sil:.4f}")

# %%
# Sort clusters by speed (low speed = high congestion)
sample_df = sample.copy()
sample_df["cluster"] = labels
speed_by_cluster = sample_df.groupby("cluster")["avg_speed_mph"].mean().sort_values()
label_map = {c: i for i, c in enumerate(speed_by_cluster.index[::-1])}  # 2=high congestion
sample_df["congestion"] = sample_df["cluster"].map(label_map)

COLOR_MAP = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for level in [0, 1, 2]:
    sub = sample_df[sample_df["congestion"] == level]
    axes[0].scatter(sub["trip_density"].clip(0, 300), sub["avg_speed_mph"],
                    alpha=0.3, s=8, color=COLOR_MAP[level], label=LABEL_MAP[level])

axes[0].set_xlabel("Trip Density")
axes[0].set_ylabel("Avg Speed (mph)")
axes[0].set_title("Clusters: Density vs Speed")
axes[0].legend(title="Congestion")

for level in [0, 1, 2]:
    sub = sample_df[sample_df["congestion"] == level]
    axes[1].scatter(sub["speed_deviation"].clip(-15, 15), sub["avg_speed_mph"],
                    alpha=0.3, s=8, color=COLOR_MAP[level], label=LABEL_MAP[level])

axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Speed Deviation (mph)")
axes[1].set_ylabel("Avg Speed (mph)")
axes[1].set_title("Clusters: Speed Deviation vs Avg Speed")
axes[1].legend(title="Congestion")

plt.tight_layout()
plt.show()

# %%
# Cluster summary statistics
summary = sample_df.groupby("congestion")[FEATURES].agg(["mean", "std"]).round(3)
summary.index = [LABEL_MAP[i] for i in summary.index]
print("Cluster Summary Statistics:")
print(summary)

# %% [markdown]
# ## Summary
#
# | Finding | Detail |
# |---|---|
# | Peak congestion hours | Typically 7–10 AM and 4–8 PM |
# | Most congested zones | Midtown Manhattan (Times Square, Penn Station area) |
# | Weekend vs Weekday | Weekday AM/PM peaks; weekend late-night activity |
# | Speed deviation | Effective discriminator between natural slow zones and congestion |
# | Preliminary Silhouette | Check printed above — should be > 0.5 |
#
# **Next step**: Run the full pipeline (`data_loader → preprocessing → feature_engineering → clustering → visualization → app`).
