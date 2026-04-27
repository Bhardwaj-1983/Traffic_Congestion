"""
scripts/make_centroids.py
-------------------------
Generate ``data/raw/taxi_zone_centroids.csv`` from the official NYC TLC
Taxi Zone shapefile.

This file is consumed by the Streamlit app (app/app.py) to place each
zone's congestion level on the map.

The shapefile is publicly hosted at:
    https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip

Usage:
    python scripts/make_centroids.py
    python scripts/make_centroids.py --shapefile path/to/taxi_zones.shp
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR, ZONE_CENTROIDS_FILE  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("make_centroids")

SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def _download_and_extract_shapefile(dest_dir: Path) -> Path:
    """Download the TLC zone shapefile zip and extract to ``dest_dir``.
    Returns the path to the .shp file."""
    logger.info(f"Downloading {SHAPEFILE_URL} …")
    r = requests.get(SHAPEFILE_URL, timeout=60)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest_dir)
    # Recursive search — the TLC zip now nests files inside a subfolder.
    shp_candidates = list(dest_dir.rglob("*.shp"))
    if not shp_candidates:
        raise FileNotFoundError(
            f"No .shp file found inside extracted zip at {dest_dir}"
        )
    # Prefer a file named like taxi_zones.shp if multiple are present.
    shp_candidates.sort(key=lambda p: (0 if "taxi_zone" in p.stem.lower() else 1, len(p.parts)))
    return shp_candidates[0]


def generate_centroids(shp_path: Path, out_csv: Path) -> None:
    """Read the shapefile, project to WGS84, compute centroids, write CSV."""
    try:
        import geopandas as gpd
    except ImportError as e:
        raise SystemExit(
            "geopandas is required. Install with: pip install geopandas shapely"
        ) from e

    logger.info(f"Reading shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        raise ValueError(
            "Shapefile is missing a CRS — cannot project to lat/lon."
        )

    # Project to WGS84 (EPSG:4326) so centroids are in lon/lat.
    # Centroids are computed on the original projection first for geographic
    # correctness, then each centroid is individually re-projected.
    centroids_proj = gdf.geometry.centroid
    centroids_wgs84 = centroids_proj.to_crs("EPSG:4326")

    zone_id_col = next(
        (c for c in ["LocationID", "locationid", "OBJECTID"] if c in gdf.columns),
        None,
    )
    if zone_id_col is None:
        raise KeyError(
            f"Could not find LocationID column in {list(gdf.columns)}"
        )

    out = gdf[[zone_id_col]].copy()
    out = out.rename(columns={zone_id_col: "zone_id"})
    out["longitude"] = centroids_wgs84.x.values
    out["latitude"] = centroids_wgs84.y.values
    out = out.sort_values("zone_id").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info(f"✓ Wrote {len(out):,} centroids → {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--shapefile",
        type=Path,
        default=None,
        help="Local path to taxi_zones.shp (skip download).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=RAW_DIR / ZONE_CENTROIDS_FILE,
        help=f"Output CSV path (default: {RAW_DIR / ZONE_CENTROIDS_FILE})",
    )
    args = p.parse_args()

    if args.shapefile is not None:
        shp_path = args.shapefile
        if not shp_path.exists():
            p.error(f"Shapefile not found: {shp_path}")
        generate_centroids(shp_path, args.out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            shp_path = _download_and_extract_shapefile(Path(tmp))
            generate_centroids(shp_path, args.out)


if __name__ == "__main__":
    main()
