"""
scripts/download_data.py
------------------------
Download the NYC TLC Yellow Taxi Trip Record monthly parquet files into
``data/raw/``. Defaults to the first six months of the most recent
year for which the TLC has finished publishing (2023 at the time of
writing), matching PRD §5.1 (six-month scope).

Example usages
--------------
    # Default: 2023-01 through 2023-06
    python scripts/download_data.py

    # Custom year + month range
    python scripts/download_data.py --year 2022 --start 7 --end 12

    # Green taxi instead of yellow
    python scripts/download_data.py --cab green

The TLC publishes URLs like:
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Allow this script to run both as `python scripts/download_data.py` and
# `python -m scripts.download_data`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DIR  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("download_data")

TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def _url_for(cab: str, year: int, month: int) -> str:
    fname = f"{cab}_tripdata_{year}-{month:02d}.parquet"
    return f"{TLC_BASE}/{fname}"


def _download_one(url: str, dest: Path, overwrite: bool = False) -> bool:
    """Stream-download a single file to ``dest``. Returns True on success."""
    if dest.exists() and not overwrite:
        logger.info(f"✓ {dest.name} already present — skipping")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=dest.name, ncols=80,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        tmp.rename(dest)
        logger.info(f"✓ Saved {dest.name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed {dest.name}: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--year", type=int, default=2023, help="Trip-record year (default 2023)")
    p.add_argument("--start", type=int, default=1, help="Starting month (1-12, default 1)")
    p.add_argument("--end", type=int, default=6, help="Ending month inclusive (default 6)")
    p.add_argument("--cab", choices=["yellow", "green", "fhv", "fhvhv"], default="yellow",
                   help="Cab type (default yellow)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download files that already exist.")
    args = p.parse_args()

    if not (1 <= args.start <= 12 and 1 <= args.end <= 12 and args.start <= args.end):
        p.error("--start and --end must satisfy 1 ≤ start ≤ end ≤ 12")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Downloading {args.cab} trip data for {args.year}-"
        f"{args.start:02d}..{args.year}-{args.end:02d} → {RAW_DIR}"
    )

    failures = 0
    for month in range(args.start, args.end + 1):
        url = _url_for(args.cab, args.year, month)
        dest = RAW_DIR / f"{args.cab}_tripdata_{args.year}-{month:02d}.parquet"
        ok = _download_one(url, dest, overwrite=args.overwrite)
        if not ok:
            failures += 1

    logger.info(
        f"Done. Requested {args.end - args.start + 1} files, failures: {failures}."
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
