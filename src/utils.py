"""
utils.py
--------
Shared utilities used across the pipeline modules:
  - Consistent logger configuration
  - Timing decorator for stage instrumentation
  - Parquet I/O helpers with defensive error messaging
  - Zone-month label parsing from filenames
"""

from __future__ import annotations

import functools
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

# ── Logger ────────────────────────────────────────────────────────────────────

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a module-scoped logger with a consistent formatter.

    Safe to call repeatedly — handlers are only attached once per process.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FMT, _LOG_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


# ── Timing decorator ──────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])


def timeit(label: str | None = None) -> Callable[[F], F]:
    """
    Decorator that logs the wall-clock duration of a function.

    Example:
        @timeit("clean_dataframe")
        def clean_dataframe(...): ...
    """

    def decorator(fn: F) -> F:
        tag = label or fn.__name__
        log = get_logger(fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                dt = time.perf_counter() - t0
                log.info(f"⏱  {tag}: {dt:.2f}s")

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Parquet I/O helpers ───────────────────────────────────────────────────────

def read_parquet_safe(path: Path, label: str | None = None) -> pd.DataFrame:
    """Read a parquet file with a descriptive error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required parquet file not found: {path}"
            + (f" (expected from stage: {label})" if label else "")
        )
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


# ── Month-label parsing ───────────────────────────────────────────────────────

_YM_RE = re.compile(r"(\d{4}-\d{2})")


def parse_year_month(filename: str | Path) -> str:
    """
    Extract a 'YYYY-MM' stamp from a filename such as
    ``yellow_tripdata_2023-01.parquet``. Raises ValueError on failure.
    """
    stem = Path(filename).stem
    m = _YM_RE.search(stem)
    if not m:
        raise ValueError(f"Could not parse YYYY-MM stamp from '{filename}'")
    return m.group(1)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def write_json(obj: dict[str, Any], path: Path) -> None:
    """Serialize ``obj`` to JSON at ``path`` (pretty-printed, UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path) -> dict[str, Any]:
    """Load JSON from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
