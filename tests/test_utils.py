"""Tests for src/utils.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import utils


def test_parse_year_month_extracts_stamp():
    assert utils.parse_year_month("yellow_tripdata_2023-01.parquet") == "2023-01"
    assert utils.parse_year_month("cleaned_trips_2022-12.parquet") == "2022-12"


def test_parse_year_month_raises_on_bad_filename():
    with pytest.raises(ValueError):
        utils.parse_year_month("no-year-month-here.parquet")


def test_read_parquet_safe_raises_on_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        utils.read_parquet_safe(missing, label="test")


def test_write_and_read_parquet_roundtrip(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = tmp_path / "nested" / "dir" / "sample.parquet"
    utils.write_parquet(df, out)
    assert out.exists()
    recovered = utils.read_parquet_safe(out)
    pd.testing.assert_frame_equal(recovered, df)


def test_write_and_read_json_roundtrip(tmp_path: Path):
    payload = {"silhouette_score": 0.62, "k": 3, "levels": [0, 1, 2]}
    out = tmp_path / "metrics.json"
    utils.write_json(payload, out)
    recovered = utils.read_json(out)
    assert recovered == payload


def test_timeit_decorator_passes_through_return_value():
    @utils.timeit("double")
    def double(x: int) -> int:
        return x * 2

    assert double(3) == 6
