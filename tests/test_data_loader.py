"""Tests for dataset loading and validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import data_loader
from tests.conftest import synthetic_nsl_kdd, requires_dataset


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(header=False, index=False).encode("utf-8")


def test_validate_dataframe_maps_labels(synthetic_df):
    df, report = data_loader.validate_dataframe(synthetic_df.copy())
    assert report.is_valid
    assert config.TARGET_COLUMN in df.columns
    assert set(df[config.TARGET_COLUMN].unique()) <= set(
        config.ATTACK_CATEGORIES)
    assert report.duplicate_rows >= 0
    assert report.label_distribution


def test_validate_reports_unknown_labels(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, config.LABEL_COLUMN] = "made_up_attack"
    _, report = data_loader.validate_dataframe(df)
    assert "made_up_attack" in report.unknown_labels
    assert report.warnings  # explicitly warned, not silently dropped


def test_validate_reports_malformed_rows(synthetic_df):
    df = synthetic_df.copy()
    df.loc[5, config.FEATURE_NAMES] = np.nan
    df2, report = data_loader.validate_dataframe(df)
    assert report.malformed_rows == 1
    assert len(df2) == len(df) - 1


def test_validate_reports_invalid_numeric(synthetic_df):
    df = synthetic_df.copy()
    df["src_bytes"] = df["src_bytes"].astype(object)  # pandas 3 is strict
    df.loc[3, "src_bytes"] = "not_a_number"
    _, report = data_loader.validate_dataframe(df)
    assert report.invalid_numeric_cells.get("src_bytes", 0) >= 1


def test_upload_rejects_bad_extension(synthetic_df):
    df, report = data_loader.load_uploaded_file(
        _to_csv_bytes(synthetic_df), "traffic.exe")
    assert df is None
    assert any("Unsupported file type" in e for e in report.errors)


def test_upload_rejects_oversize(synthetic_df):
    big = b"0," * (config.MAX_UPLOAD_BYTES // 2 + 10)
    df, report = data_loader.load_uploaded_file(big, "big.csv")
    assert df is None
    assert any("too large" in e for e in report.errors)


def test_upload_rejects_wrong_column_count():
    df, report = data_loader.load_uploaded_file(
        b"1,2,3\n4,5,6\n", "tiny.csv")
    assert df is None
    assert any("columns" in e for e in report.errors)


def test_upload_accepts_nsl_kdd_layout(synthetic_df):
    df, report = data_loader.load_uploaded_file(
        _to_csv_bytes(synthetic_df), "traffic.csv")
    assert df is not None
    assert report.is_valid
    assert config.TARGET_COLUMN in df.columns


def test_upload_accepts_header_layout(synthetic_df):
    content = synthetic_df.to_csv(index=False).encode("utf-8")
    df, report = data_loader.load_uploaded_file(content, "traffic.csv")
    assert df is not None
    assert report.is_valid


def test_upload_rejects_empty():
    df, report = data_loader.load_uploaded_file(b"   ", "empty.csv")
    assert df is None
    assert any("empty" in e for e in report.errors)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download"):
        data_loader.load_nsl_kdd_file(tmp_path / "nope.txt")


@requires_dataset
def test_load_real_training_file():
    df, report = data_loader.load_nsl_kdd_file(config.TRAIN_FILE)
    assert len(df) > 100_000
    assert report.is_valid
    assert "normal" in df[config.LABEL_COLUMN].values


@requires_dataset
def test_mapping_covers_all_dataset_labels():
    """Every attack label in both NSL-KDD files must be in the documented
    mapping - no label may silently become 'Unknown'."""
    for path in (config.TRAIN_FILE, config.TEST_FILE):
        df, report = data_loader.load_nsl_kdd_file(path)
        assert report.unknown_labels == [], (
            f"{path.name}: labels missing from ATTACK_CATEGORY_MAP: "
            f"{report.unknown_labels}")
        assert "Unknown" not in set(df[config.TARGET_COLUMN])
