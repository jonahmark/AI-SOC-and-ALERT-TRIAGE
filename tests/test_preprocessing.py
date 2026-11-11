"""Tests for the preprocessing pipeline."""
from __future__ import annotations

import numpy as np
import pytest

import config
from src.preprocessing import Preprocessor


def test_fit_transform_shape(synthetic_df):
    pre = Preprocessor().fit(synthetic_df)
    X = pre.transform(synthetic_df)
    n_expected = (
        len(config.FEATURE_NAMES) - len(config.CATEGORICAL_FEATURES)
        + sum(len(pre.categorical_values[c])
              for c in config.CATEGORICAL_FEATURES)
    )
    assert X.shape == (len(synthetic_df), n_expected)
    assert len(pre.feature_names_) == X.shape[1]
    assert np.isfinite(X).all()


def test_onehot_encoding_values(synthetic_df):
    pre = Preprocessor().fit(synthetic_df)
    X = pre.transform(synthetic_df)
    # each categorical block sums to <= 1 per row
    offset = 0
    for col in config.FEATURE_NAMES:
        if col in config.CATEGORICAL_FEATURES:
            width = len(pre.categorical_values[col])
            block = X[:, offset:offset + width]
            assert set(np.unique(block.sum(axis=1))) <= {0.0, 1.0}
            offset += width
        else:
            offset += 1
    assert offset == X.shape[1]


def test_unseen_category_becomes_zero_row(synthetic_df):
    pre = Preprocessor().fit(synthetic_df)
    df2 = synthetic_df.head(5).copy()
    df2.loc[df2.index[0], "protocol_type"] = "carrier_pigeon"
    X = pre.transform(df2)
    width = len(pre.categorical_values["protocol_type"])
    # protocol_type is the 2nd feature (index 1) -> block starts at col 1
    start = config.FEATURE_NAMES.index("protocol_type")
    assert X[0, start:start + width].sum() == 0.0


def test_nan_numeric_imputed_with_median(synthetic_df):
    df = synthetic_df.copy()
    df.loc[df.index[0], "src_bytes"] = np.nan
    pre = Preprocessor().fit(synthetic_df)
    X = pre.transform(df)
    col = pre.feature_names_.index("src_bytes")
    assert X[0, col] == pytest.approx(pre.numeric_medians["src_bytes"])


def test_transform_before_fit_raises(synthetic_df):
    with pytest.raises(RuntimeError, match="not been fitted"):
        Preprocessor().transform(synthetic_df)


def test_missing_columns_raise(synthetic_df):
    pre = Preprocessor().fit(synthetic_df)
    with pytest.raises(ValueError, match="missing required columns"):
        pre.transform(synthetic_df.drop(columns=["duration"]))


def test_save_load_roundtrip(synthetic_df, tmp_path):
    pre = Preprocessor().fit(synthetic_df)
    path = tmp_path / "preprocessing.pkl"
    pre.save(path, json_path=tmp_path / "preprocessing.json")
    loaded = Preprocessor.load(path)
    X1 = pre.transform(synthetic_df)
    X2 = loaded.transform(synthetic_df)
    assert np.array_equal(X1, X2)
    assert (tmp_path / "preprocessing.json").exists()


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        Preprocessor.load(tmp_path / "absent.pkl")


def test_no_leakage_fit_only_on_train(synthetic_df):
    """Categories present only in the test split must not appear in config."""
    train = synthetic_df[synthetic_df[config.LABEL_COLUMN] == "normal"]
    pre = Preprocessor().fit(train)
    assert "neptune" not in str(pre.categorical_values)
    X = pre.transform(synthetic_df)  # still transforms unseen rows
    assert np.isfinite(X).all()
