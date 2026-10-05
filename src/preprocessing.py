"""Feature preprocessing for NSL-KDD data.

Fits on training data only (no leakage): one-hot encoding of categorical
features (protocol_type, service, flag) and median imputation of numeric
NaNs. The fitted state is persisted with joblib plus a human-readable JSON
configuration for reproducibility.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config


@dataclass
class Preprocessor:
    """Fit/transform encoder producing a dense numeric feature matrix."""

    categorical_values: dict = field(default_factory=dict)   # col -> categories
    numeric_medians: dict = field(default_factory=dict)      # col -> median
    feature_names_: list = field(default_factory=list)       # encoded names
    fitted_: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self._check_columns(df)
        self.categorical_values = {
            col: sorted(df[col].astype(str).unique().tolist())
            for col in config.CATEGORICAL_FEATURES
        }
        numeric = [
            f for f in config.FEATURE_NAMES
            if f not in config.CATEGORICAL_FEATURES
        ]
        self.numeric_medians = {
            col: float(pd.to_numeric(df[col], errors="coerce").median())
            for col in numeric
        }
        names: list[str] = []
        for col in config.FEATURE_NAMES:
            if col in config.CATEGORICAL_FEATURES:
                names += [f"{col}={v}" for v in self.categorical_values[col]]
            else:
                names.append(col)
        self.feature_names_ = names
        self.fitted_ = True
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a dataframe to a float matrix.

        Unseen categorical values become all-zero one-hot rows (logged via
        the returned data itself, not silently altered numerics). NaN
        numerics are imputed with training medians.
        """
        if not self.fitted_:
            raise RuntimeError("Preprocessor has not been fitted.")
        self._check_columns(df)
        blocks = []
        for col in config.FEATURE_NAMES:
            if col in config.CATEGORICAL_FEATURES:
                cats = self.categorical_values[col]
                values = df[col].astype(str)
                mat = np.zeros((len(df), len(cats)), dtype=np.float32)
                index = {c: i for i, c in enumerate(cats)}
                for row_i, v in enumerate(values):
                    j = index.get(v)
                    if j is not None:
                        mat[row_i, j] = 1.0
                blocks.append(mat)
            else:
                series = pd.to_numeric(df[col], errors="coerce")
                series = series.fillna(self.numeric_medians[col])
                blocks.append(series.to_numpy(dtype=np.float32).reshape(-1, 1))
        return np.hstack(blocks)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # ----------------------------------------------------------------- save
    def save(self, path: str | Path = config.MODEL_FILES["preprocessing"],
             json_path: str | Path | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "categorical_values": self.categorical_values,
                "numeric_medians": self.numeric_medians,
                "feature_names": self.feature_names_,
            },
            path,
        )
        json_path = json_path or path.with_suffix(".json")
        Path(json_path).write_text(
            json.dumps(
                {
                    "dataset": config.DATASET_NAME,
                    "random_seed": config.RANDOM_SEED,
                    "categorical_values": self.categorical_values,
                    "numeric_medians": self.numeric_medians,
                    "n_encoded_features": len(self.feature_names_),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path = config.MODEL_FILES["preprocessing"]
             ) -> "Preprocessor":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessing artefact not found at {path}. Run "
                "'python train.py' first."
            )
        blob = joblib.load(path)
        pre = cls()
        pre.categorical_values = blob["categorical_values"]
        pre.numeric_medians = blob["numeric_medians"]
        pre.feature_names_ = blob["feature_names"]
        pre.fitted_ = True
        return pre

    # -------------------------------------------------------------- helpers
    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in config.FEATURE_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"Dataframe missing required columns: {missing}")
