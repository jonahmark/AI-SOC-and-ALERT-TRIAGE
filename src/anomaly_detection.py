"""Stage A: unsupervised anomaly detection with Isolation Forest.

Trained on NORMAL training traffic only. Produces a continuous anomaly
score (IsolationForest.score_samples; more negative = more anomalous) and
a binary Normal/Anomaly decision against a configurable threshold.

NOTE: an anomaly decision is NOT an attack classification - Stage B
(attack_classification.py) makes that judgement separately.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config


@dataclass
class AnomalyResult:
    scores: np.ndarray            # score_samples(): lower = more anomalous
    is_anomaly: np.ndarray        # bool
    threshold: float


class AnomalyDetector:
    """Thin wrapper around IsolationForest with an explicit threshold."""

    def __init__(self, threshold: float = config.ANOMALY_THRESHOLD, **params):
        self.threshold = threshold
        self.model = IsolationForest(**(params or config.ISOLATION_FOREST_PARAMS))
        self.train_time_seconds: float | None = None
        self.n_train_samples: int | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X_normal: np.ndarray) -> "AnomalyDetector":
        """Fit on normal traffic only (per research design)."""
        start = time.perf_counter()
        self.model.fit(X_normal)
        self.train_time_seconds = time.perf_counter() - start
        self.n_train_samples = int(len(X_normal))
        return self

    # ------------------------------------------------------------ inference
    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.score_samples(X)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> AnomalyResult:
        thr = self.threshold if threshold is None else threshold
        scores = self.score(X)
        return AnomalyResult(
            scores=scores,
            is_anomaly=scores <= thr,
            threshold=thr,
        )

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path = config.MODEL_FILES["isolation_forest"]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "threshold": self.threshold,
             "train_time_seconds": self.train_time_seconds,
             "n_train_samples": self.n_train_samples},
            path,
        )

    @classmethod
    def load(cls, path: str | Path = config.MODEL_FILES["isolation_forest"]
             ) -> "AnomalyDetector":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Isolation Forest model not found at {path}. Run "
                "'python train.py' first."
            )
        blob = joblib.load(path)
        det = cls(threshold=blob["threshold"])
        det.model = blob["model"]
        det.train_time_seconds = blob.get("train_time_seconds")
        det.n_train_samples = blob.get("n_train_samples")
        return det


def anomaly_score_to_risk_component(
    score: float, threshold: float = config.ANOMALY_THRESHOLD,
    spread: float = config.ANOMALY_SCORE_SPREAD,
) -> float:
    """Map a raw anomaly score to [0, 1] for the risk formula (transparent).

    Records at/above the threshold contribute 0; records `spread` below the
    threshold contribute 1; linear in between.
    """
    if np.isnan(score):
        return 0.0
    return float(np.clip((threshold - score) / spread, 0.0, 1.0))


def summarise_anomalies(result: AnomalyResult) -> pd.DataFrame:
    """Small helper for dashboard summaries."""
    return pd.DataFrame({
        "anomaly_score": result.scores,
        "is_anomaly": result.is_anomaly,
    })
