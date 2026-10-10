"""Stage B: supervised attack classification (Random Forest and XGBoost).

Both models classify preprocessed records into the five categories
Normal / DoS / Probe / R2L / U2R and expose class probabilities.
Anomaly detection and attack classification are deliberately separate
analytical decisions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config


def class_sample_weights(y: np.ndarray, scheme: str | None) -> np.ndarray | None:
    """Per-record weights counteracting class imbalance (None = unweighted).

    "balanced" gives each record weight n_samples / (n_classes * count of
    its class), so rare R2L/U2R attacks influence the loss as much as the
    abundant Normal/DoS records.
    """
    if not scheme:
        return None
    return compute_sample_weight(class_weight=scheme, y=y)


@dataclass
class ClassificationResult:
    predictions: np.ndarray          # category labels
    probabilities: np.ndarray        # (n, n_classes)
    confidences: np.ndarray          # max probability per record
    classes: list                    # label order of probabilities


class AttackClassifier:
    """Wrapper unifying RandomForest and XGBoost behind one interface."""

    SUPPORTED = {"random_forest", "xgboost"}

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in self.SUPPORTED:
            raise ValueError(
                f"model_type must be one of {sorted(self.SUPPORTED)}"
            )
        self.model_type = model_type
        self.model = self._build_model(model_type)
        self._label_encoder: LabelEncoder | None = None
        self.classes_: list[str] = []
        self.train_time_seconds: float | None = None
        self.n_train_samples: int | None = None

    @staticmethod
    def _build_model(model_type: str):
        if model_type == "random_forest":
            return RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
        return XGBClassifier(**config.XGBOOST_PARAMS)

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttackClassifier":
        known = set(config.ATTACK_CATEGORIES)
        y = np.asarray(y)
        mask = np.array([label in known for label in y])
        if not mask.all():
            # 'Unknown' (unmapped labels) excluded from supervised training.
            X, y = X[mask], y[mask]
        if self.model_type == "xgboost":
            # XGBoost requires integer-encoded classes; the encoder's
            # (sorted) class order is kept as classes_ so probabilities
            # map back to the right labels.
            self._label_encoder = LabelEncoder().fit(y)
            y_fit = self._label_encoder.transform(y)
        else:
            y_fit = y
        start = time.perf_counter()
        if self.model_type == "xgboost":
            # XGBoost has no class_weight parameter; cost-sensitive
            # learning is applied through per-record sample weights.
            self.model.fit(
                X, y_fit,
                sample_weight=class_sample_weights(
                    y, config.XGBOOST_CLASS_WEIGHTING),
            )
        else:
            self.model.fit(X, y_fit)
        self.train_time_seconds = time.perf_counter() - start
        self.n_train_samples = int(len(y))
        if self._label_encoder is not None:
            self.classes_ = [str(c) for c in self._label_encoder.classes_]
        else:
            self.classes_ = [str(c) for c in self.model.classes_]
        return self

    # ------------------------------------------------------------ inference
    def predict(self, X: np.ndarray) -> ClassificationResult:
        if not self.classes_:
            raise RuntimeError("Classifier has not been fitted/loaded.")
        probs = self.model.predict_proba(X)
        preds = np.asarray(self.classes_)[probs.argmax(axis=1)]
        return ClassificationResult(
            predictions=preds,
            probabilities=probs,
            confidences=probs.max(axis=1),
            classes=list(self.classes_),
        )

    # ---------------------------------------------------------- persistence
    @property
    def _default_path(self) -> Path:
        key = self.model_type
        return Path(config.MODEL_FILES[key])

    def save(self, path: str | Path | None = None):
        path = Path(path or self._default_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "classes": self.classes_,
             "model_type": self.model_type,
             "train_time_seconds": self.train_time_seconds,
             "n_train_samples": self.n_train_samples},
            path,
        )

    @classmethod
    def load(cls, model_type: str = "random_forest",
             path: str | Path | None = None) -> "AttackClassifier":
        clf = cls(model_type)
        path = Path(path or clf._default_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{model_type} model not found at {path}. Run "
                "'python train.py' first."
            )
        blob = joblib.load(path)
        if blob.get("model_type", model_type) != model_type:
            raise ValueError(
                f"Artefact at {path} is a '{blob.get('model_type')}' model, "
                f"but '{model_type}' was requested."
            )
        clf.model = blob["model"]
        clf.classes_ = blob["classes"]
        clf.train_time_seconds = blob.get("train_time_seconds")
        clf.n_train_samples = blob.get("n_train_samples")
        return clf
