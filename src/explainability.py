"""Explainable AI with SHAP (tree explainers for RF / XGBoost).

Two clearly separated capabilities:

1. GLOBAL feature importance  - mean(|SHAP|) over a sample of records.
   This is dataset-level, NOT a per-alert explanation.
2. LOCAL per-alert explanation - SHAP values for one specific record and
   the predicted class, showing which features pushed the prediction
   toward or away from that class.

Both are computed on demand; global importance is cached to
results/metrics/ so the alert pipeline is not slowed down.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shap

import config

GLOBAL_SHAP_FILE = config.METRICS_DIR / "global_shap_importance.json"


@dataclass
class LocalExplanation:
    """Per-record SHAP explanation for the predicted class."""

    predicted_class: str
    base_value: float
    contributions: list = field(default_factory=list)  # (feature, value, shap)
    explanation_time_ms: float | None = None
    model_type: str = ""

    def to_plain_language(self, top_k: int = 5) -> str:
        if not self.contributions:
            return "No local SHAP explanation available."
        top = sorted(
            self.contributions, key=lambda c: abs(c[2]), reverse=True
        )[:top_k]
        push = [
            f"'{f}' (value {v:g}, contribution {s:+.3f})"
            for f, v, s in top if s > 0
        ]
        against = [
            f"'{f}' (value {v:g}, contribution {s:+.3f})"
            for f, v, s in top if s <= 0
        ]
        text = (
            f"The classifier predicted '{self.predicted_class}'. "
        )
        if push:
            text += (
                "Features pushing TOWARD this prediction: "
                + "; ".join(push) + ". "
            )
        if against:
            text += (
                "Features pushing AGAINST it: " + "; ".join(against) + ". "
            )
        text += (
            "Contributions are SHAP values relative to the model's base "
            "expectation; they explain THIS record only."
        )
        return text


class Explainer:
    """SHAP TreeExplainer wrapper for the attack classifiers."""

    def __init__(self, classifier, feature_names: list[str]):
        self.classifier = classifier          # AttackClassifier
        self.feature_names = list(feature_names)
        self.explainer = shap.TreeExplainer(classifier.model)
        self.model_type = classifier.model_type

    # ---------------------------------------------------------------- global
    def global_importance(
        self, X_sample: np.ndarray, use_cache: bool = True
    ) -> tuple[pd.DataFrame, float]:
        """Mean |SHAP| per feature, averaged over classes.

        Returns (dataframe sorted desc, computation time in seconds).
        """
        cache_key = GLOBAL_SHAP_FILE
        if use_cache and cache_key.exists():
            try:
                blob = pd.read_json(cache_key)
                if len(blob) and str(blob.iloc[0].get("model_type", "")) == self.model_type:
                    return (
                        blob.drop(columns=["model_type", "compute_seconds"]),
                        float(blob.iloc[0].get("compute_seconds", np.nan)),
                    )
            except Exception:
                pass  # corrupt cache -> recompute

        start = time.perf_counter()
        shap_values = self.explainer.shap_values(X_sample)
        sv = self._to_3d(shap_values, X_sample)
        mean_abs = np.abs(sv).mean(axis=(0, 2))  # over records and classes
        elapsed = time.perf_counter() - start

        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        cache_key.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out["model_type"] = self.model_type
        out["compute_seconds"] = elapsed
        out.to_json(cache_key, indent=2)
        return df, elapsed

    # ----------------------------------------------------------------- local
    def explain_record(
        self, x: np.ndarray, class_index: int | None = None
    ) -> LocalExplanation:
        """Local SHAP explanation for ONE record (row vector)."""
        start = time.perf_counter()
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        shap_values = self.explainer.shap_values(x)
        sv = self._to_3d(shap_values, x)          # (1, n_features, n_classes)
        classes = self.classifier.classes_

        if class_index is None:
            probs = self.classifier.model.predict_proba(x)[0]
            class_index = int(np.argmax(probs))
        class_index = int(np.clip(class_index, 0, len(classes) - 1))

        vals = sv[0, :, class_index]
        base = self.explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.ravel(base)[class_index])
        base = float(base)

        contributions = [
            (self.feature_names[i], float(x[0, i]), float(vals[i]))
            for i in range(len(self.feature_names))
        ]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return LocalExplanation(
            predicted_class=classes[class_index],
            base_value=base,
            contributions=contributions,
            explanation_time_ms=elapsed_ms,
            model_type=self.model_type,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _to_3d(shap_values, X: np.ndarray) -> np.ndarray:
        """Normalise sklearn/xgboost SHAP outputs to (n, features, classes)."""
        if isinstance(shap_values, list):
            return np.stack(shap_values, axis=2)
        arr = np.asarray(shap_values)
        if arr.ndim == 2:  # binary or single-output
            return arr[:, :, None]
        return arr
