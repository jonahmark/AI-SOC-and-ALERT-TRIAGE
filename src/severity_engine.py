"""Severity mapping and transparent risk prioritisation.

Severity comes from a documented, configurable category -> severity map.
The risk score is a transparent weighted formula (weights in config.py):

    risk = 100 * (w_sev   * severity_weight(category)
                + w_conf  * model_confidence
                + w_anom  * anomaly_component(anomaly_score))

This is a research prioritisation heuristic, NOT a validated real-world
risk measurement (see README limitations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from src.anomaly_detection import anomaly_score_to_risk_component


@dataclass
class SeverityEngine:
    severity_map: dict = None  # category -> severity label

    def __post_init__(self):
        if self.severity_map is None:
            self.severity_map = dict(config.SEVERITY_MAP)

    def severity_for(self, category: str) -> str:
        return self.severity_map.get(category, "Informational")

    def risk_score(
        self,
        category: str,
        confidence: float,
        anomaly_score: float | None = None,
        weights: dict | None = None,
        threshold: float | None = None,
    ) -> float:
        """Compute a 0-100 prioritisation risk score (transparent formula).

        `threshold` is the anomaly threshold actually used for the detection
        decision, so the normalised anomaly component is consistent with it.
        """
        w = weights or config.RISK_WEIGHTS
        sev_weight = config.SEVERITY_WEIGHTS.get(
            self.severity_for(category), 0.0
        )
        if anomaly_score is None:
            anom = 0.0
        elif threshold is None:
            anom = anomaly_score_to_risk_component(anomaly_score)
        else:
            anom = anomaly_score_to_risk_component(anomaly_score, threshold)
        conf = 0.0 if confidence is None else float(confidence)
        risk = 100.0 * (
            w["severity"] * sev_weight + w["confidence"] * conf + w["anomaly"] * anom
        )
        return round(min(max(risk, 0.0), 100.0), 2)

    # ------------------------------------------------------- batch scoring
    def severity_for_batch(self, categories) -> np.ndarray:
        """Vectorised `severity_for` (identical mapping, including default)."""
        cats = np.asarray(categories, dtype=object)
        mapping = {c: self.severity_map.get(c, "Informational")
                   for c in np.unique(cats)}
        return np.array([mapping[c] for c in cats], dtype=object)

    def risk_score_batch(
        self,
        categories,
        confidences,
        anomaly_scores=None,
        weights: dict | None = None,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Vectorised `risk_score`; returns the same 0-100 values, element-wise.

        Mirrors the scalar formula exactly (same severity weights, anomaly
        component, clipping and rounding) so batch and per-row results agree.
        """
        cats = np.asarray(categories, dtype=object)
        n = len(cats)
        w = weights or config.RISK_WEIGHTS
        sev_labels = self.severity_for_batch(cats)
        sev_weight = np.array(
            [config.SEVERITY_WEIGHTS.get(s, 0.0) for s in sev_labels],
            dtype=float,
        )
        conf = np.asarray(confidences, dtype=float)
        if anomaly_scores is None:
            anom = np.zeros(n, dtype=float)
        else:
            scores = np.asarray(anomaly_scores, dtype=float)
            if threshold is None:
                thr = config.ANOMALY_THRESHOLD
                spread = config.ANOMALY_SCORE_SPREAD
            else:
                thr = float(threshold)
                spread = config.ANOMALY_SCORE_SPREAD
            anom = np.clip((thr - scores) / spread, 0.0, 1.0)
            anom = np.where(np.isnan(scores), 0.0, anom)
        risk = 100.0 * (
            w["severity"] * sev_weight + w["confidence"] * conf
            + w["anomaly"] * anom
        )
        return np.round(np.clip(risk, 0.0, 100.0), 2)

    @staticmethod
    def recommended_action(category: str, severity: str) -> str:
        """Rule-based analyst guidance (research prototype heuristics)."""
        actions = {
            "U2R": (
                "CRITICAL: possible local privilege escalation. Isolate the "
                "host, capture forensic image, review account activity and "
                "escalate to incident response immediately."
            ),
            "R2L": (
                "Possible remote unauthorised access. Check authentication "
                "logs for the target service, verify source IP reputation "
                "and block/monitor the source pending investigation."
            ),
            "DoS": (
                "Possible denial-of-service. Verify service availability, "
                "inspect traffic volume patterns toward the target and "
                "prepare rate-limiting/filtering if sustained."
            ),
            "Probe": (
                "Possible reconnaissance. Review scan patterns from the "
                "source, confirm no follow-on exploitation, and add the "
                "source to a watch list."
            ),
            "Normal": (
                "No attack category predicted. If flagged anomalous, "
                "compare against baseline behaviour before closing."
            ),
        }
        base = actions.get(category, "Review alert details and correlate "
                                     "with related events.")
        if severity == "Informational" and category == "Normal":
            return "No action required unless correlated with other alerts."
        return base
