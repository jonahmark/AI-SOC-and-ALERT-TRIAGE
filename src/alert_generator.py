"""Alert generation and the alert store (persistence + triage workflow).

An alert is generated when EITHER stage flags a record:
- Isolation Forest decision == Anomaly, OR
- predicted attack category != Normal.

The two stages remain separate decisions; the alert records both.
Alerts persist to CSV so dashboard sessions share state. Investigation
statuses and notes persist separately (JSON files, or a private PostgreSQL
database when one is configured) so regenerated alerts keep analyst workflow.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import triage
from src.anomaly_detection import AnomalyResult
from src.attack_classification import ClassificationResult
from src.severity_engine import SeverityEngine

ALERTS_FILE = config.DATA_PROCESSED_DIR / "alerts.csv"
STATUS_FILE = config.ALERT_STATUS_FILE
NOTES_FILE = config.ALERT_NOTES_FILE

ALERT_COLUMNS = [
    "alert_id", "timestamp", "record_index", "detection_status",
    "predicted_category", "true_label", "true_category", "severity",
    "risk_score", "model_confidence", "anomaly_score", "explanation",
    "recommended_action", "investigation_status",
]


def _make_explanation(
    is_anomaly: bool, anomaly_score: float, category: str,
    confidence: float, threshold: float,
) -> str:
    parts = []
    if is_anomaly:
        parts.append(
            f"Stage A (Isolation Forest) flagged this record as ANOMALOUS "
            f"(score {anomaly_score:.3f} <= threshold {threshold:.2f})."
        )
    else:
        parts.append(
            f"Stage A (Isolation Forest) scored this record as normal "
            f"(score {anomaly_score:.3f} > threshold {threshold:.2f})."
        )
    if category != "Normal":
        parts.append(
            f"Stage B classifier predicts attack category '{category}' "
            f"with {confidence:.1%} confidence."
        )
    else:
        parts.append(
            f"Stage B classifier predicts 'Normal' with {confidence:.1%} "
            "confidence."
        )
    return " ".join(parts)


class AlertGenerator:
    """Turns per-record model outputs into prioritised alerts."""

    def __init__(self, severity_engine: SeverityEngine | None = None,
                 risk_weights: dict | None = None):
        self.severity_engine = severity_engine or SeverityEngine()
        self.risk_weights = risk_weights

    def generate(
        self,
        anomaly_result: AnomalyResult,
        classification: ClassificationResult,
        record_indices: np.ndarray | None = None,
        true_labels: np.ndarray | None = None,
        timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        n = len(anomaly_result.scores)
        if len(classification.predictions) != n:
            raise ValueError("Anomaly and classification results must align.")
        ts = timestamp or datetime.now(timezone.utc)
        is_anomaly = np.asarray(anomaly_result.is_anomaly, dtype=bool)
        categories = np.asarray(classification.predictions).astype(str)
        confidences = np.asarray(classification.confidences, dtype=float)
        scores = np.asarray(anomaly_result.scores, dtype=float)

        # An alert is raised when EITHER stage flags the record.
        flagged = is_anomaly | (categories != "Normal")
        idx = np.flatnonzero(flagged)

        # Severity and risk are computed once for the flagged subset (the
        # vectorised engine mirrors the per-row formula exactly).
        severities = self.severity_engine.severity_for_batch(categories[idx])
        risks = self.severity_engine.risk_score_batch(
            categories[idx], confidences[idx], scores[idx],
            weights=self.risk_weights, threshold=anomaly_result.threshold,
        )
        rec_idx = (np.asarray(record_indices)[idx]
                   if record_indices is not None else idx)
        ts_iso = ts.isoformat()
        threshold = anomaly_result.threshold

        rows = []
        for j, i in enumerate(idx):
            category = categories[i]
            confidence = float(confidences[i])
            score = float(scores[i])
            severity = str(severities[j])
            rows.append({
                "alert_id": f"ALT-{uuid.uuid4().hex[:10].upper()}",
                "timestamp": ts_iso,
                "record_index": int(rec_idx[j]),
                "detection_status": "Anomaly" if is_anomaly[i] else "Normal",
                "predicted_category": category,
                "true_label": str(true_labels[i]) if true_labels is not None
                else "",
                "true_category": config.ATTACK_CATEGORY_MAP.get(
                    str(true_labels[i]).lower(), ""
                ) if true_labels is not None else "",
                "severity": severity,
                "risk_score": float(risks[j]),
                "model_confidence": round(confidence, 4),
                "anomaly_score": round(score, 4),
                "explanation": _make_explanation(
                    bool(is_anomaly[i]), score, category, confidence, threshold,
                ),
                "recommended_action":
                    self.severity_engine.recommended_action(category, severity),
                "investigation_status": "New",
            })
        alerts = pd.DataFrame(rows, columns=ALERT_COLUMNS)
        return alerts.sort_values("risk_score", ascending=False)


class AlertStore:
    """CSV-backed alert persistence with investigation-status workflow.

    Statuses and notes are held by a triage backend: JSON files by default, or
    a private PostgreSQL database when one is configured (src/triage.py). The
    JSON paths are unused in database mode.
    """

    def __init__(self, alerts_file: Path = ALERTS_FILE,
                 status_file: Path = STATUS_FILE,
                 notes_file: Path = NOTES_FILE):
        self.alerts_file = Path(alerts_file)
        self.status_file = Path(status_file)
        self.notes_file = Path(notes_file)
        self.state = triage.get_state(self.status_file, self.notes_file)

    # ----------------------------------------------------------- persistence
    def save(self, alerts: pd.DataFrame, merge: bool = True) -> None:
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)
        if merge and self.alerts_file.exists():
            existing = pd.read_csv(self.alerts_file)
            alerts = pd.concat([alerts, existing], ignore_index=True)
            alerts = alerts.drop_duplicates(
                subset="alert_id", keep="first"
            ).reset_index(drop=True)
        alerts.to_csv(self.alerts_file, index=False)
        self.state.retain(set(alerts["alert_id"]))

    def load(self) -> pd.DataFrame:
        if not self.alerts_file.exists():
            return pd.DataFrame(columns=ALERT_COLUMNS)
        alerts = pd.read_csv(self.alerts_file)
        statuses = self.state.statuses()
        if statuses:
            alerts["investigation_status"] = alerts["alert_id"].map(
                statuses
            ).fillna(alerts["investigation_status"])
        return alerts

    def clear(self) -> None:
        if self.alerts_file.exists():
            self.alerts_file.unlink()
        self.state.clear()

    # --------------------------------------------------------------- triage
    def update_status(self, alert_id: str, status: str) -> None:
        if status not in config.INVESTIGATION_STATUSES:
            raise ValueError(
                f"Status must be one of {config.INVESTIGATION_STATUSES}"
            )
        alerts = self.load()
        if alert_id not in set(alerts["alert_id"]):
            raise KeyError(f"Alert '{alert_id}' not found in the store.")
        self.state.set_status(alert_id, status)

    # ---------------------------------------------------------------- notes
    def add_note(self, alert_id: str, text: str, author: str = "analyst") -> dict:
        """Append a timestamped investigation note for an alert."""
        text = str(text).strip()
        if not text:
            raise ValueError("Note text must not be empty.")
        if len(text) > 4000:
            raise ValueError("Note text must be 4000 characters or fewer.")
        alerts = self.load()
        if alert_id not in set(alerts["alert_id"]):
            raise KeyError(f"Alert '{alert_id}' not found in the store.")
        entry = {
            "alert_id": alert_id,
            "author": str(author)[:80],
            "text": text,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.state.add_note(alert_id, entry)
        return entry

    def get_notes(self, alert_id: str) -> list:
        return self.state.notes(alert_id)

    def note_counts(self) -> dict:
        """Map alert_id -> number of notes, for table indicators."""
        return self.state.note_counts()

    # -------------------------------------------------------------- queries
    @staticmethod
    def filter_alerts(
        alerts: pd.DataFrame,
        severity: str | None = None,
        category: str | None = None,
        status: str | None = None,
        search: str | None = None,
        sort_by: str = "risk_score",
        ascending: bool = False,
    ) -> pd.DataFrame:
        out = alerts
        if severity:
            out = out[out["severity"] == severity]
        if category:
            out = out[out["predicted_category"] == category]
        if status:
            out = out[out["investigation_status"] == status]
        if search:
            out = out[out["alert_id"].str.contains(search, case=False,
                                                      na=False)]
        if sort_by in out.columns:
            out = out.sort_values(sort_by, ascending=ascending)
        return out.reset_index(drop=True)
