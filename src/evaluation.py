"""Evaluation metrics for anomaly detection, classification and triage.

Targets and actuals are always reported separately; 'target_met' is a
simple comparison for transparency, never an automatic claim of success.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

import config

EVALUATION_FILE = config.METRICS_DIR / "evaluation.json"


# ---------------------------------------------------------------- anomaly
def anomaly_metrics(
    y_true_is_attack: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Binary anomaly-detection metrics.

    y_true_is_attack: True where the ground-truth category != Normal.
    scores: IsolationForest.score_samples() (lower = more anomalous).
    """
    y_true = np.asarray(y_true_is_attack).astype(int)
    y_pred = (np.asarray(scores) <= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": threshold,
        "n_records": int(len(y_true)),
        "accuracy": round((tp + tn) / max(len(y_true), 1), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "false_negative_rate": round(fn / max(fn + tp, 1), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }
    try:
        # ROC with negated scores: higher = more anomalous.
        auc = roc_auc_score(y_true, -np.asarray(scores))
        fpr, tpr, _ = roc_curve(y_true, -np.asarray(scores))
        if auc is None or not np.isfinite(auc):
            raise ValueError("ROC-AUC undefined for this ground truth.")
        out["roc_auc"] = round(float(auc), 4)
        out["roc_curve"] = {
            "fpr": [round(float(v), 4) for v in fpr[::max(1, len(fpr) // 200)]],
            "tpr": [round(float(v), 4) for v in tpr[::max(1, len(tpr) // 200)]],
        }
    except ValueError:
        out["roc_auc"] = None  # single-class ground truth: not measurable
        out["roc_curve"] = None
    return out


# ----------------------------------------------------------- classification
def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    probabilities: np.ndarray | None = None, classes: list | None = None,
) -> dict:
    labels = classes or config.ATTACK_CATEGORIES
    present = [c for c in labels if c in set(y_true) | set(y_pred)]
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0
    )
    per_class = {
        cls: {
            "precision": round(float(pi), 4),
            "recall": round(float(ri), 4),
            "f1": round(float(fi), 4),
            "support": int(s),
        }
        for cls, pi, ri, fi, s in zip(present, p, r, f1, support)
    }
    cm = confusion_matrix(y_true, y_pred, labels=present)
    out = {
        "n_records": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(
            f1_score(y_true, y_pred, labels=present, average="macro",
                     zero_division=0), 4),
        "weighted_f1": round(
            f1_score(y_true, y_pred, labels=present, average="weighted",
                     zero_division=0), 4),
        "macro_precision": round(
            precision_score(y_true, y_pred, labels=present, average="macro",
                            zero_division=0), 4),
        "macro_recall": round(
            recall_score(y_true, y_pred, labels=present, average="macro",
                         zero_division=0), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": present,
            "matrix": cm.tolist(),
        },
    }
    if per_class:
        strongest = max(per_class, key=lambda c: per_class[c]["f1"])
        weakest = min(per_class, key=lambda c: per_class[c]["f1"])
        out["strongest_category"] = {"name": strongest, **per_class[strongest]}
        out["weakest_category"] = {"name": weakest, **per_class[weakest]}
    if probabilities is not None and classes is not None:
        try:
            mask = np.array([c in classes for c in y_true])
            y_idx = np.array(
                [classes.index(c) for c in np.asarray(y_true)[mask]])
            probs = np.asarray(probabilities)[mask]
            if len(classes) == 2:
                # sklearn's multi_class="ovr" rejects 2-column scores
                auc = roc_auc_score(y_idx, probs[:, 1])
            else:
                auc = roc_auc_score(y_idx, probs, multi_class="ovr")
            out["roc_auc_ovr"] = round(float(auc), 4)
        except (ValueError, KeyError):
            out["roc_auc_ovr"] = None
    return out


# ---------------------------------------------------------------- triage
def triage_metrics(alerts: pd.DataFrame) -> dict:
    """Alert-level triage statistics from the generated alert table."""
    if alerts.empty:
        return {"n_alerts": 0}
    false_positives = None
    if (alerts["true_category"] != "").any():
        # Ground truth available: FP = alert where truth is Normal.
        labelled = alerts[alerts["true_category"] != ""]
        false_positives = int((labelled["true_category"] == "Normal").sum())
    return {
        "n_alerts": int(len(alerts)),
        "by_severity": alerts["severity"].value_counts().to_dict(),
        "by_predicted_category":
            alerts["predicted_category"].value_counts().to_dict(),
        "by_investigation_status":
            alerts["investigation_status"].value_counts().to_dict(),
        "n_prioritised_top50": int(
            alerts.nlargest(50, "risk_score")["alert_id"].nunique()
        ),
        "n_investigated": int(
            (alerts["investigation_status"] != "New").sum()
        ),
        "false_positives_vs_ground_truth": false_positives,
        "mean_risk_score": round(float(alerts["risk_score"].mean()), 2),
        "max_risk_score": round(float(alerts["risk_score"].max()), 2),
    }


# -------------------------------------------------------------- comparison
def compare_with_targets(metrics: dict) -> dict:
    """Return targets vs actuals side by side. Never claims automatic success."""
    comparison = {}
    t = config.EVALUATION_TARGETS
    ad = metrics.get("anomaly_detection", {})
    ac = metrics.get("attack_classification", {})
    comparison["anomaly_detection"] = {
        "roc_auc": {"target": t["anomaly_detection"]["roc_auc"],
                    "actual": ad.get("roc_auc")},
        "recall": {"target": t["anomaly_detection"]["recall"],
                   "actual": ad.get("recall")},
    }
    # Use the better of RF / XGB as the headline classifier actual.
    def best(key):
        vals = [m.get(key) for m in ac.values() if isinstance(m, dict)
                and m.get(key) is not None]
        return max(vals) if vals else None
    comparison["attack_classification"] = {
        "accuracy": {"target": t["attack_classification"]["accuracy"],
                     "actual": best("accuracy")},
        "macro_f1": {"target": t["attack_classification"]["macro_f1"],
                     "actual": best("macro_f1")},
    }
    for section in comparison.values():
        for entry in section.values():
            a, tg = entry["actual"], entry["target"]
            entry["target_met"] = (a is not None and a >= tg)
    return comparison


def save_evaluation(payload: dict,
                    path: str | Path = EVALUATION_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_evaluation(path: str | Path = EVALUATION_FILE) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
