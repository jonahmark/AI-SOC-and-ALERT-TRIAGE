"""Tests for evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import evaluation


def test_anomaly_metrics_perfect_detector():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.5, 0.3, -0.9, -0.8])  # lower = anomalous
    m = evaluation.anomaly_metrics(y_true, scores, threshold=-0.45)
    assert m["recall"] == 1.0
    assert m["false_positive_rate"] == 0.0
    assert m["false_negative_rate"] == 0.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_anomaly_metrics_single_class_no_auc():
    y_true = np.array([0, 0, 0])
    scores = np.array([0.1, 0.2, -0.9])
    m = evaluation.anomaly_metrics(y_true, scores, threshold=-0.45)
    assert m["roc_auc"] is None  # not measurable, reported honestly


def test_classification_metrics_basic():
    y_true = np.array(["Normal", "DoS", "DoS", "Probe", "Normal"])
    y_pred = np.array(["Normal", "DoS", "Probe", "Probe", "Normal"])
    m = evaluation.classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.8
    assert set(m["per_class"]) == {"Normal", "DoS", "Probe"}
    assert m["per_class"]["Normal"]["f1"] == 1.0
    assert m["strongest_category"]["name"] in ("Normal", "Probe")
    assert m["weakest_category"]["name"] == "DoS"
    cm = m["confusion_matrix"]
    assert cm["labels"] and len(cm["matrix"]) == len(cm["labels"])


def test_classification_metrics_with_probabilities():
    classes = ["Normal", "DoS"]
    y_true = np.array(["Normal", "DoS", "Normal", "DoS"])
    y_pred = np.array(["Normal", "DoS", "DoS", "DoS"])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
    m = evaluation.classification_metrics(
        y_true, y_pred, probabilities=probs, classes=classes)
    assert m["accuracy"] == 0.75
    assert m["roc_auc_ovr"] is not None


def test_triage_metrics_counts_false_positives():
    alerts = pd.DataFrame({
        "alert_id": ["A1", "A2", "A3"],
        "severity": ["High", "Critical", "Informational"],
        "predicted_category": ["DoS", "U2R", "Normal"],
        "true_category": ["DoS", "Normal", "Normal"],
        "investigation_status": ["New", "Resolved", "New"],
        "risk_score": [70.0, 95.0, 10.0],
    })
    m = evaluation.triage_metrics(alerts)
    assert m["n_alerts"] == 3
    # FP definition: any alert whose ground-truth category is Normal
    # (A2 predicted U2R on normal traffic, A3 flagged although normal).
    assert m["false_positives_vs_ground_truth"] == 2
    assert m["n_investigated"] == 1
    assert m["by_severity"]["Critical"] == 1


def test_triage_metrics_empty():
    m = evaluation.triage_metrics(pd.DataFrame())
    assert m == {"n_alerts": 0}


def test_compare_with_targets_separate_fields():
    metrics = {
        "anomaly_detection": {"roc_auc": 0.99, "recall": 0.5},
        "attack_classification": {
            "random_forest": {"accuracy": 0.8, "macro_f1": 0.75},
            "xgboost": {"accuracy": 0.7, "macro_f1": 0.6},
        },
    }
    comp = evaluation.compare_with_targets(metrics)
    ad = comp["anomaly_detection"]
    assert ad["roc_auc"]["actual"] == 0.99
    assert ad["roc_auc"]["target"] == config.EVALUATION_TARGETS[
        "anomaly_detection"]["roc_auc"]
    assert ad["roc_auc"]["target_met"] is True
    assert ad["recall"]["target_met"] is False
    ac = comp["attack_classification"]
    assert ac["accuracy"]["actual"] == 0.8  # best of both models


def test_save_load_evaluation_roundtrip(tmp_path):
    path = tmp_path / "eval.json"
    evaluation.save_evaluation({"x": 1}, path)
    loaded = evaluation.load_evaluation(path)
    assert loaded["x"] == 1
    assert "evaluated_at" in loaded
    assert evaluation.load_evaluation(tmp_path / "absent.json") is None
