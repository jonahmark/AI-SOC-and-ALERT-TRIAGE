"""Tests for report assembly and real PDF/CSV export.

The fixtures below are deliberately small hand-built frames: they verify that
the report copies the values it is given (and labels missing ones), never that
the models produce good numbers. Model correctness is covered elsewhere.
"""
from __future__ import annotations

import io
import json

import pandas as pd
import pytest

import config
from src import pipeline, reporting, settings as S


@pytest.fixture
def records():
    return pd.DataFrame({
        "record_index": [0, 1, 2, 3],
        "detection_status": ["Anomaly", "Normal", "Anomaly", "Normal"],
        "predicted_category": ["DoS", "Normal", "Probe", "Normal"],
        "true_category": ["DoS", "Normal", "Normal", "Normal"],
        "severity": ["High", "Informational", "Medium", "Informational"],
        "risk_score": [80.0, 10.0, 55.0, 12.0],
        "model_confidence": [0.9, 0.99, 0.7, 0.95],
    })


@pytest.fixture
def alerts(records):
    return pd.DataFrame({
        "alert_id": ["ALT-1", "ALT-2", "ALT-3"],
        "predicted_category": ["DoS", "Probe", "DoS"],
        "true_category": ["DoS", "Normal", "Normal"],
        "severity": ["High", "Medium", "High"],
        "risk_score": [80.0, 55.0, 40.0],
        "investigation_status": ["New", "Investigating", "Resolved"],
    })


@pytest.fixture
def analysis(records, alerts):
    return pipeline.AnalysisResult(
        records=records, alerts=alerts,
        stages=[pipeline.StageResult(name, pipeline.STATUS_DONE, 0.5, "measured")
                for name in pipeline.STAGES],
        meta={"n_records": 4, "classifier": "xgboost", "total_seconds": 3.5,
              "anomaly_threshold": -0.4049, "ran_at": "2026-11-07T12:00:00+00:00"},
    )


@pytest.fixture
def evaluation():
    return {
        "anomaly_detection": {"roc_auc": 0.9483, "recall": 0.7464,
                              "precision": 0.916},
        "attack_classification": {
            "xgboost": {"accuracy": 0.7807, "macro_f1": 0.5594,
                        "per_class": {"U2R": {"precision": 0.6364,
                                              "recall": 0.1045, "f1": 0.1795,
                                              "support": 67}}},
        },
    }


# ------------------------------------------------------------------- payload
def test_payload_always_carries_methodology_and_limitations():
    payload = reporting.collect_payload(S.defaults())
    assert payload["title"] == reporting.PROJECT_TITLE
    assert payload["app_version"] == config.APP_VERSION
    assert payload["limitations"] == reporting.LIMITATIONS
    assert len(payload["limitations"]) >= 5
    assert set(payload["methodology"]) == {
        "split", "anomaly", "classification", "triage", "explainability"}
    assert "live_analysis" not in payload and "alerts" not in payload


def test_payload_uses_the_configured_severity_map_and_weights():
    settings = {**S.defaults(),
                "severity_map": {**config.SEVERITY_MAP, "Probe": "Critical"},
                "risk_weights": {"severity": 0.6, "confidence": 0.2,
                                 "anomaly": 0.2}}
    payload = reporting.collect_payload(settings)
    assert payload["severity_map"]["Probe"] == "Critical"
    assert payload["risk_weights"]["severity"] == 0.6


def test_dataset_block_reports_recorded_values_only():
    state = {"n_train_records": 125973, "n_test_records": 22544,
             "n_encoded_features": 122, "random_seed": 42,
             "anomaly_threshold_method": "5% quantile of normal training scores"}
    payload = reporting.collect_payload(S.defaults(), state=state)
    assert payload["dataset"]["train_rows"] == 125973
    assert payload["dataset"]["random_seed"] == 42
    assert "5% quantile" in payload["methodology"]["anomaly"]
    empty = reporting.collect_payload(S.defaults())["dataset"]
    assert empty["train_rows"] is None       # not invented


def test_live_analysis_distributions_match_the_records(analysis, records):
    payload = reporting.collect_payload(S.defaults(), analysis=analysis)
    live = payload["live_analysis"]
    assert live["meta"] == analysis.meta
    assert sum(live["by_category"].values()) == len(records)
    assert sum(live["detection_status"].values()) == len(records)
    assert sum(live["ground_truth"].values()) == len(records)
    assert live["by_category"]["DoS"] == 1
    assert len(live["stages"]) == 7
    assert all(s["seconds"] == 0.5 for s in live["stages"])


def test_failed_analysis_is_not_reported_as_a_result(analysis):
    analysis.errors.append("preprocessing failed")
    payload = reporting.collect_payload(S.defaults(), analysis=analysis)
    assert "live_analysis" not in payload


def test_alert_summary_counts_and_ranks_real_alerts(alerts):
    payload = reporting.collect_payload(S.defaults(), alerts=alerts)
    summary = payload["alerts"]
    assert summary["total"] == 3
    assert summary["by_severity"] == {"High": 2, "Medium": 1}
    assert summary["by_status"]["Investigating"] == 1
    assert summary["mean_risk"] == pytest.approx((80 + 55 + 40) / 3)
    assert summary["max_risk"] == 80.0
    assert [a["risk_score"] for a in summary["top"]] == [80.0, 55.0, 40.0]
    assert summary["false_positives_vs_ground_truth"] == 2


def test_empty_alert_store_adds_no_summary():
    payload = reporting.collect_payload(S.defaults(), alerts=pd.DataFrame())
    assert "alerts" not in payload


# -------------------------------------------------------------------- exports
def test_csv_exports_contain_the_supplied_values(analysis, alerts, records,
                                               evaluation):
    payload = reporting.collect_payload(S.defaults(), analysis=analysis,
                                        alerts=alerts, evaluation=evaluation)
    exports = reporting.csv_exports(payload, alerts, records)
    assert set(exports) == {"model_evaluation.csv", "analysis_summary.json",
                            "alerts.csv", "processed_records.csv"}
    metrics = pd.read_csv(io.BytesIO(exports["model_evaluation.csv"]))
    row = metrics[(metrics["model"] == "isolation_forest")
                  & (metrics["metric"] == "roc_auc")]
    assert float(row["value"].iloc[0]) == pytest.approx(0.9483)
    per_class = metrics[metrics["metric"] == "U2R_f1"]
    assert float(per_class["value"].iloc[0]) == pytest.approx(0.1795)
    exported_alerts = pd.read_csv(io.BytesIO(exports["alerts.csv"]))
    assert list(exported_alerts["alert_id"]) == list(alerts["alert_id"])
    exported_records = pd.read_csv(io.BytesIO(exports["processed_records.csv"]))
    assert len(exported_records) == len(records)
    summary = json.loads(exports["analysis_summary.json"])
    assert summary["meta"]["n_records"] == 4


def test_resource_measurements_are_exported_verbatim():
    resources = {"measurements": {
        "train_xgboost": {"wall_time_seconds": 86.14, "memory_delta_mb": -1.2},
        "xgboost_prediction_latency": {"mean_ms_per_record": 0.0305,
                                       "n_records": 5000},
    }}
    payload = reporting.collect_payload(S.defaults(), resources=resources)
    exports = reporting.csv_exports(payload)
    frame = pd.read_csv(io.BytesIO(exports["resource_measurements.csv"]))
    assert set(frame["stage"]) == {"train_xgboost", "xgboost_prediction_latency"}
    assert float(frame.loc[frame["stage"] == "train_xgboost",
                           "wall_time_seconds"].iloc[0]) == 86.14


def test_no_artefacts_means_no_empty_placeholder_exports():
    payload = reporting.collect_payload(S.defaults())
    assert reporting.csv_exports(payload) == {}


# ------------------------------------------------------------------------ pdf
def test_pdf_is_a_real_document(analysis, alerts, evaluation):
    payload = reporting.collect_payload(S.defaults(), analysis=analysis,
                                        alerts=alerts, evaluation=evaluation)
    pdf = reporting.generate_pdf(payload)
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 5000
    assert pdf.count(b"/Type /Page") >= 2


def test_pdf_renders_with_missing_data_instead_of_failing():
    payload = reporting.collect_payload(S.defaults())
    pdf = reporting.generate_pdf(payload)
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 3000


def test_save_pdf_writes_the_document(tmp_path, analysis, alerts):
    payload = reporting.collect_payload(S.defaults(), analysis=analysis,
                                        alerts=alerts)
    path = reporting.save_pdf(payload, tmp_path / "nested" / "report.pdf")
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF-")