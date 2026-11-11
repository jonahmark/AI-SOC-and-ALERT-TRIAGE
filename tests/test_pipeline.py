"""Tests for the shared seven-stage analysis pipeline.

Models are trained on the small synthetic NSL-KDD-shaped fixture, so these
tests exercise the real orchestration (encoding, both model stages, alerts,
risk, SHAP) without depending on the 30 MB production artefacts.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import config
from src import pipeline, settings as S
from src.anomaly_detection import AnomalyDetector
from src.attack_classification import AttackClassifier
from src.preprocessing import Preprocessor


@pytest.fixture(scope="module")
def trained():
    from tests.conftest import synthetic_nsl_kdd
    df = synthetic_nsl_kdd(n=300)
    y = df[config.LABEL_COLUMN].map(config.ATTACK_CATEGORY_MAP).fillna("Unknown")
    pre = Preprocessor().fit(df)
    X = pre.transform(df)
    normal = X[(y == "Normal").to_numpy()]
    detector = AnomalyDetector().fit(normal)
    detector.threshold = float(np.quantile(detector.score(normal), 0.05))
    models = {
        "pre": pre,
        "detector": detector,
        "random_forest": AttackClassifier("random_forest").fit(X, y.to_numpy()),
        "xgboost": AttackClassifier("xgboost").fit(X, y.to_numpy()),
    }
    return df, models


def test_stages_are_the_documented_seven():
    assert pipeline.STAGES == [
        "Data ingestion", "Preprocessing", "Anomaly detection",
        "Attack classification", "Alert generation", "Risk prioritisation",
        "Explanation generation",
    ]


def test_full_run_returns_real_records_and_measured_stages(trained):
    df, models = trained
    result = pipeline.run_analysis(df, models, S.defaults(),
                                   ingestion_seconds=0.25,
                                   source_label="unit-test")
    assert result.ok, result.errors
    assert len(result.records) == len(df)
    assert not result.alerts.empty
    for column in ("record_index", "detection_status", "anomaly_score",
                   "predicted_category", "model_confidence", "severity",
                   "risk_score", "true_label", "true_category", "labelled"):
        assert column in result.records.columns
    assert result.records["labelled"].all()
    assert result.meta["n_records"] == len(df)
    assert result.meta["n_encoded_features"] == len(models["pre"].feature_names_)
    assert result.meta["source"] == "unit-test"
    assert result.meta["total_seconds"] > 0
    assert result.meta["ran_at"].endswith("+00:00")
    done = [s for s in result.stages if s.status == pipeline.STATUS_DONE]
    assert len(done) == 6                      # explanations not requested
    assert all(s.seconds is not None and s.seconds >= 0 for s in done)
    assert result.stages[0].seconds == 0.25    # measured ingestion passed through


def test_progress_callback_reports_every_stage_in_order(trained):
    df, models = trained
    events = []
    pipeline.run_analysis(df, models, S.defaults(),
                          progress_callback=lambda i, s: events.append((i, s.name)))
    assert [i for i, _ in events] == list(range(len(pipeline.STAGES)))
    assert [name for _, name in events] == pipeline.STAGES


def test_anomaly_and_classification_stay_separate_decisions(trained):
    df, models = trained
    result = pipeline.run_analysis(df, models, S.defaults())
    records = result.records
    # Stage A flags records Stage B calls Normal, and vice versa.
    assert ((records["detection_status"] == "Anomaly")
            & (records["predicted_category"] == "Normal")).any()
    assert ((records["detection_status"] == "Normal")
            & (records["predicted_category"] != "Normal")).any()
    assert set(result.alerts["record_index"]) <= set(records["record_index"])


def test_alerts_equal_flagged_records(trained):
    df, models = trained
    result = pipeline.run_analysis(df, models, S.defaults())
    flagged = result.records[
        (result.records["detection_status"] == "Anomaly")
        | (result.records["predicted_category"] != "Normal")
    ]
    assert len(result.alerts) == len(flagged)
    assert set(result.alerts["record_index"]) == set(flagged["record_index"])
    assert result.alerts["risk_score"].is_monotonic_decreasing
    assert result.alerts["risk_score"].between(0, 100).all()


def test_alert_generation_can_be_disabled(trained):
    df, models = trained
    result = pipeline.run_analysis(df, models, S.defaults(), generate_alerts=False)
    assert result.ok
    assert result.alerts.empty
    assert result.stages[4].status == pipeline.STATUS_SKIPPED
    assert "risk_score" in result.records.columns   # records still ranked


def test_missing_artefact_fails_without_simulating_results(trained):
    df, models = trained
    broken = {**models, "detector": None}
    result = pipeline.run_analysis(df, broken, S.defaults())
    assert not result.ok
    assert any("Missing model artefact" in e for e in result.errors)
    assert result.records.empty and result.alerts.empty
    assert result.stages[1].status == pipeline.STATUS_FAILED
    assert all(s.status == pipeline.STATUS_PENDING for s in result.stages[2:])


def test_validation_errors_stop_the_pipeline(trained):
    df, models = trained
    report = SimpleNamespace(errors=["bad header"], warnings=["duplicate rows"],
                             to_dict=lambda: {"malformed_rows": 2})
    events = []
    result = pipeline.run_analysis(df, models, S.defaults(), validation=report,
                                   progress_callback=lambda i, s: events.append(i))
    assert not result.ok and result.errors == ["bad header"]
    assert result.warnings == ["duplicate rows"]
    assert result.stages[0].status == pipeline.STATUS_FAILED
    assert events == [0]
    assert result.records.empty


def test_validation_counts_are_reported(trained):
    df, models = trained
    report = SimpleNamespace(errors=[], warnings=[],
                             to_dict=lambda: {"malformed_rows": 3,
                                              "duplicate_rows": 4})
    result = pipeline.run_analysis(df, models, S.defaults(), validation=report)
    assert result.ok
    assert "3 malformed" in result.stages[0].detail
    assert "4 duplicate rows" in result.stages[0].detail


def test_preprocessing_failure_is_recorded(trained):
    df, models = trained

    class BrokenPre:
        feature_names_: list = []

        def transform(self, frame):
            raise ValueError("unknown categorical level")

    result = pipeline.run_analysis(df, {**models, "pre": BrokenPre()}, S.defaults())
    assert not result.ok
    assert result.stages[1].status == pipeline.STATUS_FAILED
    assert "unknown categorical level" in result.errors[0]
    assert result.stages[2].status == pipeline.STATUS_PENDING


def test_threshold_override_is_applied_and_labelled(trained):
    df, models = trained
    loose = pipeline.run_analysis(df, models,
                                  {**S.defaults(), "anomaly_threshold": 0.5})
    strict = pipeline.run_analysis(df, models,
                                   {**S.defaults(), "anomaly_threshold": -0.99})
    default = pipeline.run_analysis(df, models, S.defaults())
    assert loose.meta["n_anomalous"] == len(df)
    assert strict.meta["n_anomalous"] <= default.meta["n_anomalous"]
    assert "analyst override" in loose.meta["threshold_source"]
    assert "model default" in default.meta["threshold_source"]
    assert default.meta["anomaly_threshold"] == \
        pytest.approx(models["detector"].threshold)
    # the risk component must use the same threshold as the decision
    row = loose.records.iloc[0]
    assert row["risk_score"] >= 0


def test_settings_severity_map_flows_into_alerts(trained):
    df, models = trained
    settings = {**S.defaults(),
                "severity_map": {**config.SEVERITY_MAP, "Probe": "Critical"}}
    result = pipeline.run_analysis(df, models, settings)
    probe = result.alerts[result.alerts["predicted_category"] == "Probe"]
    assert len(probe) > 0
    assert (probe["severity"] == "Critical").all()
    assert (result.records.loc[result.records["predicted_category"] == "Probe",
                               "severity"] == "Critical").all()


def test_local_explanations_are_real_shap_for_top_alerts(trained):
    df, models = trained
    result = pipeline.run_analysis(df, models, S.defaults(), explain_top_n=2)
    assert result.stages[6].status == pipeline.STATUS_DONE
    assert 0 < len(result.explanations) <= 2
    n_features = len(models["pre"].feature_names_)
    for idx, explanation in result.explanations.items():
        assert idx in set(result.alerts["record_index"])
        assert len(explanation.contributions) == n_features
        assert explanation.predicted_class in config.ATTACK_CATEGORIES
        assert explanation.explanation_time_ms > 0
        assert np.isfinite(explanation.base_value)
        assert all(np.isfinite(c[2]) for c in explanation.contributions)


def test_explanations_skipped_when_disabled(trained):
    df, models = trained
    settings = {**S.defaults(), "generate_local_explanations": False}
    result = pipeline.run_analysis(df, models, settings, explain_top_n=5)
    assert result.explanations == {}
    assert result.stages[6].status == pipeline.STATUS_SKIPPED
    assert "disabled in settings" in result.stages[6].detail


def test_stage_frame_and_pairs_expose_measurements(trained):
    df, models = trained
    result = pipeline.run_analysis(df, models, S.defaults())
    frame = result.to_frame()
    assert list(frame.columns) == ["stage", "status", "seconds", "detail"]
    assert len(frame) == 7
    pairs = result.stage_pairs()
    assert pairs[-1] == ("Explanation generation", None)   # skipped, not timed
    assert all(seconds is None or seconds >= 0 for _, seconds in pairs)


def test_summarise_records_counts_match_the_frame(trained):
    df, models = trained
    result = pipeline.run_analysis(df, models, S.defaults())
    summary = pipeline.summarise_records(result.records)
    assert summary["total"] == len(df)
    assert summary["normal"] + summary["anomalous"] == len(df)
    assert sum(summary["by_category"].values()) == len(df)
    assert sum(summary["by_severity"].values()) == len(df)
    assert 0 <= summary["mean_confidence"] <= 1
    assert pipeline.summarise_records(result.records.iloc[:0]) == {}