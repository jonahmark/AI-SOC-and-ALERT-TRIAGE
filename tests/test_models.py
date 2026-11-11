"""Tests for anomaly detection and attack classification models."""
from __future__ import annotations

import numpy as np
import pytest

import config
from src.anomaly_detection import (
    AnomalyDetector, anomaly_score_to_risk_component,
)
from src.attack_classification import AttackClassifier, class_sample_weights
from src.preprocessing import Preprocessor


@pytest.fixture(scope="module")
def encoded():
    """Session-cheap module fixture: fit preprocessing once on synthetic data."""
    from tests.conftest import synthetic_nsl_kdd
    df = synthetic_nsl_kdd()
    labelled = df.copy()
    labelled[config.TARGET_COLUMN] = labelled[config.LABEL_COLUMN].map(
        config.ATTACK_CATEGORY_MAP).fillna("Unknown")
    pre = Preprocessor().fit(df)
    return pre, pre.transform(df), labelled


# ------------------------------------------------------------- anomaly
def test_isolation_forest_fit_predict(encoded):
    _, X, labelled = encoded
    normal = X[labelled[config.TARGET_COLUMN] == "Normal"]
    det = AnomalyDetector().fit(normal)
    res = det.predict(X)
    assert res.scores.shape == (len(X),)
    assert res.is_anomaly.dtype == bool
    assert res.threshold == config.ANOMALY_THRESHOLD
    assert det.train_time_seconds > 0


def test_threshold_is_configurable(encoded):
    _, X, labelled = encoded
    normal = X[labelled[config.TARGET_COLUMN] == "Normal"]
    det = AnomalyDetector(threshold=-0.1).fit(normal)
    strict = det.predict(X, threshold=-0.9).is_anomaly.sum()
    loose = det.predict(X, threshold=0.5).is_anomaly.sum()
    assert loose >= strict


def test_anomaly_detector_detects_outliers():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(500, 5)).astype(np.float32)
    det = AnomalyDetector().fit(normal)
    # data-driven threshold: flag the most anomalous 5% of training normals
    det.threshold = float(np.quantile(det.score(normal), 0.05))
    outliers = rng.normal(0, 1, size=(10, 5)).astype(np.float32) + 25
    assert det.predict(outliers).is_anomaly.all()
    assert det.predict(normal).is_anomaly.mean() <= 0.06


def test_anomaly_score_component_bounds():
    thr = config.ANOMALY_THRESHOLD
    assert anomaly_score_to_risk_component(thr + 0.1, thr) == 0.0
    assert anomaly_score_to_risk_component(
        thr - config.ANOMALY_SCORE_SPREAD, thr) == pytest.approx(1.0)
    assert 0 < anomaly_score_to_risk_component(thr - 0.1, thr) < 1
    assert anomaly_score_to_risk_component(float("nan"), thr) == 0.0


def test_detector_save_load(encoded, tmp_path):
    _, X, labelled = encoded
    normal = X[labelled[config.TARGET_COLUMN] == "Normal"]
    det = AnomalyDetector(threshold=-0.33).fit(normal)
    path = tmp_path / "iforest.pkl"
    det.save(path)
    loaded = AnomalyDetector.load(path)
    assert loaded.threshold == pytest.approx(-0.33)
    assert np.allclose(loaded.score(X[:20]), det.score(X[:20]))


def test_detector_load_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        AnomalyDetector.load(tmp_path / "absent.pkl")


# ------------------------------------------------------ classification
@pytest.mark.parametrize("model_type", ["random_forest", "xgboost"])
def test_classifier_fit_predict(model_type, encoded):
    pre, X, labelled = encoded
    y = labelled[config.TARGET_COLUMN].to_numpy()
    clf = AttackClassifier(model_type).fit(X, y)
    res = clf.predict(X)
    assert set(res.predictions) <= set(config.ATTACK_CATEGORIES)
    assert res.probabilities.shape == (len(X), len(res.classes))
    assert np.allclose(res.probabilities.sum(axis=1), 1.0)
    assert (res.confidences >= 0).all() and (res.confidences <= 1).all()
    # synthetic attacks are made strongly separable
    assert (res.predictions == y).mean() > 0.9


def test_classifier_excludes_unknown_labels(encoded):
    pre, X, labelled = encoded
    y = labelled[config.TARGET_COLUMN].to_numpy().astype(object)
    y[0] = "Unknown"
    clf = AttackClassifier("random_forest").fit(X, y)
    assert clf.n_train_samples == len(y) - 1


def test_classifier_invalid_type():
    with pytest.raises(ValueError):
        AttackClassifier("deep_magic")


# ------------------------------------------------- cost-sensitive weighting
def test_class_sample_weights_balanced():
    y = np.array(["Normal"] * 80 + ["DoS"] * 15 + ["U2R"] * 5)
    w = class_sample_weights(y, "balanced")
    assert w is not None and len(w) == len(y)
    # balanced weighting: every class contributes the same total weight
    totals = {c: float(w[y == c].sum()) for c in np.unique(y)}
    assert totals["Normal"] == pytest.approx(totals["U2R"])
    assert w[y == "U2R"][0] > w[y == "Normal"][0]
    assert class_sample_weights(y, None) is None


def test_xgboost_fit_applies_configured_sample_weights(encoded, monkeypatch):
    _, X, labelled = encoded
    y = labelled[config.TARGET_COLUMN].to_numpy()
    clf = AttackClassifier("xgboost")
    captured = {}
    original = clf.model.fit

    def spy(X_, y_, **kwargs):
        captured["sample_weight"] = kwargs.get("sample_weight")
        return original(X_, y_, **kwargs)

    monkeypatch.setattr(clf.model, "fit", spy)
    monkeypatch.setattr(config, "XGBOOST_CLASS_WEIGHTING", "balanced")
    clf.fit(X, y)
    sw = captured["sample_weight"]
    assert sw is not None and len(sw) == len(y)
    # the rarer attack classes must receive larger per-record weights
    assert sw[y == "DoS"][0] > sw[y == "Normal"][0]


def test_xgboost_weighting_can_be_disabled(encoded, monkeypatch):
    _, X, labelled = encoded
    y = labelled[config.TARGET_COLUMN].to_numpy()
    clf = AttackClassifier("xgboost")
    captured = {}
    original = clf.model.fit

    def spy(X_, y_, **kwargs):
        captured["sample_weight"] = kwargs.get("sample_weight")
        return original(X_, y_, **kwargs)

    monkeypatch.setattr(clf.model, "fit", spy)
    monkeypatch.setattr(config, "XGBOOST_CLASS_WEIGHTING", None)
    clf.fit(X, y)
    assert captured["sample_weight"] is None


def test_random_forest_uses_configured_class_weight(encoded):
    _, X, labelled = encoded
    y = labelled[config.TARGET_COLUMN].to_numpy()
    clf = AttackClassifier("random_forest").fit(X, y)
    assert clf.model.class_weight == config.RANDOM_FOREST_PARAMS["class_weight"]
    assert clf.model.class_weight == "balanced"


def test_classifier_predict_before_fit(encoded):
    _, X, _ = encoded
    with pytest.raises(RuntimeError, match="not been fitted"):
        AttackClassifier("random_forest").predict(X[:5])


def test_classifier_save_load(encoded, tmp_path):
    pre, X, labelled = encoded
    y = labelled[config.TARGET_COLUMN].to_numpy()
    clf = AttackClassifier("random_forest").fit(X, y)
    path = tmp_path / "rf.pkl"
    clf.save(path)
    loaded = AttackClassifier.load("random_forest", path)
    a = clf.predict(X[:20]).predictions
    b = loaded.predict(X[:20]).predictions
    assert (a == b).all()


def test_classifier_load_type_mismatch(encoded, tmp_path):
    pre, X, labelled = encoded
    y = labelled[config.TARGET_COLUMN].to_numpy()
    clf = AttackClassifier("random_forest").fit(X, y)
    path = tmp_path / "rf.pkl"
    clf.save(path)
    with pytest.raises(ValueError, match="but 'xgboost' was requested"):
        AttackClassifier.load("xgboost", path)


def test_stages_are_independent(encoded):
    """Anomaly decision and attack category are separate outputs."""
    pre, X, labelled = encoded
    normal = X[labelled[config.TARGET_COLUMN] == "Normal"]
    det = AnomalyDetector(threshold=-0.1).fit(normal)
    clf = AttackClassifier("random_forest").fit(
        X, labelled[config.TARGET_COLUMN].to_numpy())
    a = det.predict(X)
    c = clf.predict(X)
    # there exist records anomalous but classified Normal, or vice versa,
    # demonstrating the two decisions are not identical by construction
    assert len(a.is_anomaly) == len(c.predictions)
    combined = set(zip(a.is_anomaly.tolist(), c.predictions.tolist()))
    assert len(combined) >= 2
