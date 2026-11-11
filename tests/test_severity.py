"""Tests for the severity engine and risk prioritisation."""
from __future__ import annotations

import numpy as np
import pytest

import config
from src.severity_engine import SeverityEngine


@pytest.fixture
def engine():
    return SeverityEngine()


def test_default_severity_mapping(engine):
    assert engine.severity_for("Normal") == "Informational"
    assert engine.severity_for("Probe") == "Medium"
    assert engine.severity_for("DoS") == "High"
    assert engine.severity_for("R2L") == "High"
    assert engine.severity_for("U2R") == "Critical"


def test_unknown_category_defaults_informational(engine):
    assert engine.severity_for("SomethingNew") == "Informational"


def test_severity_map_is_configurable():
    engine = SeverityEngine(severity_map={"Normal": "Low", "DoS": "Critical"})
    assert engine.severity_for("Normal") == "Low"
    assert engine.severity_for("DoS") == "Critical"


def test_risk_score_range(engine):
    for category in config.ATTACK_CATEGORIES:
        for conf in (0.0, 0.5, 1.0):
            for score in (0.0, -0.45, -0.95):
                r = engine.risk_score(category, conf, score)
                assert 0.0 <= r <= 100.0


def test_risk_score_monotonic_in_inputs(engine):
    base = engine.risk_score("DoS", 0.5, -0.5)
    assert engine.risk_score("DoS", 0.9, -0.5) > base   # confidence
    assert engine.risk_score("U2R", 0.5, -0.5) > base   # severity
    assert engine.risk_score("DoS", 0.5, -0.9) > base   # anomaly score


def test_risk_score_formula_transparent(engine):
    """Verify the documented weighted formula exactly."""
    w = config.RISK_WEIGHTS
    expected = 100 * (
        w["severity"] * config.SEVERITY_WEIGHTS["Critical"]
        + w["confidence"] * 0.8
        + w["anomaly"] * 0.5
    )
    assert engine.risk_score("U2R", 0.8,
                             config.ANOMALY_THRESHOLD - 0.25) == \
        pytest.approx(round(expected, 2))


def test_normal_low_confidence_low_risk(engine):
    assert engine.risk_score("Normal", 0.5, 0.0) < 20.0


def test_recommended_actions(engine):
    for category in config.ATTACK_CATEGORIES:
        action = SeverityEngine.recommended_action(
            category, config.SEVERITY_MAP[category])
        assert isinstance(action, str) and len(action) > 20


# ------------------------------------------------- batch / scalar parity
def test_severity_batch_matches_scalar(engine):
    cats = np.array(config.ATTACK_CATEGORIES + ["SomethingNew", "Normal"])
    batch = engine.severity_for_batch(cats)
    assert list(batch) == [engine.severity_for(c) for c in cats]


@pytest.mark.parametrize("threshold", [None, config.ANOMALY_THRESHOLD, -0.9, 0.3])
def test_risk_batch_matches_scalar_exactly(engine, threshold):
    """The vectorised path must return the same values as the per-row formula."""
    rng = np.random.default_rng(42)
    cats = rng.choice(config.ATTACK_CATEGORIES + ["Unknown"], size=2000)
    conf = rng.uniform(0.0, 1.0, size=2000)
    scores = rng.uniform(-1.5, 0.5, size=2000)
    # exercise the documented edge cases: NaN score, clipping, boundaries
    scores[0] = np.nan
    scores[1] = 10.0          # far above threshold -> component 0
    scores[2] = -10.0         # far below threshold -> component 1
    conf[3] = np.nan          # NaN propagates identically in both paths

    batch = engine.risk_score_batch(cats, conf, scores, threshold=threshold)
    scalar = np.array([
        engine.risk_score(str(c), float(cf), float(s), threshold=threshold)
        for c, cf, s in zip(cats, conf, scores)
    ])
    assert np.array_equal(batch, scalar, equal_nan=True)


def test_risk_batch_custom_weights_match_scalar():
    engine = SeverityEngine(severity_map={"DoS": "Critical", "Normal": "Low"})
    weights = {"severity": 0.5, "confidence": 0.3, "anomaly": 0.2}
    cats = np.array(["DoS", "Normal", "U2R", "Probe"])
    conf = np.array([0.9, 0.4, 0.7, 0.2])
    scores = np.array([-0.8, 0.1, -0.5, -0.45])
    batch = engine.risk_score_batch(cats, conf, scores, weights=weights,
                                    threshold=-0.45)
    scalar = [engine.risk_score(c, cf, s, weights=weights, threshold=-0.45)
              for c, cf, s in zip(cats, conf, scores)]
    assert list(batch) == scalar


def test_risk_batch_no_anomaly_scores(engine):
    cats = np.array(["DoS", "U2R", "Normal"])
    conf = np.array([0.8, 0.6, 0.9])
    batch = engine.risk_score_batch(cats, conf)
    scalar = [engine.risk_score(c, cf) for c, cf in zip(cats, conf)]
    assert list(batch) == scalar


def test_risk_batch_empty(engine):
    assert engine.risk_score_batch([], []).shape == (0,)
    assert engine.severity_for_batch([]).shape == (0,)
