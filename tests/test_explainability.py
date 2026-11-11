"""Tests for SHAP explainability (global vs local)."""
from __future__ import annotations

import numpy as np
import pytest

import config
from src.attack_classification import AttackClassifier
from src.explainability import Explainer, LocalExplanation
from src.preprocessing import Preprocessor


@pytest.fixture(scope="module")
def fitted():
    from tests.conftest import synthetic_nsl_kdd
    df = synthetic_nsl_kdd(n=300)
    y = df[config.LABEL_COLUMN].map(config.ATTACK_CATEGORY_MAP).to_numpy()
    pre = Preprocessor().fit(df)
    X = pre.transform(df)
    clf = AttackClassifier("random_forest").fit(X, y)
    return Explainer(clf, pre.feature_names_), X


def test_local_explanation_structure(fitted):
    explainer, X = fitted
    exp = explainer.explain_record(X[0])
    assert isinstance(exp, LocalExplanation)
    assert exp.predicted_class in config.ATTACK_CATEGORIES
    assert len(exp.contributions) == X.shape[1]
    assert exp.explanation_time_ms is not None
    assert exp.explanation_time_ms > 0  # real measured latency


def test_local_explanation_plain_language(fitted):
    explainer, X = fitted
    exp = explainer.explain_record(X[1])
    text = exp.to_plain_language()
    assert exp.predicted_class in text
    assert "SHAP" in text
    assert len(text) > 50


def test_local_explanations_differ_per_record(fitted):
    """Local SHAP must vary by record (unlike global importance)."""
    explainer, X = fitted
    e1 = explainer.explain_record(X[0])
    e2 = explainer.explain_record(X[-1])
    c1 = dict((f, s) for f, _, s in e1.contributions)
    c2 = dict((f, s) for f, _, s in e2.contributions)
    assert any(abs(c1[k] - c2[k]) > 1e-9 for k in c1)


def test_global_importance_shape_and_cache(fitted, tmp_path, monkeypatch):
    import src.explainability as ex
    monkeypatch.setattr(ex, "GLOBAL_SHAP_FILE", tmp_path / "global.json")
    explainer, X = fitted
    df_imp, seconds = explainer.global_importance(X[:100], use_cache=False)
    assert list(df_imp.columns) == ["feature", "mean_abs_shap"]
    assert len(df_imp) == X.shape[1]
    assert df_imp["mean_abs_shap"].is_monotonic_decreasing
    assert seconds > 0
    # cached second call returns the same features
    df_cached, _ = explainer.global_importance(X[:100], use_cache=True)
    assert df_cached["feature"].tolist() == df_imp["feature"].tolist()
