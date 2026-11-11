"""Dashboard tests: every page must render against the real artefacts.

The page smoke tests execute the actual page modules through Streamlit's
AppTest harness, so a broken import, a missing artefact path or a runtime
error in any page fails the suite instead of silently showing an empty UI.
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
from dashboard import common, components as ui, nav

PAGE_MODULES = [
    "overview", "traffic_monitor", "network_analysis", "alert_management",
    "attack_intelligence", "explainability", "model_performance",
    "resource_monitoring", "reports", "settings_page", "user_management",
]

requires_artefacts = pytest.mark.skipif(
    not all(Path(p).exists() for p in config.MODEL_FILES.values())
    or not config.TEST_FILE.exists(),
    reason="trained model artefacts / NSL-KDD test file not present",
)


def _page_app(module_name: str, token: str | None = None,
              timeout: float = 300.0) -> AppTest:
    """Run a whole page module (imports included) as a Streamlit script.

    Pages enforce role permissions server-side, so an authenticated session
    token is injected when one is supplied.
    """
    module = importlib.import_module(f"dashboard.pages.{module_name}")
    source = Path(module.__file__).read_text(encoding="utf-8")
    at = AppTest.from_string(source + "\nrender()\n", default_timeout=timeout)
    if token:
        at.session_state["auth_token"] = token
    return at


# ---------------------------------------------------------------- navigation
def test_registry_covers_every_page_once():
    pages = nav.build()
    assert set(pages) == set(PAGE_MODULES)
    assert len(pages) == len(PAGE_MODULES)


@requires_artefacts
def test_app_shell_registers_navigation_and_status_footer(admin_token):
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(root / "app.py", default_timeout=300)
    at.session_state["auth_token"] = admin_token
    at = at.run()
    assert not at.exception, [e.message for e in at.exception]
    sidebar = " ".join(str(m.value) for m in at.sidebar.markdown)
    assert "AI-SOC" in sidebar
    assert "Intelligent Security Operations" in sidebar
    assert "Operational" in sidebar or "unavailable" in sidebar
    assert config.APP_VERSION in sidebar
    assert "Test Administrator" in sidebar
    assert "Administrator" in sidebar


@requires_artefacts
def test_startup_warming_makes_the_next_visit_fast():
    """app.py loads the models once; the next run reuses the shared cache.

    Guards the startup warming: without it every fresh session re-paid the
    cold model-load cost. The first run may be slow (cold load); the second
    must be near-instant because the artefacts are already cached.
    """
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(root / "app.py", default_timeout=300)
    at = at.run()
    assert not at.exception, [e.message for e in at.exception]
    start = time.perf_counter()
    at = at.run()
    elapsed = time.perf_counter() - start
    assert not at.exception, [e.message for e in at.exception]
    assert elapsed < 10.0, f"second visit took {elapsed:.1f}s; models not warmed"


# ------------------------------------------------------------- page rendering
@requires_artefacts
@pytest.mark.parametrize("module_name", PAGE_MODULES)
def test_page_renders_without_exception(module_name, admin_token):
    at = _page_app(module_name, admin_token).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.main.children) > 0


@requires_artefacts
def test_overview_metrics_match_the_alert_store(admin_token):
    at = _page_app("overview", admin_token).run()
    assert not at.exception
    alerts = common.get_alerts()
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert f"{len(alerts):,}" in rendered
    assert common.get_settings()["dataset"] in rendered


@requires_artefacts
def test_model_performance_shows_current_evaluation(admin_token):
    from src import evaluation
    report = evaluation.load_evaluation()
    assert report is not None, "run train.py first"
    at = _page_app("model_performance", admin_token).run()
    assert not at.exception
    rendered = " ".join(str(m.value) for m in at.markdown)
    for name, metrics in report["attack_classification"].items():
        assert f"{metrics['macro_f1']:.4f}" in rendered, (
            f"{name} macro-F1 from evaluation.json not shown on the page")


@requires_artefacts
def test_alert_management_exposes_triage_controls(admin_token):
    at = _page_app("alert_management", admin_token).run()
    assert not at.exception
    assert at.selectbox, "alert selector missing"
    assert at.button, "no workflow actions rendered"


@requires_artefacts
def test_settings_form_roundtrip(admin_token):
    at = _page_app("settings_page", admin_token).run()
    assert not at.exception
    assert at.form, "settings editor is not a form"
    labels = [s.label for s in at.selectbox]
    assert any("dataset" in str(label).lower() for label in labels)


# ------------------------------------------------------------------ helpers
def test_severity_badge_is_not_colour_only():
    for severity in config.SEVERITY_ORDER:
        html = ui.severity_badge(severity)
        assert severity in html
        assert ui.SEVERITY_GLYPH[severity] in html
        assert ui.SEVERITY_COLORS[severity] in html


def test_timestamp_label_converts_offsets_to_utc():
    assert ui.timestamp_label("2026-11-07T18:25:34+03:00") == \
        "2026-11-07 15:25:34 UTC"
    assert ui.timestamp_label(None) == "not recorded"
    assert ui.timestamp_label("not a date") == "not a date"


def test_missing_measurements_are_labelled_not_zeroed():
    assert ui.fmt(None) == "not measured"
    assert ui.pct(None) == "not measured"
    assert ui.fmt(0.0) == "0.00"


def test_notification_count_uses_untriaged_high_severity_only():
    alerts = pd.DataFrame({
        "severity": ["Critical", "High", "High", "Medium", "Critical"],
        "investigation_status": ["New", "New", "Resolved", "New",
                                 "False Positive"],
    })
    assert common.notification_count(alerts) == 2
    assert common.notification_count(pd.DataFrame()) == 0


def test_charts_carry_the_dark_theme_and_transparent_background():
    fig = ui.doughnut(["Normal", "Anomaly"], [80, 20])
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.font.color == ui.TEXT_MUTED