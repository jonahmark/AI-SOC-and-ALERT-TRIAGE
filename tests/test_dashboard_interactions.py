"""Interaction tests for the dashboard's write paths.

These drive the real pages through Streamlit's AppTest harness (widget values,
button clicks, form submits) while the alert store, settings file, activity log
and report directory are redirected to a temporary folder, so verifying the
workflow never mutates the project's persisted analyst state.
"""
from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
from dashboard import common
from src import settings as S
from src.alert_generator import AlertStore

SUBMIT_KEY = "FormSubmitter:settings_form-Save configuration"

requires_artefacts = pytest.mark.skipif(
    not all(Path(p).exists() for p in config.MODEL_FILES.values())
    or not (config.DATA_PROCESSED_DIR / "alerts.csv").exists(),
    reason="trained models / generated alert store not present",
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every persisted write to tmp_path and seed real alerts."""
    alerts_file = tmp_path / "alerts.csv"
    shutil.copy(config.DATA_PROCESSED_DIR / "alerts.csv", alerts_file)
    monkeypatch.setattr(AlertStore.__init__, "__defaults__",
                        (alerts_file, tmp_path / "statuses.json",
                         tmp_path / "notes.json"))
    monkeypatch.setattr(S, "SETTINGS_FILE", tmp_path / "app_settings.json")
    monkeypatch.setattr(S, "LOG_FILE", tmp_path / "activity.log")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    common.get_alerts.clear()
    yield tmp_path
    common.get_alerts.clear()


def _run(module_name: str, token: str, timeout: float = 600.0) -> AppTest:
    module = importlib.import_module(f"dashboard.pages.{module_name}")
    source = Path(module.__file__).read_text(encoding="utf-8")
    at = AppTest.from_string(source + "\nrender()\n",
                             default_timeout=timeout)
    at.session_state["auth_token"] = token
    return at.run()


def _widget(elements, key):
    for element in elements:
        if element.key == key:
            return element
    raise AssertionError(f"widget '{key}' not rendered; keys="
                         f"{[e.key for e in elements]}")


def _picked_alert_id(at: AppTest) -> str:
    """Alert ID of the first option in the investigation selectbox."""
    return _widget(at.selectbox, "am_pick").value.split("  |")[0]


# ------------------------------------------------------------------ settings
@requires_artefacts
def test_settings_form_persists_valid_changes(sandbox, admin_token):
    at = _run("settings_page", admin_token)
    _widget(at.selectbox, "st_classifier").set_value("random_forest")
    _widget(at.number_input, "st_page_size").set_value(50)
    _widget(at.selectbox, "st_sev_Probe").set_value("High")
    at = _widget(at.button, SUBMIT_KEY).click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    saved = S.load()
    assert saved["classifier"] == "random_forest"
    assert saved["table_page_size"] == 50
    assert saved["severity_map"]["Probe"] == "High"
    assert any("Configuration saved" in s.value for s in at.success)


@requires_artefacts
def test_settings_form_rejects_weights_that_do_not_sum_to_one(sandbox, admin_token):
    at = _run("settings_page", admin_token)
    _widget(at.number_input, "st_w_severity").set_value(0.9)
    at = _widget(at.button, SUBMIT_KEY).click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("sum to exactly 1.0" in e.value for e in at.error)
    assert any("Nothing was saved" in w.value for w in at.warning)
    assert not S.SETTINGS_FILE.exists(), "invalid settings must not be written"


@requires_artefacts
def test_settings_reset_restores_research_defaults(sandbox, admin_token):
    S.save({**S.defaults(), "classifier": "random_forest"})
    at = _run("settings_page", admin_token)
    at = _widget(at.button, "st_reset").click().run()
    at = _widget(at.button, "st_reset_yes").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert S.load()["classifier"] == S.defaults()["classifier"]


# ------------------------------------------------------------ alert workflow
@requires_artefacts
def test_alert_status_change_persists(sandbox, admin_token):
    at = _run("alert_management", admin_token)
    alert_id = _picked_alert_id(at)
    _widget(at.selectbox, f"am_status_{alert_id}").set_value("Investigating")
    at = at.run()
    at = _widget(at.button, f"am_update_{alert_id}").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    store = AlertStore()
    assert store.load().set_index("alert_id").loc[
        alert_id, "investigation_status"] == "Investigating"


@requires_artefacts
def test_closing_an_alert_requires_confirmation(sandbox, admin_token):
    at = _run("alert_management", admin_token)
    alert_id = _picked_alert_id(at)
    _widget(at.selectbox, f"am_status_{alert_id}").set_value("Resolved")
    at = at.run()
    assert _widget(at.button, f"am_update_{alert_id}").disabled
    _widget(at.checkbox, f"am_confirm_{alert_id}").check()
    at = at.run()
    assert not _widget(at.button, f"am_update_{alert_id}").disabled
    at = _widget(at.button, f"am_update_{alert_id}").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert AlertStore().load().set_index("alert_id").loc[
        alert_id, "investigation_status"] == "Resolved"


@requires_artefacts
def test_analyst_note_persists_and_is_displayed(sandbox, admin_token):
    at = _run("alert_management", admin_token)
    alert_id = _picked_alert_id(at)
    _widget(at.text_area, f"am_note_{alert_id}").set_value(
        "Checked src_bytes against the baseline; consistent with a scan.")
    at = at.run()
    at = _widget(at.button, f"am_add_note_{alert_id}").click().run()
    assert not at.exception, [e.message for e in at.exception]
    notes = AlertStore().get_notes(alert_id)
    assert len(notes) == 1 and notes[0]["text"].startswith("Checked src_bytes")
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Checked src_bytes" in rendered


@requires_artefacts
def test_local_shap_button_produces_a_real_explanation(sandbox, admin_token):
    at = _run("alert_management", admin_token)
    at = _widget(at.button, "am_shap_btn").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "The classifier predicted" in rendered
    assert "SHAP" in rendered


# --------------------------------------------------------- network analysis
@requires_artefacts
def test_builtin_dataset_analysis_and_alert_export(sandbox, admin_token):
    at = _run("network_analysis", admin_token)
    _widget(at.selectbox, "na_builtin_limit").set_value("First 1,000")
    at = _widget(at.button, "na_load").click().run()
    assert not at.exception, [e.message for e in at.exception]
    loaded = at.session_state["na_builtin_state"]
    assert len(loaded["df"]) == 1000 and not loaded["report"].errors
    at = _widget(at.button, "na_run").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    result = at.session_state["na_result"]
    assert result.ok and result.meta["n_records"] == 1000
    assert not result.records.empty
    assert any(b.key == "na_download" for b in at.download_button)
    before = len(AlertStore().load())
    at = _widget(at.checkbox, "na_confirm_store").check().run()
    at = _widget(at.button, "na_save").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(AlertStore().load()) > before


# ------------------------------------------------------------------ reports
@requires_artefacts
def test_report_page_generates_and_saves_a_real_pdf(sandbox, admin_token):
    at = _run("reports", admin_token)
    assert not at.exception, [e.message for e in at.exception]
    at = _widget(at.button, "rp_pdf_btn").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    pdf = at.session_state["rp_pdf_bytes"]
    assert pdf.startswith(b"%PDF-") and len(pdf) > 5000
    assert any(b.key == "rp_pdf_download" for b in at.download_button)
    at = _widget(at.button, "rp_pdf_save").click().run()
    assert not at.exception, [e.message for e in at.exception]
    saved = list((sandbox / "reports").glob("*.pdf"))
    assert len(saved) == 1 and saved[0].read_bytes().startswith(b"%PDF-")
