"""Shared cached loaders and backend access helpers for the dashboard pages.

All data shown in the UI comes from these functions: trained model artefacts,
evaluation JSON, resource-metric JSON, the alert store and the raw NSL-KDD
files. Nothing here fabricates values.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from src import analysis_cache, pipeline, settings as settings_module

# Timestamp of the most recent live analysis computed in this server process
# (used only for the "Last analysis" label; no result data is kept here).
_last_analysis_ran_at: str | None = None


# ---------------------------------------------------------------------------
# Models (loaded once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_models() -> dict:
    from src.preprocessing import Preprocessor
    from src.anomaly_detection import AnomalyDetector
    from src.attack_classification import AttackClassifier

    errors: dict[str, str] = {}
    models: dict = {}
    try:
        models["pre"] = Preprocessor.load()
    except FileNotFoundError as exc:
        errors["preprocessing"] = str(exc)
    try:
        models["detector"] = AnomalyDetector.load()
    except FileNotFoundError as exc:
        errors["isolation_forest"] = str(exc)
    for name in ("random_forest", "xgboost"):
        try:
            models[name] = AttackClassifier.load(name)
        except FileNotFoundError as exc:
            errors[name] = str(exc)
    models["_errors"] = errors
    return models


def models_available(models: dict) -> bool:
    return not models.get("_errors")


def render_missing_models_notice(models: dict) -> None:
    from dashboard import components as ui
    ui.empty_state(
        "Trained model artefacts are missing",
        "The dashboard can only show real results once the models exist. "
        "Run <code>python train.py</code> from the project root, then refresh.",
        "alert",
    )
    for name, msg in models.get("_errors", {}).items():
        st.markdown(f"- **{name}**: {msg}")


# ---------------------------------------------------------------------------
# Settings (cheap JSON read; never cached so analyst edits apply immediately)
# ---------------------------------------------------------------------------
def get_settings() -> dict:
    return settings_module.load()


def save_settings(payload: dict) -> tuple[dict, list[str]]:
    clean, errors = settings_module.save(payload)
    st.cache_data.clear()
    return clean, errors


# ---------------------------------------------------------------------------
# Artefacts on disk
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict | None:
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


@st.cache_data(show_spinner=False)
def get_pipeline_state() -> dict | None:
    return load_json(config.PIPELINE_STATE_FILE)


@st.cache_data(show_spinner=False)
def get_evaluation() -> dict | None:
    return load_json(config.METRICS_DIR / "evaluation.json")


@st.cache_data(show_spinner=False)
def get_resource_metrics() -> dict | None:
    return load_json(config.METRICS_DIR / "resource_metrics.json")


def get_alert_store():
    from src.alert_generator import AlertStore
    return AlertStore()


@st.cache_data(show_spinner="Loading alerts")
def get_alerts() -> pd.DataFrame:
    return get_alert_store().load()


def invalidate_alerts() -> None:
    get_alerts.clear()


@st.cache_data(show_spinner="Loading dataset")
def dataset_frame(dataset_name: str) -> pd.DataFrame | None:
    """Raw NSL-KDD frame for the selected dataset (labels included)."""
    path = settings_module.DATASETS.get(dataset_name)
    if path is None:
        return None
    from src import data_loader
    try:
        df, _ = data_loader.load_nsl_kdd_file(path)
        return df
    except FileNotFoundError:
        return None


def dataset_validation(dataset_name: str):
    path = settings_module.DATASETS.get(dataset_name)
    if path is None:
        return None
    from src import data_loader
    try:
        _, report = data_loader.load_nsl_kdd_file(path)
        return report
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def _compute_dataset_analysis(dataset: str, classifier: str, anomaly_threshold,
                              severity_map_json: str, risk_weights_json: str,
                              max_records, explain_top_n: int):
    """Run the real pipeline once per server process for a given configuration.

    Cached process-wide (not per browser session) so signing in or moving
    between pages reuses the result instead of re-running the pipeline. The
    key covers every setting that changes the output, so edited settings
    naturally produce a fresh computation.
    """
    settings = get_settings()
    key = analysis_cache.cache_key(
        settings, max_records=max_records, explain_top_n=explain_top_n,
        model_fingerprint=analysis_cache.model_fingerprint(),
    )
    # Reuse a persisted result across server restarts when the configuration
    # and models are unchanged; otherwise run the real pipeline.
    persisted = analysis_cache.load(key)
    if persisted is not None:
        settings_module.log(
            f"Analysis restored from disk cache ({dataset}, "
            f"{len(persisted.records):,} records).")
        return persisted
    models = get_models()
    df = dataset_frame(dataset)
    if df is None:
        return None
    if max_records is not None:
        df = df.head(int(max_records))
    report = dataset_validation(dataset)
    result = pipeline.run_analysis(
        df, models, settings, validation=report,
        explain_top_n=explain_top_n, source_label=dataset,
    )
    # Persist only the default dashboard view (full dataset, no on-demand SHAP)
    # so restarts reuse it; variant runs stay in-memory only.
    if result.ok and explain_top_n == 0 and max_records is None:
        analysis_cache.save(result, key)
    return result


def invalidate_analysis() -> None:
    """Drop the shared analysis cache (called when settings change)."""
    global _last_analysis_ran_at
    _compute_dataset_analysis.clear()
    analysis_cache.clear()
    _last_analysis_ran_at = None


def _copy_analysis(result):
    """Defensive copy so a caller cannot mutate the shared cached result.

    The DataFrames are copied (a few MB); the read-only stage list and meta
    are shared. This keeps the expensive computation cached while giving each
    caller independent frames.
    """
    if result is None:
        return None
    from dataclasses import replace
    return replace(result, records=result.records.copy(),
                   alerts=result.alerts.copy())


def run_dataset_analysis(settings: dict, explain_top_n: int = 0,
                         max_records: int | None = None):
    """Return the real pipeline result for the selected dataset.

    Computed once per configuration and shared across sessions; each call
    gets its own copy of the result frames.
    """
    global _last_analysis_ran_at
    models = get_models()
    if not models_available(models):
        return None
    if dataset_frame(settings["dataset"]) is None:
        return None
    result = _compute_dataset_analysis(
        settings.get("dataset"), settings.get("classifier"),
        settings.get("anomaly_threshold"),
        json.dumps(settings.get("severity_map"), sort_keys=True),
        json.dumps(settings.get("risk_weights"), sort_keys=True),
        max_records, explain_top_n,
    )
    if result is not None and result.ok:
        _last_analysis_ran_at = (result.meta or {}).get("ran_at")
    return _copy_analysis(result)


def model_status_label(models: dict) -> tuple[str, str]:
    """(text, colour) for the top bar / sidebar model indicator."""
    if models_available(models):
        state = get_pipeline_state() or {}
        trained = state.get("trained_at")
        return (f"Ready &middot; {trained[:10]}" if trained else "Ready", "#22C55E")
    missing = len(models.get("_errors", {}))
    return (f"{missing} missing", "#EF4444")


def last_analysis_label() -> str:
    """Most recent real training/analysis timestamp, normalised to UTC."""
    from dashboard.components import timestamp_label
    state = get_pipeline_state() or {}
    stamps = [state.get("trained_at"), _last_analysis_ran_at]
    latest = None
    for stamp in stamps:
        if not stamp:
            continue
        try:
            dt = datetime.fromisoformat(str(stamp)).astimezone(timezone.utc)
        except ValueError:
            continue
        if latest is None or dt > latest:
            latest = dt
    return timestamp_label(latest.isoformat()) if latest else "not run yet"


def notification_count(alerts: pd.DataFrame) -> int:
    """Real count: untriaged Critical/High alerts."""
    if alerts is None or alerts.empty:
        return 0
    mask = (
        alerts["severity"].isin(["Critical", "High"])
        & (alerts["investigation_status"] == "New")
    )
    return int(mask.sum())


def artefact_status(settings: dict | None = None) -> tuple[str, str, str]:
    """(system, models, dataset) status strings from real file existence.

    Used by the sidebar footer; does not load models (cheap and honest).
    """
    settings = settings or get_settings()
    missing_models = [name for name, path in config.MODEL_FILES.items()
                      if not Path(path).exists()]
    if not Path(config.PIPELINE_STATE_FILE).exists():
        missing_models.append("pipeline_state")
    model_state = "Ready" if not missing_models else f"{len(missing_models)} missing"

    dataset_path = settings_module.dataset_path(settings)
    if dataset_path.exists():
        dataset_state = f"{settings['dataset']} present"
    else:
        dataset_state = f"{settings['dataset']} missing"

    metrics_missing = not Path(config.METRICS_DIR / "evaluation.json").exists()
    if not missing_models and not metrics_missing and dataset_path.exists():
        system_state = "Operational"
    elif missing_models:
        system_state = "Models unavailable"
    else:
        system_state = "Degraded"
    return system_state, model_state, dataset_state


def live_system_snapshot() -> dict:
    """Real CPU/memory reading taken at call time (never simulated)."""
    import platform

    import psutil
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "system_memory_percent": vm.percent,
        "system_memory_used_mb": round((vm.total - vm.available) / 1024 / 1024, 1),
        "system_memory_total_mb": round(vm.total / 1024 / 1024, 1),
        "process_memory_mb": round(proc.memory_info().rss / 1024 / 1024, 1),
        "platform": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "sampled_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
