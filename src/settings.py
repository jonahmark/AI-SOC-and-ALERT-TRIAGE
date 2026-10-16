"""Persisted, validated application settings for the dashboard.

`config.py` holds the research defaults and reproducibility constants. This
module holds the analyst-editable runtime configuration that the dashboard
reads and writes (`data/processed/app_settings.json`).

Rules enforced here:
- every value is validated before it is written (bad input never reaches the
  model pipeline);
- defaults always come from `config.py`, so a deleted settings file restores
  the documented research configuration;
- nothing sensitive (paths, credentials, internal state) is stored.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import config

SETTINGS_FILE: Path = config.APP_SETTINGS_FILE
LOG_FILE: Path = config.APP_LOG_FILE

DATASETS = {
    "KDDTest+": config.TEST_FILE,
    "KDDTrain+": config.TRAIN_FILE,
}
CLASSIFIERS = ["xgboost", "random_forest"]
MAX_LOG_LINES = 500


def defaults() -> dict:
    """Documented research defaults (never mutated)."""
    return {
        "dataset": "KDDTest+",
        "classifier": "xgboost",
        "anomaly_threshold": None,          # None = threshold saved with the model
        "severity_map": dict(config.SEVERITY_MAP),
        "risk_weights": dict(config.RISK_WEIGHTS),
        "shap_top_k": int(config.SHAP_LOCAL_TOP_K),
        "shap_global_sample_size": int(config.SHAP_GLOBAL_SAMPLE_SIZE),
        "generate_local_explanations": True,
        "show_ground_truth": True,
        "table_page_size": 25,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _num(value, lo: float, hi: float, errors: list, label: str):
    try:
        out = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number.")
        return None
    if not (lo <= out <= hi):
        errors.append(f"{label} must be between {lo} and {hi}.")
        return None
    return out


def _int(value, lo: int, hi: int, errors: list, label: str):
    try:
        out = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a whole number.")
        return None
    if not (lo <= out <= hi):
        errors.append(f"{label} must be between {lo} and {hi}.")
        return None
    return out


def validate(payload: dict) -> tuple[dict, list[str]]:
    """Return (clean settings, errors). `clean` is usable when errors is empty."""
    errors: list[str] = []
    base = defaults()
    out = dict(base)
    if not isinstance(payload, dict):
        return base, ["Settings must be a mapping of option names to values."]

    dataset = payload.get("dataset", base["dataset"])
    if dataset in DATASETS:
        out["dataset"] = dataset
    else:
        errors.append(f"Dataset must be one of {sorted(DATASETS)}.")

    classifier = payload.get("classifier", base["classifier"])
    if classifier in CLASSIFIERS:
        out["classifier"] = classifier
    else:
        errors.append(f"Classifier must be one of {CLASSIFIERS}.")

    threshold = payload.get("anomaly_threshold", base["anomaly_threshold"])
    if threshold in (None, "", "auto"):
        out["anomaly_threshold"] = None
    else:
        value = _num(threshold, -20.0, 20.0, errors, "Anomaly threshold")
        if value is not None:
            out["anomaly_threshold"] = round(value, 6)

    severity_map = payload.get("severity_map", base["severity_map"])
    if isinstance(severity_map, dict):
        clean_map = {}
        for category in config.ATTACK_CATEGORIES:
            severity = severity_map.get(category, base["severity_map"][category])
            if severity in config.SEVERITY_ORDER:
                clean_map[category] = severity
            else:
                errors.append(
                    f"Severity for '{category}' must be one of "
                    f"{config.SEVERITY_ORDER}."
                )
                clean_map[category] = base["severity_map"][category]
        out["severity_map"] = clean_map
    else:
        errors.append("Severity mapping must be a mapping of category to severity.")

    weights = payload.get("risk_weights", base["risk_weights"])
    if isinstance(weights, dict):
        clean_w = {}
        ok = True
        for key in ("severity", "confidence", "anomaly"):
            value = _num(weights.get(key, base["risk_weights"][key]), 0.0, 1.0,
                         errors, f"Risk weight '{key}'")
            if value is None:
                ok = False
            else:
                clean_w[key] = value
        if ok:
            if abs(sum(clean_w.values()) - 1.0) > 1e-6:
                errors.append(
                    "Risk weights must sum to exactly 1.0 "
                    f"(currently {sum(clean_w.values()):.4f})."
                )
            else:
                out["risk_weights"] = clean_w
    else:
        errors.append("Risk weights must be a mapping of severity/confidence/anomaly.")

    out["shap_top_k"] = _int(payload.get("shap_top_k", base["shap_top_k"]),
                             1, 25, errors, "SHAP top-K") or base["shap_top_k"]
    out["shap_global_sample_size"] = _int(
        payload.get("shap_global_sample_size", base["shap_global_sample_size"]),
        50, 5000, errors, "SHAP global sample size",
    ) or base["shap_global_sample_size"]
    out["generate_local_explanations"] = bool(
        payload.get("generate_local_explanations",
                    base["generate_local_explanations"])
    )
    out["show_ground_truth"] = bool(
        payload.get("show_ground_truth", base["show_ground_truth"])
    )
    out["table_page_size"] = _int(payload.get("table_page_size",
                                              base["table_page_size"]),
                                  10, 500, errors,
                                  "Table page size") or base["table_page_size"]
    return out, errors


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load() -> dict:
    """Load settings, silently falling back to defaults for unknown keys."""
    settings = defaults()
    if SETTINGS_FILE.exists():
        try:
            stored = json.loads(SETTINGS_FILE.read_text())
        except json.JSONDecodeError:
            log("WARNING: app_settings.json was unreadable; using defaults.")
            return settings
        clean, errors = validate(stored)
        if errors:
            log("WARNING: stored settings invalid; defaults applied: "
                + " ".join(errors))
        settings = clean
        settings["saved_at"] = stored.get("saved_at")
    return settings


def save(payload: dict) -> tuple[dict, list[str]]:
    """Validate and persist. Returns (settings, errors); nothing is written
    when errors is non-empty."""
    clean, errors = validate(payload)
    if errors:
        return clean, errors
    clean["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(clean, indent=2))
    log("Settings updated.")
    return clean, []


def reset() -> dict:
    """Restore documented defaults (deletes the stored file)."""
    if SETTINGS_FILE.exists():
        SETTINGS_FILE.unlink()
    log("Settings reset to research defaults.")
    return defaults()


# ---------------------------------------------------------------------------
# Helpers used by the dashboard
# ---------------------------------------------------------------------------
def effective_threshold(settings: dict, model_threshold: float) -> float:
    """Threshold actually applied to the anomaly decision."""
    override = settings.get("anomaly_threshold")
    return float(override) if override is not None else float(model_threshold)


def severity_engine(settings: dict):
    from src.severity_engine import SeverityEngine
    return SeverityEngine(severity_map=dict(settings["severity_map"]))


def alert_generator(settings: dict):
    from src.alert_generator import AlertGenerator
    return AlertGenerator(
        severity_engine=severity_engine(settings),
        risk_weights=dict(settings["risk_weights"]),
    )


def dataset_path(settings: dict) -> Path:
    return DATASETS[settings.get("dataset", "KDDTest+")]


def settings_diff(settings: dict) -> list[str]:
    """Human-readable list of options that differ from research defaults."""
    base = defaults()
    diffs = []
    for key, value in base.items():
        current = settings.get(key)
        if current != value:
            diffs.append(f"{key}: {value!r} -> {current!r}")
    return diffs


# ---------------------------------------------------------------------------
# Activity log (plain text, capped; no secrets, no absolute paths)
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()
    lines.append(f"{stamp}  {message}")
    LOG_FILE.write_text("\n".join(lines[-MAX_LOG_LINES:]) + "\n")


def read_log(n: int = 100) -> list[str]:
    if not LOG_FILE.exists():
        return []
    return LOG_FILE.read_text().splitlines()[-n:]
