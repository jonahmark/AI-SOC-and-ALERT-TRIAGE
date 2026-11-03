"""Disk persistence for a completed analysis (survives server restarts).

The dashboard's default analysis (full dataset, no on-demand SHAP) is written
to ``data/processed/analysis_cache/`` so a restarted server can reuse the last
real result instead of re-running the pipeline. Frames are stored as parquet
(exact dtypes, no new dependency - pyarrow is already installed); stage timings
and meta are JSON.

Nothing here fabricates results: a persisted entry is only reused when its key
(every output-affecting setting plus the model ``trained_at`` fingerprint)
matches the current configuration. Missing, corrupt or stale entries are
ignored and the pipeline runs again. Local SHAP explanations are computed on
demand elsewhere and are not part of the persisted default analysis.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import config
from src.pipeline import AnalysisResult, StageResult

CACHE_DIR: Path = config.DATA_PROCESSED_DIR / "analysis_cache"


def cache_key(settings: dict, max_records=None, explain_top_n: int = 0,
              model_fingerprint: str = "") -> str:
    """Stable key for an analysis configuration.

    Includes the model ``trained_at`` fingerprint so retraining invalidates any
    persisted result automatically.
    """
    payload = {
        "dataset": settings.get("dataset"),
        "classifier": settings.get("classifier"),
        "anomaly_threshold": settings.get("anomaly_threshold"),
        "severity_map": settings.get("severity_map"),
        "risk_weights": settings.get("risk_weights"),
        "max_records": max_records,
        "explain_top_n": explain_top_n,
        "model_fingerprint": model_fingerprint,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_fingerprint() -> str:
    """The current models' training timestamp (empty when unavailable)."""
    try:
        state = json.loads(config.PIPELINE_STATE_FILE.read_text(encoding="utf-8"))
        return str(state.get("trained_at") or "")
    except (OSError, json.JSONDecodeError):
        return ""


def load(key: str) -> AnalysisResult | None:
    """Return the persisted analysis for `key`, or None if absent/invalid."""
    try:
        meta_doc = json.loads((CACHE_DIR / "result.json").read_text(encoding="utf-8"))
        if meta_doc.get("key") != key:
            return None
        records = pd.read_parquet(CACHE_DIR / "records.parquet")
        alerts = pd.read_parquet(CACHE_DIR / "alerts.parquet")
        stages = [StageResult(**s) for s in meta_doc.get("stages", [])]
        return AnalysisResult(
            records=records,
            alerts=alerts,
            stages=stages,
            errors=list(meta_doc.get("errors", [])),
            warnings=list(meta_doc.get("warnings", [])),
            explanations={},
            meta=dict(meta_doc.get("meta", {})),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(result: AnalysisResult, key: str) -> None:
    """Persist a successful analysis. Best-effort: a write failure is ignored.

    ``result.json`` (which carries the key) is written last so a partial write
    from a crash is treated as invalid on the next load.
    """
    if not result.ok:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.records.to_parquet(CACHE_DIR / "records.parquet", index=False)
        result.alerts.to_parquet(CACHE_DIR / "alerts.parquet", index=False)
        doc = {
            "key": key,
            "stages": [asdict(s) for s in result.stages],
            "errors": result.errors,
            "warnings": result.warnings,
            "meta": result.meta,
        }
        (CACHE_DIR / "result.json").write_text(
            json.dumps(doc, indent=2, default=str), encoding="utf-8")
    except (OSError, ValueError):
        pass


def clear() -> None:
    """Remove the persisted analysis (called when settings change)."""
    try:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
    except OSError:
        pass