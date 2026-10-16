"""End-to-end analysis pipeline shared by the dashboard pages and the tests.

The seven stages mirror the documented research pipeline:

    1 Data ingestion -> 2 Preprocessing -> 3 Anomaly detection (Isolation
    Forest) -> 4 Attack classification (Random Forest / XGBoost) -> 5 Alert
    generation -> 6 Risk prioritisation -> 7 Explanation generation (SHAP).

Stages 3 and 4 remain separate analytical decisions: an anomaly is not
assumed to be any particular attack. Every stage records its real elapsed
time so the UI can show measured progress rather than a simulated bar.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from src import settings as settings_module
from src.alert_generator import AlertGenerator
from src.anomaly_detection import AnomalyResult
from src.attack_classification import ClassificationResult

STAGES = [
    "Data ingestion",
    "Preprocessing",
    "Anomaly detection",
    "Attack classification",
    "Alert generation",
    "Risk prioritisation",
    "Explanation generation",
]

STATUS_PENDING, STATUS_DONE, STATUS_SKIPPED, STATUS_FAILED = (
    "pending", "done", "skipped", "failed",
)


@dataclass
class StageResult:
    name: str
    status: str = STATUS_PENDING
    seconds: float | None = None
    detail: str = ""


@dataclass
class AnalysisResult:
    records: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    stages: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    explanations: dict = field(default_factory=dict)   # record_index -> LocalExplanation
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def stage_pairs(self) -> list:
        """(label, seconds|None) for the UI stage tracker."""
        return [(s.name, s.seconds if s.status == STATUS_DONE else None)
                for s in self.stages]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"stage": s.name, "status": s.status,
             "seconds": s.seconds, "detail": s.detail}
            for s in self.stages
        ])


def load_dataset_frame(settings: dict) -> tuple[pd.DataFrame | None, dict]:
    """Load the dataset selected in settings (KDDTest+ / KDDTrain+)."""
    from src import data_loader
    path = settings_module.dataset_path(settings)
    start = time.perf_counter()
    try:
        df, report = data_loader.load_nsl_kdd_file(path)
    except FileNotFoundError:
        return None, {"error": f"Dataset file '{path.name}' was not found.",
                      "seconds": 0.0}
    return df, {"seconds": time.perf_counter() - start, "report": report,
                "path_name": path.name}


def run_analysis(
    raw_df: pd.DataFrame,
    models: dict,
    settings: dict,
    ingestion_seconds: float | None = None,
    validation=None,
    generate_alerts: bool = True,
    explain_top_n: int = 0,
    source_label: str = "",
    progress_callback=None,
) -> AnalysisResult:
    """Run the full pipeline on `raw_df` and return measured results.

    `models` is the dict produced by the dashboard's cached loader (keys:
    pre, detector, random_forest, xgboost). Nothing is simulated: if a stage
    fails, the failure is recorded and later stages are skipped.

    `progress_callback(stage_index, stage_result)` is invoked after each stage
    so the UI can drive a progress bar from real completion events.
    """
    result = AnalysisResult(stages=[StageResult(name) for name in STAGES])
    stages = {s.name: s for s in result.stages}

    def report(index: int) -> None:
        if progress_callback is not None:
            progress_callback(index, result.stages[index])

    # ---------------------------------------------------- 1 data ingestion
    ing = stages["Data ingestion"]
    ing.seconds = float(ingestion_seconds) if ingestion_seconds is not None else None
    if validation is not None:
        result.warnings.extend(getattr(validation, "warnings", []) or [])
        errors = getattr(validation, "errors", []) or []
        if errors:
            ing.status = STATUS_FAILED
            ing.detail = f"{len(errors)} validation error(s)"
            result.errors.extend(errors)
            report(0)
            return result
        counts = getattr(validation, "to_dict", lambda: {})()
        ing.detail = (
            f"{len(raw_df):,} records, "
            f"{counts.get('malformed_rows', 0)} malformed, "
            f"{counts.get('duplicate_rows', 0)} duplicate rows"
        )
    else:
        ing.detail = f"{len(raw_df):,} records"
    ing.status = STATUS_DONE
    report(0)

    classifier_name = settings.get("classifier", "xgboost")
    pre = models.get("pre")
    detector = models.get("detector")
    clf = models.get(classifier_name)
    missing = [name for name, obj in
               (("preprocessing", pre), ("isolation_forest", detector),
                (classifier_name, clf)) if obj is None]
    if missing:
        result.errors.append(
            f"Missing model artefact(s): {', '.join(missing)}. Run "
            "'python train.py' to create them."
        )
        stages["Preprocessing"].status = STATUS_FAILED
        report(0)
        report(1)
        return result

    # --------------------------------------------------- 2 preprocessing
    stage = stages["Preprocessing"]
    start = time.perf_counter()
    try:
        X = pre.transform(raw_df)
    except ValueError as exc:
        stage.status = STATUS_FAILED
        stage.detail = str(exc)
        result.errors.append(f"Preprocessing failed: {exc}")
        report(1)
        return result
    stage.seconds = time.perf_counter() - start
    stage.status = STATUS_DONE
    stage.detail = f"{X.shape[0]:,} x {X.shape[1]} encoded features"
    report(1)

    # --------------------------------------------- 3 anomaly detection
    stage = stages["Anomaly detection"]
    model_threshold = float(detector.threshold)
    threshold = settings_module.effective_threshold(settings, model_threshold)
    source = ("analyst override" if settings.get("anomaly_threshold") is not None
              else f"model default ({config.ANOMALY_NORMAL_FPR_QUANTILE:.0%} "
                   "quantile of normal training scores)")
    start = time.perf_counter()
    anomaly_res: AnomalyResult = detector.predict(X, threshold=threshold)
    stage.seconds = time.perf_counter() - start
    stage.status = STATUS_DONE
    n_anom = int(anomaly_res.is_anomaly.sum())
    stage.detail = f"{n_anom:,} anomalous of {len(X):,} (threshold {threshold:.4f})"
    report(2)

    # ------------------------------------------ 4 attack classification
    stage = stages["Attack classification"]
    start = time.perf_counter()
    cls_res: ClassificationResult = clf.predict(X)
    stage.seconds = time.perf_counter() - start
    stage.status = STATUS_DONE
    n_attack = int((cls_res.predictions != "Normal").sum())
    stage.detail = (f"{classifier_name}: {n_attack:,} attack-classified of "
                    f"{len(X):,}")

    records = pd.DataFrame({
        "record_index": np.arange(len(raw_df)),
        "detection_status": np.where(anomaly_res.is_anomaly, "Anomaly", "Normal"),
        "anomaly_score": np.round(anomaly_res.scores, 4),
        "predicted_category": cls_res.predictions,
        "model_confidence": np.round(cls_res.confidences, 4),
    })
    for i, c in enumerate(cls_res.classes):
        records[f"p_{c}"] = np.round(cls_res.probabilities[:, i], 4)
    if config.LABEL_COLUMN in raw_df.columns:
        records["true_label"] = raw_df[config.LABEL_COLUMN].to_numpy()
        records["true_category"] = raw_df[config.LABEL_COLUMN].astype(str) \
            .str.lower().map(config.ATTACK_CATEGORY_MAP).fillna("Unknown").to_numpy()
        records["labelled"] = True
    else:
        records["labelled"] = False
    report(3)

    # ------------------------------------------------- 5 alert generation
    stage = stages["Alert generation"]
    if not generate_alerts:
        stage.status = STATUS_SKIPPED
        stage.detail = "disabled for this run"
        report(4)
    else:
        start = time.perf_counter()
        true_labels = (raw_df[config.LABEL_COLUMN].to_numpy()
                       if config.LABEL_COLUMN in raw_df.columns else None)
        generator: AlertGenerator = settings_module.alert_generator(settings)
        alerts = generator.generate(
            anomaly_res, cls_res,
            record_indices=records["record_index"].to_numpy(),
            true_labels=true_labels,
        )
        stage.seconds = time.perf_counter() - start
        stage.status = STATUS_DONE
        stage.detail = f"{len(alerts):,} alerts raised"
        result.alerts = alerts
        report(4)

    # ---------------------------------------------- 6 risk prioritisation
    stage = stages["Risk prioritisation"]
    start = time.perf_counter()
    engine = settings_module.severity_engine(settings)
    weights = settings["risk_weights"]
    records["severity"] = engine.severity_for_batch(
        records["predicted_category"].to_numpy())
    records["risk_score"] = engine.risk_score_batch(
        records["predicted_category"].to_numpy(),
        records["model_confidence"].to_numpy(),
        records["anomaly_score"].to_numpy(),
        weights=weights, threshold=threshold,
    )
    result.records = records
    if not result.alerts.empty:
        result.alerts = result.alerts.sort_values(
            "risk_score", ascending=False).reset_index(drop=True)
        sev_counts = result.alerts["severity"].value_counts().to_dict()
        stage.detail = ("ranked by risk; " + ", ".join(
            f"{k}={v:,}" for k, v in
            sorted(sev_counts.items(),
                   key=lambda kv: config.SEVERITY_ORDER.index(kv[0])
                   if kv[0] in config.SEVERITY_ORDER else 99)))
    else:
        stage.detail = "no alerts to rank"
    stage.seconds = time.perf_counter() - start
    stage.status = STATUS_DONE
    report(5)

    # ------------------------------------------- 7 explanation generation
    stage = stages["Explanation generation"]
    if explain_top_n <= 0 or not settings.get("generate_local_explanations", True):
        stage.status = STATUS_SKIPPED
        stage.detail = ("not requested" if explain_top_n <= 0
                        else "disabled in settings")
        report(6)
    elif result.alerts.empty:
        stage.status = STATUS_SKIPPED
        stage.detail = "no alerts to explain"
        report(6)
    else:
        start = time.perf_counter()
        try:
            from src.explainability import Explainer
            explainer = Explainer(clf, pre.feature_names_)
            targets = result.alerts.head(int(explain_top_n))
            for _, row in targets.iterrows():
                idx = int(row["record_index"])
                if idx >= len(X):
                    continue
                result.explanations[idx] = explainer.explain_record(X[idx])
            stage.detail = (f"{len(result.explanations)} local SHAP "
                            f"explanations ({classifier_name})")
            stage.status = STATUS_DONE
        except Exception as exc:                        # noqa: BLE001
            stage.status = STATUS_FAILED
            stage.detail = f"SHAP unavailable: {exc}"
            result.warnings.append(
                f"Local explanations could not be generated: {exc}")
        stage.seconds = time.perf_counter() - start
        report(6)

    measured = [s.seconds for s in result.stages if s.seconds is not None]
    total = float(sum(measured))
    n = max(len(raw_df), 1)
    result.meta = {
        "n_records": int(len(raw_df)),
        "n_encoded_features": int(X.shape[1]),
        "classifier": classifier_name,
        "anomaly_threshold": round(threshold, 6),
        "threshold_source": source,
        "model_threshold": round(model_threshold, 6),
        "total_seconds": round(total, 4),
        "ms_per_record": round(total / n * 1000, 4),
        "n_anomalous": int(anomaly_res.is_anomaly.sum()),
        "n_attack_classified": int(n_attack),
        "n_alerts": int(len(result.alerts)),
        "source": source_label,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return result


def summarise_records(records: pd.DataFrame) -> dict:
    """Compact counts for metric cards (all derived from real outputs)."""
    if records.empty:
        return {}
    return {
        "total": int(len(records)),
        "normal": int((records["detection_status"] == "Normal").sum()),
        "anomalous": int((records["detection_status"] == "Anomaly").sum()),
        "attack_classified": int((records["predicted_category"] != "Normal").sum()),
        "mean_confidence": float(records["model_confidence"].mean()),
        "by_category": records["predicted_category"].value_counts().to_dict(),
        "by_severity": records["severity"].value_counts().to_dict(),
    }
