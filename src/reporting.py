"""Report generation (CSV exports + a real PDF summary).

Every value written into a report is read from the persisted artefacts
(evaluation.json, resource_metrics.json, pipeline_state.json, the alert store)
or from a live pipeline run passed in by the caller. Nothing is templated with
placeholder numbers: missing data is reported as "not available".
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

PROJECT_TITLE = (
    "AI-Assisted Network Anomaly Detection and Alert Triage for "
    "Resource-Constrained Security Operations Centres"
)

LIMITATIONS = [
    "NSL-KDD is a historical benchmark dataset (1999 DARPA traffic with 2009 "
    "deduplication). Results do not demonstrate detection performance on live "
    "enterprise traffic.",
    "The prototype has not been deployed in a real SOC and no human study was "
    "conducted, so no claim is made about reduced analyst workload.",
    "The severity mapping and the weighted risk score are transparent research "
    "heuristics, not a validated real-world risk framework.",
    "Extreme class imbalance (52 U2R and 995 R2L training records) leads to low "
    "recall for those categories; weak results are reported rather than hidden.",
    "KDDTest+ has a different attack distribution from KDDTrain+, so the "
    "anomaly threshold selected on normal training traffic transfers "
    "imperfectly (documented in the evaluation).",
    "Resource measurements come from one general-purpose machine under "
    "uncontrolled load; they are indicative, not a controlled benchmark. Energy "
    "consumption was not measured.",
    "This prototype is not a replacement for a production IDS, SIEM or a human "
    "SOC analyst. Closing an alert records an analyst decision, not verified "
    "remediation.",
]


def collect_payload(settings: dict, analysis=None, alerts: pd.DataFrame | None = None,
                    evaluation: dict | None = None, resources: dict | None = None,
                    state: dict | None = None) -> dict:
    """Assemble the real data used by both the preview and the PDF."""
    evaluation = evaluation or {}
    resources = resources or {}
    state = state or {}
    alerts = alerts if alerts is not None else pd.DataFrame()

    payload = {
        "title": PROJECT_TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_version": config.APP_VERSION,
        "dataset": {
            "name": config.DATASET_NAME,
            "source": config.DATASET_SOURCE,
            "selected": settings.get("dataset"),
            "train_file": state.get("train_file", "KDDTrain+.txt"),
            "test_file": state.get("test_file", "KDDTest+.txt"),
            "train_rows": state.get("n_train_records"),
            "test_rows": state.get("n_test_records"),
            "normal_train_rows": state.get("n_normal_train_records"),
            "encoded_features": state.get("n_encoded_features"),
            "random_seed": state.get("random_seed", config.RANDOM_SEED),
            "trained_at": state.get("trained_at"),
        },
        "methodology": {
            "split": "Models trained on KDDTrain+.txt and evaluated on the "
                     "held-out KDDTest+.txt; preprocessing fitted on training "
                     "data only.",
            "anomaly": "Isolation Forest trained on normal training traffic "
                       "only; decision threshold = "
                       + str(state.get("anomaly_threshold_method", "not recorded")),
            "classification": "Random Forest and XGBoost trained separately "
                              "for five-class attack categorisation; anomaly "
                              "detection and attack classification remain "
                              "separate decisions.",
            "triage": "An alert is raised when Stage A flags an anomaly OR "
                      "Stage B predicts a non-Normal category. Severity comes "
                      "from a configurable mapping; risk is a weighted formula.",
            "explainability": "SHAP TreeExplainer. Global mean |SHAP| over a "
                              "sample (model level) is kept separate from "
                              "local per-record explanations.",
        },
        "severity_map": settings.get("severity_map", dict(config.SEVERITY_MAP)),
        "risk_weights": settings.get("risk_weights", dict(config.RISK_WEIGHTS)),
        "evaluation": evaluation,
        "resources": resources,
        "state": state,
        "limitations": LIMITATIONS,
    }

    if analysis is not None and getattr(analysis, "ok", False):
        records = analysis.records
        payload["live_analysis"] = {
            "meta": analysis.meta,
            "by_category": records["predicted_category"].value_counts().to_dict(),
            "by_severity": records["severity"].value_counts().to_dict(),
            "detection_status": records["detection_status"].value_counts().to_dict(),
            "ground_truth": (records["true_category"].value_counts().to_dict()
                             if "true_category" in records.columns else {}),
            "stages": [
                {"stage": s.name, "status": s.status, "seconds": s.seconds,
                 "detail": s.detail} for s in analysis.stages
            ],
        }
    if alerts is not None and not alerts.empty:
        payload["alerts"] = {
            "total": int(len(alerts)),
            "by_severity": alerts["severity"].value_counts().to_dict(),
            "by_category": alerts["predicted_category"].value_counts().to_dict(),
            "by_status": alerts["investigation_status"].value_counts().to_dict(),
            "mean_risk": float(alerts["risk_score"].mean()),
            "max_risk": float(alerts["risk_score"].max()),
            "top": alerts.sort_values("risk_score", ascending=False).head(10)[
                ["alert_id", "predicted_category", "severity", "risk_score",
                 "investigation_status"]
            ].to_dict(orient="records"),
        }
        if "true_category" in alerts.columns:
            payload["alerts"]["false_positives_vs_ground_truth"] = int(
                (alerts["true_category"] == "Normal").sum())
    return payload


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------
def csv_exports(payload: dict, alerts: pd.DataFrame | None = None,
                records: pd.DataFrame | None = None) -> dict[str, bytes]:
    exports: dict[str, bytes] = {}
    evaluation = payload.get("evaluation", {})
    rows = []
    anom = evaluation.get("anomaly_detection", {})
    for key in ("accuracy", "precision", "recall", "false_positive_rate",
                "false_negative_rate", "roc_auc"):
        if key in anom:
            rows.append({"model": "isolation_forest", "metric": key,
                         "value": anom[key]})
    for name, metrics in evaluation.get("attack_classification", {}).items():
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "weighted_f1", "roc_auc_ovr"):
            if key in metrics:
                rows.append({"model": name, "metric": key,
                             "value": metrics[key]})
        for cat, values in metrics.get("per_class", {}).items():
            for key in ("precision", "recall", "f1", "support"):
                rows.append({"model": name, "metric": f"{cat}_{key}",
                             "value": values.get(key)})
    if rows:
        exports["model_evaluation.csv"] = pd.DataFrame(rows).to_csv(
            index=False).encode("utf-8")

    resources = payload.get("resources", {}).get("measurements", {})
    res_rows = [{"stage": k, **{kk: vv for kk, vv in v.items()}}
                for k, v in resources.items() if isinstance(v, dict)]
    if res_rows:
        exports["resource_measurements.csv"] = pd.DataFrame(
            res_rows).to_csv(index=False).encode("utf-8")

    live = payload.get("live_analysis")
    if live:
        exports["analysis_summary.json"] = json.dumps(
            live, indent=2, default=str).encode("utf-8")
    if alerts is not None and not alerts.empty:
        exports["alerts.csv"] = alerts.to_csv(index=False).encode("utf-8")
    if records is not None and not records.empty:
        exports["processed_records.csv"] = records.to_csv(
            index=False).encode("utf-8")
    return exports


# ---------------------------------------------------------------------------
# PDF summary (reportlab)
# ---------------------------------------------------------------------------
def generate_pdf(payload: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (HRFlowable, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    navy = colors.HexColor("#0B1424")
    accent = colors.HexColor("#1D6FA5")
    grey = colors.HexColor("#5A6B82")
    light = colors.HexColor("#EDF2F7")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SocTitle", parent=styles["Title"], fontSize=18, leading=23,
        textColor=navy, spaceAfter=6, alignment=0,
    )
    h1 = ParagraphStyle("SocH1", parent=styles["Heading1"], fontSize=13,
                        leading=17, textColor=accent, spaceBefore=12,
                        spaceAfter=5)
    h2 = ParagraphStyle("SocH2", parent=styles["Heading2"], fontSize=11,
                        leading=14, textColor=navy, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("SocBody", parent=styles["BodyText"], fontSize=9,
                          leading=13, textColor=colors.HexColor("#1F2937"),
                          spaceAfter=4)
    small = ParagraphStyle("SocSmall", parent=body, fontSize=7.8, leading=11,
                           textColor=grey)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="AI-SOC Research Report", author="AI-SOC prototype",
    )
    story: list = []

    def table(headers: list[str], rows: list[list], widths=None) -> Table:
        data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle(
            "th", parent=body, textColor=colors.white, fontSize=8.4))
            for h in headers]]
        for row in rows:
            data.append([Paragraph(str(c) if c is not None else "not available",
                                   body) for c in row])
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ]))
        return t

    def fmt(value, digits=4):
        if value is None:
            return "not available"
        try:
            return f"{float(value):,.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    story.append(Paragraph(payload["title"], title_style))
    story.append(Paragraph(
        f"Generated {payload['generated_at']} &middot; AI-SOC prototype "
        f"v{payload['app_version']} &middot; research prototype report",
        small))
    story.append(HRFlowable(width="100%", thickness=0.8, color=accent,
                            spaceBefore=6, spaceAfter=8))

    # 1 dataset
    ds = payload["dataset"]
    story.append(Paragraph("1. Dataset and configuration", h1))
    story.append(table(
        ["Item", "Value"],
        [["Dataset", ds["name"]], ["Source", ds["source"]],
         ["Active analysis dataset", ds["selected"]],
         ["Training file / records",
          f"{ds['train_file']} / {fmt(ds['train_rows'], 0)}"],
         ["Test file / records", f"{ds['test_file']} / {fmt(ds['test_rows'], 0)}"],
         ["Normal training records", fmt(ds["normal_train_rows"], 0)],
         ["Encoded features", fmt(ds["encoded_features"], 0)],
         ["Random seed", fmt(ds["random_seed"], 0)],
         ["Models trained at", str(ds["trained_at"] or "not recorded")]],
        widths=[55 * mm, 115 * mm]))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Severity mapping (research rules)", h2))
    story.append(table(
        ["Predicted category", "Severity"],
        [[k, v] for k, v in payload["severity_map"].items()],
        widths=[55 * mm, 115 * mm]))
    w = payload["risk_weights"]
    story.append(Paragraph(
        f"Risk score = 100 x ({w['severity']} x severity weight + "
        f"{w['confidence']} x model confidence + {w['anomaly']} x normalised "
        "anomaly score). This is a transparent prioritisation heuristic, not a "
        "validated real-world risk measurement.", small))

    # 2 methodology
    story.append(Paragraph("2. Evaluation methodology", h1))
    for key in ("split", "anomaly", "classification", "triage", "explainability"):
        story.append(Paragraph(
            f"<b>{key.capitalize()}:</b> {payload['methodology'][key]}", body))

    # 3 live analysis
    live = payload.get("live_analysis")
    if live:
        meta = live["meta"]
        story.append(Paragraph("3. Analysis results (this run)", h1))
        story.append(table(
            ["Measure", "Value"],
            [["Records processed", fmt(meta.get("n_records"), 0)],
             ["Anomalous records (Stage A)", fmt(meta.get("n_anomalous"), 0)],
             ["Attack-classified records (Stage B)",
              fmt(meta.get("n_attack_classified"), 0)],
             ["Alerts generated", fmt(meta.get("n_alerts"), 0)],
             ["Classifier", str(meta.get("classifier"))],
             ["Anomaly threshold", fmt(meta.get("anomaly_threshold"))],
             ["Threshold source", str(meta.get("threshold_source"))],
             ["Total processing time (s)", fmt(meta.get("total_seconds"), 3)],
             ["Processing time per record (ms)",
              fmt(meta.get("ms_per_record"), 4)],
             ["Run timestamp", str(meta.get("ran_at"))]],
            widths=[75 * mm, 95 * mm]))
        story.append(Paragraph("Pipeline stages (measured)", h2))
        story.append(table(
            ["Stage", "Status", "Seconds", "Detail"],
            [[s["stage"], s["status"], fmt(s["seconds"], 3), s["detail"]]
             for s in live["stages"]],
            widths=[38 * mm, 18 * mm, 20 * mm, 94 * mm]))
        story.append(Paragraph("Attack distribution", h2))
        cats = config.ATTACK_CATEGORIES
        story.append(table(
            ["Category", "Predicted (Stage B)", "Ground truth (labels)",
             "Severity counts"],
            [[c, fmt(live["by_category"].get(c, 0), 0),
              fmt(live["ground_truth"].get(c, 0), 0) if live["ground_truth"]
              else "not labelled",
              fmt(live["by_severity"].get(
                  payload["severity_map"].get(c, ""), 0), 0)] for c in cats],
            widths=[32 * mm, 46 * mm, 46 * mm, 46 * mm]))
        story.append(Paragraph(
            "Predictions come from the trained classifier; ground truth comes "
            "from the dataset labels mapped to categories. They are reported "
            "separately and never substituted for one another.", small))
    else:
        story.append(Paragraph("3. Analysis results", h1))
        story.append(Paragraph(
            "No live analysis run was included in this report. Generate the "
            "report from the Reports page after running an analysis to include "
            "per-record results and measured stage timings.", body))

    # 4 alerts
    alerts = payload.get("alerts")
    if alerts:
        story.append(Paragraph("4. Alert triage summary", h1))
        story.append(table(
            ["Measure", "Value"],
            [["Total alerts", fmt(alerts["total"], 0)],
             ["Mean risk score", fmt(alerts["mean_risk"], 2)],
             ["Maximum risk score", fmt(alerts["max_risk"], 2)],
             ["False positives vs ground truth",
              fmt(alerts.get("false_positives_vs_ground_truth"), 0)]],
            widths=[75 * mm, 95 * mm]))
        story.append(Paragraph("By severity", h2))
        story.append(table(
            ["Severity", "Alerts"],
            [[k, fmt(v, 0)] for k, v in alerts["by_severity"].items()],
            widths=[55 * mm, 115 * mm]))
        story.append(Paragraph("By investigation status", h2))
        story.append(table(
            ["Status", "Alerts"],
            [[k, fmt(v, 0)] for k, v in alerts["by_status"].items()],
            widths=[55 * mm, 115 * mm]))
        story.append(Paragraph("Highest-risk alerts", h2))
        story.append(table(
            ["Alert ID", "Category", "Severity", "Risk", "Status"],
            [[a["alert_id"], a["predicted_category"], a["severity"],
              fmt(a["risk_score"], 2), a["investigation_status"]]
             for a in alerts["top"]],
            widths=[42 * mm, 28 * mm, 30 * mm, 22 * mm, 48 * mm]))

    story.append(PageBreak())

    # 5 model performance
    evaluation = payload.get("evaluation", {})
    story.append(Paragraph("5. Model performance (held-out test set)", h1))
    story.append(Paragraph(
        f"Evaluated at {evaluation.get('evaluated_at', 'not recorded')} with "
        f"random seed {evaluation.get('random_seed', 'n/a')}. These are TEST-set "
        "results; training-set metrics were not recorded by this pipeline.",
        small))
    anom = evaluation.get("anomaly_detection", {})
    if anom:
        story.append(Paragraph("Isolation Forest (anomaly detection)", h2))
        cm = anom.get("confusion_matrix", {})
        story.append(table(
            ["Metric", "Value"],
            [["ROC-AUC", fmt(anom.get("roc_auc"))],
             ["Accuracy", fmt(anom.get("accuracy"))],
             ["Precision", fmt(anom.get("precision"))],
             ["Recall", fmt(anom.get("recall"))],
             ["False positive rate", fmt(anom.get("false_positive_rate"))],
             ["False negative rate", fmt(anom.get("false_negative_rate"))],
             ["Threshold", fmt(anom.get("threshold"))],
             ["Confusion matrix (TP/FP/FN/TN)",
              f"{cm.get('tp', 0):,} / {cm.get('fp', 0):,} / "
              f"{cm.get('fn', 0):,} / {cm.get('tn', 0):,}"]],
            widths=[75 * mm, 95 * mm]))
    for name, metrics in evaluation.get("attack_classification", {}).items():
        story.append(Paragraph(f"{name} (attack classification)", h2))
        story.append(table(
            ["Metric", "Value"],
            [["Accuracy", fmt(metrics.get("accuracy"))],
             ["Macro precision", fmt(metrics.get("macro_precision"))],
             ["Macro recall", fmt(metrics.get("macro_recall"))],
             ["Macro F1", fmt(metrics.get("macro_f1"))],
             ["Weighted F1", fmt(metrics.get("weighted_f1"))],
             ["ROC-AUC (one-vs-rest)", fmt(metrics.get("roc_auc_ovr"))]],
            widths=[75 * mm, 95 * mm]))
        per_class = metrics.get("per_class", {})
        story.append(table(
            ["Category", "Precision", "Recall", "F1", "Support"],
            [[c, fmt(per_class.get(c, {}).get("precision")),
              fmt(per_class.get(c, {}).get("recall")),
              fmt(per_class.get(c, {}).get("f1")),
              fmt(per_class.get(c, {}).get("support"), 0)]
             for c in config.ATTACK_CATEGORIES],
            widths=[38 * mm, 33 * mm, 33 * mm, 33 * mm, 33 * mm]))
        cm = metrics.get("confusion_matrix", {})
        if cm.get("matrix"):
            labels = cm.get("labels", config.ATTACK_CATEGORIES)
            story.append(Paragraph("Confusion matrix (rows = actual)", small))
            story.append(table(
                ["Actual \\ Predicted"] + labels,
                [[labels[i]] + [fmt(v, 0) for v in row]
                 for i, row in enumerate(cm["matrix"])],
                widths=[38 * mm] + [26 * mm] * len(labels)))
        weakest = metrics.get("weakest_category", {})
        if weakest:
            story.append(Paragraph(
                f"Weakest category: {weakest.get('name')} "
                f"(F1 {fmt(weakest.get('f1'))}, recall "
                f"{fmt(weakest.get('recall'))}). Reported openly rather than "
                "hidden.", small))

    targets = evaluation.get("targets_vs_actuals", {})
    if targets:
        story.append(Paragraph("Research targets vs achieved results", h2))
        rows = []
        for group, metrics in targets.items():
            for metric, values in metrics.items():
                rows.append([group.replace("_", " "), metric,
                             fmt(values.get("target")), fmt(values.get("actual")),
                             "met" if values.get("target_met") else "NOT met"])
        story.append(table(
            ["Area", "Metric", "Target", "Actual", "Status"], rows,
            widths=[42 * mm, 34 * mm, 28 * mm, 28 * mm, 38 * mm]))

    # 6 explainability
    story.append(Paragraph("6. Explainability", h1))
    gs = evaluation.get("global_shap", {})
    story.append(Paragraph(
        f"Global feature importance (mean |SHAP|) computed for model "
        f"<b>{gs.get('model', 'not recorded')}</b> over "
        f"{gs.get('sample_size', 'n/a')} sampled records in "
        f"{fmt(gs.get('compute_seconds'), 3)} s. This is a model-level "
        "explanation, not a per-alert explanation.", body))
    top = gs.get("top_15", [])
    if top:
        story.append(table(
            ["Feature", "Mean |SHAP|"],
            [[t.get("feature"), fmt(t.get("mean_abs_shap"))] for t in top[:10]],
            widths=[100 * mm, 70 * mm]))
    story.append(Paragraph(
        "Local per-alert explanations are generated on demand with SHAP "
        "TreeExplainer for the specific record and the class the model "
        "predicted. A single-record example measured during the training run "
        f"took {fmt(evaluation.get('local_shap_latency_ms_example'), 2)} ms. "
        "The dashboard shows base value, contributions and a waterfall for the "
        "selected alert; it never substitutes global importance for a local "
        "explanation.", body))

    # 7 resources
    resources = payload.get("resources", {})
    story.append(Paragraph("7. Resource efficiency measurements", h1))
    sys_info = resources.get("system", {})
    story.append(table(
        ["Item", "Value"],
        [["Platform", sys_info.get("platform", "not recorded")],
         ["Python", sys_info.get("python_version", "not recorded")],
         ["Logical CPUs", str(sys_info.get("cpu_count_logical", "not recorded"))],
         ["System memory (MB)", fmt(sys_info.get("system_memory_total_mb"), 1)],
         ["Process memory at collection (MB)",
          fmt(sys_info.get("process_memory_mb"), 1)],
         ["Collected at", resources.get("collected_at", "not recorded")],
         ["Energy consumption", "not measured"]],
        widths=[75 * mm, 95 * mm]))
    sizes = resources.get("model_files", {})
    if sizes:
        story.append(table(
            ["Artefact", "Size (MB)"],
            [[k.replace("_size_mb", ""), fmt(v, 3)] for k, v in sizes.items()],
            widths=[100 * mm, 70 * mm]))
    measurements = resources.get("measurements", {})
    stage_rows = [[k, fmt(v.get("wall_time_seconds"), 3),
                   fmt(v.get("memory_delta_mb"), 1),
                   fmt(v.get("cpu_percent_during"), 1)]
                  for k, v in measurements.items()
                  if isinstance(v, dict) and "wall_time_seconds" in v]
    if stage_rows:
        story.append(Paragraph("Measured pipeline stages", h2))
        story.append(table(
            ["Stage", "Wall seconds", "Memory delta MB", "CPU % during"],
            stage_rows,
            widths=[62 * mm, 34 * mm, 38 * mm, 36 * mm]))
    latency_rows = []
    for name in ("isolation_forest", "random_forest", "xgboost"):
        rec = measurements.get(f"{name}_prediction_latency")
        if rec:
            latency_rows.append([
                name, fmt(rec.get("mean_ms_per_record"), 5),
                fmt(rec.get("throughput_records_per_second"), 1),
                fmt(rec.get("n_records"), 0), str(rec.get("repeats", ""))])
    if latency_rows:
        story.append(Paragraph("Batched prediction latency", h2))
        story.append(table(
            ["Model", "ms / record", "records / s", "Sample", "Repeats"],
            latency_rows,
            widths=[45 * mm, 32 * mm, 38 * mm, 28 * mm, 27 * mm]))
    story.append(Paragraph(
        "Latency is batched inference amortised per record (best of N repeats), "
        "not single-request latency. Measurements were taken on one "
        "general-purpose machine under uncontrolled load; they do not by "
        "themselves establish suitability for a production SOC.", small))

    # 8 limitations
    story.append(Paragraph("8. Limitations and data integrity statement", h1))
    for item in payload["limitations"]:
        story.append(Paragraph(f"&bull; {item}", body))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All figures in this report were read from the prototype's persisted "
        "artefacts or from a live run of its models. No values were simulated, "
        "hardcoded or interpolated. Missing measurements are reported as 'not "
        "available'.", small))

    doc.build(story)
    return buffer.getvalue()


def save_pdf(payload: dict, path: Path | None = None) -> Path:
    path = Path(path or (config.REPORTS_DIR /
                         f"ai_soc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(generate_pdf(payload))
    return path
