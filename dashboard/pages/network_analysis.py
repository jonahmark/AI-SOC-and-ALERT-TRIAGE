"""Page 3 - Network Analysis (upload, validate, run the real pipeline).

Uploaded files are parsed strictly as data (CSV/TXT in NSL-KDD layout) and are
never executed. Validation runs before any model is invoked, and the progress
indicator is driven by real stage-completion events from the pipeline.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import config
from src import pipeline
from src import settings as settings_module
from src.auth import rbac
from dashboard import auth_ui, common
from dashboard import components as ui


def _validation_view(report) -> None:
    data = report.to_dict()
    if report.errors:
        for err in report.errors:
            st.error(err)
    if report.warnings:
        for warn in report.warnings:
            st.warning(warn)
    if not report.errors and not report.warnings:
        st.success("Validation passed: required columns, dtypes and label "
                   "mapping are all acceptable.")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.kv_table([
            ("Rows", f"{data['n_rows']:,}"),
            ("Columns", f"{data['n_columns']}"),
            ("Malformed rows", f"{data['malformed_rows']}"),
        ])
    with c2:
        ui.kv_table([
            ("Duplicate rows", f"{data['duplicate_rows']}"),
            ("Unknown labels", f"{len(data['unknown_labels'])}"),
            ("Invalid numeric cells", f"{len(data['invalid_numeric_cells'])}"),
        ])
    with c3:
        missing = {k: v for k, v in data["missing_values"].items() if v}
        ui.kv_table([("Columns with missing values", f"{len(missing)}"),
                     ("Status", "REJECTED" if report.errors else "ACCEPTED")])
    if data["label_distribution"]:
        st.markdown("**Label distribution (mapped categories)**")
        dist = pd.DataFrame(
            sorted(data["label_distribution"].items(), key=lambda kv: -kv[1]),
            columns=["category", "records"],
        )
        st.dataframe(dist, width="stretch", hide_index=True, height=180)


def render() -> None:
    settings = common.get_settings()

    ui.topbar(
        ["AI-SOC", "Network Analysis"],
        "Network Analysis",
        "Upload traffic records, validate them, and run the full detection "
        "pipeline with measured per-stage timings.",
        chips=[
            ui.chip("Max upload", f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                    "upload"),
            ui.chip("Accepted",
                    ", ".join(sorted(config.ALLOWED_UPLOAD_EXTENSIONS)), "file"),
            ui.chip("Classifier", settings["classifier"], "cpu"),
        ],
        notifications=common.notification_count(common.get_alerts()),
        refresh_key="na_refresh",
        icon_name="network",
    )

    models = common.get_models()
    if not common.models_available(models):
        common.render_missing_models_notice(models)
        return

    ui.note(
        "Uploaded files are read as tabular data only and are never executed. "
        "Records are validated before any model runs; invalid files are "
        "rejected with a specific reason rather than silently truncated.",
        "lock", ui.ACCENT_2,
    )

    source_tab, builtin_tab = st.tabs(["Upload a file", "Use a built-in dataset"])

    raw_df, report, source_label, ingestion_seconds = None, None, "", None

    with source_tab:
        uploaded = st.file_uploader(
            "CSV or NSL-KDD .txt file",
            type=[e.lstrip(".") for e in config.ALLOWED_UPLOAD_EXTENSIONS],
            key="na_upload",
            help="41 features (+ optional label and difficulty columns), or a "
                 "CSV with the documented feature names as a header row.",
        )
        if uploaded is not None:
            from src import data_loader
            start = time.perf_counter()
            content = uploaded.read()
            raw_df, report = data_loader.load_uploaded_file(content, uploaded.name)
            ingestion_seconds = time.perf_counter() - start
            source_label = uploaded.name
            size_kb = len(content) / 1024
            ui.metric_row([
                ui.metric_card("File name", uploaded.name[:28],
                               "Accepted extension and size", "file", ui.ACCENT),
                ui.metric_card("File size", f"{size_kb:,.1f} KB",
                               f"Limit {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                               "drive", ui.ACCENT_2),
                ui.metric_card("Records",
                               f"{len(raw_df):,}" if raw_df is not None else "0",
                               "Rows accepted by the parser", "layers", "#22C55E"),
                ui.metric_card("Features",
                               f"{raw_df.shape[1]}" if raw_df is not None else "0",
                               "Columns present in the file", "list", "#38BDF8"),
                ui.metric_card("Validation",
                               "Rejected" if raw_df is None else
                               ("Warnings" if report.warnings else "Passed"),
                               f"{len(report.errors)} error(s), "
                               f"{len(report.warnings)} warning(s)",
                               "check-circle",
                               "#EF4444" if raw_df is None else "#22C55E"),
            ])
            with ui.card("Validation report", "filter",
                         "Structural and content checks run before modelling",
                         key="na_validation"):
                _validation_view(report)
            if raw_df is None:
                st.stop()

    with builtin_tab:
        limits = {"Full dataset": None, "First 20,000": 20000,
                  "First 5,000": 5000, "First 1,000": 1000}
        choice = st.selectbox("Dataset", list(settings_module.DATASETS),
                              key="na_builtin")
        limit = st.selectbox("Records", list(limits), key="na_builtin_limit")
        if st.button("Load dataset", icon=":material/database:", key="na_load"):
            from src import data_loader
            path = settings_module.DATASETS[choice]
            start = time.perf_counter()
            try:
                df, rep = data_loader.load_nsl_kdd_file(path)
            except FileNotFoundError as exc:
                st.error(str(exc))
                df, rep = None, None
            seconds = time.perf_counter() - start
            if df is not None and limits[limit]:
                df = df.head(limits[limit])
            st.session_state["na_builtin_state"] = {
                "df": df, "report": rep, "seconds": seconds,
                "source": f"{choice} ({limit.lower()})",
            }
        state = st.session_state.get("na_builtin_state")
        if uploaded is None and state is not None and state["df"] is not None:
            raw_df = state["df"]
            report = state["report"]
            ingestion_seconds = state["seconds"]
            source_label = state["source"]
            ui.metric_row([
                ui.metric_card("Dataset", choice, f"Loaded in "
                               f"{ingestion_seconds:.2f} s (measured)",
                               "database", ui.ACCENT),
                ui.metric_card("Records", f"{len(raw_df):,}", limit,
                               "layers", ui.ACCENT_2),
                ui.metric_card("Columns", f"{raw_df.shape[1]}",
                               "41 features + label + difficulty", "list",
                               "#38BDF8"),
                ui.metric_card("Validation",
                               "Warnings" if report.warnings else "Passed",
                               f"{len(report.errors)} error(s), "
                               f"{len(report.warnings)} warning(s)",
                               "check-circle", "#22C55E"),
            ])
            with ui.card("Validation report", "filter", source_label,
                         key="na_validation2"):
                _validation_view(report)

    if raw_df is None:
        ui.empty_state(
            "No dataset loaded",
            "Upload a CSV / NSL-KDD file, or load one of the built-in datasets "
            "to run the pipeline.",
            "upload",
        )
        return

    with ui.card("Analysis configuration", "settings",
                 "Applies to this run only (persistent defaults live in "
                 "System Settings)", key="na_config"):
        c1, c2, c3 = st.columns(3)
        with c1:
            classifier = st.selectbox(
                "Attack classifier (Stage B)",
                ["xgboost", "random_forest"],
                index=0 if settings["classifier"] == "xgboost" else 1,
                key="na_classifier")
        with c2:
            model_threshold = float(models["detector"].threshold)
            threshold = st.number_input(
                "Anomaly threshold (Stage A)",
                min_value=-5.0, max_value=5.0, value=model_threshold,
                step=0.005, format="%.4f", key="na_threshold",
                help=f"Model default {model_threshold:.4f} = "
                     f"{config.ANOMALY_NORMAL_FPR_QUANTILE:.0%} quantile of "
                     "normal training scores. Lower = stricter.",
            )
        with c3:
            explain_n = st.number_input(
                "SHAP explanations (top-N alerts)", min_value=0, max_value=25,
                value=0, step=1, key="na_explain")

    run_settings = dict(settings)
    run_settings["classifier"] = classifier
    run_settings["anomaly_threshold"] = (
        None if abs(threshold - model_threshold) < 1e-9 else float(threshold)
    )

    if st.button("Run analysis pipeline", type="primary",
                 icon=":material/play_arrow:", key="na_run"):
        bar = st.progress(0.0, text="Starting pipeline...")

        def on_stage(index: int, stage) -> None:
            bar.progress(min((index + 1) / len(pipeline.STAGES), 1.0),
                         text=f"Stage {index + 1}/{len(pipeline.STAGES)}: "
                              f"{stage.name} - {stage.detail or stage.status}")

        with st.spinner("Preprocessing, anomaly detection and classification "
                        "are running on your records..."):
            result = pipeline.run_analysis(
                raw_df, models, run_settings,
                ingestion_seconds=ingestion_seconds, validation=report,
                explain_top_n=int(explain_n), source_label=source_label,
                progress_callback=on_stage,
            )
        bar.empty()
        st.session_state["na_result"] = result

    result = st.session_state.get("na_result")
    if result is None:
        return
    if not result.ok:
        for err in result.errors:
            st.error(err)
        return
    for warn in result.warnings:
        st.warning(warn)

    meta = result.meta
    with ui.card("Pipeline stages", "layers",
                 "Real elapsed time per stage for this run", key="na_stages"):
        ui.stage_tracker([(s.name, s.seconds if s.status == "done" else None)
                          for s in result.stages])
        st.dataframe(result.to_frame(), width="stretch", hide_index=True,
                     height=270, key="na_stage_table")

    ui.metric_row([
        ui.metric_card("Records processed", f"{meta['n_records']:,}",
                       f"Encoded to {meta['n_encoded_features']} features",
                       "layers", ui.ACCENT),
        ui.metric_card("Anomalous", f"{meta['n_anomalous']:,}",
                       f"Threshold {meta['anomaly_threshold']:.4f} "
                       f"({meta['threshold_source']})", "activity", "#F97316"),
        ui.metric_card("Attack predictions", f"{meta['n_attack_classified']:,}",
                       f"Classifier: {meta['classifier']}", "crosshair", "#EF4444"),
        ui.metric_card("Alerts generated", f"{meta['n_alerts']:,}",
                       "Raised by either stage", "bell", ui.ACCENT_2),
        ui.metric_card("Total processing time", f"{meta['total_seconds']:.2f} s",
                       f"{meta['ms_per_record']:.3f} ms per record", "clock",
                       "#22C55E"),
    ])

    with ui.card("Results", "list", "Per-record model outputs", key="na_results"):
        f1, f2 = st.columns(2)
        with f1:
            cat_filter = st.multiselect("Predicted category",
                                        config.ATTACK_CATEGORIES,
                                        default=config.ATTACK_CATEGORIES,
                                        key="na_f_cat")
        with f2:
            det_filter = st.multiselect("Detection result", ["Normal", "Anomaly"],
                                        default=["Normal", "Anomaly"],
                                        key="na_f_det")
        view = result.records[
            result.records["predicted_category"].isin(cat_filter)
            & result.records["detection_status"].isin(det_filter)
        ]
        columns = ["record_index", "detection_status", "predicted_category",
                   "anomaly_score", "model_confidence", "severity", "risk_score"]
        if settings["show_ground_truth"] and "true_category" in view.columns:
            columns += ["true_label", "true_category"]
        prob_cols = [c for c in view.columns if c.startswith("p_")]
        if st.toggle("Show class probabilities", value=False, key="na_probs"):
            columns += prob_cols
        view = view[columns]
        ui.show_table(ui.paginate(view, settings["table_page_size"], "na"),
                      height=400, key="na_table")
        st.caption(f"{len(view):,} of {len(result.records):,} records match "
                   "the current filters.")
        st.download_button(
            "Download processed results (CSV)",
            result.records.to_csv(index=False).encode("utf-8"),
            file_name="analysis_results.csv", mime="text/csv",
            icon=":material/download:", key="na_download",
        )

    if not result.alerts.empty:
        with ui.card("Alerts from this run", "bell",
                     f"{len(result.alerts):,} alerts, ranked by risk score",
                     key="na_alerts"):
            preview = result.alerts[[
                "alert_id", "predicted_category", "severity", "risk_score",
                "model_confidence", "anomaly_score", "detection_status",
            ]].head(50)
            ui.show_table(preview, height=320, key="na_alert_table")
            st.caption("Showing the 50 highest-risk alerts of this run.")
            st.download_button(
                "Download alerts (CSV)",
                result.alerts.to_csv(index=False).encode("utf-8"),
                file_name="generated_alerts.csv", mime="text/csv",
                icon=":material/download:", key="na_alert_download",
            )
            confirm = st.checkbox(
                f"I understand this will append {len(result.alerts):,} alerts to "
                "the shared alert store (existing investigation statuses and "
                "notes are preserved for matching alert IDs).",
                key="na_confirm_store",
            )
            actor = auth_ui.current_user()
            can_store = actor is not None and rbac.can(actor["role"],
                                                       "triage_alerts")
            if not can_store:
                st.caption("Read-only role: downloading results is allowed, "
                           "but writing to the shared alert store requires "
                           "the SOC Analyst or Administrator role.")
            if st.button("Save alerts to the store", type="primary",
                         icon=":material/save:",
                         disabled=not confirm or not can_store,
                         key="na_save"):
                if actor is None or not rbac.can(actor["role"], "triage_alerts"):
                    st.error("Your role cannot write to the alert store.")
                    st.stop()
                store = common.get_alert_store()
                store.save(result.alerts, merge=True)
                common.invalidate_alerts()
                settings_module.log(
                    f"{len(result.alerts):,} alerts saved from "
                    f"{meta['source'] or 'uploaded file'} "
                    f"(classifier={meta['classifier']})."
                )
                st.success(f"{len(result.alerts):,} alerts appended to the "
                           "store. Open Alert Management to triage them.")
