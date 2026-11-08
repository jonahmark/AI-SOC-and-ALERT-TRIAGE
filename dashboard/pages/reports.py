"""Page 9 - Reports (CSV exports and a real PDF summary).

The report is assembled by src/reporting.py from persisted artefacts and an
optional live analysis run, so exported documents contain actual system
results, methodology and limitations.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src import reporting, settings as settings_module
from dashboard import common
from dashboard import components as ui


def render() -> None:
    settings = common.get_settings()
    evaluation = common.get_evaluation()
    resources = common.get_resource_metrics()
    state = common.get_pipeline_state()
    alerts = common.get_alerts()

    ui.topbar(
        ["AI-SOC", "Reports"],
        "Reports",
        "Export research-ready summaries of the dataset, detections, model "
        "evaluation and resource measurements.",
        chips=[
            ui.chip("Dataset", settings["dataset"], "database"),
            ui.chip("Alerts", f"{len(alerts):,}", "bell"),
            ui.chip("Evaluation", "available" if evaluation else "missing",
                    "chart",
                    value_color="#22C55E" if evaluation else "#EF4444"),
        ],
        notifications=common.notification_count(alerts),
        refresh_key="rp_refresh",
        icon_name="file",
    )

    with ui.card("Report contents", "settings",
                 "Choose what to include - all content comes from real artefacts",
                 key="rp_options"):
        c1, c2, c3 = st.columns(3)
        with c1:
            include_live = st.toggle(
                "Include a live analysis run", value=True, key="rp_live",
                help="Runs the real pipeline over the selected dataset and "
                     "includes per-record counts and measured stage timings.")
        with c2:
            include_alerts = st.toggle(
                "Include the alert store", value=not alerts.empty,
                key="rp_alerts",
                help="Adds the triage summary and the 10 highest-risk alerts.")
        with c3:
            include_records = st.toggle(
                "Include processed records CSV", value=False, key="rp_records",
                help="Adds a full per-record CSV export (can be several MB).")
        max_records = st.select_slider(
            "Records for the live run", options=[1000, 5000, 20000, None],
            value=None, format_func=lambda v: "Full dataset" if v is None
            else f"{v:,}", key="rp_limit")

    analysis = None
    if include_live:
        models = common.get_models()
        if not common.models_available(models):
            common.render_missing_models_notice(models)
        else:
            with st.spinner("Running the pipeline for the report..."):
                analysis = common.run_dataset_analysis(settings,
                                                       max_records=max_records)
            if analysis is not None and not analysis.ok:
                for err in analysis.errors:
                    st.error(err)
                analysis = None

    payload = reporting.collect_payload(
        settings, analysis=analysis,
        alerts=alerts if include_alerts else pd.DataFrame(),
        evaluation=evaluation, resources=resources, state=state,
    )

    # ------------------------------------------------------------- preview
    with ui.card("Report preview", "file",
                 f"Generated {payload['generated_at']} - content below is "
                 "exactly what the PDF contains", key="rp_preview"):
        ds = payload["dataset"]
        train_rows = f"{ds['train_rows']:,}" if ds["train_rows"] else "n/a"
        test_rows = f"{ds['test_rows']:,}" if ds["test_rows"] else "n/a"
        ui.kv_table([
            ("Project", payload["title"]),
            ("Dataset", f"{ds['name']} - {ds['selected']}"),
            ("Train / test records", f"{train_rows} / {test_rows}"),
            ("Encoded features", str(ds["encoded_features"] or "n/a")),
            ("Random seed", str(ds["random_seed"])),
        ])
        sections = ["Dataset and configuration", "Evaluation methodology"]
        if payload.get("live_analysis"):
            meta = payload["live_analysis"]["meta"]
            sections.append(
                f"Analysis results ({meta['n_records']:,} records, "
                f"{meta['n_alerts']:,} alerts, {meta['total_seconds']:.2f} s)")
        if payload.get("alerts"):
            sections.append(
                f"Alert triage summary ({payload['alerts']['total']:,} alerts)")
        if evaluation:
            sections.append("Model performance (Isolation Forest, Random "
                            "Forest, XGBoost) + targets vs actuals")
            sections.append("Explainability (global importance + local SHAP "
                            "latency)")
        if resources:
            sections.append("Resource efficiency measurements and environment")
        sections.append("Limitations and data integrity statement")
        st.markdown("**Sections**")
        for i, section in enumerate(sections, start=1):
            st.markdown(f"{i}. {section}")

        if payload.get("live_analysis"):
            live = payload["live_analysis"]
            st.markdown("**Attack distribution in this report**")
            table = pd.DataFrame({
                "category": config.ATTACK_CATEGORIES,
                "predicted": [live["by_category"].get(c, 0)
                              for c in config.ATTACK_CATEGORIES],
                "ground_truth": [live["ground_truth"].get(c, 0)
                                 for c in config.ATTACK_CATEGORIES],
            })
            ui.show_table(table, height=220, key="rp_preview_table")

    # ------------------------------------------------------------- exports
    col_a, col_b = st.columns([1, 1.2])
    with col_a, ui.card("PDF summary", "download",
                        "Professional research report layout", key="rp_pdf"):
        if st.button("Generate PDF report", type="primary",
                     icon=":material/picture_as_pdf:", key="rp_pdf_btn",
                     width="stretch"):
            with st.spinner("Building the PDF from real artefacts..."):
                try:
                    pdf_bytes = reporting.generate_pdf(payload)
                except Exception as exc:                        # noqa: BLE001
                    st.error(f"PDF generation failed: {exc}")
                    pdf_bytes = None
            if pdf_bytes:
                st.session_state["rp_pdf_bytes"] = pdf_bytes
                st.session_state["rp_pdf_payload"] = payload
                settings_module.log(
                    f"PDF report generated ({len(pdf_bytes) / 1024:.1f} KB)."
                )
        if st.session_state.get("rp_pdf_bytes"):
            pdf_bytes = st.session_state["rp_pdf_bytes"]
            st.success(f"PDF ready: {len(pdf_bytes) / 1024:.1f} KB.")
            st.download_button(
                "Download PDF summary", pdf_bytes,
                file_name="ai_soc_research_report.pdf", mime="application/pdf",
                icon=":material/download:", key="rp_pdf_download",
                width="stretch",
            )
            if st.button("Save a copy to results/reports/",
                         icon=":material/save:", key="rp_pdf_save",
                         width="stretch"):
                path = reporting.save_pdf(st.session_state["rp_pdf_payload"])
                st.success(f"Saved to {path.relative_to(config.PROJECT_ROOT)}")

    with col_b, ui.card("CSV / JSON exports", "list",
                        "Machine-readable results for further analysis",
                        key="rp_csv"):
        records = (analysis.records if analysis is not None
                   and include_records else None)
        exports = reporting.csv_exports(
            payload, alerts=alerts if include_alerts else None, records=records)
        if not exports:
            ui.empty_state("Nothing to export",
                           "Run the training pipeline or include a live "
                           "analysis.", "file")
        for name, data in exports.items():
            mime = "text/csv" if name.endswith(".csv") else "application/json"
            st.download_button(
                f"{name}  ({len(data) / 1024:.1f} KB)", data, file_name=name,
                mime=mime, icon=":material/download:", key=f"rp_dl_{name}",
                width="stretch",
            )

    ui.note(
        "Reports contain only measured results. Sections whose artefacts are "
        "missing state that explicitly instead of filling in example numbers. "
        "The limitations section is always included.",
        "info", ui.ACCENT_2,
    )
