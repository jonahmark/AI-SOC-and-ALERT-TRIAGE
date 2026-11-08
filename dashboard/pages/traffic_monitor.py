"""Page 2 - Traffic Analysis Monitor.

Honest scope: this prototype analyses stored NSL-KDD records (or records the
analyst uploads on the Network Analysis page). No packet capture or live
ingestion component exists, so nothing on this page is streamed or simulated.
All timings are measured on this machine during the run.
"""
from __future__ import annotations

import streamlit as st

import config
from dashboard import common
from dashboard import components as ui

LIMITS = {"Full dataset": None, "First 20,000 records": 20000,
          "First 5,000 records": 5000, "First 1,000 records": 1000}


def render() -> None:
    settings = common.get_settings()

    ui.topbar(
        ["AI-SOC", "Traffic Analysis Monitor"],
        "Traffic Analysis Monitor",
        "Run the detection pipeline over stored NSL-KDD records and inspect "
        "per-record results.",
        chips=[
            ui.chip("Dataset", settings["dataset"], "database"),
            ui.chip("Classifier", settings["classifier"], "cpu"),
            ui.chip("Anomaly threshold",
                    f"{settings['anomaly_threshold']:.4f}"
                    if settings["anomaly_threshold"] is not None else "model default",
                    "filter"),
            ui.chip("Source", "stored records (no live capture)", "info"),
        ],
        notifications=common.notification_count(common.get_alerts()),
        refresh_key="tm_refresh",
        icon_name="activity",
    )

    ui.note(
        "<b>Scope.</b> NSL-KDD is a static benchmark dataset with no capture "
        "timestamps and no live feed. This page therefore analyses records "
        "already on disk and reports real, measured processing times - it does "
        "not simulate live network traffic.",
        "info", ui.ACCENT_2,
    )

    models = common.get_models()
    if not common.models_available(models):
        common.render_missing_models_notice(models)
        return

    c1, c2, c3 = st.columns([1.4, 1.4, 1])
    with c1:
        limit_label = st.selectbox("Records to process", list(LIMITS), key="tm_limit")
    with c2:
        explain_n = st.number_input(
            "Local SHAP explanations for the top-N alerts (0 = off)",
            min_value=0, max_value=50, value=0, step=1, key="tm_explain",
            help="Explanations are generated on demand and add real latency "
                 "(reported in the stage tracker).",
        )
    with c3:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        run_now = st.button("Run analysis", type="primary",
                            icon=":material/play_arrow:", key="tm_run",
                            width="stretch")

    if run_now:
        with st.status("Running the detection pipeline...", expanded=True) as status:
            st.write(f"Loading {settings['dataset']} and running all seven stages.")
            analysis = common.run_dataset_analysis(
                settings, explain_top_n=int(explain_n),
                max_records=LIMITS[limit_label],
            )
            status.update(label="Analysis complete.", state="complete")
        st.session_state["tm_result"] = analysis

    analysis = st.session_state.get("tm_result")
    if analysis is None:
        ui.empty_state(
            "No analysis run in this session",
            "Choose how many records to process and press <b>Run analysis</b>. "
            "The Overview page runs the same pipeline automatically.",
            "play",
        )
        return

    if not analysis.ok:
        for err in analysis.errors:
            st.error(err)
        return

    meta = analysis.meta
    records = analysis.records
    st.success(
        f"Processed {meta['n_records']:,} records in {meta['total_seconds']:.2f} s "
        f"({meta['ms_per_record']:.3f} ms/record) using "
        f"{meta['classifier']} and threshold {meta['anomaly_threshold']:.4f}."
    )

    with ui.card("Pipeline stages (measured)", "layers",
                 "Real elapsed time per stage on this machine", key="tm_stages"):
        pairs = []
        for stage in analysis.stages:
            pairs.append((f"{stage.name}", stage.seconds
                          if stage.status == "done" else None))
        ui.stage_tracker(pairs, active_index=None)
        st.dataframe(analysis.to_frame(), width="stretch", hide_index=True,
                     height=260, key="tm_stage_table")

    summary = {
        "Total records processed": (f"{meta['n_records']:,}", "layers", ui.ACCENT,
                                    f"Source: {settings['dataset']}"),
        "Normal records": (f"{meta['n_records'] - meta['n_anomalous']:,}",
                           "check-circle", "#22C55E",
                           "Score above the anomaly threshold (Stage A)"),
        "Anomalous records": (f"{meta['n_anomalous']:,}", "activity", "#F97316",
                              "Score at or below the anomaly threshold (Stage A)"),
        "Attack predictions": (f"{meta['n_attack_classified']:,}", "crosshair",
                               "#EF4444",
                               "Stage B classifier predicted a non-Normal category"),
        "Alerts raised": (f"{meta['n_alerts']:,}", "bell", ui.ACCENT_2,
                          "Flagged by either stage; not yet written to the store"),
        "Processing duration": (f"{meta['total_seconds']:.2f} s", "clock", "#38BDF8",
                                f"{meta['ms_per_record']:.3f} ms per record (measured)"),
    }
    cards = [ui.metric_card(label, value[0], value[3], value[1], value[2])
             for label, value in summary.items()]
    ui.metric_row(cards[:3])
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ui.metric_row(cards[3:])

    # ---------------------------------------------------------- filters
    with ui.card("Processed records", "list",
                 "Filter, sort and page through per-record model outputs",
                 key="tm_records"):
        f1, f2, f3 = st.columns([1.2, 1.6, 1])
        with f1:
            status_filter = st.multiselect(
                "Detection result", ["Normal", "Anomaly"],
                default=["Normal", "Anomaly"], key="tm_f_status")
        with f2:
            category_filter = st.multiselect(
                "Predicted category", config.ATTACK_CATEGORIES,
                default=config.ATTACK_CATEGORIES, key="tm_f_cat")
        with f3:
            search = st.text_input("Record index", placeholder="e.g. 42",
                                   key="tm_f_search")

        view = records[records["detection_status"].isin(status_filter)
                       & records["predicted_category"].isin(category_filter)]
        if search.strip():
            try:
                view = view[view["record_index"] == int(search.strip())]
            except ValueError:
                st.warning("Record index must be a whole number.")

        columns = ["record_index", "detection_status", "predicted_category",
                   "anomaly_score", "model_confidence", "severity", "risk_score"]
        if settings["show_ground_truth"] and "true_category" in view.columns:
            columns.append("true_category")
        view = view[columns]

        page_size = settings["table_page_size"]
        ui.show_table(ui.paginate(view, page_size, "tm"), height=380,
                      key="tm_table")
        st.caption(
            f"{len(view):,} records match the filters. NSL-KDD provides no "
            "capture timestamps, so no timestamp column is shown and no "
            "activity timeline is rendered - fabricating one would misrepresent "
            "the data."
        )
        st.download_button(
            "Download these results (CSV)", view.to_csv(index=False).encode("utf-8"),
            file_name=f"traffic_analysis_{settings['dataset'].replace('+', '')}.csv",
            mime="text/csv", icon=":material/download:", key="tm_download",
        )

    if analysis.explanations:
        with ui.card("Local explanations generated", "sparkles",
                     "Per-alert SHAP for the highest-risk records in this run",
                     key="tm_shap"):
            for idx, exp in list(analysis.explanations.items())[:5]:
                st.markdown(
                    f"**Record {idx}** &middot; predicted "
                    f"{ui.category_badge(exp.predicted_class)} &middot; "
                    f"{exp.explanation_time_ms:.1f} ms",
                    unsafe_allow_html=True,
                )
                ui.plain_language(exp.to_plain_language(settings["shap_top_k"]))
            st.caption("Open the Explainable AI page for the full waterfall "
                       "view of any of these records.")
