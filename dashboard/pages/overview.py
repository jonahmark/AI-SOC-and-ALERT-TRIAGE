"""Page 1 - Security Operations Overview.

Every number on this page comes from either (a) a live run of the real
pipeline over the selected NSL-KDD file, or (b) the persisted alert store /
evaluation artefacts. Sources are labelled on each card.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import config
from src.settings import settings_diff
from dashboard import common
from dashboard import components as ui
from dashboard import nav


def _grouped_hbar(labels, series: dict[str, list[float]], height=320):
    fig = go.Figure()
    for (name, values), colour in zip(series.items(), [ui.ACCENT, ui.ACCENT_2]):
        fig.add_trace(go.Bar(
            name=name, y=list(labels), x=list(values), orientation="h",
            marker=dict(color=colour, line=dict(color=ui.BORDER, width=0.5)),
            hovertemplate=f"{name}<br>%{{y}}: %{{x:,}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1,
                      xaxis=dict(title="records"))
    return ui.style_fig(fig, height=height)


def render() -> None:
    settings = common.get_settings()
    alerts = common.get_alerts()

    ui.topbar(
        ["AI-SOC", "Overview"],
        "Security Operations Overview",
        "Monitor network activity, detect anomalies and prioritise security alerts.",
        chips=[
            ui.chip("Dataset", settings["dataset"], "database"),
            ui.chip("Classifier", settings["classifier"], "cpu"),
            ui.chip("Models", common.model_status_label(common.get_models())[0],
                    "shield-check"),
            ui.chip("Last analysis", common.last_analysis_label(), "clock"),
        ],
        notifications=common.notification_count(alerts),
        refresh_key="ov_refresh",
        icon_name="gauge",
    )

    models = common.get_models()
    if not common.models_available(models):
        common.render_missing_models_notice(models)
        return

    with st.spinner(f"Running the real pipeline over {settings['dataset']}..."):
        analysis = common.run_dataset_analysis(settings)
    if analysis is None:
        ui.empty_state(
            "Dataset unavailable",
            f"<code>{settings['dataset']}</code> was not found in "
            "<code>data/raw/</code>. Run <code>python scripts/download_dataset.py</code>.",
            "database",
        )
        return
    if not analysis.ok:
        for err in analysis.errors:
            st.error(err)
        return

    records = analysis.records
    meta = analysis.meta
    n_total = meta["n_records"]
    n_anom = meta["n_anomalous"]
    n_normal = n_total - n_anom

    labelled = bool(records["labelled"].iloc[0]) if len(records) else False
    if labelled:
        attack_mask = records["true_category"].isin(["DoS", "Probe", "R2L", "U2R"])
        detection_rate = (
            float((records.loc[attack_mask, "detection_status"] == "Anomaly").mean())
            if attack_mask.any() else None
        )
        rate_caption = (
            f"Isolation Forest recall on the {int(attack_mask.sum()):,} "
            f"ground-truth attack records in {settings['dataset']} (live run)"
        )
    else:
        detection_rate = n_anom / n_total if n_total else None
        rate_caption = ("No ground-truth labels in this dataset: showing the "
                        "share of records flagged anomalous (live run)")

    n_critical = (int((alerts["severity"] == "Critical").sum())
                  if not alerts.empty else 0)
    store_note = "persisted alert store"
    if not alerts.empty and "timestamp" in alerts.columns:
        store_note = ("alert store &middot; generated "
                      + ui.timestamp_label(alerts["timestamp"].iloc[0]))

    # ------------------------------------------------------- metric cards
    ui.metric_row([
        ui.metric_card(
            "Total records analysed", f"{n_total:,}",
            f"{settings['dataset']} &middot; live pipeline run "
            f"{meta['ran_at'][11:19]} UTC",
            "layers", ui.ACCENT,
        ),
        ui.metric_card(
            "Normal traffic", f"{n_normal:,}",
            f"{n_normal / n_total:.1%} of records scored above the anomaly "
            f"threshold ({meta['anomaly_threshold']:.4f})",
            "check-circle", "#22C55E",
        ),
        ui.metric_card(
            "Anomalous traffic", f"{n_anom:,}",
            f"{n_anom / n_total:.1%} flagged by Isolation Forest - "
            f"threshold source: {meta['threshold_source']}",
            "activity", "#F97316",
        ),
    ])
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ui.metric_row([
        ui.metric_card(
            "Total security alerts", f"{len(alerts):,}", store_note,
            "bell", ui.ACCENT_2,
        ),
        ui.metric_card(
            "Critical alerts", f"{n_critical:,}",
            "Predicted U2R (privilege escalation) under the current severity map",
            "alert", "#EF4444",
        ),
        ui.metric_card(
            "Detection rate", ui.pct(detection_rate), rate_caption,
            "trend", "#38BDF8",
        ),
    ])

    drift = settings_diff(settings)
    if drift and not alerts.empty:
        ui.note(
            "The stored alerts were generated with the research defaults; the "
            "current configuration differs (" + ", ".join(
                d.split(":")[0] for d in drift) + "). Regenerate alerts on the "
            "<b>Alert Management</b> page to keep counts consistent.",
            "info",
        )

    # ------------------------------------------------ traffic + attacks
    col_a, col_b = st.columns([1, 1.35])
    with col_a, ui.card("Traffic overview", "activity",
                        "Stage A decision (Isolation Forest) on the live run",
                        key="ov_traffic"):
        ui.show(ui.doughnut(
            ["Normal", "Anomalous"], [n_normal, n_anom],
            colors=["#22C55E", "#F97316"], height=300,
            center_label=f"{n_anom / n_total:.1%}" if n_total else "n/a",
            center_sub="anomalous",
        ), key="ov_traffic_chart")
        st.caption(f"Threshold {meta['anomaly_threshold']:.4f} - source: "
                   f"{meta['threshold_source']}.")

    with col_b, ui.card("Attack distribution", "crosshair",
                        "Predicted categories vs. dataset ground truth",
                        key="ov_attacks"):
        view = st.radio(
            "Show", ["Predicted vs ground truth", "Model predictions only",
                     "Ground-truth labels only"],
            horizontal=True, key="ov_attack_view", label_visibility="collapsed",
        )
        pred_counts = records["predicted_category"].value_counts()
        series = {"Model predictions (Stage B)": [
            float(pred_counts.get(c, 0)) for c in config.ATTACK_CATEGORIES]}
        if labelled and view != "Model predictions only":
            true_counts = records["true_category"].value_counts()
            series["Ground truth (dataset labels)"] = [
                float(true_counts.get(c, 0)) for c in config.ATTACK_CATEGORIES]
        if view == "Ground-truth labels only":
            series = {k: v for k, v in series.items() if "Ground" in k}
        ui.show(_grouped_hbar(config.ATTACK_CATEGORIES, series, height=320),
                key="ov_attack_chart")
        st.caption(
            "Ground truth is the NSL-KDD label mapped to a category; it is "
            "shown for comparison only and is never presented as a model output."
        )

    # --------------------------------------- severity + recent alerts
    col_c, col_d = st.columns([1, 1.6])
    with col_c, ui.card("Alert severity distribution", "bell", store_note,
                        key="ov_sev"):
        if alerts.empty:
            ui.empty_state("No alerts stored",
                           "Generate alerts on the Network Analysis page.",
                           "bell")
        else:
            counts = alerts["severity"].value_counts()
            order = [s for s in config.SEVERITY_ORDER if s in counts.index]
            ui.show(ui.doughnut(
                order, [float(counts[s]) for s in order],
                colors=[ui.SEVERITY_COLORS[s] for s in order], height=300,
                center_label=f"{len(alerts):,}", center_sub="stored alerts",
            ), key="ov_sev_chart")
            st.caption("Severity comes from the configurable mapping on the "
                       "System Settings page (research rules, not a validated "
                       "SOC risk framework).")

    with col_d, ui.card("Recent security alerts", "list",
                        "Highest-risk alerts currently in the store",
                        key="ov_recent"):
        if alerts.empty:
            ui.empty_state("No alerts stored", "Nothing to triage yet.", "list")
        else:
            recent = alerts.sort_values("risk_score", ascending=False).head(8)
            ui.alert_table(recent)
            st.caption(
                f"Showing the {len(recent):,} highest-risk of "
                f"{len(alerts):,} stored alerts. Risk scores are on the "
                "documented 0-100 scale produced by the configurable "
                "severity/confidence/anomaly weights."
            )
            pick_col, act_col = st.columns([2.4, 1])
            with pick_col:
                options = [
                    f"{r.alert_id}  ({r.predicted_category} / {r.severity})"
                    for r in recent.itertuples()]
                picked = st.selectbox("Open an alert for investigation", options,
                                      key="ov_pick")
            with act_col:
                st.markdown("<div style='height:28px'></div>",
                            unsafe_allow_html=True)
                if st.button("Investigate", icon=":material/open_in_new:",
                             key="ov_open", width="stretch"):
                    st.session_state["focus_alert"] = picked.split("  ")[0]
                    st.switch_page(nav.page("alert_management"))
            if st.button("View all alerts", icon=":material/arrow_forward:",
                         key="ov_view_all", width="stretch"):
                st.switch_page(nav.page("alert_management"))

    # ------------------------------------------------------ system health
    with ui.card("System health", "server",
                 "Live readings and artefact availability", key="ov_health"):
        snap = common.live_system_snapshot()
        state = common.get_pipeline_state() or {}
        evaluation = common.get_evaluation() or {}
        missing = [n for n, p in config.MODEL_FILES.items()
                   if not p.exists()]
        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            st.markdown("**Host resources (measured now)**")
            ui.progress("CPU usage", snap["cpu_percent"], ui.ACCENT, suffix="%",
                        sub=f"{snap['cpu_count_logical']} logical cores "
                            f"&middot; 0.1 s sample")
            ui.progress("System memory", snap["system_memory_percent"],
                        ui.ACCENT_2, suffix="%",
                        sub=f"{snap['system_memory_used_mb']:,.0f} MB of "
                            f"{snap['system_memory_total_mb']:,.0f} MB used")
            ui.progress("This process", snap["process_memory_mb"], "#22C55E",
                        max_value=max(snap["system_memory_total_mb"], 1.0),
                        digits=0, suffix=" MB",
                        sub="Dashboard process memory &middot; sampled "
                            f"{ui.timestamp_label(snap['sampled_at'])}")
        with c2:
            st.markdown("**Pipeline artefacts**")
            ui.kv_table([
                ("Models available", "All present" if not missing
                 else f"Missing: {', '.join(missing)}"),
                ("Primary classifier", state.get("primary_classifier", "n/a")),
                ("Trained at", ui.timestamp_label(state.get("trained_at"))),
                ("Last analysis", common.last_analysis_label()),
            ])
        with c3:
            st.markdown("**Dataset**")
            ui.kv_table([
                ("File", f"{settings['dataset']} "
                         f"({'present' if config.DATA_RAW_DIR.exists() else 'missing'})"),
                ("Records (train / test)",
                 f"{state.get('n_train_records', 'n/a'):,} / "
                 f"{state.get('n_test_records', 'n/a'):,}"),
                ("Evaluation artefact",
                 "present" if evaluation else "not generated"),
                ("Alert store", f"{len(alerts):,} alerts"),
            ])
        st.caption(
            "CPU and memory are real readings from this machine taken when the "
            "page rendered (0.1 s CPU sample). They describe the dashboard "
            "host, not a deployed SOC environment."
        )
