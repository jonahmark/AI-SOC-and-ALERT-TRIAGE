"""Page 4 - Alert Management (triage workspace).

Alerts come from the persisted store produced by real model runs. Analyst
workflow changes (status, notes) are written back to disk. Recommendations are
rule-based suggestions generated from the predicted category - they are
labelled as suggestions, never as confirmed incident-response actions.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src import settings as settings_module
from src.auth import rbac
from src.severity_engine import SeverityEngine
from dashboard import auth_ui, common
from dashboard import components as ui
from dashboard import nav

SORT_OPTIONS = {
    "Risk score (high to low)": ("risk_score", False),
    "Risk score (low to high)": ("risk_score", True),
    "Model confidence": ("model_confidence", False),
    "Anomaly score (most anomalous first)": ("anomaly_score", True),
    "Detection timestamp": ("timestamp", False),
    "Alert ID": ("alert_id", False),
}


def _set_status(store, alert_id: str, new_status: str) -> None:
    """Persist a workflow transition and reload the queue."""
    actor = auth_ui.current_user()
    if actor is None or not rbac.can(actor["role"], "triage_alerts"):
        st.error("Your role has read-only access to the alert queue.")
        return
    try:
        store.update_status(alert_id, new_status)
    except (KeyError, ValueError) as exc:
        st.error(str(exc))
        return
    common.invalidate_alerts()
    settings_module.log(f"Alert {alert_id} status -> {new_status}.")
    st.success(f"{alert_id} marked as {new_status}.")
    st.rerun()


def _local_explanation(alert: pd.Series, models: dict, settings: dict):
    """Real per-alert SHAP for the record behind this alert (cached)."""
    cache = st.session_state.setdefault("shap_cache", {})
    key = (alert["alert_id"], settings["classifier"], settings["dataset"])
    if key in cache:
        return cache[key]

    dataset = common.dataset_frame(settings["dataset"])
    if dataset is None:
        return {"error": f"{settings['dataset']} is not available, so the "
                         "underlying record cannot be re-explained."}
    idx = int(alert["record_index"])
    if idx >= len(dataset):
        return {"error": f"Record index {idx} is outside {settings['dataset']} "
                         f"({len(dataset):,} rows). The alert was generated "
                         "from a different dataset."}
    clf = models.get(settings["classifier"])
    pre = models.get("pre")
    if clf is None or pre is None:
        return {"error": "Classifier or preprocessor artefact is missing."}
    try:
        from src.explainability import Explainer
        X = pre.transform(dataset.iloc[[idx]])
        explainer = Explainer(clf, pre.feature_names_)
        explanation = explainer.explain_record(X[0])
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"SHAP explanation failed: {exc}"}
    cache[key] = explanation
    return explanation


def render() -> None:
    settings = common.get_settings()
    models = common.get_models()
    store = common.get_alert_store()
    alerts = common.get_alerts()
    actor = auth_ui.current_user()
    can_triage = actor is not None and rbac.can(actor["role"], "triage_alerts")

    generated = alerts["timestamp"].iloc[0][:19].replace("T", " ") + " UTC" \
        if not alerts.empty else "no alerts stored"

    ui.topbar(
        ["AI-SOC", "Alert Management"],
        "Alert Management",
        "Triage, investigate and document security alerts raised by the "
        "detection pipeline.",
        chips=[
            ui.chip("Alerts", f"{len(alerts):,}", "bell"),
            ui.chip("Generated", generated, "clock"),
            ui.chip("Severity map", "custom" if settings_diff(settings)
                    else "research defaults", "filter"),
        ],
        notifications=common.notification_count(alerts),
        refresh_key="am_refresh",
        icon_name="bell",
    )

    if not can_triage:
        ui.note("Read-only access: the Researcher / Viewer role can inspect "
                "alerts and explanations but cannot change statuses, add "
                "notes or regenerate the store. Ask an administrator for "
                "the SOC Analyst role to triage.", "lock", ui.ACCENT_2)

    if alerts.empty:
        ui.empty_state(
            "No alerts in the store",
            "Run the pipeline on the <b>Network Analysis</b> page and save the "
            "generated alerts, or re-run <code>python train.py</code> to "
            "regenerate alerts for the full benchmark dataset.",
            "bell",
        )
        return

    # ------------------------------------------------------- summary cards
    counts = alerts["investigation_status"].value_counts()
    sev = alerts["severity"].value_counts()
    ui.metric_row([
        ui.metric_card("Total alerts", f"{len(alerts):,}",
                       "Raised by Stage A or Stage B", "bell", ui.ACCENT),
        ui.metric_card("New", f"{int(counts.get('New', 0)):,}",
                       "Not yet triaged by an analyst", "info", "#38BDF8"),
        ui.metric_card("Critical", f"{int(sev.get('Critical', 0)):,}",
                       "Predicted U2R under the current severity map",
                       "alert", "#EF4444"),
        ui.metric_card("High priority", f"{int(sev.get('High', 0)):,}",
                       "Predicted DoS or R2L", "zap", "#F97316"),
        ui.metric_card("Investigating", f"{int(counts.get('Investigating', 0)):,}",
                       "Analyst has opened the alert", "eye", "#EAB308"),
        ui.metric_card("Resolved / false positive",
                       f"{int(counts.get('Resolved', 0) + counts.get('False Positive', 0)):,}",
                       "Closed by an analyst decision (label change only - "
                       "not verified remediation)", "check-circle", "#22C55E"),
    ])

    # ------------------------------------------------------------ filters
    with ui.card("Filters", "filter", "Narrow the alert queue", key="am_filters"):
        c1, c2, c3, c4, c5 = st.columns([1.4, 1.1, 1.1, 1.1, 1.3])
        with c1:
            search = st.text_input("Search alert ID", placeholder="ALT-...",
                                   key="am_search")
        with c2:
            sev_filter = st.multiselect("Severity", config.SEVERITY_ORDER,
                                        default=list(config.SEVERITY_ORDER),
                                        key="am_sev")
        with c3:
            cat_filter = st.multiselect("Attack category", config.ATTACK_CATEGORIES,
                                        default=list(config.ATTACK_CATEGORIES),
                                        key="am_cat")
        with c4:
            status_filter = st.multiselect("Investigation status",
                                           config.INVESTIGATION_STATUSES,
                                           default=list(config.INVESTIGATION_STATUSES),
                                           key="am_status")
        with c5:
            sort_choice = st.selectbox("Sort by", list(SORT_OPTIONS),
                                       key="am_sort")

        from src.alert_generator import AlertStore
        sort_key, ascending = SORT_OPTIONS[sort_choice]
        filtered = AlertStore.filter_alerts(
            alerts, severity=None, category=None, status=None,
            search=search.strip() or None, sort_by=sort_key, ascending=ascending,
        )
        filtered = filtered[
            filtered["severity"].isin(sev_filter)
            & filtered["predicted_category"].isin(cat_filter)
            & filtered["investigation_status"].isin(status_filter)
        ].reset_index(drop=True)

    st.caption(f"{len(filtered):,} of {len(alerts):,} alerts match the filters.")
    if filtered.empty:
        ui.empty_state("No matching alerts",
                       "Adjust the filters to widen the search.", "search")
        return

    columns = ["alert_id", "timestamp", "predicted_category", "severity",
               "risk_score", "model_confidence", "detection_status",
               "investigation_status"]
    if settings["show_ground_truth"] and "true_category" in filtered.columns:
        columns.append("true_category")
    table_view = filtered[columns].copy()
    table_view["timestamp"] = table_view["timestamp"].str[:19].str.replace("T", " ")
    with ui.card("Alert queue", "list", "Select an alert below to investigate",
                 key="am_queue"):
        ui.show_table(ui.paginate(table_view, settings["table_page_size"], "am"),
                      height=380, key="am_table")

    # --------------------------------------------------------- selection
    focus = st.session_state.pop("focus_alert", None)
    head = filtered.head(300)
    options = [
        f"{r.alert_id}  |  {r.predicted_category}  |  {r.severity}  |  "
        f"risk {r.risk_score:.1f}" for r in head.itertuples()
    ]
    index = 0
    if focus:
        match = [i for i, o in enumerate(options) if o.startswith(focus)]
        if match:
            index = match[0]
    picked = st.selectbox(
        f"Alert to investigate (showing the {len(options)} highest-ranked "
        "matching alerts)", options, index=index, key="am_pick")
    alert_id = picked.split("  |")[0]
    alert = filtered[filtered["alert_id"] == alert_id].iloc[0]
    current = str(alert["investigation_status"])

    # --------------------------------------------- details header + actions
    st.markdown(
        '<div class="soc-crumb" style="margin-top:6px">'
        f'{ui.icon("chevron", 11)}<span>Alert details</span>'
        f'<span class="sep">{ui.icon("chevron", 11)}</span>'
        f'<span class="current">{alert_id}</span></div>',
        unsafe_allow_html=True,
    )
    a1, a2, a3, a4, a5 = st.columns([1.15, 1.15, 1, 1.15, 1.35],
                                     vertical_alignment="bottom", gap="small")
    with a1:
        closure_confirmed = st.checkbox(
            "Confirm closure", key=f"am_confirm_{alert_id}",
            help="Required before an alert can be marked Resolved or False "
                 "Positive. Closing records an analyst decision; it does not "
                 "verify that an incident was remediated.",
        )
    with a2:
        if st.button("Mark investigating", type="primary",
                     icon=":material/manage_search:", width="stretch",
                     key=f"am_quick_inv_{alert_id}",
                     disabled=current == "Investigating" or not can_triage):
            _set_status(store, alert_id, "Investigating")
    with a3:
        if st.button("Mark resolved", icon=":material/check_circle:",
                     width="stretch", key=f"am_quick_res_{alert_id}",
                     disabled=current == "Resolved" or not closure_confirmed
                     or not can_triage):
            _set_status(store, alert_id, "Resolved")
    with a4:
        if st.button("Flag as false positive", icon=":material/block:",
                     width="stretch", key=f"am_quick_fp_{alert_id}",
                     disabled=current == "False Positive"
                     or not closure_confirmed or not can_triage):
            _set_status(store, alert_id, "False Positive")
    with a5:
        if st.button("Explain in Explainable AI", icon=":material/psychology:",
                     width="stretch", key=f"am_quick_xai_{alert_id}"):
            st.session_state["focus_record"] = int(alert["record_index"])
            st.switch_page(nav.page("explainability"))

    # ------------------------------------------------------------ details
    explanation = None
    if common.models_available(models) \
            and st.session_state.get(f"shap_show_{alert_id}"):
        with st.spinner("Computing SHAP values for this record..."):
            explanation = _local_explanation(alert, models, settings)

    left, right = st.columns([1, 1.6])

    with left:
        with ui.card("Alert information", "info", alert_id, key="am_info"):
            ui.kv_table([
                ("Alert ID", alert["alert_id"]),
                ("Detection timestamp",
                 ui.timestamp_label(alert["timestamp"])),
                ("Attack category (predicted)",
                 ui.category_badge(str(alert["predicted_category"]))),
                ("Severity", ui.severity_badge(str(alert["severity"]))),
                ("Risk score", f"{alert['risk_score']:.2f} / 100"),
                ("Model confidence", f"{alert['model_confidence']:.1%}"),
                ("Anomaly score", f"{alert['anomaly_score']:.4f}"),
                ("Stage A decision", str(alert["detection_status"])),
                ("Investigation status", ui.status_pill(current)),
                ("Record index", f"{int(alert['record_index']):,}"),
            ])
            ui.progress("Risk score", float(alert["risk_score"]),
                        ui.SEVERITY_COLORS.get(str(alert["severity"]), ui.ACCENT),
                        max_value=100.0, digits=2, suffix=" / 100",
                        sub="0-100 scale from the configured severity, "
                            "confidence and anomaly weights")
            if settings["show_ground_truth"] and alert.get("true_category"):
                st.caption(
                    f"Ground-truth label for this record: "
                    f"{alert['true_label']} -> {alert['true_category']} "
                    "(dataset label, shown for research evaluation only)."
                )
            w = settings["risk_weights"]
            st.caption(
                f"Risk formula: 100 x ({w['severity']} x severity weight + "
                f"{w['confidence']} x confidence + {w['anomaly']} x normalised "
                "anomaly score). Research prioritisation heuristic, not a "
                "validated risk measurement."
            )

        with ui.card("Analyst recommendations", "note",
                     "Rule-based suggestions - not confirmed response actions",
                     key="am_rec"):
            ui.plain_language(f"<b>Suggested next step:</b> "
                              f"{alert['recommended_action']}")
            sev_engine = SeverityEngine(settings["severity_map"])
            indicators = {
                "DoS": ["Connection volume features (count, srv_count, "
                        "serror_rate) are the usual drivers.",
                        "Check whether the same source appears in other "
                        "high-severity alerts in this queue."],
                "Probe": ["Scan indicators (srv_diff_host_rate, "
                          "dst_host_srv_diff_host_rate) typically dominate.",
                          "Look for follow-on alerts from the same record "
                          "neighbourhood before closing."],
                "R2L": ["Authentication-related features (num_failed_logins, "
                        "logged_in, su_attempted) are worth reviewing.",
                        "R2L recall is low for both classifiers on NSL-KDD "
                        "(see Model Performance), so verify carefully."],
                "U2R": ["Local privilege-escalation indicators (root_shell, "
                        "num_root, num_compromised, num_shells).",
                        "U2R is extremely rare in NSL-KDD; false positives are "
                        "common - inspect the SHAP contributions."],
                "Normal": ["No attack category predicted; the alert was raised "
                           "by the anomaly detector alone.",
                           "Compare the anomaly score against the threshold "
                           "before escalating."],
            }
            for line in indicators.get(str(alert["predicted_category"]), []):
                st.markdown(
                    f'<div style="display:flex;gap:8px;align-items:flex-start;'
                    f'margin:5px 0;color:#D7E0EC;font-size:.88rem">'
                    f'<span style="color:#22C55E;flex:0 0 auto">'
                    f'{ui.icon("check", 14, "#22C55E")}</span>'
                    f'<span>{line}</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"Recommendations are generated from the predicted category and "
                f"severity ({sev_engine.severity_for(str(alert['predicted_category']))}) "
                "using documented NSL-KDD feature semantics. They require "
                "analyst judgement."
            )

        with ui.card("Related information", "layers",
                     "Where this alert came from", key="am_related"):
            rows = [
                ("Predicted category", str(alert["predicted_category"])),
                ("Model used (Stage B)", settings["classifier"]),
                ("Anomaly model (Stage A)", "Isolation Forest"),
                ("Dataset", settings["dataset"]),
                ("Record index", f"{int(alert['record_index']):,}"),
                ("Alert generated", ui.timestamp_label(alert["timestamp"])),
            ]
            if not isinstance(explanation, dict) and explanation is not None:
                rows.append(("Local SHAP time (measured)",
                             f"{explanation.explanation_time_ms:.1f} ms"))
                rows.append(("Explained class", explanation.predicted_class))
            ui.kv_table(rows)

    with right:
        with ui.card("Detection explanation", "sparkles",
                     "Why the pipeline raised this alert", key="am_explain"):
            ui.plain_language(str(alert["explanation"]))
            st.divider()
            if not common.models_available(models):
                ui.note("Model artefacts are missing, so a local SHAP "
                        "explanation cannot be generated.", "alert", "#EF4444")
            elif st.button("Generate local SHAP explanation",
                           icon=":material/psychology:", key="am_shap_btn",
                           help=f"Explains this record with the "
                                f"{settings['classifier']} classifier "
                                "(computed on demand, ~tens of ms)."):
                st.session_state[f"shap_show_{alert_id}"] = True
                st.rerun()

            if isinstance(explanation, dict):
                st.error(explanation["error"])
                st.caption(
                    "No explanation is displayed rather than an "
                    "approximated or fabricated one."
                )
            elif explanation is not None:
                top_k = int(settings["shap_top_k"])
                st.markdown(
                    f"**Predicted class:** {explanation.predicted_class} "
                    f"&middot; **model:** {explanation.model_type} "
                    f"&middot; **base value:** "
                    f"{explanation.base_value:.4f} &middot; "
                    f"**computed in "
                    f"{explanation.explanation_time_ms:.1f} ms**",
                    unsafe_allow_html=True,
                )
                c_bars, c_flow = st.columns([1, 1.15], gap="small")
                with c_bars:
                    st.markdown("**Key contributing features (local SHAP)**")
                    ui.show(ui.shap_bars(explanation, top_k),
                            key=f"am_shap_{alert_id}")
                with c_flow:
                    st.markdown("**SHAP waterfall (base value to output)**")
                    ui.show(ui.shap_waterfall(explanation, top_k, height=400),
                            key=f"am_shap_wf_{alert_id}")
                st.caption(
                    "Green segments push the prediction toward "
                    f"'{explanation.predicted_class}', red segments push "
                    "against it. Values are SHAP contributions for this single "
                    "record, not global feature importance."
                )
                ui.plain_language(
                    explanation.to_plain_language(top_k)
                )
                st.caption(
                    "Global (model-level) importance and the interactive "
                    "summary plot: Explainable AI page."
                )

        with ui.card("Investigation workflow", "user",
                     "Status changes and notes are persisted to disk",
                     key="am_workflow"):
            c1, c2 = st.columns([1.4, 1])
            with c1:
                new_status = st.selectbox(
                    "Investigation status", config.INVESTIGATION_STATUSES,
                    index=config.INVESTIGATION_STATUSES.index(current),
                    key=f"am_status_{alert_id}")
            with c2:
                st.markdown("<div style='height:28px'></div>",
                            unsafe_allow_html=True)
                closing = new_status in ("Resolved", "False Positive")
                confirmed = closure_confirmed if closing else True
                if st.button("Update status", icon=":material/save:",
                             key=f"am_update_{alert_id}", width="stretch",
                             disabled=(new_status == current) or not confirmed
                             or not can_triage):
                    _set_status(store, alert_id, new_status)

    with ui.card("Investigation notes", "note",
                 "Persisted with the alert; visible after a reload",
                 key="am_notes"):
        notes = store.get_notes(alert_id)
        if notes:
            for n in notes:
                st.markdown(
                    f'<div class="soc-note" style="border-left-color:#22C55E">'
                    f'{ui.icon("note", 14, "#22C55E")}'
                    f'<div><b>{n["created_at"]}</b> &middot; {n["author"]}'
                    f'<br>{n["text"]}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No notes recorded for this alert yet.")
        note_text = st.text_area(
            "Add a note", key=f"am_note_{alert_id}", height=80,
            placeholder="What did you check? What did you find?",
        )
        if st.button("Add note", icon=":material/add_comment:",
                     key=f"am_add_note_{alert_id}",
                     disabled=not note_text.strip() or not can_triage):
            if actor is None or not rbac.can(actor["role"], "triage_alerts"):
                st.error("Your role has read-only access to the alert queue.")
                return
            try:
                store.add_note(alert_id, note_text,
                               author=actor["username"] if actor else "analyst")
                settings_module.log(f"Note added to alert {alert_id}.")
                st.success("Note saved.")
                st.rerun()
            except (KeyError, ValueError) as exc:
                st.error(str(exc))

    # ------------------------------------------------- regenerate alerts
    with ui.card("Regenerate the alert store", "refresh",
                 "Rebuild alerts from the selected dataset using the current "
                 "configuration", key="am_regen", elevated=True):
        drift = settings_diff(settings)
        if drift:
            ui.note("Current configuration differs from the research defaults: "
                    + "; ".join(drift), "info", "#EAB308")
        st.caption(
            "This replaces the stored alerts with a fresh run over "
            f"{settings['dataset']} using {settings['classifier']} and the "
            "configured threshold, severity map and risk weights. Statuses and "
            "notes belonging to alerts that no longer exist are discarded."
        )
        confirm = st.checkbox("I understand the existing alert queue will be "
                              "replaced.", key="am_regen_confirm")
        if st.button("Regenerate alerts", type="primary",
                     icon=":material/autorenew:",
                     disabled=not confirm or not can_triage,
                     key="am_regen_run"):
            if actor is None or not rbac.can(actor["role"], "triage_alerts"):
                st.error("Your role has read-only access to the alert queue.")
                return
            if not common.models_available(models):
                common.render_missing_models_notice(models)
                return
            bar = st.progress(0.0, text="Starting regeneration...")

            def on_stage(index, stage):
                bar.progress(min((index + 1) / 7, 1.0),
                             text=f"Stage {index + 1}/7: {stage.name} - "
                                  f"{stage.detail or stage.status}")

            from src import pipeline
            df = common.dataset_frame(settings["dataset"])
            if df is None:
                st.error(f"{settings['dataset']} is not available.")
                return
            result = pipeline.run_analysis(
                df, models, settings,
                validation=common.dataset_validation(settings["dataset"]),
                source_label=settings["dataset"], progress_callback=on_stage,
            )
            bar.empty()
            if not result.ok:
                for err in result.errors:
                    st.error(err)
                return
            store.clear()
            store.save(result.alerts, merge=False)
            common.invalidate_alerts()
            settings_module.log(
                f"Alert store regenerated: {len(result.alerts):,} alerts from "
                f"{settings['dataset']} "
                f"(classifier={settings['classifier']}, "
                f"threshold={result.meta['anomaly_threshold']:.4f})."
            )
            st.success(
                f"{len(result.alerts):,} alerts regenerated in "
                f"{result.meta['total_seconds']:.2f} s."
            )
            st.rerun()


def settings_diff(settings: dict) -> list[str]:
    return settings_module.settings_diff(settings)
