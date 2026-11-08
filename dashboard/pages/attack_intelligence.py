"""Page 5 - Attack Intelligence.

Per-category statistics come from three real sources, clearly labelled:
dataset ground-truth counts, live Stage B predictions, the persisted alert
store, and the stored evaluation metrics. Category descriptions are the
documented NSL-KDD attack-family definitions - no invented operational
intelligence or threat feeds.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from dashboard import common
from dashboard import components as ui

CHARACTERISTICS = {
    "Normal": {
        "family": "Benign traffic",
        "description": "Legitimate connection records with no attack label. "
                       "Used as the sole training source for the Isolation "
                       "Forest, so the anomaly model learns a normal-traffic "
                       "baseline rather than attack signatures.",
        "examples": "normal",
    },
    "DoS": {
        "family": "Denial of Service",
        "description": "Attempts to disrupt availability, typically visible as "
                       "high connection counts, large byte volumes and "
                       "connection-error rates (count, srv_count, serror_rate).",
        "examples": "back, land, neptune, pod, smurf, teardrop, apache2, "
                    "udpstorm, processtable, worm, mailbomb",
    },
    "Probe": {
        "family": "Surveillance / reconnaissance",
        "description": "Scanning and probing for hosts, ports and services. "
                       "Often shows host-diversity features "
                       "(dst_host_srv_diff_host_rate, srv_diff_host_rate) and "
                       "many short connections.",
        "examples": "satan, ipsweep, nmap, portsweep, mscan, saint",
    },
    "R2L": {
        "family": "Remote to Local",
        "description": "Unauthorised access from a remote host, commonly "
                       "through credential guessing or service exploitation. "
                       "Authentication-related features (num_failed_logins, "
                       "logged_in, su_attempted) are the usual signals. Rare "
                       "in the training set, which is why recall is low.",
        "examples": "guess_passwd, ftp_write, imap, phf, multihop, warezmaster, "
                    "warezclient, spy, xlock, xsnoop, snmpgetattack, httptunnel, "
                    "sendmail, named, snmpguess",
    },
    "U2R": {
        "family": "User to Root",
        "description": "Local privilege escalation after an attacker already "
                       "has user-level access. Indicators include root_shell, "
                       "num_root, num_compromised and num_shells. Extremely "
                       "rare (52 training records), so both classifiers "
                       "under-detect it.",
        "examples": "buffer_overflow, loadmodule, perl, rootkit, sqlattack, "
                    "xterm, ps",
    },
}


def render() -> None:
    settings = common.get_settings()
    evaluation = common.get_evaluation() or {}
    alerts = common.get_alerts()
    classifier = settings["classifier"]

    ui.topbar(
        ["AI-SOC", "Attack Intelligence"],
        "Attack Intelligence",
        "Category-level detection behaviour, classifier comparison and "
        "misclassification analysis on the NSL-KDD benchmark.",
        chips=[
            ui.chip("Dataset", settings["dataset"], "database"),
            ui.chip("Evaluation classifier", classifier, "cpu"),
            ui.chip("Evaluated", ui.timestamp_label(evaluation.get("evaluated_at"))[:19],
                    "clock"),
        ],
        notifications=common.notification_count(alerts),
        refresh_key="ai_refresh",
        icon_name="crosshair",
    )

    models = common.get_models()
    if not common.models_available(models):
        common.render_missing_models_notice(models)
        return
    with st.spinner("Running the pipeline to obtain live per-record predictions..."):
        analysis = common.run_dataset_analysis(settings)
    if analysis is None or not analysis.ok:
        ui.empty_state("Analysis unavailable",
                       "The selected dataset could not be analysed.", "alert")
        return
    records = analysis.records
    labelled = bool(records["labelled"].iloc[0]) if len(records) else False

    clf_metrics = evaluation.get("attack_classification", {}).get(classifier, {})
    per_class = clf_metrics.get("per_class", {})
    alert_counts = (alerts["predicted_category"].value_counts().to_dict()
                    if not alerts.empty else {})
    pred_counts = records["predicted_category"].value_counts().to_dict()
    true_counts = (records["true_category"].value_counts().to_dict()
                   if labelled else {})

    # ------------------------------------------------- category overview
    with ui.card("Supported categories", "crosshair",
                 f"Ground truth and predictions from {settings['dataset']}; "
                 f"performance from the stored {classifier} evaluation",
                 key="ai_categories"):
        rows = []
        for cat in config.ATTACK_CATEGORIES:
            metrics = per_class.get(cat, {})
            rows.append({
                "Category": cat,
                "Ground-truth records": true_counts.get(cat, 0) if labelled else None,
                "Predicted records (live)": pred_counts.get(cat, 0),
                "Alerts in store": alert_counts.get(cat, 0),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1-score": metrics.get("f1"),
                "Support (test)": metrics.get("support"),
            })
        table = pd.DataFrame(rows)
        display = table.copy()
        for col in ("Precision", "Recall", "F1-score"):
            display[col] = display[col].map(
                lambda v: f"{v:.4f}" if isinstance(v, (int, float))
                and pd.notna(v) else "not evaluated")
        st.dataframe(display, width="stretch", hide_index=True, height=240,
                     key="ai_cat_table")
        if not per_class:
            st.caption("No stored evaluation for this classifier - run "
                       "`python train.py` to populate performance metrics.")
        else:
            weakest = clf_metrics.get("weakest_category", {})
            strongest = clf_metrics.get("strongest_category", {})
            st.caption(
                f"{classifier}: strongest category "
                f"{strongest.get('name', '?')} (F1 {strongest.get('f1', 0):.4f}), "
                f"weakest category {weakest.get('name', '?')} "
                f"(F1 {weakest.get('f1', 0):.4f}). Poor categories are shown "
                "rather than hidden."
            )

    # ------------------------------------------- per-family characteristics
    with ui.card("Category characteristics", "file",
                 "Documented NSL-KDD attack families (not threat intelligence)",
                 key="ai_chars"):
        tabs = st.tabs(config.ATTACK_CATEGORIES)
        for tab, cat in zip(tabs, config.ATTACK_CATEGORIES):
            with tab:
                info = CHARACTERISTICS[cat]
                ui.plain_language(
                    f"<b>{info['family']}</b><br>{info['description']}"
                )
                st.caption(f"NSL-KDD labels mapped to this category: "
                           f"{info['examples']}")
                cols = st.columns(3)
                metrics = per_class.get(cat, {})
                cols[0].metric("Precision",
                               ui.pct(metrics.get("precision"), 2, "n/a"))
                cols[1].metric("Recall", ui.pct(metrics.get("recall"), 2, "n/a"))
                cols[2].metric("F1", ui.fmt(metrics.get("f1"), 4, missing="n/a"))
                st.caption(f"Source: {config.DATASET_SOURCE} Attack names and "
                           "groupings follow the documented NSL-KDD mapping in "
                           "config.py / README.")

    # ------------------------------------------------- comparison + matrix
    col_a, col_b = st.columns([1.2, 1])
    with col_a, ui.card("Performance across categories", "chart",
                        "Per-class F1 from the stored test-set evaluation",
                        key="ai_compare"):
        rf = evaluation.get("attack_classification", {}).get("random_forest", {})
        xgb = evaluation.get("attack_classification", {}).get("xgboost", {})
        if rf and xgb:
            series = {
                "Random Forest": [
                    float(rf.get("per_class", {}).get(c, {}).get("f1", 0))
                    for c in config.ATTACK_CATEGORIES],
                "XGBoost": [
                    float(xgb.get("per_class", {}).get(c, {}).get("f1", 0))
                    for c in config.ATTACK_CATEGORIES],
            }
            ui.show(ui.grouped_bar(config.ATTACK_CATEGORIES, series,
                                   height=340, y_range=(0, 1.05),
                                   colors=[ui.ACCENT, ui.ACCENT_2]),
                    key="ai_compare_chart")
            st.caption("F1 is low for R2L and U2R with both models - a known "
                       "consequence of extreme class imbalance in NSL-KDD.")
        else:
            ui.empty_state("No stored comparison",
                           "Run `python train.py` to generate evaluation "
                           "metrics for both classifiers.", "chart")

    with col_b, ui.card("Confusion matrix", "grid",
                        f"{classifier} on {evaluation.get('data_quality', {}).get('test', {}).get('n_rows', 0):,} test records",
                        key="ai_cm"):
        cm = clf_metrics.get("confusion_matrix", {})
        if cm:
            labels = cm.get("labels", config.ATTACK_CATEGORIES)
            matrix = cm.get("matrix", [])
            mode = st.radio("Display", ["Counts", "Row-normalised"],
                            horizontal=True, key="ai_cm_mode",
                            label_visibility="collapsed")
            if mode == "Row-normalised":
                import numpy as np
                arr = np.asarray(matrix, dtype=float)
                row_sum = arr.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1.0
                shown = (arr / row_sum).round(4)
                ui.show(ui.heatmap(shown, labels, labels, height=380,
                                   colorscale="Viridis", text=False),
                        key="ai_cm_chart")
            else:
                ui.show(ui.heatmap(matrix, labels, labels, height=380,
                                   colorscale="Blues"), key="ai_cm_chart")
            st.caption("Rows = actual category, columns = predicted category.")
        else:
            ui.empty_state("No confusion matrix stored",
                           "Run `python train.py`.", "grid")

    # ------------------------------------------- misclassified examples
    with ui.card("Misclassified examples", "search",
                 "Inspect real errors made by the live run (ground truth vs "
                 "prediction)", key="ai_errors", elevated=True):
        if not labelled:
            ui.empty_state(
                "Unlabelled data",
                "Misclassification analysis needs ground-truth labels. Load a "
                "labelled NSL-KDD file to inspect errors.", "search")
            return
        errors_only = records[records["true_category"] != records["predicted_category"]]
        st.caption(f"{len(errors_only):,} of {len(records):,} records "
                   f"({len(errors_only) / max(len(records), 1):.1%}) were "
                   f"misclassified by {classifier} in this run.")
        c1, c2, c3 = st.columns(3)
        with c1:
            true_cat = st.selectbox("Actual category",
                                    config.ATTACK_CATEGORIES, key="ai_true_cat")
        with c2:
            pred_options = ["Any prediction"] + config.ATTACK_CATEGORIES
            pred_cat = st.selectbox("Predicted as", pred_options, key="ai_pred_cat")
        with c3:
            error_kind = st.selectbox(
                "Error type", ["All errors", "False negatives (attack missed)",
                               "False positives (benign flagged)"],
                key="ai_kind")
        subset = errors_only[errors_only["true_category"] == true_cat]
        if pred_cat != "Any prediction":
            subset = subset[subset["predicted_category"] == pred_cat]
        if error_kind == "False negatives (attack missed)":
            subset = subset[(subset["true_category"] != "Normal")
                            & (subset["predicted_category"] == "Normal")]
        elif error_kind == "False positives (benign flagged)":
            subset = subset[(subset["true_category"] == "Normal")
                            & (subset["predicted_category"] != "Normal")]
        if subset.empty:
            ui.empty_state("No records match",
                           f"{classifier} made no errors of this kind in "
                           f"{settings['dataset']}.", "check-circle")
        else:
            columns = ["record_index", "true_label", "true_category",
                       "predicted_category", "model_confidence",
                       "anomaly_score", "detection_status"]
            ui.show_table(ui.paginate(subset[columns], 15, "ai_errors"),
                          height=300, key="ai_error_table")
            st.caption(f"{len(subset):,} matching records.")
            idx = st.number_input("Record index to inspect", min_value=0,
                                  max_value=int(len(records) - 1),
                                  value=int(subset["record_index"].iloc[0]),
                                  step=1, key="ai_inspect_idx")
            dataset = common.dataset_frame(settings["dataset"])
            if dataset is not None and idx < len(dataset):
                row = dataset.iloc[int(idx)]
                feat_df = pd.DataFrame({
                    "feature": config.FEATURE_NAMES,
                    "value": ["" if pd.isna(row.get(f, "")) else str(row.get(f, ""))
                              for f in config.FEATURE_NAMES],
                })
                st.dataframe(feat_df, width="stretch", hide_index=True,
                             height=300, key="ai_features")
                if st.button("Explain this record with SHAP",
                             icon=":material/psychology:", key="ai_to_xai"):
                    st.session_state["focus_record"] = int(idx)
                    from dashboard import nav
                    st.switch_page(nav.page("explainability"))
