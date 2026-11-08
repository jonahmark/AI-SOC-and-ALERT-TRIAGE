"""Page 7 - Model Performance.

All headline metrics come from the stored evaluation artefact produced by
`train.py` (test-set evaluation on KDDTest+). ROC / precision-recall curves
that are not stored can be computed live from the saved models on demand, and
are labelled as such. Training-set metrics are not recorded by this pipeline
and are explicitly reported as unavailable rather than invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from dashboard import common
from dashboard import components as ui

CLASSIFIER_LABELS = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


TARGET_LABELS = {"roc_auc": "ROC-AUC", "recall": "Recall",
                 "accuracy": "Accuracy", "macro_f1": "Macro F1"}


def _target_chip(targets: dict, group: str, metric: str) -> str:
    """Honest 'target met / not met' chip built from the evaluation artefact."""
    entry = (targets.get(group) or {}).get(metric)
    if not entry or entry.get("actual") is None:
        return ""
    met = bool(entry.get("target_met"))
    colour = "#22C55E" if met else "#EAB308"
    label = TARGET_LABELS.get(metric, metric.replace("_", " "))
    actual = entry.get("actual")
    return ui.badge(
        f"{ui.icon('target', 11)} {label} target &ge; {entry['target']:g} "
        f"&middot; actual {float(actual):.4f} &middot; "
        f"{'met' if met else 'not met'}", colour)


def _chips_row(chips: list[str]) -> None:
    chips = [c for c in chips if c]
    if not chips:
        return
    st.markdown(
        '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px">'
        + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def _model_summary(evaluation: dict, state: dict) -> None:
    """One card per model with its real test-set headline metrics."""
    targets = evaluation.get("targets_vs_actuals", {})
    anom = evaluation.get("anomaly_detection", {})
    clf = evaluation.get("attack_classification", {})
    primary = state.get("primary_classifier")

    c1, c2, c3 = st.columns(3, gap="small")
    with c1, ui.card("Isolation Forest", "activity",
                     "Stage A - anomaly detection, test set", key="mp_sum_if"):
        ui.kv_table([
            ("ROC-AUC", ui.fmt(anom.get("roc_auc"), 4)),
            ("Precision", ui.pct(anom.get("precision"))),
            ("Recall", ui.pct(anom.get("recall"))),
            ("False positive rate", ui.pct(anom.get("false_positive_rate"))),
        ])
        _chips_row([_target_chip(targets, "anomaly_detection", m)
                    for m in ("roc_auc", "recall")])
        st.caption(f"Threshold {anom.get('threshold', 0):.4f} on "
                   f"{anom.get('n_records', 0):,} held-out test records.")

    for col, name in ((c2, "random_forest"), (c3, "xgboost")):
        metrics = clf.get(name, {})
        subtitle = "Stage B - five-class classification, test set"
        if primary == name:
            subtitle += " &middot; primary"
        with col, ui.card(CLASSIFIER_LABELS[name], "cpu", subtitle,
                          key=f"mp_sum_{name}"):
            if not metrics:
                ui.empty_state("Not evaluated", "Run <code>python train.py</code>.",
                               "chart")
                continue
            ui.kv_table([
                ("Accuracy", ui.pct(metrics.get("accuracy"))),
                ("Macro F1", ui.fmt(metrics.get("macro_f1"), 4)),
                ("Macro recall", ui.pct(metrics.get("macro_recall"))),
                ("ROC-AUC (one-vs-rest)", ui.fmt(metrics.get("roc_auc_ovr"), 4)),
            ])
            if primary == name:
                _chips_row([_target_chip(targets, "attack_classification", m)
                            for m in ("accuracy", "macro_f1")])
            weakest = metrics.get("weakest_category", {})
            st.caption(
                f"{metrics.get('n_records', 0):,} test records &middot; "
                f"weakest category {weakest.get('name', '?')} "
                f"(F1 {weakest.get('f1', 0):.4f}).")

    st.caption(
        "Chips compare the measured value with the research targets in "
        "<code>config.EVALUATION_TARGETS</code>; they are project goals, not "
        "validated requirements, and are assessed against the classifier "
        "selected as primary during training. Unmet targets are shown as "
        "unmet.", unsafe_allow_html=True,
    )


def _metric_cards(metrics: dict) -> None:
    ui.metric_row([
        ui.metric_card("Accuracy", ui.pct(metrics.get("accuracy")),
                       "Correctly classified records / all test records",
                       "check-circle", ui.ACCENT),
        ui.metric_card("Macro precision", ui.pct(metrics.get("macro_precision")),
                       "Unweighted mean over the five categories",
                       "crosshair", ui.ACCENT_2),
        ui.metric_card("Macro recall", ui.pct(metrics.get("macro_recall")),
                       "Unweighted mean - exposes weak minority classes",
                       "eye", "#22C55E"),
        ui.metric_card("Macro F1", ui.fmt(metrics.get("macro_f1"), 4),
                       "Harmonic mean of macro precision and recall",
                       "chart", "#EAB308"),
        ui.metric_card("Weighted F1", ui.fmt(metrics.get("weighted_f1"), 4),
                       "Support-weighted; dominated by Normal and DoS",
                       "trend", "#F97316"),
    ])


def _confusion_card(cm: dict, title: str, key: str) -> None:
    with ui.card(title, "grid", "Rows = actual, columns = predicted", key=key):
        if not cm:
            ui.empty_state("No confusion matrix stored",
                           "Run `python train.py`.", "grid")
            return
        labels = cm.get("labels", config.ATTACK_CATEGORIES)
        matrix = cm.get("matrix", [])
        mode = st.radio("Display", ["Counts", "Row-normalised"], horizontal=True,
                        key=f"{key}_mode", label_visibility="collapsed")
        if mode == "Row-normalised":
            arr = np.asarray(matrix, dtype=float)
            rows = arr.sum(axis=1, keepdims=True)
            rows[rows == 0] = 1.0
            ui.show(ui.heatmap((arr / rows).round(4), labels, labels,
                               height=400, colorscale="Viridis", text=False),
                    key=f"{key}_chart")
        else:
            ui.show(ui.heatmap(matrix, labels, labels, height=400,
                               colorscale="Blues"), key=f"{key}_chart")


def _live_curves(models: dict, settings: dict, classifier: str) -> None:
    """ROC + PR curves computed on demand from the saved model (real data)."""
    with ui.card("ROC and precision-recall curves", "trend",
                 "Computed live from the saved model on the selected dataset",
                 key=f"{classifier}_curves", elevated=True):
        st.caption(
            "These curves are calculated now from real predictions; they are "
            "not part of the stored training-run evaluation. One-vs-rest per "
            "class."
        )
        class_name = st.selectbox("Class (one-vs-rest)", config.ATTACK_CATEGORIES,
                                  key=f"{classifier}_curve_class")
        if not st.button("Compute curves", icon=":material/show_chart:",
                         key=f"{classifier}_curve_btn"):
            return
        dataset = common.dataset_frame(settings["dataset"])
        if dataset is None or config.LABEL_COLUMN not in dataset.columns:
            st.error("A labelled dataset is required to compute curves.")
            return
        with st.spinner(f"Predicting probabilities for {len(dataset):,} records..."):
            from sklearn.metrics import (auc, precision_recall_curve,
                                         roc_curve)
            pre = models["pre"]
            clf = models[classifier]
            X = pre.transform(dataset)
            proba = clf.model.predict_proba(X)
            y_true = dataset[config.LABEL_COLUMN].astype(str).str.lower() \
                .map(config.ATTACK_CATEGORY_MAP).fillna("Unknown").to_numpy()
            classes = list(clf.classes_)
            idx = classes.index(class_name)
            y_bin = (y_true == class_name).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, proba[:, idx])
            precision, recall, _ = precision_recall_curve(y_bin, proba[:, idx])
            roc_auc = float(auc(fpr, tpr))
            pr_auc = float(auc(recall, precision))
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{class_name} (AUC {roc_auc:.4f})",
                                     line=dict(color=ui.ACCENT, width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Chance", line=dict(color=ui.BORDER,
                                                              width=1,
                                                              dash="dash")))
            fig.update_layout(xaxis_title="False positive rate",
                              yaxis_title="True positive rate",
                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]))
            ui.show(fig, height=340, key=f"{classifier}_roc")
            st.caption(f"ROC-AUC (one-vs-rest, {class_name}): {roc_auc:.4f}")
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                     name=f"{class_name} (AUC {pr_auc:.4f})",
                                     line=dict(color=ui.ACCENT_2, width=2)))
            baseline = float(y_bin.mean())
            fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline],
                                     mode="lines", name=f"Prevalence {baseline:.3f}",
                                     line=dict(color=ui.BORDER, width=1,
                                               dash="dash")))
            fig.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]))
            ui.show(fig, height=340, key=f"{classifier}_pr")
            st.caption(f"Precision-recall AUC ({class_name}): {pr_auc:.4f}. "
                       f"Prevalence of this class in the dataset: {baseline:.4f}.")


def render() -> None:
    settings = common.get_settings()
    evaluation = common.get_evaluation() or {}
    models = common.get_models()
    state = common.get_pipeline_state() or {}

    ui.topbar(
        ["AI-SOC", "Model Performance"],
        "Model Performance",
        "Detection quality for Isolation Forest, Random Forest and XGBoost, "
        "evaluated separately on held-out test data.",
        chips=[
            ui.chip("Evaluation set",
                    state.get("test_file", "KDDTest+.txt"), "database"),
            ui.chip("Evaluated",
                    ui.timestamp_label(evaluation.get("evaluated_at"))[:19], "clock"),
            ui.chip("Seed", str(evaluation.get("random_seed", "n/a")), "lock"),
        ],
        notifications=common.notification_count(common.get_alerts()),
        refresh_key="mp_refresh",
        icon_name="chart",
    )

    if not evaluation:
        ui.empty_state("No evaluation artefact",
                       "Run <code>python train.py</code> to train the models "
                       "and write results/metrics/evaluation.json.", "chart")
        return

    test_dq = evaluation.get("data_quality", {}).get("test", {})
    train_dq = evaluation.get("data_quality", {}).get("train", {})
    with ui.card("Evaluation methodology", "info",
                 "Protocol and data used for every metric on this page",
                 key="mp_method"):
        ui.kv_table([
            ("Training data", f"{state.get('train_file', 'KDDTrain+.txt')} "
                              f"({train_dq.get('n_rows', 0):,} records)"),
            ("Test data", f"{state.get('test_file', 'KDDTest+.txt')} "
                          f"({test_dq.get('n_rows', 0):,} records, held out)"),
            ("Preprocessing", "One-hot encoding + median imputation fitted on "
                              "training data only (no test leakage)"),
            ("Anomaly model", "Isolation Forest trained on normal training "
                              "traffic only; threshold = "
                              f"{evaluation.get('anomaly_detection', {}).get('threshold', 0):.4f}"),
            ("Random seed", str(evaluation.get("random_seed", config.RANDOM_SEED))),
            ("Training-set metrics", "Not recorded by this pipeline - only "
                                     "held-out test metrics are reported"),
        ])
        st.caption(
            "All figures below are TEST-set results unless explicitly marked "
            "'computed live'. Training and testing metrics are never mixed."
        )

    _model_summary(evaluation, state)

    tabs = st.tabs(["Isolation Forest", "Random Forest", "XGBoost",
                    "Comparison", "Targets vs actuals"])

    # ------------------------------------------------------- Isolation Forest
    with tabs[0]:
        anom = evaluation.get("anomaly_detection", {})
        if not anom:
            ui.empty_state("No anomaly evaluation stored", "Run train.py.",
                           "activity")
        else:
            ui.metric_row([
                ui.metric_card("ROC-AUC", ui.fmt(anom.get("roc_auc"), 4),
                               "Score-based separation of attack vs normal",
                               "trend", ui.ACCENT),
                ui.metric_card("Precision", ui.pct(anom.get("precision")),
                               "Of flagged records, share that are attacks",
                               "crosshair", ui.ACCENT_2),
                ui.metric_card("Recall", ui.pct(anom.get("recall")),
                               "Of attack records, share flagged", "eye",
                               "#22C55E"),
                ui.metric_card("False positive rate",
                               ui.pct(anom.get("false_positive_rate")),
                               "Normal records incorrectly flagged", "alert",
                               "#F97316"),
                ui.metric_card("False negative rate",
                               ui.pct(anom.get("false_negative_rate")),
                               "Attacks missed by Stage A", "x-circle",
                               "#EF4444"),
            ])
            st.caption(
                f"Accuracy {ui.pct(anom.get('accuracy'))} on "
                f"{anom.get('n_records', 0):,} test records. Threshold "
                f"{anom.get('threshold', 0):.4f} "
                f"({state.get('anomaly_threshold_method', 'selected on normal training scores')})."
            )
            c1, c2 = st.columns([1, 1.2])
            with c1, ui.card("Anomaly confusion matrix", "grid",
                             "Binary: attack (positive) vs normal (negative)",
                             key="mp_anom_cm"):
                cm = anom.get("confusion_matrix", {})
                if cm:
                    matrix = [[cm.get("tn", 0), cm.get("fp", 0)],
                              [cm.get("fn", 0), cm.get("tp", 0)]]
                    ui.show(ui.heatmap(matrix, ["Normal", "Anomaly"],
                                       ["Normal", "Attack"], height=340,
                                       colorscale="Blues"),
                            key="mp_anom_cm_chart")
                    ui.kv_table([
                        ("True positives", f"{cm.get('tp', 0):,}"),
                        ("False positives", f"{cm.get('fp', 0):,}"),
                        ("False negatives", f"{cm.get('fn', 0):,}"),
                        ("True negatives", f"{cm.get('tn', 0):,}"),
                    ])
            with c2, ui.card("ROC curve", "trend",
                             "Stored curve from the evaluation run",
                             key="mp_anom_roc"):
                curve = anom.get("roc_curve", {})
                if curve.get("fpr") and curve.get("tpr"):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=curve["fpr"], y=curve["tpr"], mode="lines",
                        name=f"Isolation Forest (AUC {anom.get('roc_auc', 0):.4f})",
                        line=dict(color=ui.ACCENT, width=2),
                        hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                             name="Chance",
                                             line=dict(color=ui.BORDER,
                                                       dash="dash")))
                    fig.update_layout(xaxis_title="False positive rate",
                                      yaxis_title="True positive rate")
                    ui.show(fig, height=380, key="mp_anom_roc_chart")
                else:
                    ui.empty_state("No stored ROC curve",
                                   "Run train.py to regenerate.", "trend")

    # -------------------------------------------------- RF / XGBoost tabs
    for i, name in enumerate(("random_forest", "xgboost"), start=1):
        with tabs[i]:
            metrics = evaluation.get("attack_classification", {}).get(name, {})
            if not metrics:
                ui.empty_state(f"No stored evaluation for {CLASSIFIER_LABELS[name]}",
                               "Run `python train.py`.", "chart")
                continue
            st.markdown(f"#### {CLASSIFIER_LABELS[name]} - test-set evaluation")
            _metric_cards(metrics)
            st.caption(
                f"ROC-AUC (one-vs-rest): "
                f"{ui.fmt(metrics.get('roc_auc_ovr'), 4)} &middot; "
                f"{metrics.get('n_records', 0):,} test records &middot; "
                f"five-class classification (Normal, DoS, Probe, R2L, U2R)."
            )
            strongest = metrics.get("strongest_category", {})
            weakest = metrics.get("weakest_category", {})
            ui.note(
                f"<b>Strongest:</b> {strongest.get('name', '?')} "
                f"(F1 {strongest.get('f1', 0):.4f}, recall "
                f"{ui.pct(strongest.get('recall'))}). "
                f"<b>Weakest:</b> {weakest.get('name', '?')} "
                f"(F1 {weakest.get('f1', 0):.4f}, recall "
                f"{ui.pct(weakest.get('recall'))}). Weak results are reported, "
                "not hidden.",
                "alert", "#EAB308",
            )
            with ui.card("Per-class performance", "list",
                         "Precision, recall and F1 for each category",
                         key=f"mp_{name}_perclass"):
                per_class = metrics.get("per_class", {})
                rows = [{
                    "category": cat,
                    "precision": per_class.get(cat, {}).get("precision"),
                    "recall": per_class.get(cat, {}).get("recall"),
                    "f1": per_class.get(cat, {}).get("f1"),
                    "support": per_class.get(cat, {}).get("support", 0),
                } for cat in config.ATTACK_CATEGORIES]
                table = pd.DataFrame(rows)
                ui.show_table(table, height=230, key=f"mp_{name}_table")
                ui.show(ui.grouped_bar(
                    config.ATTACK_CATEGORIES,
                    {"Precision": [r["precision"] or 0 for r in rows],
                     "Recall": [r["recall"] or 0 for r in rows],
                     "F1": [r["f1"] or 0 for r in rows]},
                    height=330, y_range=(0, 1.05),
                    colors=[ui.ACCENT, ui.ACCENT_2, "#22C55E"]),
                    key=f"mp_{name}_bars")
            _confusion_card(metrics.get("confusion_matrix", {}),
                            "Confusion matrix", f"mp_{name}_cm")
            if common.models_available(models):
                _live_curves(models, settings, name)

    # ------------------------------------------------------- Comparison
    with tabs[3]:
        rf = evaluation.get("attack_classification", {}).get("random_forest", {})
        xgb = evaluation.get("attack_classification", {}).get("xgboost", {})
        if rf and xgb:
            with ui.card("Head-to-head", "chart",
                         "Stored test-set metrics for both classifiers",
                         key="mp_compare"):
                keys = ["accuracy", "macro_precision", "macro_recall",
                        "macro_f1", "weighted_f1", "roc_auc_ovr"]
                labels = ["Accuracy", "Macro precision", "Macro recall",
                          "Macro F1", "Weighted F1", "ROC-AUC (OvR)"]
                ui.show(ui.grouped_bar(
                    labels,
                    {"Random Forest": [float(rf.get(k, 0)) for k in keys],
                     "XGBoost": [float(xgb.get(k, 0)) for k in keys]},
                    height=360, y_range=(0, 1.05),
                    colors=[ui.ACCENT, ui.ACCENT_2]),
                    key="mp_compare_chart")
                comp = pd.DataFrame({
                    "metric": labels,
                    "random_forest": [rf.get(k) for k in keys],
                    "xgboost": [xgb.get(k) for k in keys],
                    "difference (XGB - RF)": [
                        round(float(xgb.get(k, 0)) - float(rf.get(k, 0)), 4)
                        for k in keys],
                })
                ui.show_table(comp, height=280, key="mp_compare_table")
            primary = state.get("primary_classifier")
            if primary:
                ui.note(
                    f"<b>Primary classifier: {primary}.</b> Chosen by macro F1 "
                    "on the test set during the training run and used for "
                    "global SHAP importance and the stored alert set.",
                    "info", ui.ACCENT,
                )
            st.caption(
                "Weighted F1 favours the majority classes (Normal, DoS); macro "
                "F1 treats R2L and U2R equally and is therefore much lower. "
                "Both are shown so the trade-off is visible."
            )
        else:
            ui.empty_state("Comparison unavailable",
                           "Both classifiers need a stored evaluation.", "chart")

    # ------------------------------------------------- Targets vs actuals
    with tabs[4]:
        targets = evaluation.get("targets_vs_actuals", {})
        with ui.card("Research targets vs achieved results", "target",
                     "Targets are project goals, not validated requirements",
                     key="mp_targets"):
            rows = []
            for group, metrics in targets.items():
                for metric, values in metrics.items():
                    rows.append({
                        "area": group.replace("_", " "),
                        "metric": metric,
                        "target": values.get("target"),
                        "actual": values.get("actual"),
                        "met": "Yes" if values.get("target_met") else "No",
                    })
            if rows:
                ui.show_table(pd.DataFrame(rows), height=260, key="mp_target_table")
                st.caption(
                    "A target is marked met only when the measured value "
                    "reaches it. Unmet targets are shown as unmet - the "
                    "dashboard does not claim success."
                )
            else:
                ui.empty_state("No targets recorded",
                               "See config.EVALUATION_TARGETS.", "target")
