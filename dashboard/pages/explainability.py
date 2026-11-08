"""Page 6 - Explainable AI & Model Transparency.

Two strictly separated capabilities, kept in two separate tabs so they can
never be confused:

* GLOBAL: mean |SHAP| over a sample of records (dataset-level importance).
  Stored artefact from the training run, or recomputed on demand.
* LOCAL: real per-record SHAP values for one selected alert / record and the
  class the model actually predicted, with base value, reconstructed output,
  ranked contributions, a waterfall chart and plain-language text derived
  from those values.

Nothing is approximated: if SHAP fails, the page says so instead of showing
placeholder numbers.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st

import config
from src.explainability import Explainer, GLOBAL_SHAP_FILE, LocalExplanation
from dashboard import common
from dashboard import components as ui

RISK_LEGEND = (
    '<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;'
    f'font-size:.78rem;color:{ui.TEXT_MUTED};margin:2px 0 8px 0">'
    '<span style="display:inline-flex;align-items:center;gap:6px">'
    '<span class="soc-dot" style="background:#22C55E"></span>'
    'pushes toward the predicted class (higher risk for that class)</span>'
    '<span style="display:inline-flex;align-items:center;gap:6px">'
    '<span class="soc-dot" style="background:#EF4444"></span>'
    'pushes against it (lower risk for that class)</span></div>'
)


def _encode_record(models: dict, settings: dict, idx: int):
    dataset = common.dataset_frame(settings["dataset"])
    if dataset is None:
        raise ValueError(f"{settings['dataset']} is not available in data/raw/.")
    if not 0 <= idx < len(dataset):
        raise ValueError(
            f"Record index {idx} is outside {settings['dataset']} "
            f"({len(dataset):,} rows)."
        )
    pre = models["pre"]
    X = pre.transform(dataset.iloc[[idx]])
    return X[0], dataset.iloc[idx], pre.feature_names_


def render() -> None:
    settings = common.get_settings()
    models = common.get_models()
    classifier = settings["classifier"]

    ui.topbar(
        ["AI-SOC", "Explainable AI"],
        "Explainable AI & Model Transparency",
        "Understand why the machine learning system generated each security "
        "prediction.",
        chips=[
            ui.chip("Model", classifier, "cpu"),
            ui.chip("Explainer", "SHAP TreeExplainer", "sparkles"),
            ui.chip("Dataset", settings["dataset"], "database"),
        ],
        notifications=common.notification_count(common.get_alerts()),
        refresh_key="xai_refresh",
        icon_name="brain",
    )

    if not common.models_available(models):
        common.render_missing_models_notice(models)
        return
    if models.get(classifier) is None:
        st.error(f"The selected classifier '{classifier}' is not available.")
        return

    tab_global, tab_local = st.tabs(
        ["Global feature importance", "Individual alert explanation"])
    with tab_global:
        _render_global(settings, models, classifier)
    with tab_local:
        _render_local_section(settings, models, classifier)


# ===========================================================================
# A. GLOBAL explainability (model-level)
# ===========================================================================
def _render_global(settings: dict, models: dict, classifier: str) -> None:
    ui.note(
        "<b>Model-level, not per-alert.</b> Global importance is the mean "
        "absolute SHAP value per feature across a sample of records. It "
        "describes the model in general and must not be read as the reason "
        "for any individual alert.",
        "info", ui.ACCENT_2,
    )

    cache = common.load_json(GLOBAL_SHAP_FILE)
    col_a, col_b = st.columns([1.35, 1])

    with col_a, ui.card("Global SHAP feature importance", "chart",
                        "Mean |SHAP| averaged over classes and sampled records",
                        key="xai_global"):
        top_n = st.slider("Features shown", 5, 30, 15, key="xai_global_n")
        importance = None
        meta_line = ""
        if cache is not None:
            try:
                df = pd.read_json(GLOBAL_SHAP_FILE)
                model_type = str(df.iloc[0].get("model_type", ""))
                compute_seconds = float(df.iloc[0].get("compute_seconds", np.nan))
                importance = df.drop(columns=[c for c in
                                              ("model_type", "compute_seconds")
                                              if c in df.columns])
                meta_line = (f"Cached artefact from the training run &middot; "
                             f"model <b>{model_type}</b> &middot; computed in "
                             f"{compute_seconds:.2f} s")
                if model_type != classifier:
                    st.warning(
                        f"The cached importance belongs to '{model_type}', but "
                        f"'{classifier}' is selected. Recompute below for the "
                        "selected model."
                    )
            except Exception as exc:                            # noqa: BLE001
                st.warning(f"Could not read the cached importance: {exc}")
        if importance is not None and len(importance):
            head = importance.head(top_n)
            ui.show(ui.hbar(head["feature"].tolist()[::-1],
                            head["mean_abs_shap"].round(4).tolist()[::-1],
                            colors=[ui.ACCENT] * len(head), height=440,
                            text_auto=False),
                    key="xai_global_chart")
            st.caption(meta_line, unsafe_allow_html=True)
        else:
            ui.empty_state("No global importance stored",
                           "Run `python train.py`, or recompute below.",
                           "chart")

    with col_b, ui.card("Recompute / summary plot", "refresh",
                        "Real SHAP computation on a random sample",
                        key="xai_recompute"):
        sample_size = st.number_input(
            "Sample size (records)", min_value=50, max_value=2000,
            value=int(settings["shap_global_sample_size"]), step=50,
            key="xai_sample",
            help="Larger samples give more stable global importance and cost "
                 "more time. The measurement is reported after the run.",
        )
        if st.button("Recompute global importance",
                     icon=":material/autorenew:", key="xai_recompute_btn",
                     width="stretch"):
            dataset = common.dataset_frame(settings["dataset"])
            if dataset is None:
                st.error(f"{settings['dataset']} is not available.")
            else:
                with st.spinner("Computing SHAP values (this runs the real "
                                "explainer, not a cached estimate)..."):
                    start = time.perf_counter()
                    pre = models["pre"]
                    rng = np.random.default_rng(config.RANDOM_SEED)
                    rows = rng.choice(len(dataset),
                                      size=min(int(sample_size), len(dataset)),
                                      replace=False)
                    X = pre.transform(dataset.iloc[rows])
                    explainer = Explainer(models[classifier], pre.feature_names_)
                    importance, seconds = explainer.global_importance(
                        X, use_cache=False)
                    elapsed = time.perf_counter() - start
                st.success(
                    f"Computed mean |SHAP| for {len(rows)} records with "
                    f"{classifier} in {seconds:.2f} s (total {elapsed:.2f} s). "
                    "The cached artefact has been updated."
                )
                st.dataframe(importance.head(15), width="stretch",
                             hide_index=True, height=320, key="xai_new_importance")
                st.session_state["xai_summary_sample"] = rows
        if st.button("Generate SHAP summary plot",
                     icon=":material/scatter_plot:", key="xai_summary_btn",
                     width="stretch"):
            dataset = common.dataset_frame(settings["dataset"])
            if dataset is None:
                st.error(f"{settings['dataset']} is not available.")
            else:
                with st.spinner("Computing SHAP values for the summary plot..."):
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        import shap
                        pre = models["pre"]
                        rows = st.session_state.get("xai_summary_sample")
                        if rows is None:
                            rng = np.random.default_rng(config.RANDOM_SEED)
                            rows = rng.choice(
                                len(dataset),
                                size=min(int(sample_size), len(dataset)),
                                replace=False)
                        X = pre.transform(dataset.iloc[rows])
                        explainer = Explainer(models[classifier], pre.feature_names_)
                        start = time.perf_counter()
                        values = explainer.explainer.shap_values(X)
                        seconds = time.perf_counter() - start
                        if isinstance(values, list):
                            values = np.stack(values, axis=2)
                            # Multiclass: show the class with the largest
                            # overall mean |SHAP| (explicitly labelled).
                            class_idx = int(np.abs(values).mean(axis=(0, 1)).argmax())
                            values = values[:, :, class_idx]
                            class_label = models[classifier].classes_[class_idx]
                        else:
                            class_label = None
                        fig = plt.figure(figsize=(7, 6))
                        shap.summary_plot(
                            values, X, feature_names=pre.feature_names_,
                            max_display=15, show=False,
                            plot_size=(7, 6),
                        )
                        st.session_state["xai_summary_fig"] = (
                            plt.gcf(), seconds, len(rows), class_label)
                    except Exception as exc:                    # noqa: BLE001
                        st.error(f"Summary plot failed: {exc}")
        if "xai_summary_fig" in st.session_state:
            fig, seconds, n, class_label = st.session_state["xai_summary_fig"]
            st.pyplot(fig, width="stretch")
            st.caption(
                f"SHAP summary (beeswarm) for {n:,} sampled records, computed "
                f"in {seconds:.2f} s"
                + (f" &middot; class '{class_label}'" if class_label else "")
                + ". Global model behaviour, not a per-alert explanation."
            )

# ===========================================================================
# B. LOCAL per-alert explanation
# ===========================================================================
def _render_local_section(settings: dict, models: dict,
                          classifier: str) -> None:
    ui.note(
        "<b>Local SHAP.</b> Select one alert (or one record) and generate its "
        "own explanation: the contributions below are computed for that "
        "specific record and the class the model predicted for it.",
        "sparkles", ui.ACCENT,
    )

    source = st.radio("Explain", ["An alert from the store", "A record by index"],
                      horizontal=True, key="xai_source",
                      label_visibility="collapsed")

    alert_id = None
    record_idx = int(st.session_state.pop("focus_record", 0) or 0)
    alerts = common.get_alerts()

    if source == "An alert from the store":
        if alerts.empty:
            ui.empty_state("No alerts stored",
                           "Generate alerts on the Network Analysis page, or "
                           "switch to 'A record by index'.", "bell")
            return
        pool = alerts.sort_values("risk_score", ascending=False).head(500)
        options = [f"{r.alert_id}  |  {r.predicted_category}  |  "
                   f"{r.severity}  |  record {r.record_index}"
                   for r in pool.itertuples()]
        picked = st.selectbox(
            f"Alert (showing the {len(options)} highest-risk stored alerts)",
            options, key="xai_alert_pick")
        alert_id = picked.split("  |")[0]
        row = pool[pool["alert_id"] == alert_id].iloc[0]
        record_idx = int(row["record_index"])
        st.caption(f"Alert {alert_id} &middot; predicted category "
                   f"{row['predicted_category']} &middot; severity "
                   f"{row['severity']} &middot; risk {row['risk_score']:.2f}")
    else:
        dataset_len = len(common.dataset_frame(settings["dataset"]) or [])
        record_idx = st.number_input(
            "Record index", min_value=0, max_value=max(dataset_len - 1, 0),
            value=min(record_idx, max(dataset_len - 1, 0)), step=1,
            key="xai_record_idx")

    cache = st.session_state.setdefault("shap_cache", {})
    cache_key = (alert_id or f"record-{record_idx}", classifier, settings["dataset"])
    generate = st.button("Show explanation", type="primary",
                         icon=":material/psychology:", key="xai_generate")
    if not (generate or cache_key in cache):
        ui.empty_state(
            "No explanation generated yet",
            "Press <b>Show explanation</b> to compute real SHAP values for the "
            "selected alert or record. Nothing is displayed until they exist.",
            "sparkles",
        )
        return
    explanation = cache.get(cache_key)
    if explanation is None:
        with st.spinner("Computing local SHAP values for this record..."):
            try:
                start = time.perf_counter()
                x, _, feature_names = _encode_record(models, settings,
                                                     record_idx)
                explainer = Explainer(models[classifier], feature_names)
                explanation = explainer.explain_record(x)
                cache[cache_key] = explanation
                st.session_state["xai_last_time"] = (
                    time.perf_counter() - start, record_idx)
            except Exception as exc:                            # noqa: BLE001
                st.error(f"Local explanation failed: {exc}")
                st.caption(
                    "No values are displayed rather than fabricated ones. "
                    "Check that the record index exists in the selected "
                    "dataset and that the model artefacts are present."
                )
                return
    if not isinstance(explanation, LocalExplanation):
        st.error("The cached explanation is not valid; regenerate it.")
        return
    _render_local(explanation, settings, models, classifier, record_idx,
                  alert_id)


def _render_local(explanation: LocalExplanation, settings: dict, models: dict,
                  classifier: str, record_idx: int, alert_id: str | None) -> None:
    contributions = explanation.contributions
    total = float(sum(c[2] for c in contributions))
    reconstructed = explanation.base_value + total
    clf = models[classifier]
    try:
        dataset = common.dataset_frame(settings["dataset"])
        x = models["pre"].transform(dataset.iloc[[record_idx]])
        proba = clf.model.predict_proba(x)[0]
        class_idx = list(clf.classes_).index(explanation.predicted_class)
        actual_proba = float(proba[class_idx])
    except Exception:                                          # noqa: BLE001
        actual_proba = None
        class_idx = None

    top_k = int(settings["shap_top_k"])
    with ui.card("Prediction summary", "crosshair",
                 "What the model actually output for this record",
                 key="xai_summary"):
        ui.metric_row([
            ui.metric_card(
                "Predicted class", explanation.predicted_class,
                f"Explained class index: "
                f"{class_idx if class_idx is not None else 'n/a'}"
                + (f" &middot; probability {actual_proba:.1%}"
                   if actual_proba is not None else ""),
                "crosshair",
                ui.CATEGORY_COLORS.get(explanation.predicted_class, ui.ACCENT)),
            ui.metric_card(
                "Model used", explanation.model_type or classifier,
                f"SHAP TreeExplainer &middot; {len(contributions)} encoded "
                "features", "cpu", ui.ACCENT_2),
            ui.metric_card("Baseline output", f"{explanation.base_value:.4f}",
                           "Expected model output before this record's features",
                           "layers", "#94A3B8"),
            ui.metric_card("Final output", f"{reconstructed:.4f}",
                           f"base + sum of contributions ({total:+.4f}), in "
                           "explainer units", "trend", ui.ACCENT),
            ui.metric_card("Explanation time",
                           f"{explanation.explanation_time_ms:.1f} ms",
                           "Measured for this record (real SHAP computation)",
                           "clock", "#22C55E"),
        ])

    st.markdown(RISK_LEGEND, unsafe_allow_html=True)
    left, right = st.columns([1.25, 1])
    with left, ui.card("SHAP waterfall", "chart",
                       f"Base value to reconstructed output (top {top_k})",
                       key="xai_waterfall"):
        ui.show(ui.shap_waterfall(explanation, top_k), key="xai_wf_chart")
        st.caption(
            "The final bar is the reconstructed model output (base value plus "
            "all contributions), in explainer units for the predicted class."
        )

    with right, ui.card("Key contributing features", "list",
                        "Signed SHAP values for this record only",
                        key="xai_ranked"):
        ui.show(ui.shap_bars(explanation, top_k), key="xai_ranked_chart")
        ordered = sorted(contributions, key=lambda c: abs(c[2]),
                         reverse=True)[:top_k]
        df = pd.DataFrame(
            [(f, float(v), float(s)) for f, v, s in ordered],
            columns=["feature", "feature value", "SHAP contribution"],
        )
        ui.show_table(df, height=280, key="xai_ranked_table")

    pos = [c for c in contributions if c[2] > 0]
    neg = [c for c in contributions if c[2] < 0]
    with ui.card("Plain-language explanation", "note",
                 "Generated from the actual SHAP values above",
                 key="xai_language"):
        ui.plain_language(explanation.to_plain_language(top_k))
        st.caption(
            f"{len(pos)} positive and {len(neg)} negative contributions out of "
            f"{len(contributions)} encoded features."
            + (f" The model's probability for '{explanation.predicted_class}' "
               f"is {actual_proba:.1%}." if actual_proba is not None else "")
        )
        ui.note(
            "SHAP values explain the classifier's decision for this record. "
            "They are not a root-cause analysis and do not confirm that an "
            "attack occurred.",
            "info", ui.ACCENT_2,
        )
        if alert_id:
            st.caption(f"Explanation for alert {alert_id}, record {record_idx:,} "
                       f"of {settings['dataset']}.")
        else:
            st.caption(f"Explanation for record {record_idx:,} of "
                       f"{settings['dataset']}.")
        export = {
            "record_index": record_idx,
            "alert_id": alert_id,
            "model": explanation.model_type,
            "predicted_class": explanation.predicted_class,
            "base_value": explanation.base_value,
            "reconstructed_output": reconstructed,
            "explanation_time_ms": explanation.explanation_time_ms,
            "contributions": [
                {"feature": f, "value": v, "shap": s} for f, v, s in contributions
            ],
        }
        st.download_button(
            "Download explanation (JSON)",
            json.dumps(export, indent=2).encode("utf-8"),
            file_name=f"shap_explanation_{alert_id or record_idx}.json",
            mime="application/json", icon=":material/download:",
            key="xai_download",
        )
