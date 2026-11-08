"""Page 10 - System Settings.

Analyst-editable runtime configuration, validated before it is written to
data/processed/app_settings.json. Also shows model / dataset information,
software versions and the application activity log. Only project-relative
paths are displayed - no absolute paths, credentials or internal secrets.
"""
from __future__ import annotations

import importlib.metadata
import platform

import pandas as pd
import streamlit as st

import config
from src import settings as settings_module
from src.auth import rbac
from dashboard import auth_ui, common
from dashboard import components as ui

PACKAGES = ["streamlit", "pandas", "numpy", "scikit-learn", "xgboost", "shap",
            "plotly", "joblib", "psutil", "reportlab", "pytest"]


def _versions() -> pd.DataFrame:
    rows = [{"component": "python", "version": platform.python_version()},
            {"component": "platform", "version": platform.platform()}]
    for name in PACKAGES:
        try:
            rows.append({"component": name,
                         "version": importlib.metadata.version(name)})
        except importlib.metadata.PackageNotFoundError:
            rows.append({"component": name, "version": "not installed"})
    return pd.DataFrame(rows)


def render() -> None:
    actor = auth_ui.current_user()
    if actor is None or not rbac.can(actor["role"], "manage_settings"):
        ui.empty_state(
            "Access restricted",
            "System Settings requires the <b>Administrator</b> role. "
            "Permissions are enforced on the server, not only in the "
            "navigation.", "lock")
        return
    settings = common.get_settings()
    models = common.get_models()
    state = common.get_pipeline_state() or {}
    evaluation = common.get_evaluation() or {}

    ui.topbar(
        ["AI-SOC", "System Settings"],
        "System Settings",
        "Configure detection, triage and explainability behaviour; inspect "
        "models, datasets and software versions.",
        chips=[
            ui.chip("Saved", ui.timestamp_label(settings.get("saved_at"))[:19]
                    if settings.get("saved_at") else "using research defaults",
                    "clock"),
            ui.chip("App version", config.APP_VERSION, "shield"),
            ui.chip("Seed", str(config.RANDOM_SEED), "lock"),
        ],
        notifications=common.notification_count(common.get_alerts()),
        refresh_key="st_refresh",
        icon_name="settings",
    )

    drift = settings_module.settings_diff(settings)
    if drift:
        ui.note("Active configuration differs from the research defaults: "
                + "; ".join(drift), "info", "#EAB308")
    else:
        ui.note("All options are at the documented research defaults from "
                "<code>config.py</code>.", "check-circle", "#22C55E")

    # ------------------------------------------------------ configuration form
    with ui.card("Configuration", "settings",
                 "Validated before saving; changes apply to new analyses",
                 key="st_form"):
        with st.form("settings_form"):
            st.markdown("**Dataset and models**")
            c1, c2 = st.columns(2)
            with c1:
                dataset = st.selectbox(
                    "Active dataset", list(settings_module.DATASETS),
                    index=list(settings_module.DATASETS).index(settings["dataset"]),
                    key="st_dataset",
                    help="Used for live analyses, per-alert explanations and "
                         "alert regeneration.")
            with c2:
                classifier = st.selectbox(
                    "Attack classifier (Stage B)", settings_module.CLASSIFIERS,
                    index=settings_module.CLASSIFIERS.index(settings["classifier"]),
                    key="st_classifier")

            model_threshold = (float(models["detector"].threshold)
                               if models.get("detector") is not None
                               else config.ANOMALY_THRESHOLD)
            use_default = st.toggle(
                f"Use the model's saved anomaly threshold ({model_threshold:.4f})",
                value=settings["anomaly_threshold"] is None, key="st_thr_default",
                help=f"The saved threshold is the "
                     f"{config.ANOMALY_NORMAL_FPR_QUANTILE:.0%} quantile of "
                     f"normal training scores (nominal "
                     f"{config.ANOMALY_NORMAL_FPR_QUANTILE:.0%} "
                     "false-positive rate on normal traffic).")
            threshold = st.number_input(
                "Anomaly threshold override", min_value=-5.0, max_value=5.0,
                value=float(settings["anomaly_threshold"]
                            if settings["anomaly_threshold"] is not None
                            else model_threshold),
                step=0.005, format="%.4f", key="st_threshold",
                disabled=use_default,
                help="Records with score <= threshold are flagged anomalous. "
                     "Lower = stricter.")

            st.markdown("**Severity mapping** (research rules, not a validated "
                        "SOC risk framework)")
            sev_cols = st.columns(len(config.ATTACK_CATEGORIES))
            severity_map = {}
            for col, cat in zip(sev_cols, config.ATTACK_CATEGORIES):
                with col:
                    severity_map[cat] = st.selectbox(
                        cat, config.SEVERITY_ORDER,
                        index=config.SEVERITY_ORDER.index(
                            settings["severity_map"].get(cat,
                                                         config.SEVERITY_MAP[cat])),
                        key=f"st_sev_{cat}")

            st.markdown("**Risk scoring weights** (must sum to 1.0)")
            w_cols = st.columns(3)
            weights = {}
            for col, key, label in zip(
                    w_cols, ("severity", "confidence", "anomaly"),
                    ("Severity weight", "Confidence weight", "Anomaly weight")):
                with col:
                    weights[key] = st.number_input(
                        label, min_value=0.0, max_value=1.0,
                        value=float(settings["risk_weights"][key]), step=0.05,
                        format="%.3f", key=f"st_w_{key}")
            total = sum(weights.values())
            st.caption(f"Current sum: {total:.3f} "
                       f"({'valid' if abs(total - 1.0) <= 1e-6 else 'INVALID - must be 1.000'})")
            st.caption(
                f"risk = 100 x (severity weight x category severity + confidence "
                f"weight x model confidence + anomaly weight x "
                f"clip((threshold - score)/{config.ANOMALY_SCORE_SPREAD}, 0, 1))"
            )

            st.markdown("**Explainability**")
            x1, x2, x3 = st.columns(3)
            with x1:
                top_k = st.number_input(
                    "Top-K features in local explanations", min_value=1,
                    max_value=25, value=int(settings["shap_top_k"]), step=1,
                    key="st_topk")
            with x2:
                sample = st.number_input(
                    "Global SHAP sample size", min_value=50, max_value=5000,
                    value=int(settings["shap_global_sample_size"]), step=50,
                    key="st_sample")
            with x3:
                gen_local = st.toggle(
                    "Allow local explanation generation",
                    value=bool(settings["generate_local_explanations"]),
                    key="st_gen_local")

            st.markdown("**Application preferences**")
            p1, p2 = st.columns(2)
            with p1:
                show_truth = st.toggle(
                    "Show ground-truth labels alongside predictions",
                    value=bool(settings["show_ground_truth"]),
                    key="st_truth",
                    help="Ground truth is always labelled as a dataset label, "
                         "never as a model output.")
            with p2:
                page_size = st.number_input(
                    "Table page size", min_value=10, max_value=500,
                    value=int(settings["table_page_size"]), step=5,
                    key="st_page_size")

            submitted = st.form_submit_button(
                "Save configuration", type="primary", icon=":material/save:")
            if submitted:
                payload = {
                    "dataset": dataset, "classifier": classifier,
                    "anomaly_threshold": None if use_default else float(threshold),
                    "severity_map": severity_map, "risk_weights": weights,
                    "shap_top_k": int(top_k),
                    "shap_global_sample_size": int(sample),
                    "generate_local_explanations": bool(gen_local),
                    "show_ground_truth": bool(show_truth),
                    "table_page_size": int(page_size),
                }
                clean, errors = common.save_settings(payload)
                if errors:
                    for err in errors:
                        st.error(err)
                    st.warning("Nothing was saved. Fix the highlighted values "
                               "and submit again.")
                else:
                    st.success("Configuration saved to "
                               "data/processed/app_settings.json. New analyses "
                               "will use these values; stored alerts and stored "
                               "evaluation metrics are unchanged.")
                    common.invalidate_analysis()

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Reset to research defaults", icon=":material/restart_alt:",
                     key="st_reset", width="stretch"):
            st.session_state["st_reset_confirm"] = True
    with c2:
        if st.session_state.get("st_reset_confirm"):
            st.warning("This deletes the saved configuration and restores the "
                       "documented defaults from config.py.")
            if st.button("Confirm reset", type="primary", key="st_reset_yes"):
                settings_module.reset()
                common.invalidate_analysis()
                st.session_state.pop("st_reset_confirm", None)
                common.save_settings(settings_module.defaults())
                st.success("Settings reset to research defaults.")
                st.rerun()
            if st.button("Cancel", key="st_reset_no"):
                st.session_state.pop("st_reset_confirm", None)
                st.rerun()

    # ------------------------------------------------------ model information
    with ui.card("Model information", "cpu",
                 "From models/pipeline_state.json (written by train.py)",
                 key="st_models"):
        if not state:
            ui.empty_state("No pipeline state",
                           "Run `python train.py` to create model artefacts.",
                           "cpu")
        else:
            ui.kv_table([
                ("Trained at", ui.timestamp_label(state.get("trained_at"))),
                ("Dataset", state.get("dataset", config.DATASET_NAME)),
                ("Train / test records",
                 f"{state.get('n_train_records', 0):,} / "
                 f"{state.get('n_test_records', 0):,}"),
                ("Normal training records",
                 f"{state.get('n_normal_train_records', 0):,}"),
                ("Encoded features", str(state.get("n_encoded_features", "n/a"))),
                ("Anomaly threshold", ui.fmt(state.get("anomaly_threshold"), 6)),
                ("Threshold method", state.get("anomaly_threshold_method",
                                               "not recorded")),
                ("Primary classifier", state.get("primary_classifier", "n/a")),
            ])
            times = state.get("train_times_seconds", {})
            if times:
                st.markdown("**Training times (measured)**")
                ui.show_table(pd.DataFrame(
                    [{"model": k, "seconds": v} for k, v in times.items()]),
                    height=150, key="st_train_times")
            missing = [n for n, p in config.MODEL_FILES.items() if not p.exists()]
            st.caption(
                "Artefacts: " + (", ".join(f"{n} present" for n in config.MODEL_FILES)
                                 if not missing else
                                 f"MISSING: {', '.join(missing)}")
                + ". Paths are project-relative (models/).")

    # ---------------------------------------------------- dataset information
    with ui.card("Dataset information", "database",
                 "NSL-KDD benchmark files in data/raw/", key="st_dataset_info"):
        rows = []
        dq = evaluation.get("data_quality", {})
        for name, path in settings_module.DATASETS.items():
            key = "train" if "Train" in name else "test"
            info = dq.get(key, {})
            rows.append({
                "file": path.name,
                "present": "yes" if path.exists() else "NO",
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2)
                if path.exists() else None,
                "records": info.get("n_rows"),
                "malformed_rows": info.get("malformed_rows"),
                "duplicate_rows": info.get("duplicate_rows"),
            })
        ui.show_table(pd.DataFrame(rows), height=140, key="st_dataset_table")
        st.caption(f"Source: {config.DATASET_SOURCE}")
        dist = dq.get("test", {}).get("label_distribution", {})
        if dist:
            st.markdown("**Test-set category distribution (ground truth)**")
            ui.show(ui.vbar(list(dist.keys()), list(dist.values()),
                            colors=[ui.CATEGORY_COLORS.get(k, ui.ACCENT)
                                    for k in dist], height=250),
                    key="st_dist_chart")

    # ------------------------------------------------------ versions & config
    c3, c4 = st.columns([1, 1])
    with c3, ui.card("Software versions", "layers",
                     "Installed package versions in this environment",
                     key="st_versions"):
        ui.show_table(_versions(), height=420, key="st_versions_table")
    with c4, ui.card("System configuration", "lock",
                     "Fixed research parameters from config.py", key="st_config"):
        ui.kv_table([
            ("Application version", config.APP_VERSION),
            ("Random seed", str(config.RANDOM_SEED)),
            ("Max upload size", f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB"),
            ("Accepted upload types",
             ", ".join(sorted(config.ALLOWED_UPLOAD_EXTENSIONS))),
            ("Anomaly default threshold", str(config.ANOMALY_THRESHOLD)),
            ("Anomaly score spread", str(config.ANOMALY_SCORE_SPREAD)),
            ("Severity weights", ", ".join(
                f"{k}={v}" for k, v in config.SEVERITY_WEIGHTS.items())),
            ("Isolation Forest trees",
             str(config.ISOLATION_FOREST_PARAMS["n_estimators"])),
            ("Random Forest trees",
             str(config.RANDOM_FOREST_PARAMS["n_estimators"])),
            ("XGBoost estimators / depth",
             f"{config.XGBOOST_PARAMS['n_estimators']} / "
             f"{config.XGBOOST_PARAMS['max_depth']}"),
            ("Upload handling", "parsed as data only, never executed"),
        ])

    # ------------------------------------------------------------------ logs
    with ui.card("Application logs", "list",
                 "Recent dashboard activity (no sensitive data recorded)",
                 key="st_logs"):
        lines = settings_module.read_log(60)
        if lines:
            st.code("\n".join(lines), language="text")
            st.download_button(
                "Download log", "\n".join(settings_module.read_log(500)).encode(),
                file_name="ai_soc_activity.log", mime="text/plain",
                icon=":material/download:", key="st_log_dl")
        else:
            ui.empty_state("No log entries yet",
                           "Actions such as saving settings, updating alert "
                           "statuses and generating reports are logged here.",
                           "list")
