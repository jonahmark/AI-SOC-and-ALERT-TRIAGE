"""Page 8 - Resource Monitoring.

Shows only real measurements: the artefacts written by `train.py` and fresh
readings taken from this machine on demand. Anything not measured is labelled
"not measured". The measurement environment and its limitations are stated
explicitly - running on a laptop is not evidence of SOC-grade efficiency.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.resource_monitor import ResourceMonitor
from dashboard import common
from dashboard import components as ui


def _sample_now() -> dict:
    snap = common.live_system_snapshot()
    history = st.session_state.setdefault("resource_history", [])
    history.append({
        "time": snap["sampled_at"][11:19],
        "cpu_percent": snap["cpu_percent"],
        "system_memory_percent": snap["system_memory_percent"],
        "process_memory_mb": snap["process_memory_mb"],
    })
    st.session_state["resource_history"] = history[-60:]
    return snap


def _live_measurements(models: dict, settings: dict) -> None:
    with ui.card("Measure now", "zap",
                 "Fresh, real measurements on this machine (single run unless "
                 "stated)", key="rm_live", elevated=True):
        sample_size = st.number_input(
            "Sample size (records)", min_value=100, max_value=5000, value=1000,
            step=100, key="rm_sample",
            help="Used for inference latency and the SHAP overhead comparison.")
        c1, c2 = st.columns(2)
        with c1:
            run_latency = st.button("Measure inference + SHAP overhead",
                                    icon=":material/speed:", key="rm_latency_btn",
                                    width="stretch")
        with c2:
            run_load = st.button("Measure model load time",
                                 icon=":material/download:", key="rm_load_btn",
                                 width="stretch")
        if run_load:
            from src.attack_classification import AttackClassifier
            monitor = ResourceMonitor()
            with monitor.timed("load_xgboost_from_disk"):
                AttackClassifier.load("xgboost")
            rec = monitor.records["load_xgboost_from_disk"]
            st.session_state["rm_load"] = rec
        if "rm_load" in st.session_state:
            rec = st.session_state["rm_load"]
            ui.kv_table([
                ("XGBoost model load time", f"{rec['wall_time_seconds']:.3f} s"),
                ("Process memory delta", f"{rec['memory_delta_mb']:+.1f} MB"),
                ("Process CPU during load", f"{rec['cpu_percent_during']:.1f}%"),
            ])
        if run_latency:
            dataset = common.dataset_frame(settings["dataset"])
            if dataset is None:
                st.error(f"{settings['dataset']} is not available.")
                return
            n = min(int(sample_size), len(dataset))
            monitor = ResourceMonitor()
            pre = models["pre"]
            clf = models[settings["classifier"]]
            with st.spinner(f"Measuring on {n:,} records..."):
                with monitor.timed("encode_sample"):
                    X = pre.transform(dataset.head(n))
                latency = monitor.measure_prediction_latency(
                    lambda x: clf.model.predict_proba(x), X, n_repeats=3,
                    key="live_inference")
                from src.explainability import Explainer
                with monitor.timed("live_shap_same_sample"):
                    explainer = Explainer(clf, pre.feature_names_)
                    explainer.explainer.shap_values(X)
                shap_rec = monitor.records["live_shap_same_sample"]
            overhead = (shap_rec["wall_time_seconds"] / latency["batch_seconds"]
                        if latency["batch_seconds"] else None)
            st.session_state["rm_overhead"] = {
                "n_records": n, "classifier": settings["classifier"],
                "inference": latency, "shap": shap_rec,
                "overhead_factor": overhead,
                "encode_seconds": monitor.records["encode_sample"]["wall_time_seconds"],
            }
        if "rm_overhead" in st.session_state:
            data = st.session_state["rm_overhead"]
            inf, shap = data["inference"], data["shap"]
            ui.metric_row([
                ui.metric_card("Inference latency",
                               f"{inf['mean_ms_per_record']:.4f} ms",
                               f"Per record, best of {inf['repeats']} repeats "
                               f"({data['n_records']:,} records, "
                               f"{data['classifier']})", "zap", ui.ACCENT),
                ui.metric_card("Throughput",
                               f"{inf['throughput_records_per_second']:,.0f}",
                               "Records per second (same measurement)",
                               "trend", "#22C55E"),
                ui.metric_card("SHAP time (same sample)",
                               f"{shap['wall_time_seconds']:.3f} s",
                               f"{shap['wall_time_seconds'] / data['n_records'] * 1000:.3f} "
                               "ms per record, all classes", "sparkles",
                               ui.ACCENT_2),
                ui.metric_card("Explanation overhead",
                               f"{data['overhead_factor']:.1f}x"
                               if data["overhead_factor"] else "n/a",
                               "SHAP batch time / inference batch time",
                               "chart", "#EAB308"),
            ])
            st.caption(
                f"Measured just now on this machine: encoding took "
                f"{data['encode_seconds']:.3f} s, inference "
                f"{inf['batch_seconds']:.4f} s (best of 3), SHAP "
                f"{shap['wall_time_seconds']:.3f} s (single run, process memory "
                f"delta {shap['memory_delta_mb']:+.1f} MB). Inference-only and "
                "with-explanation timings were collected on the same encoded "
                "sample, so the overhead ratio is comparable - but a single "
                "run is not a controlled benchmark."
            )


def render() -> None:
    settings = common.get_settings()
    metrics = common.get_resource_metrics() or {}
    models = common.get_models()
    system = metrics.get("system", {})
    measurements = metrics.get("measurements", {})
    evaluation = common.get_evaluation() or {}

    ui.topbar(
        ["AI-SOC", "Resource Monitoring"],
        "Resource Monitoring",
        "Measured computational cost of detection and explanation - the "
        "evidence behind the resource-constrained SOC objective.",
        chips=[
            ui.chip("Collected", ui.timestamp_label(metrics.get("collected_at"))[:19],
                    "clock"),
            ui.chip("Host", system.get("platform", "unknown")[:28], "server"),
            ui.chip("Logical CPUs", str(system.get("cpu_count_logical", "n/a")),
                    "cpu"),
        ],
        notifications=common.notification_count(common.get_alerts()),
        refresh_key="rm_refresh",
        icon_name="cpu",
    )

    with ui.card("Measurement environment", "info",
                 "What these numbers do and do not demonstrate", key="rm_env"):
        snap = common.live_system_snapshot()
        ui.kv_table([
            ("Platform (training run)", system.get("platform", "not recorded")),
            ("Platform (now)", f"{snap['platform']} / Python "
                               f"{snap['python_version']}"),
            ("Logical CPUs", str(snap["cpu_count_logical"])),
            ("Total system memory", f"{snap['system_memory_total_mb']:,.0f} MB"),
            ("Process memory now", f"{snap['process_memory_mb']:,.0f} MB"),
            ("Measurements collected", metrics.get("collected_at", "none")),
            ("Energy consumption", "not measured (no instrumentation)"),
        ])
        st.caption(
            "Limitations: readings come from a single general-purpose machine "
            "running other processes, not a controlled benchmark or a "
            "representative SOC deployment. Process CPU% can exceed 100% "
            "because it is summed across cores. Wall-clock timings depend on "
            "hardware, thread count and system load. These measurements show "
            "the prototype's cost on this host; they do not by themselves "
            "prove suitability for a resource-constrained production SOC."
        )

    # ------------------------------------------------------- live readings
    with ui.card("Live resource gauges", "gauge",
                 "Real readings from this host, sampled when the page rendered",
                 key="rm_gauges"):
        g1, g2, g3 = st.columns(3, gap="small")
        gauges = [
            (g1, snap["cpu_percent"], "CPU usage", ui.ACCENT, "rm_g_cpu",
             f"{snap['cpu_count_logical']} logical cores, 0.3 s sample"),
            (g2, snap["system_memory_percent"], "System memory", ui.ACCENT_2,
             "rm_g_mem",
             f"{snap['system_memory_used_mb']:,.0f} / "
             f"{snap['system_memory_total_mb']:,.0f} MB"),
            (g3, snap["process_memory_mb"] / max(snap["system_memory_total_mb"], 1.0)
             * 100.0, "This process", "#22C55E", "rm_g_proc",
             f"{snap['process_memory_mb']:,.0f} MB of system memory"),
        ]
        for col, value, label, colour, key, sub in gauges:
            with col:
                st.plotly_chart(
                    ui.ring_gauge(value, label, color=colour),
                    theme=None, width="stretch", key=key,
                    config={"displaylogo": False},
                )
                st.caption(sub)
        st.caption(
            f"Sampled at {ui.timestamp_label(snap['sampled_at'])}. These are "
            "host readings for the dashboard process, not measurements of a "
            "production SOC deployment."
        )

    c1, c2 = st.columns([1, 1.4])
    with c1:
        ui.metric_row([
            ui.metric_card("CPU usage (now)", f"{snap['cpu_percent']:.1f}%",
                           "Real 0.3 s sample of system-wide CPU", "cpu",
                           ui.ACCENT),
            ui.metric_card("Memory usage (now)",
                           f"{snap['system_memory_percent']:.1f}%",
                           f"{snap['system_memory_used_mb']:,.0f} MB of "
                           f"{snap['system_memory_total_mb']:,.0f} MB used",
                           "drive", ui.ACCENT_2),
        ])
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        sizes = ResourceMonitor.model_file_sizes()
        total_mb = sum(v for v in sizes.values() if v is not None)
        ui.metric_row([
            ui.metric_card("Model storage", f"{total_mb:.1f} MB",
                           "Sum of all saved model artefacts on disk",
                           "layers", "#22C55E"),
            ui.metric_card("SHAP latency (example)",
                           ui.fmt(evaluation.get("local_shap_latency_ms_example"),
                                  2, " ms"),
                           "Single-record local explanation measured during "
                           "the training run", "sparkles", "#EAB308"),
        ])
        if st.button("Take another reading", icon=":material/monitoring:",
                     key="rm_sample_btn"):
            _sample_now()

    with c2, ui.card("Session resource readings", "activity",
                     "Only readings actually taken in this session are plotted",
                     key="rm_history"):
        history = st.session_state.setdefault("resource_history", [])
        if not history:
            history.append({
                "time": snap["sampled_at"][11:19],
                "cpu_percent": snap["cpu_percent"],
                "system_memory_percent": snap["system_memory_percent"],
                "process_memory_mb": snap["process_memory_mb"],
            })
        if len(history) >= 2:
            hist = pd.DataFrame(history)
            ui.show(ui.line_chart(
                hist["time"].tolist(),
                {"CPU %": hist["cpu_percent"].tolist(),
                 "System memory %": hist["system_memory_percent"].tolist()},
                height=280, y_title="percent"), key="rm_history_chart")
        else:
            ui.empty_state(
                "One reading so far",
                "Press <b>Take another reading</b> to build a real time series. "
                "No historical data is simulated.", "activity")
        st.caption(f"{len(history)} reading(s) collected in this session.")

    # ---------------------------------------------------- model artefacts
    with ui.card("Model artefacts", "drive",
                 "Actual file sizes and training cost", key="rm_models"):
        rows = []
        for name, size in sizes.items():
            key = name.replace("_size_mb", "")
            train = measurements.get(f"train_{key}", {})
            rows.append({
                "artefact": key,
                "size_mb": ui.fmt(size, 2) if size is not None else "not present",
                "train_seconds": ui.fmt(train.get("wall_time_seconds"), 3),
                "memory_delta_mb": ui.fmt(train.get("memory_delta_mb"), 1),
            })
        ui.show_table(pd.DataFrame(rows), height=220, key="rm_model_table")
        present = [(k.replace("_size_mb", ""), v) for k, v in sizes.items()
                   if v is not None]
        if present:
            ui.show(ui.vbar([p[0] for p in present], [p[1] for p in present],
                            colors=[ui.ACCENT] * len(present), height=260,
                            title="Model size on disk (MB)"),
                    key="rm_size_chart")

    # ------------------------------------------------ processing performance
    if measurements:
        with ui.card("Processing performance (training run)", "zap",
                     "Wall-clock measurements recorded by train.py",
                     key="rm_perf"):
            stages = [(k, v) for k, v in measurements.items()
                      if isinstance(v, dict) and "wall_time_seconds" in v]
            if stages:
                table = pd.DataFrame([{
                    "stage": k,
                    "wall_seconds": v["wall_time_seconds"],
                    "memory_delta_mb": v.get("memory_delta_mb"),
                    "cpu_percent_during": v.get("cpu_percent_during"),
                } for k, v in stages])
                ui.show_table(table, height=340, key="rm_stage_table")
                infer = [(k, v["wall_time_seconds"]) for k, v in stages
                         if "inference" in k]
                if infer:
                    ui.show(ui.hbar([k for k, _ in infer][::-1],
                                    [v for _, v in infer][::-1],
                                    colors=[ui.ACCENT] * len(infer),
                                    height=220, title="Inference wall time (s)"),
                            key="rm_infer_chart")
            latency_rows = []
            for name in ("isolation_forest", "random_forest", "xgboost"):
                rec = measurements.get(f"{name}_prediction_latency")
                if rec:
                    latency_rows.append({
                        "model": name,
                        "ms_per_record": rec["mean_ms_per_record"],
                        "records_per_second": rec["throughput_records_per_second"],
                        "sample_records": rec["n_records"],
                        "repeats": rec["repeats"],
                    })
            if latency_rows:
                st.markdown("**Prediction latency (best of N repeats)**")
                ui.show_table(pd.DataFrame(latency_rows), height=160,
                              key="rm_latency_table")
                ui.show(ui.grouped_bar(
                    [r["model"] for r in latency_rows],
                    {"ms per record": [r["ms_per_record"] for r in latency_rows]},
                    height=260, colors=[ui.ACCENT_2]),
                    key="rm_latency_chart")
                st.caption(
                    "Latency is batched inference amortised per record, not "
                    "single-record request latency. Real per-request latency "
                    "would be higher because of encoding and framework overhead."
                )
        with ui.card("Explainability overhead (training run)", "sparkles",
                     "Cost of global SHAP importance and a local explanation",
                     key="rm_shap"):
            gs = measurements.get("global_shap_computation", {})
            ui.kv_table([
                ("Global SHAP computation",
                 ui.fmt(gs.get("wall_time_seconds"), 3, " s")),
                ("Sample size",
                 str((evaluation.get("global_shap", {}) or {}).get("sample_size",
                                                                  "n/a"))),
                ("Process memory delta", ui.fmt(gs.get("memory_delta_mb"), 1, " MB")),
                ("Process CPU during computation",
                 ui.fmt(gs.get("cpu_percent_during"), 1, "%")),
                ("Local SHAP example latency",
                 ui.fmt(evaluation.get("local_shap_latency_ms_example"), 2, " ms")),
                ("Inference without explanations",
                 ui.fmt(measurements.get(
                     f"{settings['classifier']}_inference_test", {}).get(
                     "wall_time_seconds"), 3, " s")),
            ])
            st.caption(
                "Global importance is computed once and cached, so it does not "
                "add per-alert cost. Local explanations are generated on "
                "demand; the comparison below measures both on the same sample."
            )
    else:
        ui.empty_state("No resource metrics recorded",
                       "Run <code>python train.py</code> to collect real "
                       "measurements.", "cpu")

    if common.models_available(models):
        _live_measurements(models, settings)
