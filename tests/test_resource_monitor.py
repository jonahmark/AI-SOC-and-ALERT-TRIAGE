"""Tests for resource measurement (must be real, never invented)."""
from __future__ import annotations

import json
import time

import numpy as np

from src.resource_monitor import ResourceMonitor


def test_timed_context_records_wall_time():
    mon = ResourceMonitor()
    with mon.timed("sleep_stage"):
        time.sleep(0.05)
    rec = mon.records["sleep_stage"]
    assert rec["wall_time_seconds"] >= 0.04
    assert rec["memory_before_mb"] > 0


def test_prediction_latency_measurement():
    mon = ResourceMonitor()
    X = np.random.default_rng(0).random((200, 10))
    res = mon.measure_prediction_latency(lambda x: x.sum(axis=1), X,
                                         n_repeats=2, key="dummy")
    assert res["n_records"] == 200
    assert res["mean_ms_per_record"] > 0
    assert res["throughput_records_per_second"] > 0


def test_system_snapshot_fields():
    snap = ResourceMonitor().system_snapshot()
    for key in ("process_memory_mb", "system_memory_total_mb",
                "cpu_count_logical", "python_version"):
        assert key in snap
    assert snap["process_memory_mb"] > 0


def test_model_file_sizes_report_none_when_absent():
    sizes = ResourceMonitor.model_file_sizes()
    # whatever the state of models/, every key must be present and either
    # a real measured number or None ('not measured') - never a guess
    assert set(sizes) == {f"{k}_size_mb" for k in
                          ("isolation_forest", "random_forest", "xgboost",
                           "preprocessing")}
    for v in sizes.values():
        assert v is None or (isinstance(v, float) and v >= 0)


def test_save_load_roundtrip(tmp_path):
    mon = ResourceMonitor()
    with mon.timed("stage"):
        pass
    path = tmp_path / "resource_metrics.json"
    mon.save(path)
    loaded = json.loads(path.read_text())
    assert "measurements" in loaded and "stage" in loaded["measurements"]
    assert ResourceMonitor.load(path) == loaded
    assert ResourceMonitor.load(tmp_path / "absent.json") is None
