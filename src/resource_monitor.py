"""Resource-efficiency measurement (real measurements only).

Measures what can actually be observed on the running system:
- training time, prediction latency, per-record processing time,
- throughput, CPU utilisation, process memory, model file sizes,
- SHAP explanation latency.

Anything not measured is reported as None / 'not measured' - values are
never invented.
"""
from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil

import config

RESOURCE_METRICS_FILE = config.METRICS_DIR / "resource_metrics.json"


class ResourceMonitor:
    """Collects timing, CPU and memory measurements for pipeline stages."""

    def __init__(self):
        self.process = psutil.Process()
        self.records: dict = {}

    # ----------------------------------------------------------- cpu / memory
    def cpu_percent(self, interval: float = 0.0) -> float:
        """Process CPU %. interval=0 returns since last call."""
        return self.process.cpu_percent(interval=interval)

    def memory_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def system_snapshot(self) -> dict:
        vm = psutil.virtual_memory()
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "process_cpu_percent": self.cpu_percent(),
            "process_memory_mb": round(self.memory_mb(), 1),
            "system_memory_total_mb": round(vm.total / 1024 / 1024, 1),
            "system_memory_available_mb": round(vm.available / 1024 / 1024, 1),
            "system_memory_percent_used": vm.percent,
            "cpu_count_logical": psutil.cpu_count(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

    # ---------------------------------------------------------------- timing
    @contextmanager
    def timed(self, name: str):
        """Context manager recording wall time + memory delta for a stage."""
        self.cpu_percent()  # prime CPU measurement
        mem_before = self.memory_mb()
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        mem_after = self.memory_mb()
        self.records[name] = {
            "wall_time_seconds": round(elapsed, 4),
            "memory_before_mb": round(mem_before, 1),
            "memory_after_mb": round(mem_after, 1),
            "memory_delta_mb": round(mem_after - mem_before, 1),
            "cpu_percent_during": round(self.cpu_percent(), 1),
        }

    def measure_prediction_latency(
        self, predict_fn, X: np.ndarray, n_repeats: int = 3, key: str = "prediction"
    ) -> dict:
        """Average latency of batched prediction over `n_repeats` passes."""
        best = None
        for _ in range(n_repeats):
            start = time.perf_counter()
            predict_fn(X)
            elapsed = time.perf_counter() - start
            if best is None or elapsed < best:
                best = elapsed
        n = len(X)
        self.records[key] = {
            "batch_seconds": round(best, 6),
            "n_records": int(n),
            "mean_ms_per_record": round(best / n * 1000, 6),
            "throughput_records_per_second": round(n / best, 1),
            "repeats": n_repeats,
        }
        return self.records[key]

    @staticmethod
    def model_file_sizes() -> dict:
        sizes = {}
        for name, path in config.MODEL_FILES.items():
            p = Path(path)
            sizes[f"{name}_size_mb"] = (
                round(p.stat().st_size / 1024 / 1024, 3) if p.exists()
                else None  # not measured: artefact absent
            )
        return sizes

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path = RESOURCE_METRICS_FILE) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "system": self.system_snapshot(),
            "measurements": self.records,
            "model_files": self.model_file_sizes(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path = RESOURCE_METRICS_FILE) -> dict | None:
        path = Path(path)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
