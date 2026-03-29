"""
Reusable monitoring / benchmarking utilities for BirdNET-Analyzer

Main features:
- Multiple timers at the same time (by name)
- Per-timer averages over multiple runs
- Process RAM usage (RSS) and peaks during timed blocks
- Approximate CPU usage based on process CPU-time
- Estimated energy based on CPU-time (approximation; see `estimate_energy_joules`)
- Confidence during prediction
- Light_mode for benchmarking on small devices (controller, ...)
"""

from __future__ import annotations

import os
import time
import datetime
import threading
import birdnet_analyzer.config as cfg
from dataclasses import dataclass, field
from typing import Any, Iterable


def _bytes_to_mb(num_bytes: float) -> float:
    return num_bytes / (1024 * 1024)


class MemorySampler(threading.Thread):
    def __init__(self, service_ref, interval=0.01): # 10 ms so ram use measurements are not disturbed
        super().__init__()
        self.service = service_ref
        self.interval = interval
        self.peak_ram = 0.0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            # Aktuellen RAM abfragen
            current_ram = self.service.get_ram_usage_mb()
            if current_ram > self.peak_ram:
                self.peak_ram = current_ram
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        return self.peak_ram

@dataclass
class _TimerRun:
    """Stores raw data for one timer run (start -> stop)."""

    wall_seconds: float
    cpu_seconds: float
    rss_start_bytes: int # Needed RAM storage at the start
    rss_end_bytes: int
    peak_interval_mb: float = 0.0 # peak ram usage during measurement

    @property
    def rss_delta_bytes(self) -> int:
        return self.rss_end_bytes - self.rss_start_bytes


@dataclass
class _TimerStats:
    """Aggregates multiple runs for a single named timer, calculates average values."""

    runs: list[_TimerRun] = field(default_factory=list)

    def add_run(self, run: _TimerRun) -> None:
        self.runs.append(run)

    @property
    def count(self) -> int:
        return len(self.runs)

    @property
    def total_wall_seconds(self) -> float:
        return sum(r.wall_seconds for r in self.runs)

    @property
    def total_cpu_seconds(self) -> float:
        return sum(r.cpu_seconds for r in self.runs)

    @property
    def avg_wall_seconds(self) -> float:
        return self.total_wall_seconds / self.count if self.count else 0.0

    @property
    def avg_cpu_seconds(self) -> float:
        return self.total_cpu_seconds / self.count if self.count else 0.0

    @property
    def peak_rss_end_bytes(self) -> int:
        """Peak RSS observed at stop() time (not continuous sampling)."""
        return max((r.rss_end_bytes for r in self.runs), default=0)


class MetricsService:
    """
    A small service class to measure performance metrics during a run.
    """

    def __init__(self, *, model_path: str | None = None, scenario: str = 'original', light_mode: bool = False, assumed_cpu_power_watts: float = 5.0) -> None:
        # Var for enabling to two stage benchmarking
        self._light_mode: bool = light_mode
        
        # Power is used for *estimated* energy. Default is a conservative small-device value.
        self.assumed_cpu_power_watts = float(assumed_cpu_power_watts)

        if not self._light_mode:
            import psutil
            self._proc = psutil.Process(os.getpid())
        else:
            self._proc = None

        self._model_path: str | None = None
        self._model_size_mb: float | None = None

        # Timer state:
        # - _active_starts stores start snapshots for currently running timers
        # - _timers aggregates completed runs per timer name
        self._active_starts: dict[str, dict[str, Any]] = {}
        self._timers: dict[str, _TimerStats] = {}

        self.set_model_path(model_path)

        # Confidence during prediction
        self._confidence: float | None = None

        self._scenario = scenario

    @staticmethod
    def get_model_size(path: str) -> float:
        """
        Returns the model file size in MB.

        Why it's important:
        - On edge devices, a smaller model is faster to load and may fit into memory easier.
        """
        size_bytes = os.path.getsize(path)
        return _bytes_to_mb(size_bytes)

    def set_model_path(self, path: str | None) -> None:
        """Stores the model path and caches its file size (MB) for the summary output."""
        if path is None:
            self._model_size_mb = None
            return

        self._model_path = path
        try:
            self._model_size_mb = self.get_model_size(path)
        except OSError:
            # If file is missing, we keep size as unknown.
            self._model_size_mb = None

    def _get_ram_usage_edge(self) -> float:
        """
        Uses unix system files
        """
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if "VmRSS" in line:
                        kb = int(line.split()[1])
                        return kb / 1024
        except Exception:
            return 0.0
        return 0.0

    def get_ram_usage_mb(self) -> float:
        """
        Current RAM usage (RSS: Resident Set Size) of the current Python process in MB.
        """
        if not self._light_mode:
            return _bytes_to_mb(self._proc.memory_info().rss)
        else:
            return self._get_ram_usage_edge()

    def get_cpu_usage_percent(self, *, interval_s: float = 0.1) -> float:
        """
        Returns process CPU usage percent (normalized to 0..100 across all cores).

        Notes:
        - This uses psutil's sampling over a short interval.
        - Good for "current CPU usage now", not for measuring a specific block.
        """ 
        if self._light_mode:
            return 0.0  # not supported in light mode

        import psutil

        # psutil returns a percent that can exceed 100 on multi-core, has to be normalized
        raw = self._proc.cpu_percent(interval=interval_s)
        cores = psutil.cpu_count(logical=True) or 1
        return raw / cores

    def get_peak_ram_usage_mb(self) -> float:
        """
        Gets the maximum memory usage
        """
        if os.name == 'posix':  # Linux/Mac
            import resource
            # ru_maxrss in KB
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        return 0.0

    def start_timer(self, name: str) -> None:
        """
        Start a named timer.

        Supports:
        - Extended mode (psutil)
        - Light mode (os.times, no psutil)
        """
        if name in self._active_starts:
            raise ValueError(f"Timer '{name}' is already running. Stop it before starting again.")

        t0 = time.perf_counter() # wall time: real world time

        if self._light_mode:
            # Lightweight CPU measurement (works on Raspberry Pi)
            cpu_times = os.times()
            cpu0 = float(cpu_times.user + cpu_times.system)
            rss0 = self._get_ram_usage_edge() * 1024 * 1024
        else:
            cpu_times = self._proc.cpu_times() # cpu time: real time value, the process was acitve 
            cpu0 = float(cpu_times.user + cpu_times.system)
            rss0 = int(self._proc.memory_info().rss)

        # Start runner to measure ram peaks during inference
        sampler = None
        if name == "inference":
            sampler = MemorySampler(self)
            sampler.start()

        self._active_starts[name] = {
            "t0": t0,
            "cpu0": cpu0,
            "rss0": rss0,
            "sampler": sampler
        }

    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and record its measurements.

        Works in both extended and light mode.
        """
        snap = self._active_starts.pop(name, None)
        if snap is None:
            raise ValueError(f"Timer '{name}' was not started.")

        t1 = time.perf_counter()

        if self._light_mode:
            cpu_times = os.times()
            cpu1 = float(cpu_times.user + cpu_times.system)
            rss1 = self._get_ram_usage_edge() * 1024 * 1024
        else:
            cpu_times = self._proc.cpu_times()
            cpu1 = float(cpu_times.user + cpu_times.system)
            rss1 = int(self._proc.memory_info().rss)

        wall = max(0.0, t1 - float(snap["t0"]))
        cpu = max(0.0, cpu1 - float(snap["cpu0"]))

        # stop inference ram measure runner
        interval_peak_mb = 0.0
        if "sampler" in snap and snap["sampler"] is not None:
            interval_peak_mb = snap["sampler"].stop()
            snap["sampler"].join()

        run = _TimerRun(
            wall_seconds=wall,
            cpu_seconds=cpu,
            rss_start_bytes=int(snap["rss0"]),
            rss_end_bytes=rss1,
            peak_interval_mb=interval_peak_mb
        )

        stats = self._timers.setdefault(name, _TimerStats())
        stats.add_run(run)

        return wall

    def get_timer_stats(self, name: str) -> dict[str, float]:
        """
        Convenience method to read aggregated stats for a timer.

        Returns a dict with:
        - count
        - total_wall_s, avg_wall_s
        - total_cpu_s, avg_cpu_s
        - cpu_util_percent (normalized 0..100 across all cores; based on CPU-time / wall-time)
        """
        stats = self._timers.get(name, _TimerStats())
        cpu_util = 0.0

        if not self._light_mode:
            import psutil

            cores = psutil.cpu_count(logical=True) or 1

            if stats.total_wall_seconds > 0:
                # CPU utilization for the block:
                # (CPU seconds / wall seconds) gives "fraction of one core".
                # Divide by core count to normalize to 0..100 for the whole machine.
                cpu_util = (stats.total_cpu_seconds / stats.total_wall_seconds) * 100.0 / cores

        return {
            "count": float(stats.count),
            "total_wall_s": float(stats.total_wall_seconds),
            "avg_wall_s": float(stats.avg_wall_seconds),
            "total_cpu_s": float(stats.total_cpu_seconds),
            "avg_cpu_s": float(stats.avg_cpu_seconds),
            "cpu_util_percent": float(cpu_util),
        }

    def estimate_energy_joules(self, cpu_seconds: float | None = None) -> float:
        """
        Estimate energy consumption in Joules (approximation).

        IMPORTANT:
        - This is NOT a real power measurement.
        - Approximate energy from CPU-time using:
              Energy (J) = CPU_time_seconds * assumed_cpu_power_watts
        - Real energy depends on CPU frequency, voltage, load, peripherals, and more.

        If cpu_seconds is None, we use the sum of CPU-time from all recorded timers.
        """
        if cpu_seconds is None:
            cpu_seconds = sum(s.total_cpu_seconds for s in self._timers.values())
        return float(cpu_seconds) * float(self.assumed_cpu_power_watts)

    def set_confidence_from_prediction(self, timestamps: list[str], result: dict[str, list]) -> float:
        """
        Compute average confidence from the prediction results
        """
        predictionsCounter = 0
        confidenceSum = 0

        for timestamp in timestamps:
            for c in result[timestamp]:
                if float(c[1]) >= cfg.MIN_CONFIDENCE:
                    predictionsCounter += 1
                    confidenceSum += float(c[1])
        
        self._confidence = confidenceSum / predictionsCounter

    def write_to_csv_log(self, file_path: str = "ownTests/performance/metrics_log.csv") -> None:
        """
        Writes the current metrics into a CSV file (appends one row per run).
        """
        # Prepare timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get values (safe defaults if not available)
        model_size = f"{self._model_size_mb:.2f}" if self._model_size_mb is not None else "NA"

        confidence = f"{self._confidence * 100:.2f}" if self._confidence is not None else "NA"
        ram_usage = f"{self.get_ram_usage_mb():.2f}"
        peak_ram_usage = f"{self.get_peak_ram_usage_mb():.2f}"

        # Helper to extract timer values
        def get_avg_time(timer_name: str) -> str:
            if timer_name not in self._timers:
                return "NA"
            stats = self.get_timer_stats(timer_name)
            return f"{stats['avg_wall_s']:.4f}"

        model_load_time = get_avg_time("model_load")
        audio_time = get_avg_time("audio_processing")
        inference_time = get_avg_time("inference")

        energy = f"{self.estimate_energy_joules():.2f}"

        # Get peak ram usage during inference
        inference_peak = "NA"
        if "inference" in self._timers:
            # Wir nehmen den höchsten Peak-Wert aller bisherigen Inferenz-Runs
            runs = self._timers["inference"].runs
            inference_peak = f"{max((r.peak_interval_mb for r in runs), default=0.0):.2f}"

        # CSV header
        header = (
            "Timestamp,Scenario,Model Size (MB),Average Confidence (%),"
            "RAM Usage (MB), Total Peak RAM Usage (MB), Inference Peak RAM Usage (MB), Model Load Time (s),Audio Processing Time (s),"
            "Inference Time (s),Energy (J)\n"
        )

        # CSV row
        row = (
            f"{timestamp},{self._scenario},{model_size},{confidence},"
            f"{ram_usage}, {peak_ram_usage}, {inference_peak},{model_load_time},{audio_time},{inference_time},{energy}\n"
        )

        # Make directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Check if file exists
        file_exists = os.path.isfile(file_path)

        # Write to file
        with open(file_path, "a") as f:
            if not file_exists:
                f.write(header)
            f.write(row)

    def print_summary(self) -> None:
        """Print all collected metrics in a structured, beginner-friendly format."""
        ram_now_mb = self.get_ram_usage_mb()
        peak_ram_usage = self.get_peak_ram_usage_mb()
        energy_j = self.estimate_energy_joules()

        print(f"\n=== PERFORMANCE METRICS for '{self._scenario}'===")

        print(f"Benchmark Light Mode: {self._light_mode}")

        if self._model_size_mb is not None:
            print(f"Model Size: {self._model_size_mb:.2f} MB")
        elif self._model_path:
            print("Model Size: (unknown - file not found)")
        else:
            print("Model Size: (not set)")

        formated_confidence = self._confidence * 100.0
        print(f"Average Confidence: {formated_confidence:.2f} %")

        print(f"RAM Usage (current RSS): {ram_now_mb:.2f} MB")
        print(f"Peak RAM Usage: {peak_ram_usage:.2f} MB")

        # Common timers
        for label, timer_name in (
            ("Model Load Time", "model_load"),
            ("Audio Processing Time", "audio_processing"),
            ("Inference Time", "inference"),
        ):
            if timer_name not in self._timers:
                continue
            stats = self.get_timer_stats(timer_name)
            avg = stats["avg_wall_s"]
            total = stats["total_wall_s"]
            n = int(stats["count"])
            cpu_util = stats["cpu_util_percent"]

            if n <= 1:
                print(f"{label}: {total:.4f} s (CPU util ~ {cpu_util:.1f} %)")
            else:
                print(f"{label}: {total:.4f} s total | {avg:.4f} s avg (n={n}) (CPU util ~ {cpu_util:.1f} %)")

        # If there are additional timers, print them too.
        extra = [k for k in self._timers.keys() if k not in {"model_load", "audio_processing", "inference"}]
        for name in sorted(extra):
            stats = self.get_timer_stats(name)
            n = int(stats["count"])
            if n <= 1:
                print(f"{name}: {stats['total_wall_s']:.4f} s")
            else:
                print(f"{name}: {stats['total_wall_s']:.4f} s total | {stats['avg_wall_s']:.4f} s avg (n={n})")

        print(f"Energy (estimated): {energy_j:.2f} J (approx.)")

        print("===========================\n")

