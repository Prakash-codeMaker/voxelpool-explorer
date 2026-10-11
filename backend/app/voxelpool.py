from __future__ import annotations

import platform
import random
import sys
import time
from typing import Callable
import numpy as np

from .workloads import Workload, occupancy_summary


def python_reference(workload: Workload) -> np.ndarray:
    """Readable sequential accumulation reference: output[voxel] += feature."""
    out = np.zeros((workload.num_voxels, workload.channels), dtype=np.float32)
    for i in range(workload.num_points):
        out[workload.voxel_ids[i]] += workload.features[i]
    return out


def numpy_add_at(workload: Workload) -> np.ndarray:
    """Vectorized scatter accumulation through NumPy's indexed update primitive."""
    out = np.zeros((workload.num_voxels, workload.channels), dtype=np.float32)
    np.add.at(out, workload.voxel_ids, workload.features)
    return out


def sort_segmented(workload: Workload) -> np.ndarray:
    """Sort by destination and sum each occupied segment, then scatter segment sums."""
    out = np.zeros((workload.num_voxels, workload.channels), dtype=np.float32)
    order = np.argsort(workload.voxel_ids, kind="stable")
    sorted_ids = workload.voxel_ids[order]
    sorted_features = workload.features[order]
    unique_ids, starts = np.unique(sorted_ids, return_index=True)
    if unique_ids.size:
        sums = np.add.reduceat(sorted_features, starts, axis=0, dtype=np.float32)
        out[unique_ids] = sums
    return out


STRATEGIES: list[tuple[str, str, Callable[[Workload], np.ndarray]]] = [
    ("python_reference", "Sequential reference", python_reference),
    ("numpy_add_at", "NumPy scatter (add.at)", numpy_add_at),
    ("sort_segmented", "Sort + segmented reduce", sort_segmented),
]


def _timing_summary(samples: list[float]) -> dict:
    arr = np.asarray(samples, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    standard_deviation = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "samples_ms": [round(float(x), 4) for x in arr],
        "median_ms": round(float(np.median(arr)), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "standard_deviation_ms": round(standard_deviation, 4),
        "coefficient_of_variation_percent": round(100.0 * standard_deviation / mean, 3) if mean > 0 else 0.0,
    }


def run_experiment(workload: Workload, repeats: int = 5) -> dict:
    if repeats < 1 or repeats > 20:
        raise ValueError("repeats must be between 1 and 20")

    reference = python_reference(workload)
    strategy_results: dict[str, dict] = {}
    eligible_functions: dict[str, Callable[[Workload], np.ndarray]] = {}

    # Correctness validation is a separate phase; candidates are not scored by speed
    # unless their output matches the reference before timing begins.
    for strategy_id, label, fn in STRATEGIES:
        try:
            output = fn(workload)
            correct = bool(np.allclose(output, reference, rtol=1e-4, atol=1e-4))
            error_max = float(np.max(np.abs(output - reference))) if output.size else 0.0
            strategy_results[strategy_id] = {
                "id": strategy_id,
                "label": label,
                "correct": correct,
                "max_abs_error": round(error_max, 8),
                "timing": None,
                "status": "passed correctness gate" if correct else "rejected: correctness mismatch",
            }
            if correct:
                eligible_functions[strategy_id] = fn
        except Exception as exc:
            strategy_results[strategy_id] = {
                "id": strategy_id,
                "label": label,
                "correct": False,
                "max_abs_error": None,
                "timing": None,
                "status": f"error during correctness validation: {type(exc).__name__}: {exc}",
            }

    # Interleave strategy measurements in a deterministic pseudorandom order. It is
    # reproducible for a given workload configuration, while avoiding a fixed global
    # strategy order that would always favour the first/last implementation in a run.
    pattern_code = {"uniform": 1, "hotspot": 2, "unique": 3}.get(workload.pattern, 0)
    order_seed = (workload.seed + workload.num_points * 37 + workload.channels * 101 +
                  workload.grid_size * 1009 + pattern_code * 10007) & 0x7FFFFFFF
    rng = random.Random(order_seed)
    warmup_order = list(eligible_functions)
    rng.shuffle(warmup_order)
    timing_errors: dict[str, str] = {}
    for strategy_id in warmup_order:
        try:
            eligible_functions[strategy_id](workload)
        except Exception as exc:
            timing_errors[strategy_id] = f"error during warm-up: {type(exc).__name__}: {exc}"
    for strategy_id, message in timing_errors.items():
        eligible_functions.pop(strategy_id, None)
        strategy_results[strategy_id]["timing"] = None
        strategy_results[strategy_id]["status"] = message

    samples: dict[str, list[float]] = {strategy_id: [] for strategy_id in eligible_functions}
    measurement_orders: list[list[str]] = []
    for _ in range(repeats):
        schedule = list(eligible_functions)
        rng.shuffle(schedule)
        measurement_orders.append(list(schedule))
        for strategy_id in schedule:
            start = time.perf_counter()
            try:
                eligible_functions[strategy_id](workload)
                samples[strategy_id].append((time.perf_counter() - start) * 1000.0)
            except Exception as exc:
                timing_errors[strategy_id] = f"error during timing: {type(exc).__name__}: {exc}"
        # Any strategy failing a measured call is removed from subsequent rounds.
        for strategy_id in list(eligible_functions):
            if strategy_id in timing_errors:
                eligible_functions.pop(strategy_id, None)

    for strategy_id, result in strategy_results.items():
        if strategy_id in timing_errors:
            # Do not score partial samples after a runtime failure; retain the error.
            result["timing"] = None
            result["status"] = timing_errors[strategy_id]
        elif samples.get(strategy_id):
            result["timing"] = _timing_summary(samples[strategy_id])
            result["status"] = "passed" if result["correct"] else result["status"]

    reference_timing = strategy_results.get("python_reference", {}).get("timing")
    reference_median = reference_timing.get("median_ms") if reference_timing else None
    ordered_strategy_results = [strategy_results[sid] for sid, _, _ in STRATEGIES]
    for result in ordered_strategy_results:
        timing = result.get("timing")
        if timing and reference_median and timing["median_ms"] > 0:
            result["speedup_vs_reference"] = round(reference_median / timing["median_ms"], 4)
        else:
            result["speedup_vs_reference"] = None

    summary = occupancy_summary(workload)
    return {
        "schema_version": "voxelpool-experiment/v1",
        "measurement_backend": "portable_cpu_numpy",
        "measurement_unit": "milliseconds",
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or platform.machine(),
        },
        "warning": (
            "CPU-only exploratory results. These measurements are not Quadric simulator cycles, "
            "CUDA timings, or evidence of accelerator speedup."
        ),
        "workload": {
            "points": workload.num_points,
            "channels": workload.channels,
            "grid_size": workload.grid_size,
            "pattern": workload.pattern,
            "seed": workload.seed,
            "num_voxels": workload.num_voxels,
        },
        "occupancy": summary,
        "correctness": {
            "reference": "python_reference",
            "rtol": 1e-4,
            "atol": 1e-4,
            "all_candidates_passed": all(r["correct"] for r in strategy_results.values()),
        },
        "strategies": ordered_strategy_results,
        "methodology": {
            "warmup_runs_per_strategy": 1,
            "warmup_order": warmup_order,
            "measurement_order": measurement_orders,
            "ordering_policy": "Seeded pseudorandom interleaving; each recorded list is the order attempted in one measured round.",
            "measured_repeats": repeats,
            "summary_statistic": "median",
            "timing_scope": "function call including strategy-specific preprocessing",
            "reference_output_dtype": "float32",
            "timing_spread_statistics": ["min_ms", "max_ms", "p95_ms", "standard_deviation_ms", "coefficient_of_variation_percent"],
        },
    }
