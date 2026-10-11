from __future__ import annotations

import math
import random
from statistics import median
from typing import Any

STRATEGY_LABELS = {
    "python_reference": "Sequential reference",
    "numpy_add_at": "NumPy scatter (add.at)",
    "sort_segmented": "Sort + segmented reduce",
}
STRATEGY_IDS = tuple(STRATEGY_LABELS)


def _case_parts(case: dict[str, Any]) -> tuple[dict, dict] | None:
    result = case.get("result")
    if not isinstance(result, dict):
        return None
    workload = result.get("workload")
    occupancy = result.get("occupancy")
    strategies = result.get("strategies")
    if not isinstance(workload, dict) or not isinstance(occupancy, dict) or not isinstance(strategies, list):
        return None
    times: dict[str, float] = {}
    for row in strategies:
        if not isinstance(row, dict) or not row.get("correct"):
            continue
        timing = row.get("timing")
        strategy_id = row.get("id")
        if strategy_id in STRATEGY_IDS and isinstance(timing, dict):
            value = timing.get("median_ms")
            if isinstance(value, (int, float)) and value > 0 and math.isfinite(value):
                times[strategy_id] = float(value)
    if not times:
        return None
    return {**workload, **{k: occupancy.get(k) for k in (
        "occupancy_percent", "collision_rate_percent", "max_points_per_voxel"
    )}}, times


def _feature_vector(row: dict[str, Any]) -> list[float]:
    points = max(1, int(row.get("points", 1)))
    channels = max(1, int(row.get("channels", 1)))
    grid = max(2, int(row.get("grid_size", 2)))
    return [
        math.log2(points),
        math.log2(channels),
        math.log2(grid),
        float(row.get("occupancy_percent") or 0) / 100.0,
        float(row.get("collision_rate_percent") or 0) / 100.0,
        math.log2(max(1, int(row.get("max_points_per_voxel") or 1))),
    ]


def _winner(times: dict[str, float]) -> str:
    return min(times, key=lambda key: times[key])


def _normalised_score(records: list[dict[str, Any]], strategy_id: str) -> float:
    ratios: list[float] = []
    for record in records:
        times = record["times"]
        if strategy_id in times and times:
            ratios.append(times[strategy_id] / min(times.values()))
    return sum(ratios) / len(ratios) if ratios else float("inf")


def evaluate_strategy_selector(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a simple 3-nearest-neighbour strategy selector on held-out configurations.

    Split is by (pattern, points, channels, grid size), not by repeated seed, so seed
    replicates for one configuration cannot leak between train and test partitions.
    Times are normalised within each workload to avoid a large workload dominating
    the scoring just because all implementations take longer on it.
    """
    grouped: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for case in cases:
        parsed = _case_parts(case)
        if not parsed:
            continue
        meta, times = parsed
        pattern = str(meta.get("pattern", "unknown"))
        key = (pattern, int(meta.get("points", 0)), int(meta.get("channels", 0)), int(meta.get("grid_size", 0)))
        if key not in grouped:
            grouped[key] = {"pattern": key[0], "points": key[1], "channels": key[2], "grid_size": key[3],
                            "features": [], "runs": [], "times": {sid: [] for sid in STRATEGY_IDS}}
        grouped[key]["features"].append(_feature_vector(meta))
        grouped[key]["runs"].append(case.get("case_index"))
        for strategy_id in STRATEGY_IDS:
            if strategy_id in times:
                grouped[key]["times"][strategy_id].append(times[strategy_id])

    records: list[dict[str, Any]] = []
    for row in grouped.values():
        times = {sid: median(vals) for sid, vals in row["times"].items() if vals}
        if len(times) < 2:
            continue
        # Seed repeats share a single workload-level observation; features are identical
        # except stochastic occupancy metrics, so use their median to damp random variation.
        feature_columns = list(zip(*row["features"]))
        features = [median(list(col)) for col in feature_columns]
        records.append({**row, "features": features, "times": times, "winner": _winner(times),
                        "seed_replicates": len(row["runs"])})

    records.sort(key=lambda r: (r["pattern"], r["grid_size"], r["channels"], r["points"]))
    if len(records) < 4:
        raise ValueError("At least four distinct valid workload configurations are required for held-out evaluation. Run a sweep with multiple point counts, channel counts, grid sizes or patterns.")

    # Deterministic, configuration-level holdout. Shuffle whole config groups with a
    # fixed split seed so the sorted pattern/shape order does not itself determine
    # which families land in the test partition. Seed replicates remain grouped.
    indices = list(range(len(records)))
    random.Random(20261011).shuffle(indices)
    test_indices = set(indices[::3])
    if not test_indices or len(test_indices) == len(records):
        test_indices = {indices[-1]}
    train = [row for idx, row in enumerate(records) if idx not in test_indices]
    test = [row for idx, row in enumerate(records) if idx in test_indices]
    if len(train) < 2:
        raise ValueError("Not enough training configurations after the held-out split.")

    # Min-max scale numeric features using training data only.
    columns = list(zip(*(row["features"] for row in train)))
    mins = [min(col) for col in columns]
    spans = [max(col) - min(col) for col in columns]
    spans = [span if span > 1e-9 else 1.0 for span in spans]

    def scaled(row: dict[str, Any]) -> list[float]:
        return [(value - mins[i]) / spans[i] for i, value in enumerate(row["features"])]

    def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
        av, bv = scaled(a), scaled(b)
        squared = sum((x - y) ** 2 for x, y in zip(av, bv))
        if a["pattern"] != b["pattern"]:
            squared += 0.25  # categorical mismatch penalty, fixed before test evaluation
        return math.sqrt(squared)

    # Choose the simple baseline using training data only.
    baseline_scores = {sid: _normalised_score(train, sid) for sid in STRATEGY_IDS}
    baseline_strategy = min(baseline_scores, key=baseline_scores.get)

    evaluations: list[dict[str, Any]] = []
    selector_regrets: list[float] = []
    baseline_regrets: list[float] = []
    selector_hits = 0
    baseline_hits = 0
    for target in test:
        neighbors = sorted(train, key=lambda row: distance(target, row))[: min(3, len(train))]
        neighbor_scores = {sid: _normalised_score(neighbors, sid) for sid in STRATEGY_IDS}
        selected = min(neighbor_scores, key=neighbor_scores.get)
        oracle = target["winner"]
        oracle_time = target["times"][oracle]
        selected_regret = target["times"].get(selected, float("inf")) / oracle_time - 1.0
        baseline_regret = target["times"].get(baseline_strategy, float("inf")) / oracle_time - 1.0
        selector_regrets.append(selected_regret)
        baseline_regrets.append(baseline_regret)
        selector_hits += int(selected == oracle)
        baseline_hits += int(baseline_strategy == oracle)
        evaluations.append({
            "workload": {"pattern": target["pattern"], "points": target["points"],
                         "channels": target["channels"], "grid_size": target["grid_size"]},
            "selected_strategy": selected,
            "selected_label": STRATEGY_LABELS[selected],
            "oracle_fastest_strategy": oracle,
            "oracle_fastest_label": STRATEGY_LABELS[oracle],
            "selected_median_ms": round(target["times"].get(selected, float("inf")), 4),
            "oracle_median_ms": round(oracle_time, 4),
            "regret_percent": round(100 * selected_regret, 3),
            "baseline_strategy": baseline_strategy,
            "baseline_label": STRATEGY_LABELS[baseline_strategy],
            "baseline_regret_percent": round(100 * baseline_regret, 3),
            "nearest_neighbors": len(neighbors),
        })

    mean_selector_regret = 100 * sum(selector_regrets) / len(selector_regrets)
    mean_baseline_regret = 100 * sum(baseline_regrets) / len(baseline_regrets)
    return {
        "method": "3-nearest-neighbour workload selector",
        "status": "evaluated",
        "split_policy": "deterministic random one-third configuration-level holdout (split seed 20261011; all seeds for one configuration stay in one partition)",
        "feature_names": ["log2(points)", "log2(channels)", "log2(grid_size)", "occupancy_fraction", "collision_fraction", "log2(max_points_per_voxel)"],
        "training_configurations": len(train),
        "heldout_configurations": len(test),
        "input_configurations": len(records),
        "selector_accuracy_percent": round(100 * selector_hits / len(test), 2),
        "selector_mean_regret_percent": round(mean_selector_regret, 3),
        "selector_median_regret_percent": round(100 * median(selector_regrets), 3),
        "baseline_strategy": baseline_strategy,
        "baseline_label": STRATEGY_LABELS[baseline_strategy],
        "baseline_accuracy_percent": round(100 * baseline_hits / len(test), 2),
        "baseline_mean_regret_percent": round(mean_baseline_regret, 3),
        "evaluation_cases": evaluations,
        "interpretation": "Regret is selected median latency divided by the fastest measured valid latency for that held-out workload, minus one. Results describe this CPU experiment set only; they do not establish generalisation to other CPUs or accelerators.",
    }
