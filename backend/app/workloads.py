from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Workload:
    features: np.ndarray
    voxel_ids: np.ndarray
    grid_size: int
    channels: int
    pattern: str
    seed: int

    @property
    def num_points(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_voxels(self) -> int:
        return self.grid_size ** 3


def generate_workload(
    points: int,
    channels: int,
    grid_size: int,
    pattern: str = "uniform",
    seed: int = 42,
) -> Workload:
    """Generate deterministic feature vectors and flattened 3D voxel destinations."""
    if points < 1:
        raise ValueError("points must be >= 1")
    if channels < 1:
        raise ValueError("channels must be >= 1")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if pattern not in {"uniform", "hotspot", "unique"}:
        raise ValueError("pattern must be uniform, hotspot, or unique")

    rng = np.random.default_rng(seed)
    num_voxels = grid_size ** 3
    features = rng.normal(0.0, 1.0, size=(points, channels)).astype(np.float32)

    if pattern == "uniform":
        voxel_ids = rng.integers(0, num_voxels, size=points, dtype=np.int64)
    elif pattern == "hotspot":
        # Most points target a small hot region; the remainder are spread across the grid.
        hot_count = int(num_voxels * 0.02)
        hot_count = max(1, hot_count)
        hot_count = min(hot_count, num_voxels)
        hot_ids = rng.integers(0, hot_count, size=points, dtype=np.int64)
        background = rng.integers(0, num_voxels, size=points, dtype=np.int64)
        choose_hot = rng.random(points) < 0.85
        voxel_ids = np.where(choose_hot, hot_ids, background).astype(np.int64)
    else:
        if points > num_voxels:
            raise ValueError(
                f"unique pattern needs points <= grid_size^3 ({num_voxels}); got {points}"
            )
        voxel_ids = rng.choice(num_voxels, size=points, replace=False).astype(np.int64)

    return Workload(
        features=features,
        voxel_ids=voxel_ids,
        grid_size=grid_size,
        channels=channels,
        pattern=pattern,
        seed=seed,
    )


def occupancy_summary(workload: Workload) -> dict:
    counts = np.bincount(workload.voxel_ids, minlength=workload.num_voxels)
    occupied = int(np.count_nonzero(counts))
    collisions = int(workload.num_points - occupied)
    occupied_counts = counts[counts > 0]
    return {
        "num_voxels": int(workload.num_voxels),
        "occupied_voxels": occupied,
        "empty_voxels": int(workload.num_voxels - occupied),
        "occupancy_percent": round(100.0 * occupied / workload.num_voxels, 4),
        "collision_points": collisions,
        "collision_rate_percent": round(100.0 * collisions / workload.num_points, 4),
        "max_points_per_voxel": int(counts.max(initial=0)),
        "mean_points_per_occupied_voxel": (
            round(float(occupied_counts.mean()), 4) if occupied_counts.size else 0.0
        ),
        # A compact 2D projection (sum over Z) is useful for the UI.
        "xy_projection": counts.reshape(
            workload.grid_size, workload.grid_size, workload.grid_size
        ).sum(axis=0).tolist(),
    }
