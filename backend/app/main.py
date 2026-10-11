from __future__ import annotations

import csv
import io
import itertools
import os
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from .advisor import evaluate_strategy_selector
from .workloads import generate_workload, occupancy_summary
from .voxelpool import run_experiment

app = FastAPI(
    title="VoxelPool Explorer API",
    version="1.2.0",
    description="Portable CPU-first voxel-pooling workload, sweep and benchmark API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        *[origin.strip().rstrip("/") for origin in os.getenv("FRONTEND_ORIGINS", "").split(",") if origin.strip()],
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

SAVED_RESULTS: dict[str, dict] = {}
SWEEP_JOBS: dict[str, dict] = {}


class ExperimentRequest(BaseModel):
    points: int = Field(default=1200, ge=1, le=10000)
    channels: int = Field(default=32, ge=1, le=128)
    grid_size: int = Field(default=32, ge=2, le=64)
    pattern: Literal["uniform", "hotspot", "unique"] = "uniform"
    repeats: int = Field(default=5, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def check_output_memory_budget(self):
        # The dashboard is a local research tool. Bound the largest dense output
        # so accidental slider/API combinations cannot allocate unbounded arrays.
        output_cells = self.grid_size ** 3 * self.channels
        if output_cells > 15_000_000:
            raise ValueError(
                "grid_size³ × channels exceeds the local memory guardrail "
                "(15,000,000 float32 output elements). Reduce the grid size or channels."
            )
        return self


class SweepRequest(BaseModel):
    """Cartesian product of small, controlled benchmark dimensions."""
    points: list[int] = Field(default_factory=lambda: [256, 1200], min_length=1, max_length=6)
    channels: list[int] = Field(default_factory=lambda: [8, 32], min_length=1, max_length=6)
    grid_sizes: list[int] = Field(default_factory=lambda: [16, 32], min_length=1, max_length=6)
    patterns: list[Literal["uniform", "hotspot", "unique"]] = Field(
        default_factory=lambda: ["uniform", "hotspot", "unique"], min_length=1, max_length=3
    )
    seeds: list[int] = Field(default_factory=lambda: [42, 43], min_length=1, max_length=4)
    repeats: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def validate_sweep(self):
        dimensions = {
            "points": (self.points, 1, 5000),
            "channels": (self.channels, 1, 128),
            "grid_sizes": (self.grid_sizes, 2, 64),
            "seeds": (self.seeds, 0, 2**31 - 1),
        }
        for name, (values, lower, upper) in dimensions.items():
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            if any(value < lower or value > upper for value in values):
                raise ValueError(f"{name} values must be between {lower} and {upper}")
        case_count = len(self.points) * len(self.channels) * len(self.grid_sizes) * len(self.patterns) * len(self.seeds)
        if case_count > 48:
            raise ValueError(f"Sweep expands to {case_count} cases; the safety limit is 48. Reduce selected dimensions.")
        return self


class SelectorEvaluationRequest(BaseModel):
    cases: list[dict] = Field(min_length=4, max_length=48)


class ExportRequest(BaseModel):
    experiment_ids: list[str] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "voxelpool-explorer",
        "version": "1.2.0",
        "measurement_backend": "portable_cpu_numpy",
        "quadric_backend_available": False,
    }


@app.get("/api/defaults")
def defaults():
    return {
        "points": 1200,
        "channels": 32,
        "grid_size": 32,
        "pattern": "uniform",
        "repeats": 5,
        "seed": 42,
        "limits": {
            "points": [1, 10000],
            "channels": [1, 128],
            "grid_size": [2, 64],
            "repeats": [1, 20],
            "max_output_elements": 15000000,
        },
        "patterns": [
            {"id": "uniform", "label": "Uniform", "description": "Destinations sampled across the full grid."},
            {"id": "hotspot", "label": "Hotspot", "description": "Most points target a small subset of voxels."},
            {"id": "unique", "label": "Unique", "description": "Each point targets a distinct voxel; points must not exceed grid_size³."},
        ],
    }


@app.post("/api/preview")
def preview(request: ExperimentRequest):
    """Generate occupancy metadata without running the expensive benchmark."""
    if request.pattern == "unique" and request.points > request.grid_size ** 3:
        raise HTTPException(
            status_code=422,
            detail=f"Unique destinations require points <= grid_size³ ({request.grid_size ** 3}).",
        )
    try:
        workload = generate_workload(
            points=request.points,
            channels=request.channels,
            grid_size=request.grid_size,
            pattern=request.pattern,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "workload": {
            "points": workload.num_points,
            "channels": workload.channels,
            "grid_size": workload.grid_size,
            "pattern": workload.pattern,
            "seed": workload.seed,
            "num_voxels": workload.num_voxels,
        },
        "occupancy": occupancy_summary(workload),
        "preview_only": True,
        "warning": "Preview only; no strategy timing has been run.",
    }


@app.post("/api/experiment")
def experiment(request: ExperimentRequest):
    if request.pattern == "unique" and request.points > request.grid_size ** 3:
        raise HTTPException(
            status_code=422,
            detail=f"Unique destinations require points <= grid_size³ ({request.grid_size ** 3}).",
        )
    try:
        workload = generate_workload(
            points=request.points,
            channels=request.channels,
            grid_size=request.grid_size,
            pattern=request.pattern,
            seed=request.seed,
        )
        result = run_experiment(workload, repeats=request.repeats)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    experiment_id = str(uuid.uuid4())
    result["experiment_id"] = experiment_id
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    SAVED_RESULTS[experiment_id] = result
    return result



def _execute_sweep(sweep_id: str, request_data: dict) -> None:
    """Run a bounded sweep in the background and publish incremental progress."""
    job = SWEEP_JOBS[sweep_id]
    dimensions = itertools.product(
        request_data["points"], request_data["channels"], request_data["grid_sizes"],
        request_data["patterns"], request_data["seeds"],
    )
    try:
        for case_index, (points, channels, grid_size, pattern, seed) in enumerate(dimensions, start=1):
            config = {
                "points": points, "channels": channels, "grid_size": grid_size,
                "pattern": pattern, "seed": seed, "repeats": request_data["repeats"],
            }
            case = {"case_index": case_index, "config": config, "status": "running", "reason": None, "result": None}
            output_elements = grid_size ** 3 * channels
            if output_elements > 15_000_000:
                case.update(status="skipped", reason="Dense output exceeds the 15,000,000-element memory guardrail.")
            elif pattern == "unique" and points > grid_size ** 3:
                case.update(status="skipped", reason=f"Unique destinations need points <= grid_size³ ({grid_size ** 3}).")
            else:
                try:
                    workload = generate_workload(points, channels, grid_size, pattern, seed)
                    result = run_experiment(workload, repeats=request_data["repeats"])
                    experiment_id = str(uuid.uuid4())
                    result["experiment_id"] = experiment_id
                    result["created_at"] = datetime.now(timezone.utc).isoformat()
                    result["sweep_id"] = sweep_id
                    SAVED_RESULTS[experiment_id] = result
                    case.update(status="completed", result=result)
                except Exception as exc:
                    case.update(status="error", reason=f"{type(exc).__name__}: {exc}")
            job["cases"].append(case)
            job["progress"] = case_index
            job["completed_cases"] = sum(1 for item in job["cases"] if item["status"] == "completed")
            job["skipped_cases"] = sum(1 for item in job["cases"] if item["status"] == "skipped")
            job["error_cases"] = sum(1 for item in job["cases"] if item["status"] == "error")
        job["status"] = "completed"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/sweeps")
def start_sweep(request: SweepRequest, background_tasks: BackgroundTasks):
    sweep_id = str(uuid.uuid4())
    total = len(request.points) * len(request.channels) * len(request.grid_sizes) * len(request.patterns) * len(request.seeds)
    SWEEP_JOBS[sweep_id] = {
        "sweep_id": sweep_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "request": request.model_dump(),
        "total_cases": total,
        "progress": 0,
        "completed_cases": 0,
        "skipped_cases": 0,
        "error_cases": 0,
        "cases": [],
    }
    background_tasks.add_task(_execute_sweep, sweep_id, request.model_dump())
    SWEEP_JOBS[sweep_id]["status"] = "running"
    return {"sweep_id": sweep_id, "status": "running", "total_cases": total}


@app.get("/api/sweeps/{sweep_id}")
def get_sweep(sweep_id: str):
    job = SWEEP_JOBS.get(sweep_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Sweep not found in this server session.")
    payload = {key: value for key, value in job.items() if key != "cases"}
    if job["status"] in {"completed", "failed"}:
        payload["cases"] = job["cases"]
    return payload


@app.post("/api/selector/evaluate")
def selector_evaluation(request: SelectorEvaluationRequest):
    try:
        return evaluate_strategy_selector(request.cases)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/workload/{workload_id}")
def get_workload(workload_id: str):
    result = SAVED_RESULTS.get(workload_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found in this server session.")
    return result


@app.post("/api/export/csv")
def export_csv(request: ExportRequest):
    rows = []
    for experiment_id in request.experiment_ids:
        result = SAVED_RESULTS.get(experiment_id)
        if not result:
            continue
        workload = result["workload"]
        occupancy = result["occupancy"]
        for strategy in result["strategies"]:
            timing = strategy.get("timing") or {}
            rows.append({
                "experiment_id": experiment_id,
                "created_at": result.get("created_at", ""),
                "backend": result["measurement_backend"],
                "points": workload["points"],
                "channels": workload["channels"],
                "grid_size": workload["grid_size"],
                "pattern": workload["pattern"],
                "seed": workload["seed"],
                "repeats": result["methodology"]["measured_repeats"],
                "occupied_voxels": occupancy["occupied_voxels"],
                "collision_rate_percent": occupancy["collision_rate_percent"],
                "strategy": strategy["id"],
                "correct": strategy["correct"],
                "median_ms": timing.get("median_ms"),
                "min_ms": timing.get("min_ms"),
                "max_ms": timing.get("max_ms"),
                "p95_ms": timing.get("p95_ms"),
                "speedup_vs_reference": strategy.get("speedup_vs_reference"),
                "status": strategy["status"],
            })
    buffer = io.StringIO()
    fieldnames = list(rows[0].keys()) if rows else [
        "experiment_id", "created_at", "backend", "points", "channels", "grid_size",
        "pattern", "seed", "repeats", "occupied_voxels", "collision_rate_percent",
        "strategy", "correct", "median_ms", "min_ms", "max_ms", "p95_ms",
        "speedup_vs_reference", "status"
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=voxelpool-experiments.csv"},
    )


@app.get("/api/research-notes")
def research_notes():
    return {
        "principles": [
            "Sweep dimensions as a bounded Cartesian product and preserve the seed for every case.",
            "Evaluate selectors by holding out entire workload configurations, not seed replicates.",
            "Report strategy-selection regret against the fastest measured valid candidate on held-out workloads.",
            "Correctness-gate every candidate before accepting its timing.",
            "Include strategy-specific preprocessing in the measured scope.",
            "Use seeded workload generation and preserve configuration metadata.",
            "Report per-workload results and regressions; do not rely on one headline speedup.",
            "Keep CPU, CUDA, and Quadric simulator measurements separate.",
            "Do not infer a hardware bottleneck without profiler evidence.",
        ],
        "references": [
            {"name": "KernelBench", "url": "https://github.com/ScalingIntelligence/KernelBench"},
            {"name": "Kernel Tuner", "url": "https://github.com/KernelTuner/kernel_tuner"},
            {"name": "TritonBench", "url": "https://github.com/meta-pytorch/tritonbench"},
            {"name": "Starlight", "url": "https://github.com/albertozeni/starlight"},
            {"name": "Quadric SDK", "url": "https://quadric.ai/sdk"},
            {"name": "Quadric custom-op tutorials", "url": "https://app.quadric.ai/docs/latest/chimera-software-user-guide/tutorials-model-demos/custom-op-tutorials/index"},
        ],
    }
