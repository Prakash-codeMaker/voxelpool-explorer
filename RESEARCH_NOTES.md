# Research Notes: VoxelPool Explorer

Date: 2026-10-11

## Executive decision

Build a focused, interactive research lab for voxel-pooling workloads rather than a general-purpose kernel optimization arena. The project currently runs portable CPU implementations so that the workload generator, correctness gate, experiment schema, and UI can be developed without special hardware access.

The intended research question is:

> Can workload metadata (shape, channel count, destination occupancy and collision distribution) help predict which voxel-pooling strategy performs best on held-out workloads?

This is a hypothesis, not a result. The current implementation does not yet test a trained predictor or a held-out strategy selector. It establishes the experiment infrastructure needed to investigate that question.

## What exists already

| Project | Existing strength | How this project uses the lesson | What we do not claim |
|---|---|---|---|
| KernelBench | Reference/candidate evaluation, correctness checks, repeated timing, scoring | Explicit workload schema, correctness gate, timing metadata, benchmark artifacts | We are not a replacement for KernelBench or a general LLM-kernel benchmark |
| Kernel Tuner | Parameter search, output verification, result caching | Experiment records and a small future tuning space | We do not implement a general-purpose autotuner in v1 |
| TritonBench | Reusable operator benchmark cases and common runner | Common result schema and per-strategy comparisons | No Triton/GPU backend is included |
| FlagGems | Backend-aware operator organization, tests and benchmark configuration | Shared workload contract with backend-specific adapters as a future extension | No multi-chip backend matrix is included |
| Starlight | Profiling and roofline-oriented analysis | Evidence-led interpretation and explicit separation between measurements and hypotheses | No hardware bottleneck is inferred from CPU timings alone |
| KTdashboard | Visualization of tuning results | Interactive charts and experiment history | A dashboard alone is not treated as novel research |
| Quadric DevStudio / SDK | Quadric-specific compiler, CCL, simulator and custom-op workflows | Future target adapter, only after SDK/tutorial access and a verified implementation | Current results are not Quadric cycles and no CCL kernel is shipped |

## Why voxel pooling

Voxel pooling is a meaningful irregular accumulation workload. Feature vectors are assigned to voxel destinations and accumulated into an output grid. Multiple points may target the same destination. The resulting behavior depends on both tensor dimensions and the distribution of destination IDs.

Quadric's public material includes a voxel-pooling custom-op tutorial and source-oriented guidance, so simply porting the operation would not be a strong differentiator. This prototype instead explores workload distributions, correctness, reproducibility and the evidence needed to compare candidate strategies.

## Implemented v1.1

### Workload generator

- Deterministic float32 feature vectors with a user-provided seed.
- Flattened 3D voxel destination IDs.
- `uniform`: destinations sampled across the full grid.
- `hotspot`: most points target a small region, with some background destinations.
- `unique`: no destination repeats; rejected if points exceed grid capacity.
- Occupancy statistics: occupied/empty voxels, collision points, collision rate, maximum points per voxel, mean occupancy of occupied voxels and an XY projection.

### Implementations

1. `python_reference`: sequential, readable reference.
2. `numpy_add_at`: scatter-style accumulation using NumPy.
3. `sort_segmented`: sort destinations, reduce contiguous segments and write one sum per occupied destination.

### Evaluation

- Correctness is checked against the sequential reference with `rtol=1e-4` and `atol=1e-4`.
- Each strategy receives one warm-up run and a configurable number of timed runs.
- Median, minimum, maximum, p95 and raw timing samples are retained.
- A strategy that fails correctness is not assigned an accepted timing.
- Strategy preprocessing (such as sorting) is included in timing.
- Speedup is relative only to the Python reference on the same local CPU.
- The occupancy preview reuses the same seeded generator but does not time any implementation.
- Browser-local state keeps the current settings/result and recent experiments across refreshes; CSV export is built from that saved history.
- Benchmark records include available Python, NumPy, OS and machine metadata.
- A guardrail rejects dense outputs exceeding 15 million float32 elements to avoid accidental excessive memory allocation.

## What remains to do before making scientific claims

1. Run a broader matrix of seeds, shapes, channel counts and occupancy distributions.
2. Repeat runs under controlled system load; preserve raw samples and software/environment versions.
3. Compare held-out workload families rather than choosing and reporting only the best input.
4. Establish whether profiling evidence supports any proposed bottleneck explanation.
5. Implement and evaluate a workload-aware selector against fixed strategies and simple heuristics under the same tuning budget.
6. For a Quadric backend, obtain access, run the official tutorial unchanged, record SDK/toolchain versions, and confirm the exact CCL semantics and correctness requirements before adapting any code.
7. Keep CPU milliseconds, CUDA wall-clock timings and Quadric simulator cycles as separate metrics. Never combine them into one speedup number.
8. Publish negative results and regressions alongside wins.

## Interpretation guardrails

- The sequential Python reference is deliberately simple and is not a strong optimized CPU baseline. A speedup against it is not evidence of a GPU or NPU speedup.
- Different accumulation orders can change floating-point rounding. The current tolerance is for portable exploration only.
- Sorting/grouping can cost more than it saves for some workloads.
- A collision rate is a workload property, not proof that collisions are the runtime bottleneck.
- Synthetic distributions help isolate variables but are not substitutes for real application traces.
- The current UI displays measured data only after an experiment runs. Empty values are not fabricated benchmark numbers.

## References

- KernelBench: https://github.com/ScalingIntelligence/KernelBench
- KernelBench paper: https://arxiv.org/abs/2502.10517
- Kernel Tuner: https://github.com/KernelTuner/kernel_tuner
- TritonBench: https://github.com/meta-pytorch/tritonbench
- FlagGems: https://github.com/flagos-ai/FlagGems
- Starlight: https://github.com/albertozeni/starlight
- Quadric SDK: https://quadric.ai/sdk
- Quadric DevStudio docs: https://app.quadric.ai/docs/24.04/home
- Quadric custom-op tutorials: https://app.quadric.ai/docs/24.04/chimera-software-user-guide/tutorials-model-demos/custom-op-tutorials/index
- BEVDepth reference implementation: https://github.com/Megvii-BaseDetection/BEVDepth
- BEVFusion reference implementation: https://github.com/mit-han-lab/bevfusion
