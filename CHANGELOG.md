# Changelog

## 1.1.0 — 2026-10-11

- Make occupancy preview follow the currently selected workload configuration via a debounced API preview; previewing does not benchmark or time an implementation.
- Persist workload settings, current result, and up to 25 recent experiment records in browser local storage.
- Generate CSV from browser-local experiment history, so export still works after the API process restarts.
- Attach Python, NumPy, OS, and machine metadata to new benchmark records.
- Add a dense-output size guardrail to reject accidental allocations above 15 million float32 output elements.
- Add API tests for previews, reproducibility, metadata, unique-pattern constraints, and the memory guardrail.
- Keep all timing labels CPU-only and preserve the existing dashboard design.

## 1.0.0

- Initial CPU-first voxel-pooling research prototype with three correctness-gated strategies, seeded workloads, occupancy visualization, experiment history, and JSON/CSV exports.
