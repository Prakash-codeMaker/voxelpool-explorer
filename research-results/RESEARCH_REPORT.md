# VoxelPool Explorer — Exploratory Research Report

**Run date:** 2026-10-11  
**Machine:** x86_64 / x86_64  
**OS:** Linux-6.18.44-x86_64-with-glibc2.41  
**Python / NumPy:** 3.13.5 / 2.3.5  
**Elapsed wall time for all cases:** 0.34 seconds

## Research question
Can simple workload characteristics help predict which of three CPU implementations is fastest on held-out workload configurations?

## Method
- 72 benchmark cases: 36 distinct configurations × 2 deterministic seeds (the code computes the exact count from the run).
- Workload patterns: uniform, hotspot, unique.
- Point counts: 128/512/1200 for uniform and hotspot; 128/256/512 for unique (keeps unique destinations valid on the smallest grid).
- Feature channels: 4 and 16; grid dimensions: 8³ and 16³.
- Five measured repetitions per strategy after one warm-up; strategy order is seeded and pseudorandomly interleaved per case.
- Correctness gate: `numpy.allclose` with `rtol=1e-4`, `atol=1e-4` against sequential Python accumulation.
- Selector: 3-nearest-neighbour model; configuration-level holdout so the two seeds for a configuration cannot cross the split.

## Overall measured results
| Strategy | Valid cases | Median of per-case medians (ms) | Median CV (%) | Fastest wins |
|---|---:|---:|---:|---:|
| Sequential reference | 72 | 0.3564 | 4.50 | 0 |
| NumPy scatter (add.at) | 72 | 0.0275 | 5.81 | 48 |
| Sort + segmented reduce | 72 | 0.0441 | 10.67 | 24 |

## Results by workload pattern
| Pattern | Cases | Strategy | Median of case medians (ms) | Wins |
|---|---:|---|---:|---:|
| uniform | 24 | Sequential reference | 0.3610 | 0 |
| uniform | 24 | NumPy scatter (add.at) | 0.0432 | 16 |
| uniform | 24 | Sort + segmented reduce | 0.0653 | 8 |
| hotspot | 24 | Sequential reference | 0.3571 | 0 |
| hotspot | 24 | NumPy scatter (add.at) | 0.0429 | 14 |
| hotspot | 24 | Sort + segmented reduce | 0.0416 | 10 |
| unique | 24 | Sequential reference | 0.1780 | 0 |
| unique | 24 | NumPy scatter (add.at) | 0.0265 | 18 |
| unique | 24 | Sort + segmented reduce | 0.0364 | 6 |

## Held-out strategy selector evaluation
- Distinct configurations: 36; training: 24; held out: 12.
- Selector accuracy: 75.00%.
- Selector mean regret: 2.793% (median 0.000%).
- Best single training-set strategy baseline: NumPy scatter (add.at).
- Baseline accuracy: 58.33%; mean regret: 17.467%.

## Correctness and measurement quality
- All candidates passed correctness in all cases: True.
- Raw timing samples and per-case correctness/occupancy summaries are included in `experiment_bundle.json`; aggregated timing rows are in `strategy_timings.csv`.
- Coefficient of variation is reported per strategy and case; high CV means the result may be noisy and should be rerun.

## Interpretation
These are measurements from this one CPU environment and these synthetic workload settings. They are not evidence of GPU or Quadric performance. The Python implementation is intentionally a readable correctness reference and should not be treated as a fair optimized performance baseline. The median across case medians is a descriptive aggregate, not an end-to-end workload-weighted performance claim.

The selector evaluation is exploratory: a small number of held-out configurations can yield unstable accuracy/regret estimates. The selector should not be called a successful innovation unless it consistently beats the fixed-strategy baseline across repeated grouped splits and new workload families. If it does not, report that negative result and simplify the claim.

## Limitations and next experiments
- Import real-world traces and document their source, licensing and preprocessing.
- Add more configurations and seeds; repeat on multiple CPU models.
- Use bootstrap intervals over configurations (not individual timing repetitions as if independent workloads).
- Include a competitive optimized CPU baseline where semantics allow.
- Run multiple held-out splits and report distributions of selector regret.
- Verify target-specific operator semantics before implementing or claiming Quadric results.

## Reproducibility
Run from the project backend environment: `python -m unittest discover -s tests -v`. Reproduce an individual case by using its pattern, point count, channels, grid size and seed recorded in the JSON bundle; each strategy uses five measured repeats. The raw bundle includes Python/NumPy/OS metadata and timing samples.
