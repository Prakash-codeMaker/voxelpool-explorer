# VoxelPool Explorer

**A portable, CPU-first research lab for voxel-pooling workloads.** Explore how tensor shape and destination collisions affect accumulation strategies, compare correctness and timing, inspect voxel occupancy, and export reproducible experiment records.

> **Research honesty:** This starter runs local NumPy/Python implementations. Its timings are CPU measurements for this environment—not Quadric Chimera, CUDA, or accelerator results. It does not include a Quadric CCL kernel or claim a speedup on Quadric hardware. A Quadric backend should only be added after access to the official SDK/tutorial and a verified target-specific implementation.

## What is included

- Interactive React + Vite + TypeScript dashboard.
- FastAPI backend with a portable voxel-pooling reference and three candidate strategies:
  - `python_reference`: simple sequential reference.
  - `numpy_add_at`: vectorized scatter accumulation using `numpy.add.at`.
  - `sort_segmented`: sort by destination and reduce each occupied segment.
- Workload controls for point count, feature channels, grid size, and destination distribution (`uniform`, `hotspot`, `unique`).
- Correctness checks before performance results are accepted.
- Repeated timing measurements, median latency, speedup ratios, and per-strategy status.
- Live seeded occupancy preview that updates as workload parameters change, without timing the benchmark.
- Browser-persisted experiment history, experiment JSON export, and CSV export generated from local history.
- Environment metadata (Python, NumPy, OS and machine architecture) attached to each experiment.
- Occupancy visualization, experiment details, methodology notes, and API health status.
- API health endpoint and tests.

## Requirements

- Python 3.10+ (3.11 recommended)
- Node.js 18+ (20+ recommended)
- Internet access for the first install, so Python and npm dependencies can be installed

No GPU is required.

## Quick start (Windows)

1. Extract the ZIP.
2. Open the extracted `voxelpool-explorer` folder.
3. Double-click `start_windows.bat`.
4. Wait for both services to start. The dashboard opens at `http://localhost:5173`.

If the browser does not open automatically, visit `http://localhost:5173`.
The API runs at `http://127.0.0.1:8000`; interactive API documentation is at `http://127.0.0.1:8000/docs`. Workload settings, the current result, and the most recent 25 experiments persist in that browser profile. Export JSON/CSV backups for sharing or long-term storage; the server API lookup cache itself remains in memory.

### Manual start (Windows PowerShell)

Terminal 1 — API:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — frontend:

```powershell
cd frontend
npm install
npm run dev
```

### macOS / Linux

Run `bash start_unix.sh`, or start the backend and frontend using the manual commands in `start_unix.sh`.

## Run tests

From the project root:

```bash
python -m pip install -r backend/requirements.txt
cd backend
python -m unittest discover -s tests -v
```

## API

- `GET /api/health` — service status and active measurement backend.
- `GET /api/defaults` — default workload settings.
- `POST /api/preview` — generate the current occupancy preview without benchmarking.
- `POST /api/experiment` — run a correctness-gated CPU experiment.
- `GET /api/workload/{workload_id}` — inspect a saved in-memory workload result.
- `POST /api/export/csv` — export experiment summary rows as CSV.

Example request:

```json
{
  "points": 1200,
  "channels": 32,
  "grid_size": 32,
  "pattern": "hotspot",
  "repeats": 5,
  "seed": 42
}
```

## How the experiment works

1. Generate deterministic float32 features and destination voxel IDs.
2. Run the sequential reference and compute its output.
3. Execute each candidate and compare against the reference using explicit numerical tolerances.
4. Do not accept timing as a valid performance result if correctness fails.
5. Warm up each strategy, measure repeated runs, and report the median.
6. Calculate voxel occupancy statistics and preserve the input configuration and seed.

Floating-point accumulation order can differ between implementations. This project uses `rtol=1e-4` and `atol=1e-4` for candidate comparisons; that is a portable exploratory tolerance, not a substitute for a target-specific correctness contract.

## Interpreting the strategies

- **Sequential reference** is intentionally simple and easy to inspect. It is not expected to be fast in Python.
- **NumPy add.at** performs scatter-style accumulation in compiled NumPy code.
- **Sort + segmented reduction** sorts entries by destination and reduces each occupied segment. Sorting has overhead and may or may not pay off; this is exactly what experiments should establish.

Results vary by CPU, Python/NumPy versions, input sizes, and system load. The result records the available software/OS/machine metadata. Do not generalize a single timing to all workloads. A local output-size guardrail rejects dense output tensors larger than 15,000,000 float32 elements.

## Repository layout

```text
voxelpool-explorer/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── voxelpool.py
│   │   └── workloads.py
│   ├── tests/
│   │   └── test_voxelpool.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── RESEARCH_NOTES.md
├── start_windows.bat
└── start_unix.sh
```

## Research references and design lineage

This project borrows evaluation principles—not code wholesale—from:
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench): reference/candidate comparison, correctness gates, repeated measurement, structured scoring.
- [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner): configuration-based experiments and result recording.
- [TritonBench](https://github.com/meta-pytorch/tritonbench): reusable operator benchmark cases.
- [Starlight](https://github.com/albertozeni/starlight): evidence-led performance analysis.
- [Quadric SDK](https://quadric.ai/sdk) and [custom-op tutorials](https://app.quadric.ai/docs/latest/chimera-software-user-guide/tutorials-model-demos/custom-op-tutorials/index): target-specific follow-up path.

This is an independent research prototype, not affiliated with or endorsed by those projects.

## Next research steps

1. Verify Quadric DevStudio access and run its official voxel-pooling custom-op tutorial unchanged.
2. Record the exact operator contract and target toolchain version.
3. Add a separate Quadric CCL adapter only when the SDK and simulator are available.
4. Keep CPU, CUDA, and Quadric results in separate measurement categories.
5. Test on held-out shapes and distributions before making claims about a workload-aware selector.

## Troubleshooting

- **Port 8000 is busy:** stop the process using it, or change the API port and `VITE_API_BASE` together.
- **Port 5173 is busy:** Vite will suggest another port; use the URL shown in the terminal.
- **PowerShell blocks activation:** use `py -m pip install -r backend/requirements.txt` and `py -m uvicorn app.main:app` from the backend folder, or use Command Prompt.
- **Frontend says API offline:** make sure the backend terminal shows `Uvicorn running on http://127.0.0.1:8000`.

## Publish to GitHub and deploy live

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for the tested deployment architecture: Vercel hosts the Vite frontend, while Render hosts the FastAPI API. The backend is intentionally separate because benchmark sweeps use in-memory job state and are not a good fit for a serverless-only API. `render.yaml`, `frontend/vercel.json`, and `frontend/.env.example` are included.

Frontend dependency installation did not complete in the authoring environment due to a network timeout. Run `npm install` in `frontend/` in a network-enabled environment and commit the generated `package-lock.json` before the first deployment for reproducible dependency resolution.
