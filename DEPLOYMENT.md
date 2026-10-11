# Deployment guide — VoxelPool Explorer

## Recommended setup: Vercel frontend + Render API

The application uses a React/Vite frontend and a FastAPI backend. Deploy them as separate services. The benchmark/sweep endpoints keep job state in process memory, so a serverless-only deployment is not a good fit for long sweeps.

### 1. Push to GitHub

1. Create a **public** GitHub repository named `voxelpool-explorer` (or your preferred name).
2. Extract this project, open a terminal in the `voxelpool-explorer` directory, then run:

```bash
git init
git add .
git commit -m "Release VoxelPool Explorer research platform"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/voxelpool-explorer.git
git push -u origin main
```

Do not commit `.env` files, `.venv`, `node_modules`, or credentials.

### 2. Deploy the backend on Render

1. In Render, create a new **Blueprint** from the GitHub repository; Render will detect `render.yaml`.
2. The API service uses `backend/` as its root directory and `backend/requirements.txt` for dependencies.
3. When prompted for `FRONTEND_ORIGINS`, enter the eventual Vercel origin, e.g. `https://voxelpool-explorer.vercel.app` (no trailing slash). You can update it after deployment.
4. Wait for the service to deploy. Verify `https://YOUR-API.onrender.com/api/health` returns JSON with `status: ok`.
5. Copy the API's base URL (without `/api/health`).

The free Render service may sleep when idle, so the first request can be slow. Sweep job state is in memory and can be lost when a service restarts; export important experiment bundles.

### 3. Deploy the frontend on Vercel

1. Import the same GitHub repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Framework preset: **Vite**. Build command: `npm run build`. Output directory: `dist`.
4. Add environment variable `VITE_API_BASE` with the Render API base URL, e.g. `https://voxelpool-explorer-api.onrender.com` (no trailing slash).
5. Deploy. Copy the final Vercel origin and add it to the Render `FRONTEND_ORIGINS` variable. Redeploy/restart the API if prompted.
6. Open the public app and verify API health, preview, a small experiment, a small sweep, selector evaluation, and exports.

The frontend environment variable is embedded at build time. If `VITE_API_BASE` changes, trigger a new Vercel deployment.

## Local development

- Backend: install `backend/requirements.txt`, then run `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` from `backend/`.
- Frontend: run `npm install` then `npm run dev` from `frontend/`.
- The frontend defaults to `http://127.0.0.1:8000` when `VITE_API_BASE` is not set.

## Dependency handling

Dependencies are declared in `frontend/package.json` and `backend/requirements.txt`. Vercel and Render install dependencies during deployment. `node_modules` and virtual environments are intentionally excluded from Git because they are platform-specific, large, and should not be committed. This workspace could not complete `npm install` due to a network timeout, so no verified npm lockfile or local production build is included. Run `npm install` in a network-enabled environment and commit the resulting `frontend/package-lock.json` before relying on fully reproducible frontend builds.
