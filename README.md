# AutoFlow AI — Autonomous ML & MLOps Engine

AutoFlow AI is a self-contained machine-learning operations application:

1. **Upload** a CSV dataset — it is validated, profiled (types, missing values, duplicates), and stored in an application-managed directory.
2. **Train** a classification or regression pipeline — AutoFlow trains three candidates (a dummy baseline, a linear model, and a Random Forest), evaluates them on a held-out validation split, records everything to MLflow, and selects the best candidate by an explicit primary metric.
3. **Track & persist** — every run (including failures) is written to an MLflow tracking store with metrics, parameters, and artifacts; the winning model is persisted as a complete artifact and registered in the MLflow Model Registry, with a state file that survives application restarts.
4. **Predict** — the dashboard auto-generates input fields from the active model's trained feature contract, validates payloads against it, and serves single-row and batch (up to 1 000 rows) predictions. Models are always loaded from trusted, application-managed locations only.

All displayed numbers come from real computation or the real MLflow store — the dashboard never shows fabricated metrics.

## Architecture

```
FastAPI (main.py, single module)          templates/dashboard.html
├── POST /upload      → validate CSV → profile → save under uploads/<uuid>.csv
│                                        → persist state (autoflow_state.json)
├── POST /train       → 3 candidates (baseline / linear / Random Forest)
│                        ├ preprocessing per candidate (impute → encode/scale)
│                        ├ 80/20 validation split (stratified for classification)
│                        ├ metrics (Accuracy/Precision/Recall/F1/confusion
│                        │           or MAE/RMSE/R²) + reliability warnings
│                        ├ MLflow run per candidate (params, metrics, artifacts,
│                        │  model logging; failures become FAILED runs with details)
│                        └ selection by F1 (maximize) / RMSE (minimize)
│                        → registration in MLflow Model Registry + state update
├── GET  /api/…       → dataset preview, MLflow experiment history, model schema
├── POST /predict     → schema-validated single-row or batch inference
└── GET  /            → server-rendered dashboard (Jinja2 + vanilla JS + CSS)
```

Persisted per training run (see **PERSISTENCE.md** for the full contract):

- MLflow tracking DB (`mlflow.db`, SQLite by default) — experiments `autoflow_run_*`
- MLflow artifact store (`mlruns/`) — datasets, evaluation artifacts, inference schema, logged models
- MLflow Model Registry — `autoflow-active-models` (configurable), one READY version per successful training
- State file (`autoflow_state.json`) — dataset reference, selected model, `feature_schema`, reliability notices

## Prerequisites

- **Windows 10/11** with **PowerShell 5.1+** (examples below are PowerShell)
- **Python 3.12** (the project is developed and tested on 3.12.13)
- Optional: **Docker Desktop** (or a Linux Docker engine) for containerized deployment

## Local setup (Windows PowerShell)

```powershell
# 1. Go to the project root (the directory that contains main.py)
cd C:\Users\parth\Downloads\autoflow_project\autoflow_project

# 2. Create an isolated virtual environment
python -m venv .venv          # or: py -3.12 -m venv .venv

# 3. Install the pinned dependencies
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt

# 4. (Optional) copy the configuration template and adjust it
Copy-Item .env.example .env   # then edit .env in a text editor
```

The pinned versions (`requirements.txt`) are verified compatible with Python 3.12:

| Package | Version | Role |
|---|---|---|
| fastapi | 0.162.0 | HTTP API + routing |
| uvicorn | 0.54.0 | ASGI server |
| pandas | 3.1.0 | CSV ingestion, profiling, training data |
| numpy | 2.5.4 | numeric core |
| scikit-learn | 1.9.1 | preprocessing + models + metrics |
| mlflow | 3.17.2 | tracking, artifacts, Model Registry, model persistence |
| jinja2 | 3.1.6 | dashboard templates |
| python-multipart | 0.0.32 | multipart file uploads |
| skops | 0.16.0 | trusted model serialization/loading |

## Start the application (development)

```powershell
cd C:\Users\parth\Downloads\autoflow_project\autoflow_project
& ".venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/** — the dashboard includes Overview, Dataset
management, Model training, Experiments, and Predictions. Startup performs state
recovery: if a state file and/or registered model exists, the active model is
re-validated and reloaded from trusted locations (`autoflow.db` persists
state; MLflow registry is the fallback when the state file is missing or damaged).

## Dataset upload & training

1. In the dashboard's **Dataset management** section, select a UTF-8 CSV
   (≤ 10 MiB, unique non-empty headers, at least one data row) and upload it.
   The server rejects anything else with a clear error (see error table below)
   and stores the file under a server-generated name inside `uploads/`.
2. Inspect the **preview table** and **column summary** (types + missing values
   for the full dataset).
3. In **Model training**, pick the **target column** (populated from the dataset)
   and the **task type**:
   - *classification* — best candidate selected by **F1 (maximize)**; stratified
     split when the class distribution allows it
   - *regression* — best candidate selected by **RMSE (minimize)**
4. Press **Train models**. The comparison table fills with each evaluated
   candidate, its validation metrics, hyperparameters, run ID, and — for
   failures — the error type and message. Reliability notices (e.g. validation
   split < 30 rows, degenerate class distribution, undefined R²) are shown
   explicitly.

Training never evaluates on training data and reports failed candidates as
failures; if every candidate fails, the API returns HTTP 422 with per-candidate
error details and nothing is marked as successful.

## Prediction workflow

After a successful training, the **Predictions** section generates one input
field per required feature (numeric fields add an optional *mark as missing*
toggle, letting the trained imputer fill the value).

```powershell
# single row
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"features": {"measurement": 10.0, "load_index": 30.0, "channel_type": "alpha"}}'

# batch (max 1 000 rows; null means "missing value")
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body '{"rows": [{"measurement": 10.0, "load_index": 30.0, "channel_type": "alpha"},
                   {"measurement": null,  "load_index": 30.0, "channel_type": "gamma"}]}'
```

Responses distinguish a successful prediction (`prediction`, `model_version`,
`registry_version`) from structured validation errors
(`missing_features`, `unexpected_features`, `invalid_features` with row indices).

## MLflow tracking & model persistence

- Every training creates an experiment named `autoflow_run_…` and one MLflow run
  per candidate with: task type, dataset fingerprint, model name, hyperparameters,
  metrics, run status, and — for failed candidates — `error_type`/`error_message` tags.
- Run artifacts include the validation dataset, `evaluation/confusion_matrix.json`
  (classification), `inference/inference_schema.json`, and the complete logged model.
- The selected model is additionally registered in the MLflow Model Registry
  under `autoflow-active-models` (each success creates a new READY version) and
  referenced from the state file, so inference keeps working after restarts.
- On startup the application restores its state, validates the persisted model
  reference (32-hex run ID, existing FINISHED run, application-owned experiment),
  and falls back to the newest READY registry version when needed.
- MLflow's backend schema is initialized/repaired at startup, so a fresh SQLite
  database — or one damaged by an interrupted migration — self-heals instead of
  returning 503s.

Browse the runs with the MLflow UI if desired:

```powershell
& ".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri "sqlite:///./mlflow.db"
```

## API endpoints

| Method | Path | Purpose | Key errors |
|---|---|---|---|
| GET | `/` | Dashboard (HTML) | — |
| GET | `/health` | Liveness probe | — |
| POST | `/upload` | Validate, profile, store CSV | 400 invalid CSV · 413 > 10 MiB · 415 not a CSV |
| POST | `/train` | Train + select + register | 409 no dataset · 422 invalid request/all candidates failed |
| GET | `/api/dataset/preview?rows=N` | Real preview + column summary (N ≤ 50) | 409 no dataset |
| GET | `/api/experiments?limit=N` | Real MLflow run history (N ≤ 100) | 503 tracking store unavailable |
| GET | `/api/model/schema` | Active model's feature contract | 409 no model |
| POST | `/predict` | Single-row/batch inference | 422 validation · 409 no model/no data · 413 > 1000 rows · 503 invalid/missing artifact |

More details, together with the storage layout, trust rules, and recovery order,
are documented in **PERSISTENCE.md**.

## Tests

From the project root (the directory containing `main.py`):

```powershell
# full suite (~10–20 minutes: exercises real MLflow persistence end to end)
& ".venv\Scripts\python.exe" -m unittest tests.test_main

# quick feedback on one class
& ".venv\Scripts\python.exe" -m unittest tests.test_main.MetricComputationTests -v
& ".venv\Scripts\python.exe" -m unittest tests.test_main.StartupAndRouteTests -v
```

Covered: upload validation & security (path traversal, extension, size, empty,
malformed), dataset profiling, classification/regression training + metric
correctness, MLflow experiment tracking (including FAILED run reporting and
limits), model artifact persistence & registry registration, prediction
validation (types, missing/unexpected features, batch, row cap), validation
error contracts, restart recovery (state restore, registry fallback, tamper
rejection, broken artifacts), dashboard rendering (populated & empty states),
and application startup (fresh backend init, interrupted-migration recovery,
route inventory, `/health`).

> On Windows the test suite intentionally uses process-isolated temporary data
> directories and disposes SQLAlchemy engines afterwards to avoid file-lock
> cleanup errors; a PowerShell "NativeCommandError" banner around MLflow stderr
> log lines is harmless — trust the final `Ran N tests … OK` line.

## Docker setup

The image runs as non-root user **UID 10001**, ships a `/health` healthcheck,
and separates code (baked into the image) from data (persistent volumes:

- `/app/data` — uploads, `autoflow_state.json`, MLflow tracking database
- `/app/mlruns` — MLflow artifact store (logged models, run artifacts)

```powershell
cd C:\Users\parth\Downloads\autoflow_project\autoflow_project

# build and run with named volumes (recommended)
docker compose up -d --build

# …or without compose:
docker build -t autoflow-app .
docker volume create autoflow-data
docker volume create autoflow-mlruns
docker run -d --name autoflow -p 8000:8000 `
  -v autoflow-data:/app/data -v autoflow-mlruns:/app/mlruns `
  autoflow-app

# verify
docker ps
curl.exe http://localhost:8000/health
docker logs autoflow
```

If you prefer **bind mounts** on Windows, Docker Desktop maps them; inside the
container everything is owned by UID 10001 (the `chown` step in the Dockerfile
prepares the volume roots for anonymous/named volumes automatically).

## Known limitations

- **No authentication/HTTPS** — bind uvicorn to 127.0.0.1 locally on shared
  machines behind a reverse proxy for anything internet-facing.
- **First startup initializes the MLflow schema** — on a fresh data directory
  the first boot runs MLflow's Alembic migrations once (typically seconds;
  noticeably longer on a busy machine). Subsequent restarts are fast.
- **SQLite tracking store** — fine for this app's traffic; for concurrent
  production services point `MLFLOW_TRACKING_URI` at an MLflow server with a
  robust backend.
- **One training at a time** — the application state is process-global; start
  additional containers/processes on separate data directories to parallelize.
- **Batch predictions capped at 1 000 rows** per request by design.
- **Uploads are capped at 10 MiB**; very large datasets must be sampled first.
- R² is reported as undefined (not fabricated) for degenerate/constant targets,
  and selection notices surface whenever the evaluation set is too small to be
  statistically trustworthy.
- If `docker compose`'s named volume is recreated, state is empty and the
  dashboard starts at the empty state; MLflow history lives in the volume.
