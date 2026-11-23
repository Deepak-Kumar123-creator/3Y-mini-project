# AutoFlow AI — Autonomous ML & MLOps Engine

AutoFlow AI is a self-contained machine-learning operations application:

1. **Upload** a CSV dataset — it is validated, profiled (types, missing values, duplicates), and stored in an application-managed directory.
2. **Task 1 benchmark** — AutoFlow trains a baseline and the configured classification or regression candidate algorithms, evaluates them on a held-out validation split, records every result to MLflow, and selects the measured winner by an explicit primary metric.
3. **Task 2 meta-learning** — AutoFlow extracts dataset meta-features, predicts exactly one algorithm from historical real Task 1 experiments when enough diverse data exists, or labels a cold-start rule-based fallback clearly. It then trains only that one algorithm.
4. **Track & persist** — every run (including failures) is written to MLflow; the complete Task 1 and Task 2 pipelines are registered separately and can be downloaded or selected for prediction.
5. **Compare & predict** — the dashboard displays the measured Task 1 winner and independently trained Task 2 result, then serves predictions from either saved model using its recorded feature contract.

All displayed numbers come from real computation or the real MLflow store — the dashboard never shows fabricated metrics.

## Architecture

```
FastAPI (main.py)                         templates/dashboard.html
├── POST /upload      → validate CSV → profile → save under uploads/<uuid>.csv
│                                        → persist state (autoflow_state.json)
├── POST /train       → benchmark configured candidates and choose by weighted F1 / RMSE
│                        ├ Task 1: baseline + seven classification or six regression models
│                        ├ Task 2: historical meta-model or explicit cold-start fallback
│                        ├ both train on leakage-safe preprocessing and the same split
│                        └ compare results; separately persist both complete pipelines
├── GET  /api/…       → dataset profile, detection, experiments, saved models, comparison
├── POST /predict     → schema-validated inference using the selected trusted model run
├── GET  /api/models/{run_id}/download → download a complete pipeline artifact
└── GET  /            → server-rendered dashboard (Jinja2 + vanilla JS + CSS)

Meta-learning lives in `meta_learning/`; real Task 1 meta-examples are stored in
`meta_data/experiments.json`. A trained meta-model is used only after eight
historical experiments for that problem type and at least two observed winner
algorithms. The current dataset fingerprint is excluded from its own meta-model
training data. The probability output is withheld because it is not calibrated,
and training the meta-classifier alone is not evidence that selection
generalizes to unseen datasets.
```

Persisted per training run (see **PERSISTENCE.md** for the full contract):

- MLflow tracking DB (`mlflow.db`, SQLite by default) — experiments `autoflow_run_*`
- MLflow artifact store (`mlruns/`) — datasets, evaluation artifacts, inference schema, logged models
- MLflow Model Registry — `autoflow-active-models` for Task 1 and `autoflow-meta-models` for Task 2
- State file (`autoflow_state.json`) — dataset reference, selected model, `feature_schema`, reliability notices
- Meta-dataset (`meta_data/experiments.json`) — real dataset-level features and Task 1 winner labels

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

Then open **http://127.0.0.1:8000/** — the dashboard includes Overview, Dataset,
Training, Meta-Learning, Task Comparison, and Predictions. Startup performs state
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

## Dashboard visualizations

The dashboard provides 19 data-driven charts across dataset analysis, Task 1
benchmarking, Task 2 meta-learning, and their comparison. Dataset charts show
the selected target distribution, missing values by column, a selectable
numeric-feature histogram and IQR box plot, and Pearson correlations between
numeric features (the target is excluded; at most 20 features are shown).
Model charts show Task 1 candidate metrics, fit times, cross-validation
stability, and either a validation confusion matrix/ROC curve or regression
actual-versus-predicted and residual plots. Model-specific evaluation charts
use a maximum of 500 actual held-out validation observations; ROC curves use
the estimator's real probabilities or decision scores.

Meta-learning charts include Random Forest feature importance and raw
classifier probabilities when the meta-model is trained, historical Task 1
winners counted once per dataset fingerprint, and leave-one-dataset-out
reliability. Probabilities are explicitly uncalibrated and are not confidence
or reliability estimates. Reliability is withheld until at least 10 distinct
historical datasets are available. The comparison charts use the same held-out
split for Task 1 and Task 2.

`GET /api/visualizations` returns the source data for these charts. Optional
query parameters are `target`, `task_type`, `feature`, `bins`, and
`model_run_id`. Histogram bin counts are bounded to 5–60. Changing the selected
target or task type does not reuse model results from a different training
configuration.

Task 1 total computation time is the sum of candidate fit, validation,
cross-validation, and evaluation-chart calculation time. Task 2 total
computation time includes meta-feature/history selection and its one model's
fit, validation, and evaluation-chart calculation. Both exclude MLflow
logging/artifact persistence; individual model-fit times are shown separately.
These timing definitions are descriptive and should not be interpreted as a
hardware-independent speed guarantee.

## MLflow tracking & model persistence

- Every training creates an experiment named `autoflow_run_…` and one MLflow run
  per candidate with: task type, dataset fingerprint, model name, hyperparameters,
  metrics, run status, and — for failed candidates — `error_type`/`error_message` tags.
- Run artifacts include the validation dataset, bounded held-out
  `evaluation/evaluation_data.json`, `evaluation/confusion_matrix.json`
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
| GET | `/api/visualizations?target=NAME&task_type=TYPE&feature=NAME&bins=N&model_run_id=ID` | Real dataset, validation, comparison, and meta-learning chart data | 409 no dataset · 422 invalid target/task/data |
| GET | `/api/problem-detection?target=NAME` | Heuristic classification/regression suggestion | 409 no dataset |
| GET | `/api/overview` | Computed dashboard counts and recent activity | — |
| GET | `/api/experiments?limit=N` | Real MLflow run history (N ≤ 100) | 503 tracking store unavailable |
| GET | `/api/comparison` | Latest Task 1 / Task 2 metric and time comparison | 409 no training |
| GET | `/api/models` | Available trained Task 1 and Task 2 models | — |
| GET | `/api/models/{run_id}/download` | Download a trusted complete pipeline as Joblib | 404 unknown run · 503 unavailable artifact |
| GET | `/api/model/schema` | Active model's feature contract | 409 no model |
| POST | `/predict` | Single-row/batch inference; optional `model_run_id` selects a saved run | 422 validation · 409 no model/no data · 413 > 1000 rows · 503 invalid/missing artifact |

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
limits), Task 1/Task 2 training, separate registry persistence, selected-model
prediction and artifact download, prediction
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
