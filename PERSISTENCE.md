# AutoFlow Model Persistence & Inference

This document explains how AutoFlow stores, selects, and restores trained
models, and how predictions are served. Everything described here is
implemented in `main.py` and covered by `tests/test_main.py`.

## What is stored where

### 1. Complete pipeline artifacts (MLflow)

Each successfully trained candidate is persisted by MLflow
(`mlflow.sklearn.log_model(pipeline, artifact_path="model")`) as **one complete
artifact containing preprocessing + estimator**: median/most-frequent
imputers, optional scaler, one-hot encoder, and the fitted model are all part
of the logged `sklearn.Pipeline`. Inference therefore always reuses the
training preprocessing byte-for-byte (requirement 7 is satisfied structurally:
there is only one pipeline object).

Per-run artifacts additionally include:

- `evaluation/confusion_matrix.json` (classification)
- `inference/inference_schema.json` — the prediction contract
  (`task_type`, `target`, and every feature mapped to `numeric`/`categorical`).

### 2. Selected-model reference (MLflow Model Registry)

When training selects the best successful candidate, its run artifact is
registered in the MLflow Model Registry under
`ACTIVE_MODEL_REGISTRY_NAME = "autoflow-active-models"` (configurable via
`AUTOFLOW_ACTIVE_MODEL_REGISTRY`). The registered model version carries tags
`task_type` and `run_id`. This entry lives in the MLflow backend database
(`mlflow.db`), so it survives restarts and even loss of the state file.

### 3. Application state file

`autoflow_state.json` (path from `AUTOFLOW_DATA_DIR`) stores:

- dataset metadata: `dataset_path` (inside `uploads/` only), `rows`,
  `columns`, profile summaries, SHA-256 `dataset_fingerprint`
- latest training: `task_type`, `target`, `primary_metric`,
  `metric_direction`, `run_group`, `results` (all candidates with metrics,
  status, hyperparameters, confusion matrix, error details), `reliability`
- selection: `best_model` (name + MLflow `run_id`) and
  `active_model` = the persistence reference
  (`registry_name`, `registry_version`, `run_id`, `registered`)
- `feature_schema` — the prediction contract used to validate requests

Writes are atomic (temp file + `os.replace`), and the *selected* reference is
durable in two independent places (registry + state file).

## Restore after restart (`_restore_state`, non-fatal)

1. Read the state file. A corrupt/unsupported file is logged
   (`state_file_unreadable` / `state_file_invalid`) and **ignored instead of
   crashing startup**. A persisted dataset path that escaped `uploads/` or
   disappeared is neutralized (`persisted_dataset_rejected` /
   `persisted_dataset_missing`); the rest of the state is kept.
2. Load the active pipeline through `_load_active_model_reference`, which
   tries trusted sources in order: registered model version
   (`models:/autoflow-active-models/<version>`) first, then the training-run
   artifact (`runs:/<run_id>/model`).
3. If no valid reference survives, fall back to `_latest_registry_reference`
   and rebuild a minimal selection record from the newest `READY` registry
   version, then persist it (`active_model_recovered_from_registry`).
4. If no model can be restored, the app still boots; `/predict` will explain
   why a model is unavailable (see errors below).

## Trusted locations only (requirement 5)

- Model loading accepts **only** `runs:/<run_id>/model` or
  `models:/<registry>/<version>` URIs produced from application-managed
  references. No user input can inject a filesystem path.
- Run references must match `^[0-9a-f]{32}$` and must point to a `FINISHED`
  run of an experiment named `autoflow_run_*`
  (`_valid_application_run`). Anything else is rejected and skipped, with the
  rejection logged (`model_reference_rejected`).
- Model artifacts are only ever written by this application's training
  pipeline; user-supplied model files are never loaded.

## On-demand rebuild (requirement 3)

`inference_pipeline` is only an in-memory cache. `_get_inference_pipeline()`
rebuilds it from the persisted references whenever it is missing — after a
process restart, a cache clear, or a state recovery — so inference does not
depend on globals being warm. Predictions work immediately after a restart
without retraining.

## Prediction contract (requirements 6, 8, 9, 10)

Endpoint: `POST /predict` (`Content-Type: application/json`).

| Field | Meaning |
|---|---|
| `{"features": {...}}` | single row (original contract; response shape unchanged: `prediction`, `model_version`, `status`, plus additive `registry_version`) |
| `{"rows": [{...}, ...]}` | batch prediction (max `MAX_PREDICTION_ROWS = 1000`), response `{"predictions": [...], "count": n, ...}` |

Validation performed per row against `feature_names_in_` of the *trained
pipeline* and the persisted `feature_schema`:

- **required/unknown features** — missing or unexpected feature names fail
  with a structured 422 listing `missing_features` / `unexpected_features`.
- **numeric features** — accept `int`/`float`/numeric strings; reject booleans,
  arrays/objects, NaN/Infinity, and non-numeric strings
  (`invalid_features[].reason`).
- **categorical features** — any scalar, cast to `str` exactly as training saw
  string categories; the one-hot encoder still ignores unseen categories.
- **`null` values** — allowed; the trained imputers handle missing data.

### Error responses

| Situation | Status | Detail |
|---|---|---|
| No dataset uploaded | 409 | "No active dataset is available..." |
| No trained model recorded | 409 | "No active model is available. Train a model first." |
| Model artifact missing/invalid in MLflow store | 503 | "The active model artifact is missing or invalid..." |
| Request body not valid JSON | 400 | message + reason |
| Features missing/unexpected, incompatible type, malformed `rows`, both contracts, >1000 rows | 422 / 413 | structured JSON (see above) |

## Operational notes

- Failed backend storage (e.g. deleted MLflow database + state file) is
  reported honestly; the service never fabricates predictions.
- Registration integrity: if `register_model` fails, training still succeeds
  (run artifact is the fallback reference) and a
  `model_registration_failed` warning is logged — selection is never lost,
  only demoted to one reference path.
- Reloading models uses scikit-learn/skops through MLflow's standard loader;
  the same trust list used at `log_model` time governs it.
