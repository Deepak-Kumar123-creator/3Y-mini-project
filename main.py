import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("autoflow")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("AUTOFLOW_DATA_DIR", PROJECT_DIR)).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
MLFLOW_ARTIFACT_DIR = Path(
    os.environ.get("MLFLOW_ARTIFACT_DIR", str(DATA_DIR / "mlruns"))
).resolve()
STATE_FILE = DATA_DIR / "autoflow_state.json"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
MIN_TRAINING_ROWS = 10
STATE_VERSION = 1
# Models are generated locally; user-provided model artifacts are never loaded.
SKOPS_TRUSTED_MODEL_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]

# Primary model-selection metrics per task type. The direction states explicitly
# whether the metric must be maximized or minimized when choosing the best run.
PRIMARY_METRICS = {
    "classification": {"metric": "F1", "direction": "maximize"},
    "regression": {"metric": "RMSE", "direction": "minimize"},
}
# Validation splits smaller than this make metric estimates unstable.
MIN_RELIABLE_VALIDATION_ROWS = 30
# The selected model's complete artifact is also registered in the MLflow Model
# Registry under this name, in addition to the persisted state file.
ACTIVE_MODEL_REGISTRY_NAME = os.environ.get(
    "AUTOFLOW_ACTIVE_MODEL_REGISTRY", "autoflow-active-models"
)
# MLflow run IDs are always 32 lowercase hex characters. Model loading only
# accepts run:/ or models:/ URIs whose identifiers pass this validation, so no
# arbitrary filesystem path from user input is ever used.
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_PREDICTION_ROWS = 1000

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

default_tracking_uri = f"sqlite:///{(DATA_DIR / 'mlflow.db').as_posix()}"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_tracking_uri))

app = FastAPI(title="AutoFlow AI - Autonomous ML & MLOps Engine")
templates = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))

STATE: dict[str, Any] = {
    "dataset_path": None,
    "filename": None,
    "rows": 0,
    "columns_count": 0,
    "columns": [],
    "duplicate_rows": 0,
    "numerical_summary": {},
    "categorical_summary": {},
    "dataset_fingerprint": None,
    "target": None,
    "task_type": None,
    "primary_metric": None,
    "metric_direction": None,
    "run_group": None,
    "results": [],
    "best_model": None,
    "reliability": None,
    "feature_schema": None,
    "active_model": None,
}

inference_pipeline: Any = None


def _log_event(level: int, event: str, **fields: Any) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, sort_keys=True))


def _prepare_tracking_backend() -> None:
    """Create or reconcile the MLflow schema once before serving requests.

    On SQLite databases an interrupted schema migration (for example after a
    hard process kill) can leave Alembic temp tables behind; every later store
    initialization then fails with 'table _alembic_tmp_... already exists'.
    Startup therefore removes orphaned migration temp tables, initializes the
    schema, and retries once in case a concurrent migration raced the first
    pass.
    """
    database_uri = mlflow.get_tracking_uri()
    if not database_uri.startswith("sqlite"):
        return
    try:
        from mlflow.store.db.utils import (
            _safe_initialize_tables,
            create_sqlalchemy_engine_with_retry,
        )
        from sqlalchemy import inspect as sqlalchemy_inspect

        def drop_orphan_migration_tables(engine: Any) -> None:
            orphaned_tables = [
                table_name
                for table_name in sqlalchemy_inspect(engine).get_table_names()
                if table_name.startswith("_alembic_tmp_")
            ]
            if not orphaned_tables:
                return
            with engine.begin() as connection:
                for table_name in orphaned_tables:
                    connection.exec_driver_sql(f'DROP TABLE "{table_name}"')
            _log_event(
                logging.WARNING,
                "mlflow_orphan_migration_tables_dropped",
                tables=orphaned_tables,
            )

        engine = create_sqlalchemy_engine_with_retry(database_uri)
        try:
            drop_orphan_migration_tables(engine)
            _safe_initialize_tables(engine)
        except Exception:
            drop_orphan_migration_tables(engine)
            _safe_initialize_tables(engine)
        finally:
            try:
                engine.dispose()
            except Exception:
                pass
    except Exception as exc:
        _log_event(
            logging.WARNING,
            "tracking_backend_preparation_skipped",
            reason=type(exc).__name__,
            error=str(exc)[:300],
        )


_prepare_tracking_backend()


def _save_state(state: dict[str, Any]) -> None:
    temporary_path = STATE_FILE.with_name(f".{STATE_FILE.name}.{uuid.uuid4().hex}.tmp")
    with temporary_path.open("x", encoding="utf-8") as state_file:
        json.dump({"version": STATE_VERSION, "state": state}, state_file, allow_nan=False)
        state_file.flush()
        os.fsync(state_file.fileno())
    os.replace(temporary_path, STATE_FILE)


def _tracking_client() -> MlflowClient:
    """Client pinned to the application's own MLflow tracking/registry store."""
    return MlflowClient(tracking_uri=mlflow.get_tracking_uri())


def _valid_application_run(run_id: str) -> Any:
    """Return the MLflow run behind a training-run reference, or raise ValueError.

    This is the trusted-location gate for model loading: only identifiers that
    match an MLflow run ID pattern, exist in the application-managed MLflow
    store, belong to an AutoFlow training experiment, and reached FINISHED
    status are accepted. Arbitrary filesystem paths can never pass here, and
    users never supply this identifier directly.
    """
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Run reference is not a valid MLflow run ID: {run_id!r}")
    client = _tracking_client()
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    experiment_name = getattr(experiment, "name", None)
    if not isinstance(experiment_name, str) or not experiment_name.startswith("autoflow_run_"):
        raise ValueError(f"Run {run_id} does not belong to an AutoFlow training experiment.")
    if run.info.status != "FINISHED":
        raise ValueError(f"Run {run_id} did not finish successfully.")
    return run


def _model_artifact_sources(best_model: Any, active_model: Any) -> list[tuple[str, str]]:
    """Build the ordered list of trusted artifact locations for the active model.

    Every entry is a URI inside the application's own MLflow store: either a
    registered model version (models:/name/version) or a training run artifact
    (runs:/<run_id>/model). No plain filesystem paths are ever produced here.
    """
    sources: list[tuple[str, str]] = []
    seen_run_ids: set[str] = set()
    if isinstance(active_model, dict):
        registry_name = active_model.get("registry_name")
        registry_version = active_model.get("registry_version")
        if isinstance(registry_name, str) and registry_name and registry_version is not None:
            try:
                normalized_version = int(registry_version)
            except (TypeError, ValueError):
                normalized_version = None
            if normalized_version is not None:
                sources.append(
                    (
                        "registered_model_version",
                        f"models:/{registry_name}/{normalized_version}",
                    )
                )
        run_id = active_model.get("run_id")
        if isinstance(run_id, str):
            seen_run_ids.add(run_id)
    for run_id in (
        run_id
        for run_id in (
            best_model.get("run_id") if isinstance(best_model, dict) else None,
            active_model.get("run_id") if isinstance(active_model, dict) else None,
        )
        if isinstance(run_id, str)
    ):
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        try:
            _valid_application_run(run_id)
        except Exception as exc:
            _log_event(
                logging.WARNING,
                "model_reference_rejected",
                run_id=run_id,
                reason=type(exc).__name__,
                error=str(exc)[:300],
            )
            continue
        sources.append(("training_run", f"runs:/{run_id}/model"))
    return sources


def _latest_registry_reference() -> dict[str, Any] | None:
    """Find the newest READY model version registered by training runs.

    Used when the persisted state file is missing or unreadable, and no valid
    run reference survives: the MLflow backend database keeps the authoritative
    record of the last successfully selected model.
    """
    try:
        client = _tracking_client()
        versions = client.search_model_versions(
            f"name = '{ACTIVE_MODEL_REGISTRY_NAME}'"
        )
        ready_versions = [
            model_version
            for model_version in versions
            if getattr(model_version, "status", None) == "READY"
        ]
        if not ready_versions:
            return None
        selected = max(ready_versions, key=lambda version: int(version.version))
        complete = client.get_model_version(
            ACTIVE_MODEL_REGISTRY_NAME, selected.version
        )
        tags = complete.tags or {}
        return {
            "registry_name": ACTIVE_MODEL_REGISTRY_NAME,
            "registry_version": int(complete.version),
            "run_id": str(complete.run_id),
            "task_type": tags.get("task_type"),
            "primary_metric": tags.get("primary_metric"),
            "run_group": tags.get("run_group"),
        }
    except Exception as exc:
        _log_event(
            logging.WARNING,
            "registry_recovery_unavailable",
            reason=type(exc).__name__,
            error=str(exc)[:300],
        )
        return None


def _load_active_model_reference(
    best_model: Any, active_model: Any
) -> tuple[Any, str]:
    """Load the active inference pipeline from the first valid trusted source.

    Returns the sklearn pipeline (preprocessing plus model, exactly as trained)
    and a label of the source used. Raises RuntimeError when no trusted source
    could produce a loadable artifact.
    """
    problems: list[str] = []
    for source, model_uri in _model_artifact_sources(best_model, active_model):
        try:
            pipeline = mlflow.sklearn.load_model(model_uri)
        except Exception as exc:
            problems.append(f"{source} ({model_uri}): {type(exc).__name__}: {exc}"[:300])
            continue
        return pipeline, source
    detail = "; ".join(problems) if problems else "no valid model reference is available"
    raise RuntimeError(f"Active model artifact is missing or invalid ({detail}).")


def _persist_registry_reference(
    registry_name: str, model_version: int, run_id: str, task_type: str
) -> None:
    """Record selection metadata as tags on the registered model version."""
    client = _tracking_client()
    for key, value in {
        "task_type": task_type,
        "run_id": run_id,
    }.items():
        client.set_model_version_tag(registry_name, str(model_version), key, value)


def _register_selected_model(best_model: dict[str, Any], task_type: str) -> dict[str, Any]:
    """Register the selected run's complete pipeline in the MLflow Model Registry.

    The registry entry (kept in the MLflow backend database) is the durable
    reference to the trusted artifact store location, so the application can
    recover the selected model even if the state file is lost. When
    registration is not possible, the run artifact URI remains the fallback
    reference.
    """
    run_id = best_model["run_id"]
    try:
        model_version = mlflow.register_model(f"runs:/{run_id}/model", ACTIVE_MODEL_REGISTRY_NAME)
        _persist_registry_reference(
            ACTIVE_MODEL_REGISTRY_NAME,
            int(model_version.version),
            run_id,
            task_type,
        )
        _log_event(
            logging.INFO,
            "selected_model_registered",
            registry_name=ACTIVE_MODEL_REGISTRY_NAME,
            registry_version=int(model_version.version),
            run_id=run_id,
        )
        return {
            "registry_name": ACTIVE_MODEL_REGISTRY_NAME,
            "registry_version": int(model_version.version),
            "run_id": run_id,
            "task_type": task_type,
            "registered": True,
        }
    except Exception as exc:
        _log_event(
            logging.WARNING,
            "model_registration_failed",
            run_id=run_id,
            reason=type(exc).__name__,
            error=str(exc)[:300],
        )
        return {"registry_name": None, "registry_version": None, "run_id": run_id,
                "task_type": task_type, "registered": False}


def _restore_state() -> None:
    """Restore application state and the active model after a restart.

    Recovery order:
      1. Read and validate the persisted state file (dataset paths must stay
         inside the upload directory; invalid state files are ignored rather
         than crashing startup).
      2. Load the active model through trusted MLflow references (registered
         model version first, then the training-run artifact).
      3. If neither the state file nor the references are usable, fall back to
         the newest registered model version recorded in the MLflow backend
         database, rebuild a minimal selection record, and persist it.
    """
    global inference_pipeline

    restored_state: dict[str, Any] | None = None
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open(encoding="utf-8") as state_file:
                stored = json.load(state_file)
            if (
                isinstance(stored, dict)
                and stored.get("version") == STATE_VERSION
                and isinstance(stored.get("state"), dict)
            ):
                restored_state = stored["state"]
                dataset_path = restored_state.get("dataset_path")
                if dataset_path is not None:
                    resolved_dataset = Path(dataset_path).resolve()
                    if not resolved_dataset.is_relative_to(UPLOAD_DIR.resolve()):
                        _log_event(
                            logging.WARNING,
                            "persisted_dataset_rejected",
                            reason="dataset path is outside the upload directory",
                        )
                        restored_state["dataset_path"] = None
                    elif not resolved_dataset.is_file():
                        _log_event(
                            logging.WARNING,
                            "persisted_dataset_missing",
                            reason="dataset file is no longer present",
                        )
                        restored_state["dataset_path"] = None
            else:
                _log_event(
                    logging.WARNING,
                    "state_file_invalid",
                    reason="unsupported state format; ignoring persisted state",
                )
        except (OSError, ValueError) as exc:
            _log_event(
                logging.WARNING,
                "state_file_unreadable",
                reason=type(exc).__name__,
                error=str(exc)[:300],
            )

    STATE.update(restored_state or {})

    active_model = (
        STATE.get("active_model") if isinstance(STATE.get("active_model"), dict) else None
    )
    if (
        active_model is None
        and isinstance(STATE.get("best_model"), dict)
        and isinstance(STATE["best_model"].get("run_id"), str)
    ):
        # Legacy state files selected models by run ID only.
        active_model = {
            "registry_name": None,
            "registry_version": None,
            "run_id": STATE["best_model"]["run_id"],
            "task_type": STATE.get("task_type"),
            "registered": False,
        }
        STATE["active_model"] = active_model

    if (
        restored_state is None
        and _model_artifact_sources(STATE.get("best_model"), active_model) == []
    ):
        registry_reference = _latest_registry_reference()
        if registry_reference is not None:
            STATE.update(
                {
                    "best_model": {
                        "model": f"{registry_reference['registry_name']} v{registry_reference['registry_version']}",
                        "status": "success",
                        "primary": None,
                        "run_id": registry_reference["run_id"],
                    },
                    "active_model": registry_reference,
                    "task_type": registry_reference["task_type"],
                    "primary_metric": registry_reference["primary_metric"],
                }
            )
            active_model = registry_reference
            try:
                _save_state(dict(STATE))
            except (OSError, TypeError, ValueError) as exc:
                _log_event(
                    logging.WARNING,
                    "recovered_state_not_persisted",
                    reason=type(exc).__name__,
                )
            _log_event(
                logging.INFO,
                "active_model_recovered_from_registry",
                registry_version=registry_reference["registry_version"],
                run_id=registry_reference["run_id"],
            )

    try:
        fitted_pipeline, source = _load_active_model_reference(
            STATE.get("best_model"), active_model
        )
    except RuntimeError as exc:
        _log_event(logging.WARNING, "active_model_unavailable", error=str(exc)[:500])
        return

    inference_pipeline = fitted_pipeline
    _log_event(logging.INFO, "active_model_restored", source=source)


def _get_inference_pipeline() -> Any:
    """Return the active inference pipeline, rebuilding it on demand.

    The module-level cache is only an optimization; the persisted references in
    the state file and the MLflow registry remain the source of truth, so any
    request can rebuild the pipeline after a restart or cache loss.
    """
    global inference_pipeline
    if inference_pipeline is not None:
        return inference_pipeline

    best_model = STATE.get("best_model")
    active_model = (
        STATE.get("active_model") if isinstance(STATE.get("active_model"), dict) else None
    )
    if not isinstance(best_model, dict) and active_model is None:
        raise HTTPException(
            status_code=409,
            detail="No active model is available. Train a model first.",
        )

    try:
        fitted_pipeline, source = _load_active_model_reference(best_model, active_model)
    except RuntimeError as exc:
        _log_event(
            logging.WARNING,
            "active_model_load_failed",
            error=str(exc)[:500],
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "The active model artifact is missing or invalid in the application's"
                " MLflow store. Train a new model or restore the MLflow backend."
            ),
        ) from exc

    inference_pipeline = fitted_pipeline
    _log_event(logging.INFO, "active_model_loaded_on_demand", source=source)
    return inference_pipeline


def _parse_csv(contents: bytes) -> pd.DataFrame:
    try:
        text = contents.decode("utf-8-sig")
        records = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(records, None)
        if not header:
            raise HTTPException(status_code=400, detail="The CSV file is empty or has no header row.")
        if any(not column.strip() for column in header):
            raise HTTPException(status_code=400, detail="CSV column headers must not be empty.")

        normalized_headers = [column.strip().casefold() for column in header]
        if len(normalized_headers) != len(set(normalized_headers)):
            raise HTTPException(status_code=400, detail="CSV column headers must be unique.")

        for row_number, record in enumerate(records, start=2):
            if record and len(record) != len(header):
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV row {row_number} does not match the header column count.",
                )

        dataframe = pd.read_csv(io.BytesIO(contents), on_bad_lines="error")
    except HTTPException:
        raise
    except (UnicodeDecodeError, csv.Error, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        _log_event(logging.WARNING, "csv_rejected", reason=type(exc).__name__)
        raise HTTPException(
            status_code=400,
            detail="The file is not a valid UTF-8 CSV with consistent rows.",
        ) from exc

    if dataframe.empty:
        raise HTTPException(status_code=400, detail="The CSV file must contain at least one data row.")
    if len(dataframe.columns) == 0:
        raise HTTPException(status_code=400, detail="The CSV file must contain at least one column.")
    return dataframe


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        missing_percentage = (missing_count / len(dataframe) * 100) if len(dataframe) else 0.0
        columns.append(
            {
                "name": str(column),
                "type": "numeric" if pd.api.types.is_numeric_dtype(dataframe[column]) else "categorical",
                "data_type": str(dataframe[column].dtype),
                "missing": f"{missing_percentage:.1f}% ({missing_count})",
                "missing_count": missing_count,
                "missing_percentage": round(missing_percentage, 2),
            }
        )

    numerical_summary: dict[str, dict[str, Any]] = {}
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        statistics = dataframe[column].describe()
        summary: dict[str, Any] = {}
        for statistic, value in statistics.items():
            numeric_value = float(value)
            summary[str(statistic)] = (
                int(numeric_value)
                if statistic == "count"
                else numeric_value if math.isfinite(numeric_value) else None
            )
        numerical_summary[str(column)] = summary

    categorical_summary: dict[str, dict[str, Any]] = {}
    categorical_columns = [column for column in dataframe.columns if column not in numeric_columns]
    for column in categorical_columns:
        values = dataframe[column].dropna()
        counts = values.value_counts()
        categorical_summary[str(column)] = {
            "count": int(values.size),
            "unique": int(values.nunique()),
            "top": str(counts.index[0]) if not counts.empty else None,
            "frequency": int(counts.iloc[0]) if not counts.empty else 0,
        }

    return {
        "rows": int(len(dataframe)),
        "columns_count": int(len(dataframe.columns)),
        "columns": columns,
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary,
    }


def _training_data(
    dataframe: pd.DataFrame,
    target: str,
    task_type: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if task_type not in {"classification", "regression"}:
        raise HTTPException(
            status_code=422,
            detail="task_type must be either 'classification' or 'regression'.",
        )
    if not target or target not in dataframe.columns:
        raise HTTPException(status_code=422, detail=f"Target column '{target}' was not found in the dataset.")
    if len(dataframe) < MIN_TRAINING_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"At least {MIN_TRAINING_ROWS} rows are required for a reliable train/validation split.",
        )

    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    y = dataframe[target]
    if y.isna().any():
        raise HTTPException(status_code=422, detail="The target column contains missing or non-finite values.")

    X = dataframe.drop(columns=[target]).dropna(axis=1, how="all")
    if X.empty or not len(X.columns):
        raise HTTPException(status_code=422, detail="At least one non-empty feature column is required.")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    if task_type == "classification":
        class_counts = y.value_counts()
        class_count = len(class_counts)
        if class_count < 2:
            raise HTTPException(status_code=422, detail="Classification requires at least two target classes.")
        if int(class_counts.min()) < 2:
            raise HTTPException(
                status_code=422,
                detail="Each classification target class must contain at least two rows.",
            )
        validation_rows = math.ceil(len(y) * 0.2)
        if validation_rows < class_count or len(y) - validation_rows < class_count:
            raise HTTPException(
                status_code=422,
                detail="There are too many target classes for a stratified validation split.",
            )
    else:
        if not pd.api.types.is_numeric_dtype(y):
            raise HTTPException(status_code=422, detail="Regression requires a numeric target column.")
        if y.nunique() < 2:
            raise HTTPException(status_code=422, detail="Regression requires at least two distinct target values.")

    return X, y, numeric_features, categorical_features


def _dataset_fingerprint(path: Path) -> str:
    """SHA-256 fingerprint of the dataset file contents.

    The raw data is never logged; only this safe identifier is recorded in
    MLflow so runs can be traced back to an exact dataset without exposing it.
    """
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_models(task_type: str) -> tuple[dict[str, Any], ...]:
    """Build the candidate models to train for a task type.

    Every entry carries the model, its role (a Dummy baseline for sanity
    comparison, a linear reference model, and a Random Forest), and a flag
    telling whether numerical features should be scaled: linear models benefit
    from scaling while tree ensembles do not need it.
    """
    if task_type == "classification":
        return (
            {
                "name": "Majority-Class Baseline",
                "model": DummyClassifier(strategy="most_frequent"),
                "role": "baseline",
                "scale_numerical": False,
            },
            {
                "name": "Logistic Regression",
                "model": LogisticRegression(max_iter=1000),
                "role": "linear",
                "scale_numerical": True,
            },
            {
                "name": "Random Forest Classifier",
                "model": RandomForestClassifier(n_estimators=100, random_state=42),
                "role": "forest",
                "scale_numerical": False,
            },
        )
    return (
        {
            "name": "Median Baseline",
            "model": DummyRegressor(strategy="median"),
            "role": "baseline",
            "scale_numerical": False,
        },
        {
            "name": "Linear Regression",
            "model": LinearRegression(),
            "role": "linear",
            "scale_numerical": True,
        },
        {
            "name": "Random Forest Regressor",
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "role": "forest",
            "scale_numerical": False,
        },
    )


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numerical: bool,
) -> ColumnTransformer:
    """Create a fresh preprocessing pipeline for one candidate model.

    Imputation, one-hot encoding, and optional scaling live inside the model
    pipeline itself, and the pipeline is fitted only on the training split. The
    fitted transformers are then reused to transform the validation split, so no
    statistics ever leak from validation data into preprocessing. OneHotEncoder
    handles unseen categories safely via handle_unknown='ignore'.
    """
    numerical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numerical:
        numerical_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numerical_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent", keep_empty_features=True
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _json_safe(value: Any) -> Any:
    """Convert arbitrary values into JSON-serializable equivalents."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _model_hyperparameters(model: Any) -> dict[str, Any]:
    """Snapshot the declared hyperparameters of a candidate model."""
    try:
        return {
            str(key): _json_safe(value)
            for key, value in sorted(model.get_params(deep=False).items())
        }
    except Exception as exc:
        _log_event(logging.WARNING, "hyperparameters_unavailable", reason=type(exc).__name__)
        return {}


def _classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Validation classification metrics.

    average='weighted' with zero_division=0 keeps precision, recall, and F1
    defined for every prediction outcome, including classes that are never
    predicted or never observed in the validation split.
    """
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Macro F1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _regression_metrics(
    y_true: Any, y_pred: Any, warnings_sink: list[str]
) -> dict[str, float | None]:
    """Validation regression metrics measured on unseen data.

    R2 is only reported where it is valid: scikit-learn returns 0.0 for a
    constant validation target and NaN for a single validation row, which would
    otherwise look like real model skill. In those cases R2 is None and a
    warning is raised.
    """
    metrics: dict[str, float | None] = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": None,
    }
    y_values = np.asarray(y_true, dtype=np.float64)
    if len(y_values) < 2 or np.var(y_values) == 0.0:
        warnings_sink.append(
            "R2 is not reported: the validation target is constant (or has a single row), so R2 is undefined."
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metric_value = float(r2_score(y_true, y_pred))
        metrics["R2"] = metric_value if math.isfinite(metric_value) else None
    return metrics


def _confusion_matrix_record(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Confusion matrix over every class observed in either set."""
    labels = sorted(set(y_true) | set(y_pred), key=str)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": [str(label) for label in labels],
        "matrix": matrix.tolist(),
    }


def _reliability_warnings(task_type: str, y_train: pd.Series, y_val: pd.Series) -> list[str]:
    """Warnings that must accompany any metrics reported for this split.

    Split classes, small data, and degenerate validation targets are all
    detected so the caller can report clearly when evaluation is unreliable.
    """
    issues: list[str] = []
    if len(y_val) < MIN_RELIABLE_VALIDATION_ROWS:
        issues.append(
            f"Validation split contains only {len(y_val)} rows; metric estimates may be unreliable at this size."
        )
    if task_type == "classification":
        training_class_names = {str(value) for value in y_train.unique()}
        validation_class_names = {str(value) for value in y_val.unique()}
        missing_from_validation = sorted(training_class_names - validation_class_names)
        if missing_from_validation:
            issues.append(
                "Classes present in training but absent from validation: "
                + ", ".join(missing_from_validation)
                + "; per-class metrics for them will be reported as zero."
            )
        val_counts = y_val.value_counts()
        sparse_classes = sorted(
            str(label) for label, count in val_counts.items() if int(count) < 2
        )
        if sparse_classes:
            issues.append(
                "Validation split contains fewer than 2 rows for classes: "
                + ", ".join(sparse_classes)
                + "; per-class metrics for these classes are unstable."
            )
    else:
        y_values = np.asarray(y_val, dtype=np.float64)
        if len(y_values) < 2 or np.var(y_values) == 0.0:
            issues.append(
                "Validation targets take a single value; R2 is meaningless for this split."
            )
    return issues


def _log_json_artifact(artifact_path: str, filename: str, payload: dict[str, Any]) -> None:
    """Attach a JSON document as an artifact on the current MLflow run."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        artifact_file = Path(temporary_directory) / filename
        artifact_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(artifact_file), artifact_path=artifact_path)


def _log_confusion_matrix_artifact(record: dict[str, Any]) -> None:
    """Attach the confusion matrix as a JSON artifact on the current run."""
    _log_json_artifact("evaluation", "confusion_matrix.json", record)


_restore_state()


@app.get("/health")
def health_check():
    """Lightweight liveness endpoint for Docker/reverse-proxy health checks."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"state": STATE},
    )


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    original_filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not original_filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV datasets are currently supported.")

    contents = await file.read(MAX_UPLOAD_SIZE + 1)
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"CSV files must be no larger than {MAX_UPLOAD_SIZE // (1024 * 1024)} MiB.",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    dataframe = _parse_csv(contents)
    profile = profile_dataframe(dataframe)
    safe_filename = original_filename[:255] or "dataset.csv"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.csv"
    file_path.write_bytes(contents)

    next_state = {
        **STATE,
        "dataset_path": str(file_path),
        "filename": safe_filename,
        **profile,
        "dataset_fingerprint": None,
        "target": None,
        "task_type": None,
        "primary_metric": None,
        "metric_direction": None,
        "run_group": None,
        "results": [],
        "best_model": None,
        "reliability": None,
        "feature_schema": None,
        "active_model": None,
    }
    _save_state(next_state)

    global inference_pipeline
    inference_pipeline = None
    STATE.update(next_state)
    _log_event(
        logging.INFO,
        "dataset_uploaded",
        rows=profile["rows"],
        columns=profile["columns_count"],
        duplicate_rows=profile["duplicate_rows"],
        size_bytes=len(contents),
    )
    return JSONResponse(
        content={
            "status": "success",
            "redirect": "/",
            "profile": profile,
        }
    )


@app.post("/train")
def run_pipeline(target: str = Form(...), task_type: str = Form(...)):
    if not STATE["dataset_path"] or not Path(STATE["dataset_path"]).is_file():
        raise HTTPException(
            status_code=409,
            detail="No active dataset is available. Upload a CSV file first.",
        )

    try:
        dataframe = pd.read_csv(STATE["dataset_path"], on_bad_lines="error")
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        _log_event(logging.ERROR, "stored_dataset_unreadable", reason=type(exc).__name__)
        raise HTTPException(
            status_code=422,
            detail="The active dataset could not be read. Upload a valid CSV file again.",
        ) from exc

    X, y, numeric_features, categorical_features = _training_data(dataframe, target, task_type)
    stratify = y if task_type == "classification" else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        _log_event(logging.WARNING, "training_split_rejected", reason=type(exc).__name__)
        raise HTTPException(
            status_code=422,
            detail="The dataset cannot be split reliably for training and validation.",
        ) from exc

    metric_spec = PRIMARY_METRICS[task_type]
    primary_metric = metric_spec["metric"]
    metric_direction = metric_spec["direction"]

    dataset_fingerprint = _dataset_fingerprint(Path(STATE["dataset_path"]))
    feature_schema = {
        **{column: "numeric" for column in numeric_features},
        **{column: "categorical" for column in categorical_features},
    }
    dataset_context_params = {
        "task_type": task_type,
        "target": target,
        "primary_metric": primary_metric,
        "metric_direction": metric_direction,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_filename": STATE["filename"] or "",
        "dataset_rows": len(dataframe),
        "dataset_columns": len(dataframe.columns),
        "feature_count": len(X.columns),
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_split_fraction": 0.2,
        "split_random_state": 42,
        "stratified_split": stratify is not None,
    }

    reliability_notes = _reliability_warnings(task_type, y_train, y_val)

    run_group = f"autoflow_run_{uuid.uuid4().hex[:12]}"
    experiment_id = mlflow.create_experiment(
        run_group,
        artifact_location=MLFLOW_ARTIFACT_DIR.as_uri(),
    )
    results: list[dict[str, Any]] = []

    for candidate in candidate_models(task_type):
        name = candidate["name"]
        hyperparameters = _model_hyperparameters(candidate["model"])
        entry: dict[str, Any] = {
            "model": name,
            "role": candidate["role"],
            "status": "failed",
            "metrics": {},
            "primary": None,
            "primary_metric": primary_metric,
            "metric_direction": metric_direction,
            "hyperparameters": hyperparameters,
            "confusion_matrix": None,
            "error": None,
            "run_id": None,
        }

        active_run = mlflow.start_run(experiment_id=experiment_id, run_name=name)
        entry["run_id"] = active_run.info.run_id
        try:
            mlflow.log_params(
                {
                    **dataset_context_params,
                    "model_type": name,
                    "model_role": candidate["role"],
                    "scale_numerical_features": bool(candidate["scale_numerical"]),
                    **{
                        f"model_{param_name}": param_value
                        for param_name, param_value in hyperparameters.items()
                    },
                }
            )

            preprocessor = _build_preprocessor(
                numeric_features,
                categorical_features,
                bool(candidate["scale_numerical"]),
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", candidate["model"])])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_val)

            if task_type == "classification":
                metrics = _classification_metrics(y_val, predictions)
                confusion = _confusion_matrix_record(y_val, predictions)
                _log_confusion_matrix_artifact(confusion)
            else:
                metrics = _regression_metrics(y_val, predictions, reliability_notes)
                confusion = None

            mlflow.log_metrics(
                {key: value for key, value in metrics.items() if value is not None}
            )
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                skops_trusted_types=SKOPS_TRUSTED_MODEL_TYPES,
            )
            # Each finished run also carries the exact inference schema the
            # pipeline was trained with - the inference contract for that model.
            _log_json_artifact(
                "inference",
                "inference_schema.json",
                {
                    "task_type": task_type,
                    "target": target,
                    "features": feature_schema,
                },
            )

            entry.update(
                status="success",
                metrics=metrics,
                primary=metrics[primary_metric],
                confusion_matrix=confusion,
                error=None,
            )
            mlflow.end_run(status="FINISHED")
        except Exception as exc:
            entry["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            try:
                mlflow.set_tags(
                    {
                        "run_status": "failed",
                        "error_type": entry["error"]["type"],
                        "error_message": entry["error"]["message"],
                    }
                )
            except Exception as tag_error:
                _log_event(
                    logging.ERROR,
                    "run_error_tags_unrecorded",
                    model=name,
                    run_id=entry["run_id"],
                    reason=type(tag_error).__name__,
                )
            _log_event(
                logging.WARNING,
                "candidate_failed",
                model=name,
                run_id=entry["run_id"],
                error_type=entry["error"]["type"],
                error=entry["error"]["message"],
            )
            mlflow.end_run(status="FAILED")
        results.append(entry)

    failures = [entry for entry in results if entry["status"] == "failed"]
    successes = [entry for entry in results if entry["status"] == "success"]
    if not successes:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All candidate models failed during training. No evaluation results are available.",
                "failures": [
                    {"model": entry["model"], "error": entry["error"]} for entry in failures
                ],
                "run_group": run_group,
            },
        )

    ranking_default = float("-inf") if metric_direction == "maximize" else float("inf")
    successes.sort(
        key=lambda entry: (
            entry["primary"] if entry["primary"] is not None else ranking_default
        ),
        reverse=(metric_direction == "maximize"),
    )
    ordered_results = successes + failures
    best_model = successes[0]

    baseline_entry = next(
        (entry for entry in successes if entry["role"] == "baseline"), None
    )
    if baseline_entry is not None:
        if metric_direction == "maximize":
            improved = best_model["primary"] > baseline_entry["primary"] + 1e-12
        else:
            improved = best_model["primary"] < baseline_entry["primary"] - 1e-12
        if not improved:
            reliability_notes.append(
                f"The selected model does not outperform the {baseline_entry['model']} baseline; "
                "these metrics look suspicious and the evaluation should be treated with caution."
            )
    if failures:
        reliability_notes.append(
            f"{len(failures)} of {len(ordered_results)} candidate models failed during training; "
            "their runs are marked FAILED in MLflow with error details."
        )
    reliability_report = {
        "status": "warning" if reliability_notes else "ok",
        "warnings": reliability_notes,
    }

    active_model = _register_selected_model(best_model, task_type)
    if active_model["registered"]:
        selected_model_uri = (
            f"models:/{active_model['registry_name']}/{active_model['registry_version']}"
        )
    else:
        selected_model_uri = f"runs:/{best_model['run_id']}/model"
    new_inference_pipeline = mlflow.sklearn.load_model(selected_model_uri)
    next_state = {
        **STATE,
        "target": target,
        "task_type": task_type,
        "dataset_fingerprint": dataset_fingerprint,
        "primary_metric": primary_metric,
        "metric_direction": metric_direction,
        "run_group": run_group,
        "results": ordered_results,
        "best_model": best_model,
        "reliability": reliability_report,
        "feature_schema": feature_schema,
        "active_model": active_model,
    }
    _save_state(next_state)
    STATE.update(next_state)

    global inference_pipeline
    inference_pipeline = new_inference_pipeline
    _log_event(
        logging.INFO,
        "training_completed",
        task_type=task_type,
        candidate_count=len(ordered_results),
        failed_count=len(failures),
        reliability=reliability_report["status"],
        selected_model=best_model["model"],
        primary_metric=primary_metric,
        primary_value=best_model["primary"],
        registered=active_model["registered"],
        registry_version=active_model["registry_version"],
        run_id=best_model["run_id"],
    )
    return JSONResponse(content={"status": "success", "redirect": "/"})


def _coerce_feature_value(kind: str | None, value: Any) -> tuple[Any, str | None]:
    """Convert a scalar JSON feature value for the inference pipeline.

    Returns the converted value plus a validation reason when the value cannot
    be used. kind comes from the persisted feature schema: 'numeric' or
    'categorical'; None means the schema predates validation and values are
    passed through unchanged for the pipeline to accept or reject.
    """
    if value is None:
        # Explicit nulls are allowed: the trained imputers handle missing data.
        return float("nan"), None
    if isinstance(value, (dict, list)):
        return None, "feature values must be scalar JSON values or null"
    if kind == "numeric":
        if isinstance(value, bool):
            return None, "numeric features require a real number; booleans are rejected"
        if isinstance(value, (int, float)):
            numeric_value = float(value)
        elif isinstance(value, str):
            try:
                numeric_value = float(value.strip())
            except ValueError:
                return None, "numeric features require a number or numeric string"
        else:
            return None, "numeric features require a number or numeric string"
        if not math.isfinite(numeric_value):
            return None, "numeric features must be finite (no NaN or infinity)"
        return numeric_value, None
    if kind == "categorical":
        return str(value), None
    return value, None


def _validated_prediction_row(
    row: dict[str, Any],
    row_index: int,
    expected_features: list[str],
    feature_schema: dict[str, str],
) -> dict[str, Any]:
    """Validate one prediction row against the trained feature contract.

    Raises a single structured HTTP 422 error covering missing features,
    unexpected features, and incompatible types for that row.
    """
    missing_features = [column for column in expected_features if column not in row]
    unexpected_features = [column for column in row if column not in expected_features]
    invalid_features: list[dict[str, str]] = []
    features_by_name: dict[str, Any] = {}

    if not missing_features and not unexpected_features:
        for column in expected_features:
            value, reason = _coerce_feature_value(feature_schema.get(column), row[column])
            if reason is None:
                features_by_name[column] = value
            else:
                invalid_features.append({"feature": column, "reason": reason})

    if missing_features or unexpected_features or invalid_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Prediction features do not match the active model schema.",
                "row_index": row_index,
                "missing_features": missing_features,
                "unexpected_features": unexpected_features,
                "invalid_features": invalid_features,
            },
        )
    return features_by_name


def _prepare_prediction_rows(
    payload: Any, expected_features: list[str], feature_schema: dict[str, str]
) -> list[dict[str, Any]]:
    """Normalise a payload into validated feature rows.

    Accepts either the original single-row contract {"features": {...}} or the
    batch contract {"rows": [{...}, ...]}, and returns them in a common shape.
    """
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422,
            detail="Request JSON must contain a 'features' object or a 'rows' list.",
        )
    has_features = "features" in payload
    has_rows = "rows" in payload
    if has_features and has_rows:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'features' for one row or 'rows' for multiple rows, not both.",
        )
    if has_features:
        if not isinstance(payload["features"], dict):
            raise HTTPException(status_code=422, detail="Request JSON must contain a 'features' object.")
        rows: list[Any] = [payload["features"]]
    elif has_rows:
        if not isinstance(payload["rows"], list) or not payload["rows"]:
            raise HTTPException(
                status_code=422,
                detail="The 'rows' value must be a non-empty list of feature objects.",
            )
        if len(payload["rows"]) > MAX_PREDICTION_ROWS:
            raise HTTPException(
                status_code=413,
                detail=f"At most {MAX_PREDICTION_ROWS} prediction rows are accepted per request.",
            )
        rows = payload["rows"]
    else:
        raise HTTPException(
            status_code=422,
            detail="Request JSON must contain a 'features' object or a 'rows' list.",
        )

    for row in rows:
        if not isinstance(row, dict):
            raise HTTPException(
                status_code=422,
                detail="Every entry in 'rows' must be an object of feature values.",
            )

    return [
        _validated_prediction_row(row, index, expected_features, feature_schema)
        for index, row in enumerate(rows)
    ]


def _prediction_value(value: Any) -> Any:
    """Convert numpy scalars to plain JSON-compatible Python values."""
    return value.item() if isinstance(value, np.generic) else value


def _json_cell(value: Any) -> Any:
    """Convert a pandas cell value into a JSON-serializable value."""
    if isinstance(value, (list, dict)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return str(value)
    return _prediction_value(value)


@app.get("/api/dataset/preview")
def dataset_preview(rows: int = 10):
    """Return a preview of the persisted dataset with per-column summaries.

    Data always comes from the uploaded CSV file inside the application-managed
    uploads directory; no dataset content is ever accepted from request input.
    """
    row_limit = min(max(int(rows), 1), 50)
    dataset_path = STATE.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise HTTPException(
            status_code=409,
            detail="No dataset is available. Upload a CSV file first.",
        )
    resolved_dataset = Path(dataset_path).resolve()
    if not resolved_dataset.is_relative_to(UPLOAD_DIR.resolve()) or not resolved_dataset.is_file():
        raise HTTPException(
            status_code=409,
            detail="The stored dataset is no longer available. Upload a CSV file again.",
        )

    try:
        dataframe = pd.read_csv(resolved_dataset, on_bad_lines="error")
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        _log_event(logging.ERROR, "stored_dataset_unreadable", reason=type(exc).__name__)
        raise HTTPException(
            status_code=409,
            detail="The stored dataset could not be read. Upload a valid CSV file again.",
        ) from exc

    columns = []
    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        columns.append(
            {
                "name": str(column),
                "kind": "numeric" if pd.api.types.is_numeric_dtype(dataframe[column]) else "categorical",
                "dtype": str(dataframe[column].dtype),
                "missing": missing_count,
                "missing_percentage": round(missing_count / len(dataframe) * 100, 2)
                if len(dataframe)
                else 0.0,
            }
        )

    preview = dataframe.head(row_limit)
    records = [
        {str(column): _json_cell(cell) for column, cell in zip(dataframe.columns, row)}
        for row in preview.itertuples(index=False, name=None)
    ]

    return {
        "available": True,
        "shape": {"rows": int(len(dataframe)), "columns": int(len(dataframe.columns))},
        "preview_rows": int(len(preview)),
        "columns": columns,
        "rows": records,
    }


@app.get("/api/experiments")
def get_experiments(limit: int = 25):
    """List real MLflow runs recorded by AutoFlow training runs.

    Only experiments created by this application (names starting with
    autoflow_run_) are reported; statuses, metrics, and timestamps come straight
    from the MLflow backend store.
    """
    run_limit = min(max(int(limit), 1), 100)
    try:
        client = _tracking_client()
        experiments = [
            experiment
            for experiment in client.search_experiments(max_results=200)
            if isinstance(experiment.name, str)
            and experiment.name.startswith("autoflow_run_")
        ]
    except Exception as exc:
        _log_event(
            logging.ERROR,
            "experiments_unavailable",
            reason=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=503,
            detail="The MLflow tracking store is unavailable. Run information cannot be retrieved.",
        ) from exc

    experiments.sort(key=lambda experiment: getattr(experiment, "creation_time", 0) or 0, reverse=True)

    runs: list[dict[str, Any]] = []
    for experiment in experiments:
        if len(runs) >= run_limit:
            break
        try:
            experiment_runs = client.search_runs(
                [experiment.experiment_id],
                max_results=run_limit,
                order_by=["attributes.start_time DESC"],
            )
        except Exception as exc:
            _log_event(
                logging.WARNING,
                "experiment_runs_unavailable",
                experiment=experiment.name,
                reason=type(exc).__name__,
            )
            continue
        for run in experiment_runs:
            if len(runs) >= run_limit:
                break
            params = run.data.params
            runs.append(
                {
                    "run_id": run.info.run_id,
                    "experiment": experiment.name,
                    "run_name": params.get("model_type") or run.data.tags.get("mlflow.runName"),
                    "task_type": params.get("task_type"),
                    "target": params.get("target"),
                    "model_type": params.get("model_type"),
                    "status": run.info.status,
                    "start_time": int(run.info.start_time) if run.info.start_time else None,
                    "end_time": int(run.info.end_time) if run.info.end_time else None,
                    "primary_metric": params.get("primary_metric"),
                    "primary_value": params.get("primary_metric")
                    and run.data.metrics.get(params["primary_metric"]),
                    "metrics": dict(run.data.metrics),
                    "dataset_fingerprint": params.get("dataset_fingerprint"),
                    "error": run.data.tags.get("error_message"),
                }
            )

    return {
        "runs": runs,
        "count": len(runs),
        "experiments": len(experiments),
    }


@app.get("/api/model/schema")
def model_schema():
    """Expose the prediction contract of the active model.

    The schema is the persisted training-time contract (feature names mapped to
    numeric/categorical types); legacy state falls back to the trained
    pipeline's feature names with unspecified kinds.
    """
    best_model = STATE.get("best_model") if isinstance(STATE.get("best_model"), dict) else None
    active_model = STATE.get("active_model") if isinstance(STATE.get("active_model"), dict) else None
    if best_model is None and active_model is None:
        raise HTTPException(
            status_code=409,
            detail="No active model is available. Train a model first.",
        )

    feature_schema = STATE.get("feature_schema")
    if isinstance(feature_schema, dict) and feature_schema:
        entries = [
            {"name": str(name), "kind": str(kind)} for name, kind in feature_schema.items()
        ]
    else:
        inference_pipeline = _get_inference_pipeline()
        entries = [
            {"name": str(feature), "kind": None}
            for feature in inference_pipeline.feature_names_in_
        ]

    return {
        "available": True,
        "features": entries,
        "task_type": STATE.get("task_type"),
        "target": STATE.get("target"),
        "model_name": best_model.get("model") if best_model else None,
        "model_version": (best_model or {}).get("run_id") or (active_model or {}).get("run_id") or "",
        "registry_version": (active_model or {}).get("registry_version"),
    }


@app.post("/predict")
async def predict(request: Request):
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Request body must contain valid JSON.") from exc

    inference_pipeline = _get_inference_pipeline()
    expected_features = list(inference_pipeline.feature_names_in_)
    feature_schema = STATE.get("feature_schema")
    feature_schema = (
        {str(key): str(value) for key, value in feature_schema.items()}
        if isinstance(feature_schema, dict)
        else {}
    )

    validated_rows = _prepare_prediction_rows(payload, expected_features, feature_schema)
    single_row_mode = isinstance(payload, dict) and "features" in payload

    try:
        predictions = inference_pipeline.predict(
            pd.DataFrame(validated_rows, columns=expected_features)
        )
    except (TypeError, ValueError) as exc:
        _log_event(logging.WARNING, "prediction_rejected", reason=type(exc).__name__)
        raise HTTPException(
            status_code=422,
            detail="Feature values could not be processed by the active model.",
        ) from exc

    best_model = STATE.get("best_model") if isinstance(STATE.get("best_model"), dict) else {}
    active_model = STATE.get("active_model") if isinstance(STATE.get("active_model"), dict) else {}
    model_version = best_model.get("run_id") or active_model.get("run_id") or ""
    registry_version = active_model.get("registry_version")

    if single_row_mode:
        return {
            "prediction": _prediction_value(predictions[0]),
            "model_version": model_version,
            "registry_version": registry_version,
            "status": "success",
        }
    return {
        "predictions": [_prediction_value(value) for value in predictions],
        "count": len(validated_rows),
        "model_version": model_version,
        "registry_version": registry_version,
        "status": "success",
    }
