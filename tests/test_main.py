import asyncio
import hashlib
import json
import sqlite3
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import jinja2
import mlflow
import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile
from mlflow.tracking import MlflowClient

import main
from meta_learning.meta_feature_extractor import extract_meta_features
from meta_learning.meta_learner import (
    evaluate_meta_reliability,
    meta_model_explanation,
    predict_algorithm,
    train_meta_model,
)


class PredictionRequest:
    """Minimal request fixture exposing a JSON payload through json()."""

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def predict_aspect(payload):
    return asyncio.run(main.predict(PredictionRequest(payload)))

class BrokenFittingModel:
    """A model fixture whose fit() always fails, for testing failure reporting."""

    def get_params(self, deep=False):
        return {"broken": True}

    def fit(self, X, y=None):
        raise RuntimeError("synthetic fit BOOM failure")

    def predict(self, X):
        raise RuntimeError("synthetic predict failure")


def build_classification_dataset(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "measurement": [None if index == 0 else index / 10 for index in range(rows)],
            "category": ["warm" if index % 2 else "cool" for index in range(rows)],
            "diagnosis": ["positive" if index % 2 else "negative" for index in range(rows)],
        }
    )


def build_regression_dataset(rows: int = 60) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    measurement = index / 10.0
    secondary = ((index * 7.0) % 10.0) / 10.0
    category = np.where(np.arange(rows) % 2 == 0, "warm", "cool")
    noise = (np.arange(rows) % 5) / 20.0
    target = 3.0 * measurement + 0.8 * (category == "warm") + 0.5 * secondary + noise
    measurement[0] = np.nan
    secondary[10:15] = np.nan
    return pd.DataFrame(
        {
            "measurement": measurement,
            "secondary": secondary,
            "category": category,
            "target": target,
        }
    )


def state_template(dataset_path: str | None = None) -> dict:
    return {
        "dataset_path": dataset_path,
        "filename": None if dataset_path is None else "dataset.csv",
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


class UploadValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.patches = [
            patch.object(main, "UPLOAD_DIR", self.directory / "uploads"),
            patch.object(main, "STATE_FILE", self.directory / "state.json"),
            patch.object(main, "MAX_UPLOAD_SIZE", 1024),
            patch.dict(
                main.STATE,
                {
                    "dataset_path": None,
                    "filename": None,
                    "rows": 0,
                    "columns_count": 0,
                    "columns": [],
                    "duplicate_rows": 0,
                    "numerical_summary": {},
                    "categorical_summary": {},
                    "target": None,
                    "task_type": None,
                    "run_group": None,
                    "results": [],
                    "best_model": None,
                },
                clear=True,
            ),
        ]
        for patcher in self.patches:
            patcher.start()
        self.addCleanup(self.temporary_directory.cleanup)
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(self.patches)])

    def upload(self, filename, contents):
        return asyncio.run(
            main.upload_dataset(
                UploadFile(filename=filename, file=BytesIO(contents))
            )
        )

    def assert_upload_error(self, filename, contents, expected_status):
        with self.assertRaises(HTTPException) as raised:
            self.upload(filename, contents)
        self.assertEqual(raised.exception.status_code, expected_status)

    def test_accepts_valid_csv_and_uses_server_generated_path(self):
        response = self.upload(r"..\..\unsafe.csv", b"feature,label\n1,a\n2,b\n")

        self.assertEqual(response.status_code, 200)
        profile = json.loads(response.body)["profile"]
        self.assertEqual(profile["columns_count"], 2)
        self.assertEqual(profile["categorical_summary"]["label"]["unique"], 2)
        uploaded_path = Path(main.STATE["dataset_path"])
        self.assertEqual(uploaded_path.parent, self.directory / "uploads")
        self.assertRegex(uploaded_path.name, r"^[0-9a-f]{32}\.csv$")
        self.assertEqual(main.STATE["filename"], "unsafe.csv")
        self.assertEqual(main.STATE["rows"], 2)

    def test_rejects_non_csv_extension(self):
        self.assert_upload_error("dataset.txt", b"a,b\n1,2\n", 415)

    def test_rejects_empty_file(self):
        self.assert_upload_error("dataset.csv", b"", 400)

    def test_rejects_oversized_file(self):
        self.assert_upload_error("dataset.csv", b"a\n" + b"x" * 1024, 413)

    def test_rejects_duplicate_or_missing_headers(self):
        for contents in (b"name,Name\none,two\n", b"name, \none,two\n"):
            with self.subTest(contents=contents):
                self.assert_upload_error("dataset.csv", contents, 400)

    def test_rejects_malformed_rows_and_invalid_encoding(self):
        for contents in (b"a,b\n1,2\n3\n", b"a\n\xff\n"):
            with self.subTest(contents=contents):
                self.assert_upload_error("dataset.csv", contents, 400)


class DatasetProfileTests(unittest.TestCase):
    def test_includes_dimensions_missingness_duplicates_and_summaries(self):
        dataframe = pd.DataFrame(
            {
                "amount": [1.0, 2.0, 2.0, 2.0],
                "category": ["a", "b", "b", None],
            }
        )

        profile = main.profile_dataframe(dataframe)

        self.assertEqual(profile["rows"], 4)
        self.assertEqual(profile["columns_count"], 2)
        self.assertEqual(profile["duplicate_rows"], 1)
        self.assertEqual(profile["columns"][1]["missing_count"], 1)
        self.assertEqual(profile["columns"][1]["data_type"], str(dataframe["category"].dtype))
        self.assertEqual(profile["numerical_summary"]["amount"]["mean"], 1.75)
        self.assertEqual(
            profile["categorical_summary"]["category"],
            {"count": 3, "unique": 2, "top": "b", "frequency": 2},
        )

    def test_quality_warnings_flag_columns_without_removing_them(self):
        dataframe = pd.DataFrame(
            {
                "id": range(20),
                "empty": [None] * 20,
                "constant": ["same"] * 20,
                "high_cardinality": ["value-" + str(index) for index in range(20)],
                "infinite": [float("inf")] + list(range(19)),
                "target": [0, 1] * 10,
            }
        )

        profile = main.profile_dataframe(dataframe)
        X, _, _, _ = main._training_data(dataframe, "target", "classification")

        self.assertIn("empty", profile["empty_columns"])
        self.assertIn("constant", profile["constant_columns"])
        self.assertIn("high_cardinality", profile["high_cardinality_columns"])
        self.assertEqual(profile["infinite_values"]["infinite"], 1)
        self.assertIn("empty", X.columns)
        self.assertTrue(profile["quality_warnings"])

    def test_meta_features_and_cold_start_prediction_are_explicit(self):
        dataframe = build_classification_dataset()
        meta_features = extract_meta_features(dataframe, "diagnosis", "classification")
        meta_model, meta_status = train_meta_model([], "classification")
        predicted, confidence = predict_algorithm(
            meta_model,
            meta_status,
            meta_features,
            "classification",
        )

        self.assertIsNone(meta_model)
        self.assertIn("Cold-Start", meta_status["selection_method"])
        self.assertIsInstance(predicted, str)
        self.assertIsNone(confidence)
        self.assertGreater(len(main.candidate_models("classification")), 6)
        integer_target = pd.DataFrame(
            {"feature": range(100), "label": [index % 2 for index in range(100)]}
        )
        self.assertEqual(
            main.detect_problem_type(integer_target, "label")["detected"],
            "classification",
        )

    def test_meta_model_excludes_current_dataset_and_withholds_uncalibrated_confidence(self):
        experiments = [
            {
                "dataset_fingerprint": f"dataset-{index}",
                "task_type": "classification",
                "actual_best_algorithm": "Algorithm A" if index % 2 else "Algorithm B",
                "meta_features": {
                    "rows": 100 + index,
                    "columns": 5,
                    "numeric_features": 3,
                    "categorical_features": 2,
                    "missing_percentage": 0.1,
                    "duplicate_percentage": 0.0,
                    "mean_skewness": 0.2,
                    "mean_cardinality": 6,
                    "target_classes": 2,
                },
            }
            for index in range(9)
        ]
        model, status = train_meta_model(
            experiments,
            "classification",
            exclude_fingerprint="dataset-8",
        )

        self.assertIsNotNone(model)
        self.assertEqual(status["historical_experiments"], 8)
        prediction, confidence = predict_algorithm(
            model,
            status,
            experiments[0]["meta_features"],
            "classification",
        )
        self.assertIn(prediction, {"Algorithm A", "Algorithm B"})
        self.assertIsNone(confidence)
        explanation = meta_model_explanation(model, status, experiments[0]["meta_features"])
        self.assertEqual(len(explanation["feature_importance"]), 9)
        self.assertEqual(len(explanation["probabilities"]), 2)
        self.assertEqual(
            explanation["probability_type"],
            "raw_uncalibrated_classifier_probability",
        )


class VisualizationDataTests(unittest.TestCase):
    def test_visualization_endpoint_reads_uploaded_data_and_bounds_bins(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            upload_directory = Path(temporary_directory) / "uploads"
            upload_directory.mkdir()
            dataset_path = upload_directory / "dataset.csv"
            build_classification_dataset().to_csv(dataset_path, index=False)
            with (
                patch.object(main, "UPLOAD_DIR", upload_directory),
                patch.object(main, "META_DATASET_FILE", Path(temporary_directory) / "meta.json"),
                patch.dict(
                    main.STATE,
                    {
                        "dataset_path": str(dataset_path),
                        "target": "diagnosis",
                        "task_type": "classification",
                    },
                    clear=True,
                ),
            ):
                payload = main.dataset_visualizations(bins=100)

        self.assertEqual(len(payload["dataset"]["feature_distribution"]["bins"]), 60)
        self.assertEqual(payload["models"]["available"], False)
        with self.assertRaises(HTTPException) as error:
            main.dataset_visualizations(task_type="unknown")
        self.assertEqual(error.exception.status_code, 422)

    def test_classification_dataset_and_model_visualizations_use_real_data(self):
        dataframe = pd.DataFrame(
            {
                "signal": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
                "noise": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "label": ["a", "a", "a", "b", "b", "b"],
            }
        )
        evaluation = {
            "evaluation_rows": 6,
            "sample_rows": 6,
            "actual": ["a", "a", "a", "b", "b", "b"],
            "predicted": ["a", "a", "b", "b", "b", "a"],
            "roc_curves": [
                {
                    "label": "b",
                    "auc": 0.75,
                    "false_positive_rate": [0.0, 0.5, 1.0],
                    "true_positive_rate": [0.0, 0.8, 1.0],
                }
            ],
        }
        best = {
            "model": "Logistic Regression",
            "run_id": "best-run",
            "status": "success",
            "metrics": {"Accuracy": 0.8, "F1": 0.75},
            "training_time_seconds": 0.02,
            "cross_validation": {"mean": 0.7, "std": 0.1, "metric": "weighted F1"},
            "confusion_matrix": {
                "labels": ["a", "b"],
                "matrix": [[2, 1], [1, 2]],
            },
            "evaluation_data": evaluation,
        }
        state = {
            "target": "label",
            "task_type": "classification",
            "dataset_fingerprint": "fp-classification",
            "primary_metric": "F1",
            "results": [best],
            "best_model": best,
            "task_2": {
                "status": "success",
                "predicted_algorithm": "Decision Tree Classifier",
                "run_id": "task2-run",
                "metrics": {"Accuracy": 0.7, "F1": 0.67},
                "training_time_seconds": 0.01,
                "evaluation_data": evaluation,
                "meta_status": {"trained": False},
            },
            "comparison": {
                "task_1_algorithm": "Logistic Regression",
                "task_2_algorithm": "Decision Tree Classifier",
                "task_1_primary_metric": "F1",
                "task_1_primary_value": 0.75,
                "task_2_primary_value": 0.67,
                "performance_difference": 0.08,
                "relative_performance_difference": 0.1067,
                "algorithms_match": False,
                "task_1_execution_time_seconds": 0.2,
                "task_2_execution_time_seconds": 0.1,
                "task_1_model_fit_seconds": 0.02,
                "task_2_training_time_seconds": 0.01,
            },
        }
        historical = [
            {
                "dataset_fingerprint": "history-a",
                "task_type": "classification",
                "target": "label",
                "actual_best_algorithm": "Logistic Regression",
                "meta_features": {"rows": 20},
            },
            {
                "dataset_fingerprint": "history-a",
                "task_type": "classification",
                "target": "different_target",
                "actual_best_algorithm": "Random Forest Classifier",
                "meta_features": {"rows": 21},
            },
        ]

        payload = main.build_visualization_payload(
            dataframe,
            state,
            historical,
            selected_run="task2-run",
        )

        self.assertEqual(payload["dataset"]["target_distribution"]["classes"][0]["count"], 3)
        self.assertEqual(payload["dataset"]["missing_value_total"], 1)
        self.assertEqual(payload["dataset"]["correlation"]["features"], ["signal", "noise"])
        self.assertEqual(payload["models"]["selected_model"]["run_id"], "task2-run")
        self.assertEqual(payload["models"]["selected_model"]["evaluation_data"], evaluation)
        self.assertEqual(payload["meta_learning"]["historical_winner_frequency"], [
            {"algorithm": "Random Forest Classifier", "datasets": 1}
        ])
        self.assertFalse(payload["meta_learning"]["reliability"]["available"])

    def test_regression_target_histogram_feature_stats_and_task_outputs(self):
        dataframe = build_regression_dataset()
        dataframe.loc[dataframe.index[-1], "measurement"] = 1000.0
        evaluation = {
            "actual": [1.0, 2.0, 3.0],
            "predicted": [1.2, 1.8, 3.1],
            "roc_curves": [],
        }
        model = {
            "model": "Linear Regression",
            "run_id": "regression-run",
            "status": "success",
            "metrics": {"MAE": 0.1, "RMSE": 0.2, "R2": 0.9},
            "training_time_seconds": 0.03,
            "cross_validation": {"mean": 0.4, "std": 0.05, "metric": "RMSE"},
            "evaluation_data": evaluation,
        }
        state = {
            "target": "target",
            "task_type": "regression",
            "dataset_fingerprint": "fp-regression",
            "primary_metric": "RMSE",
            "results": [model],
            "best_model": model,
            "task_2": {},
            "comparison": {},
        }

        payload = main.build_visualization_payload(dataframe, state, [], feature="measurement")

        self.assertEqual(payload["dataset"]["target_distribution"]["type"], "regression")
        self.assertEqual(payload["dataset"]["target_distribution"]["count"], len(dataframe))
        self.assertGreater(payload["dataset"]["feature_distribution"]["statistics"]["outlier_count"], 0)
        self.assertEqual(payload["models"]["metrics"]["RMSE"][0]["value"], 0.2)
        self.assertEqual(payload["models"]["task_1_evaluation"], evaluation)
        self.assertEqual(
            payload["meta_learning"]["reliability"]["message"],
            "Insufficient historical datasets to evaluate meta-learning reliability.",
        )

    def test_meta_reliability_is_dataset_level_and_withheld_below_threshold(self):
        records = [
            {
                "dataset_fingerprint": f"unique-{index}",
                "task_type": "classification",
                "target": "label",
                "actual_best_algorithm": "A" if index % 2 else "B",
                "meta_features": {
                    "rows": index + 10,
                    "columns": 3,
                    "numeric_features": 2,
                    "categorical_features": 1,
                    "missing_percentage": 0,
                    "duplicate_percentage": 0,
                    "mean_skewness": 0,
                    "mean_cardinality": 2,
                    "target_classes": 2,
                },
            }
            for index in range(10)
        ]
        duplicate = dict(records[0], target="other-label")
        result = evaluate_meta_reliability(
            records + [duplicate],
            "classification",
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["evaluation_datasets"], 10)
        self.assertEqual(result["split_method"], "leave-one-dataset-out")
        self.assertIn("accuracy", result)
        self.assertIn("confusion_matrix", result)


class StartupAndRouteTests(unittest.TestCase):
    """Application startup behaves like a fresh boot: schema init, routes, health."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.directory = Path(self.temporary_directory.name)
        self.previous_tracking_uri = mlflow.get_tracking_uri()
        self.tracking_uri = self.previous_tracking_uri
        # Cleanup order (LIFO): restore URI, dispose engines, then delete files.
        self.addCleanup(mlflow.set_tracking_uri, self.previous_tracking_uri)
        self.addCleanup(self.dispose_tracking_engine)
        self.addCleanup(self.temporary_directory.cleanup)

    def use_fresh_backend(self, database_name: str) -> Path:
        """Point the app at a brand-new SQLite MLflow backend, as on a fresh boot."""
        database_path = self.directory / database_name
        self.tracking_uri = f"sqlite:///{database_path.as_posix()}"
        mlflow.set_tracking_uri(self.tracking_uri)
        return database_path

    def dispose_tracking_engine(self):
        if self.tracking_uri == self.previous_tracking_uri:
            return
        client = MlflowClient(tracking_uri=self.tracking_uri)
        tracking_client = client._tracking_client
        stores = [tracking_client.store]
        try:
            stores.append(tracking_client._get_registry_store())
        except Exception:
            pass
        for store in stores:
            engine = getattr(store, "engine", None)
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass

    def test_health_endpoint_reports_ok(self):
        response = main.health_check()

        self.assertEqual(response, {"status": "ok"})

    def test_app_exposes_the_documented_routes(self):
        methods_and_paths = {
            (method, route.path)
            for route in main.app.routes
            if hasattr(route, "methods")
            for method in route.methods
        }

        expected = {
            ("GET", "/"),
            ("GET", "/health"),
            ("POST", "/upload"),
            ("POST", "/train"),
            ("GET", "/api/dataset/preview"),
            ("GET", "/api/visualizations"),
            ("GET", "/api/experiments"),
            ("GET", "/api/model/schema"),
            ("POST", "/predict"),
        }
        self.assertTrue(expected.issubset(methods_and_paths))

    def test_dashboard_route_renders_the_current_state(self):
        from starlette.requests import Request

        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [],
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
            }
        )

        with patch.dict(main.STATE, {}, clear=True):
            response = asyncio.run(main.read_dashboard(request))

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AutoFlow AI", response.body)

    def test_fresh_backend_is_initialized_on_startup(self):
        self.use_fresh_backend("startup_mlflow.db")

        main._prepare_tracking_backend()

        # Before the startup reconciliation existed, this failed with a 503
        # because the first store access and the schema migration raced.
        response = main.get_experiments(limit=5)
        self.assertEqual(response["runs"], [])

    def test_interrupted_migration_temp_tables_are_recovered(self):
        database_path = self.use_fresh_backend("dirty_mlflow.db")

        # Mimic a hard process kill during MLflow's Alembic migration, which
        # leaves temp tables behind and breaks every later store access.
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                'CREATE TABLE _alembic_tmp_tags ("key" VARCHAR(250) NOT NULL, '
                "value VARCHAR(5000), run_uuid VARCHAR(32) NOT NULL, "
                'CONSTRAINT tag_pk PRIMARY KEY ("key", run_uuid))'
            )
            connection.commit()
        finally:
            connection.close()

        main._prepare_tracking_backend()

        orphaned = sqlite3.connect(database_path).execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '_alembic_tmp_%'"
        ).fetchall()
        self.assertEqual(orphaned, [])
        self.assertEqual(main.get_experiments(limit=5)["runs"], [])


class InvalidTrainingRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temporary_directory.name) / "dataset.csv"
        self.patches = [
            patch.dict(
                main.STATE,
                {
                    "dataset_path": str(self.dataset_path),
                    "filename": "dataset.csv",
                    "rows": 0,
                    "columns_count": 0,
                    "columns": [],
                    "duplicate_rows": 0,
                    "numerical_summary": {},
                    "categorical_summary": {},
                    "target": None,
                    "task_type": None,
                    "run_group": None,
                    "results": [],
                    "best_model": None,
                },
                clear=True,
            ),
        ]
        for patcher in self.patches:
            patcher.start()
        self.addCleanup(self.temporary_directory.cleanup)
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(self.patches)])

    def write_dataset(self, dataframe):
        dataframe.to_csv(self.dataset_path, index=False)

    def test_rejects_missing_target_column(self):
        self.write_dataset(pd.DataFrame({"feature": range(10), "label": ["a", "b"] * 5}))

        with self.assertRaises(HTTPException) as raised:
            main.run_pipeline("unknown", "classification")

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("was not found", raised.exception.detail)

    def test_rejects_unsupported_task_type(self):
        self.write_dataset(pd.DataFrame({"feature": range(10), "label": ["a", "b"] * 5}))

        with self.assertRaises(HTTPException) as raised:
            main.run_pipeline("label", "clustering")

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("task_type", raised.exception.detail)

    def test_rejects_too_small_dataset(self):
        self.write_dataset(pd.DataFrame({"feature": range(9), "label": ["a", "b", "a"] * 3}))

        with self.assertRaises(HTTPException) as raised:
            main.run_pipeline("label", "classification")

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("At least 10 rows", raised.exception.detail)

    def test_rejects_class_with_only_one_sample(self):
        self.write_dataset(
            pd.DataFrame(
                {
                    "feature": range(10),
                    "label": ["common"] * 9 + ["rare"],
                }
            )
        )

        with self.assertRaises(HTTPException) as raised:
            main.run_pipeline("label", "classification")

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("at least two rows", raised.exception.detail)

    def test_rejects_missing_target_values(self):
        self.write_dataset(
            pd.DataFrame(
                {
                    "feature": range(10),
                    "label": ["a", "b"] * 4 + [None, "b"],
                }
            )
        )

        with self.assertRaises(HTTPException) as raised:
            main.run_pipeline("label", "classification")

        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("missing", raised.exception.detail)


METRIC_COMPARISON_DELTA = 1e-4


class MetricComputationTests(unittest.TestCase):
    def test_classification_metrics_have_expected_values(self):
        y_true = pd.Series([1, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0])

        metrics = main._classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["Accuracy"], 0.75, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["Precision"], 0.875, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["Recall"], 0.75, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["F1"], 0.766667, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["Macro F1"], 0.733333, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["Balanced Accuracy"], 0.833333, delta=METRIC_COMPARISON_DELTA)

    def test_classification_metrics_stay_defined_when_a_class_is_never_predicted(self):
        y_true = pd.Series(["a", "a", "a", "b"])
        y_pred = np.array(["a", "a", "a", "a"])

        metrics = main._classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["Recall"], 0.75, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["Precision"], 0.5625, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["F1"], 0.642857, delta=METRIC_COMPARISON_DELTA)

    def test_regression_metrics_have_expected_values(self):
        warnings_sink: list[str] = []
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = np.array([15.0, 20.0, 25.0])

        metrics = main._regression_metrics(y_true, y_pred, warnings_sink)

        self.assertAlmostEqual(metrics["MAE"], 3.333333, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["RMSE"], 4.082483, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["R2"], 0.75, delta=METRIC_COMPARISON_DELTA)
        self.assertEqual(warnings_sink, [])

    def test_regression_metrics_report_perfect_r2_when_fit_is_exact(self):
        warnings_sink: list[str] = []
        metrics = main._regression_metrics(
            pd.Series([10.0, 20.0, 30.0]),
            np.array([10.0, 20.0, 30.0]),
            warnings_sink,
        )

        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["RMSE"], 0.0)
        self.assertEqual(metrics["R2"], 1.0)
        self.assertEqual(warnings_sink, [])

    def test_regression_metrics_report_r2_as_undefined_for_constant_targets(self):
        warnings_sink: list[str] = []
        metrics = main._regression_metrics(
            pd.Series([5.0, 5.0, 5.0, 5.0]),
            np.array([5.0, 9.0, 5.0, 5.0]),
            warnings_sink,
        )

        self.assertAlmostEqual(metrics["MAE"], 1.0, delta=METRIC_COMPARISON_DELTA)
        self.assertAlmostEqual(metrics["RMSE"], 2.0, delta=METRIC_COMPARISON_DELTA)
        self.assertIsNone(metrics["R2"])
        self.assertEqual(len(warnings_sink), 1)
        self.assertIn("R2", warnings_sink[0])

    def test_confusion_matrix_record_uses_sorted_labels_and_counts(self):
        record = main._confusion_matrix_record(
            ["a", "a", "b"], ["a", "b", "b"]
        )

        self.assertEqual(record["labels"], ["a", "b"])
        self.assertEqual(record["matrix"], [[1, 1], [0, 1]])

    def test_reliability_warnings_for_healthy_classification_split(self):
        y_train = pd.Series(["a"] * 10 + ["b"] * 20)
        y_val = pd.Series(["a"] * 10 + ["b"] * 20)

        warnings_list = main._reliability_warnings("classification", y_train, y_val)

        self.assertEqual(warnings_list, [])

    def test_reliability_warnings_for_small_or_degenerate_splits(self):
        y_train = pd.Series(["a"] * 30 + ["b"] * 30)
        sparse_val = pd.Series(["a"] * 28 + ["b"])

        classification_issues = main._reliability_warnings("classification", y_train, sparse_val)
        self.assertTrue(classification_issues)
        small_validation_issue = next(
            warning for warning in classification_issues if "only 29 rows" in warning
        )
        self.assertIn("unreliable", small_validation_issue)
        self.assertTrue(
            any("fewer than 2 rows for classes: b" in warning for warning in classification_issues)
        )

        constant_val = pd.Series([1.0] * 20)
        regression_issues = main._reliability_warnings("regression", pd.Series([1.0] * 80), constant_val)
        self.assertTrue(
            any("R2 is meaningless" in warning for warning in regression_issues)
        )

    def test_dataset_fingerprint_is_a_stable_content_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "data.csv"
            dataset_path.write_bytes(b"a,b\n1,2\n3,4\n")

            first = main._dataset_fingerprint(dataset_path)
            second = main._dataset_fingerprint(dataset_path)

        self.assertEqual(first, second)
        self.assertEqual(first, hashlib.sha256(b"a,b\n1,2\n3,4\n").hexdigest())


class TrainingTestBase(unittest.TestCase):
    """Shared setup isolating uploads, state, and MLflow tracking per test."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.directory = Path(self.temporary_directory.name)
        self.dataset_path = self.directory / "dataset.csv"
        self.original_tracking_uri = main.mlflow.get_tracking_uri()
        self.tracking_uri = f"sqlite:///{(self.directory / 'tracking.db').as_posix()}"
        self.patches = [
            patch.object(main, "UPLOAD_DIR", self.directory / "uploads"),
            patch.object(main, "STATE_FILE", self.directory / "state.json"),
            patch.object(main, "MLFLOW_ARTIFACT_DIR", self.directory / "mlruns"),
            patch.object(main, "META_DATASET_FILE", self.directory / "meta_data" / "experiments.json"),
            patch.dict(
                main.STATE,
                {
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
                    "task_2": None,
                    "comparison": None,
                    "meta_features": None,
                },
                clear=True,
            ),
        ]
        for patcher in self.patches:
            patcher.start()
        main.mlflow.set_tracking_uri(self.tracking_uri)
        main.inference_pipeline = None
        self.addCleanup(self.temporary_directory.cleanup)
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(self.patches)])
        self.addCleanup(main.mlflow.set_tracking_uri, self.original_tracking_uri)
        self.addCleanup(self.dispose_tracking_engine)

    def dispose_tracking_engine(self):
        client = MlflowClient(tracking_uri=self.tracking_uri)
        tracking_client = client._tracking_client
        stores = [tracking_client.store]
        try:
            stores.append(tracking_client._get_registry_store())
        except Exception:
            pass
        for store in stores:
            engine = getattr(store, "engine", None)
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass

    def upload(self, dataframe):
        contents = dataframe.to_csv(index=False).encode("utf-8")
        asyncio.run(
            main.upload_dataset(
                UploadFile(filename="dataset.csv", file=BytesIO(contents))
            )
        )

    def mlflow_client(self):
        return MlflowClient(tracking_uri=self.tracking_uri)

    def runs_for(self, run_group: str):
        client = self.mlflow_client()
        experiment = client.get_experiment_by_name(run_group)
        return list(client.search_runs([experiment.experiment_id]))


class TrainingPersistenceTests(TrainingTestBase):
    def test_training_prediction_and_restart_restore(self):
        self.upload(build_classification_dataset())

        response = main.run_pipeline("diagnosis", "classification")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(main.STATE["task_type"], "classification")
        self.assertEqual(main.STATE["primary_metric"], "F1")
        self.assertEqual(main.STATE["metric_direction"], "maximize")
        self.assertEqual(
            len(main.STATE["results"]),
            len(main.candidate_models("classification")),
        )
        self.assertTrue(
            all(entry["status"] == "success" for entry in main.STATE["results"])
        )
        self.assertIn("Majority-Class Baseline", [entry["model"] for entry in main.STATE["results"]])
        task_2 = main.STATE["task_2"]
        self.assertEqual(task_2["status"], "success")
        self.assertIn(
            task_2["predicted_algorithm"],
            [entry["model"] for entry in main.STATE["results"]],
        )
        self.assertIn("Cold-Start", task_2["selection_method"])
        self.assertIsNone(task_2["confidence"])
        self.assertEqual(
            main.STATE["comparison"]["task_1_algorithm"],
            main.STATE["best_model"]["model"],
        )
        task_2_prediction = predict_aspect(
            {
                "model_run_id": task_2["run_id"],
                "features": {"measurement": 0.5, "category": "unseen"},
            }
        )
        self.assertEqual(task_2_prediction["status"], "success")
        self.assertEqual(task_2_prediction["model_version"], task_2["run_id"])
        download = main.download_model(task_2["run_id"])
        self.assertEqual(download.status_code, 200)
        self.assertTrue(download.headers["content-disposition"].endswith(".joblib\""))
        self.assertIn("meta-learning", download.headers["content-disposition"])
        self.assertNotIn(task_2["run_id"], download.headers["content-disposition"])
        benchmark_download = main.download_model(main.STATE["best_model"]["run_id"])
        self.assertEqual(benchmark_download.status_code, 200)
        self.assertIn("benchmark-", benchmark_download.headers["content-disposition"])
        self.assertNotIn(
            main.STATE["best_model"]["run_id"],
            benchmark_download.headers["content-disposition"],
        )
        for entry in main.STATE["results"]:
            self.assertEqual(entry["primary_metric"], "F1")
            self.assertIn("confusion_matrix", entry)
            self.assertIsNotNone(entry["confusion_matrix"])
            confusion = entry["confusion_matrix"]
            self.assertEqual(sorted(confusion["labels"]), ["negative", "positive"])
            self.assertEqual(
                [sum(row) for row in confusion["matrix"]],
                [4, 4],
            )
        primary_values = [entry["primary"] for entry in main.STATE["results"]]
        self.assertEqual(primary_values, sorted(primary_values, reverse=True))

        fingerprints = hashlib.sha256(
            Path(main.STATE["dataset_path"]).read_bytes()
        ).hexdigest()
        self.assertEqual(main.STATE["dataset_fingerprint"], fingerprints)

        reliability = main.STATE["reliability"]
        self.assertEqual(reliability["status"], "warning")
        self.assertTrue(reliability["warnings"])

        self.assertEqual(
            main.STATE["feature_schema"],
            {"measurement": "numeric", "category": "categorical"},
        )
        active_model = main.STATE["active_model"]
        self.assertEqual(active_model["registry_name"], main.ACTIVE_MODEL_REGISTRY_NAME)
        self.assertTrue(active_model["registered"])
        self.assertIsInstance(active_model["registry_version"], int)
        self.assertEqual(active_model["run_id"], main.STATE["best_model"]["run_id"])

        registered_versions = list(
            self.mlflow_client().search_model_versions(
                f"name = '{main.ACTIVE_MODEL_REGISTRY_NAME}'"
            )
        )
        self.assertEqual(len(registered_versions), 1)
        task_1_version = registered_versions[0]
        self.assertEqual(str(task_1_version.run_id), active_model["run_id"])
        task_2_versions = list(
            self.mlflow_client().search_model_versions(
                f"name = '{main.META_MODEL_REGISTRY_NAME}'"
            )
        )
        self.assertEqual(len(task_2_versions), 1)
        self.assertEqual(str(task_2_versions[0].run_id), main.STATE["task_2"]["run_id"])
        registered_tags = self.mlflow_client().get_model_version(
            main.ACTIVE_MODEL_REGISTRY_NAME, task_1_version.version
        ).tags
        self.assertEqual(registered_tags.get("task_type"), "classification")
        self.assertEqual(registered_tags.get("run_id"), active_model["run_id"])

        prediction = predict_aspect(
            {"features": {"measurement": 0.5, "category": "unseen"}}
        )
        self.assertEqual(prediction["status"], "success")
        self.assertIn(prediction["prediction"], {"positive", "negative"})
        self.assertIsInstance(prediction["registry_version"], int)
        self.assertEqual(
            prediction["registry_version"],
            main.STATE["active_model"]["registry_version"],
        )

        main.inference_pipeline = None
        main._restore_state()
        self.assertIsNotNone(main.inference_pipeline)
        self.assertEqual(main.STATE["best_model"]["run_id"], prediction["model_version"])

        runs = self.runs_for(main.STATE["run_group"])
        self.assertEqual(len(runs), len(main.STATE["results"]) + 1)
        entries = {entry["run_id"]: entry for entry in main.STATE["results"]}
        for run in runs:
            evaluation_artifacts = {
                artifact.path
                for artifact in self.mlflow_client().list_artifacts(run.info.run_id, "evaluation")
            }
            self.assertIn("evaluation/evaluation_data.json", evaluation_artifacts)
            evaluation_data_path = self.mlflow_client().download_artifacts(
                run.info.run_id,
                "evaluation/evaluation_data.json",
            )
            evaluation_data = json.loads(
                Path(evaluation_data_path).read_text(encoding="utf-8")
            )
            self.assertLessEqual(evaluation_data["sample_rows"], 500)
            if run.data.params.get("model_role") == "task_2_meta_selection":
                self.assertEqual(run.info.status, "FINISHED")
                self.assertEqual(run.data.params["model_type"], main.STATE["task_2"]["predicted_algorithm"])
                self.assertEqual(
                    evaluation_data,
                    main.STATE["task_2"]["evaluation_data"],
                )
                continue
            entry = entries[run.info.run_id]
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.params["task_type"], "classification")
            self.assertEqual(run.data.params["model_type"], entry["model"])
            self.assertEqual(run.data.params["primary_metric"], "F1")
            self.assertEqual(run.data.params["metric_direction"], "maximize")
            self.assertEqual(
                run.data.params["dataset_fingerprint"],
                fingerprints,
            )
            self.assertEqual(run.data.params["training_rows"], "32")
            artifact_paths = {
                artifact.path
                for artifact in self.mlflow_client().list_artifacts(run.info.run_id)
            }
            self.assertIn("evaluation", artifact_paths)
            self.assertIn("inference", artifact_paths)
            self.assertIn("evaluation/confusion_matrix.json", evaluation_artifacts)
            self.assertEqual(evaluation_data, entry["evaluation_data"])
            # MLflow 3 stores logged models in the logged-model store; loading via
            # the runs:/ URI is how the application itself restores the artifact.
            loaded_model = main.mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
            self.assertTrue(hasattr(loaded_model, "predict"))
            self.assertAlmostEqual(
                run.data.metrics["F1"],
                entry["metrics"]["F1"],
                delta=METRIC_COMPARISON_DELTA,
            )


class RegressionTrainingTests(TrainingTestBase):
    def test_regression_training_records_mlflow_runs_and_metrics(self):
        self.upload(build_regression_dataset())

        response = main.run_pipeline("target", "regression")

        self.assertEqual(response.status_code, 200)
        state = main.STATE
        self.assertEqual(state["task_type"], "regression")
        self.assertEqual(state["primary_metric"], "RMSE")
        self.assertEqual(state["metric_direction"], "minimize")

        results = state["results"]
        self.assertEqual(len(results), len(main.candidate_models("regression")))
        self.assertTrue(all(entry["status"] == "success" for entry in results))
        names = [entry["model"] for entry in results]
        self.assertIn("Median Baseline", names)
        self.assertIn("Linear Regression", names)
        self.assertIn("Random Forest Regressor", names)

        rmse_values = [entry["primary"] for entry in results]
        self.assertEqual(rmse_values, sorted(rmse_values))
        self.assertEqual(state["best_model"], results[0])

        for entry in results:
            self.assertEqual(set(entry["metrics"]), {"MAE", "RMSE", "R2"})
            self.assertIsInstance(entry["metrics"]["R2"], float)
            self.assertIsNone(entry["confusion_matrix"])
            self.assertIsInstance(entry["hyperparameters"], dict)

        fingerprint = hashlib.sha256(
            Path(state["dataset_path"]).read_bytes()
        ).hexdigest()
        self.assertEqual(state["dataset_fingerprint"], fingerprint)

        reliability = state["reliability"]
        self.assertEqual(reliability["status"], "warning")
        self.assertTrue(
            any("only 12 rows" in warning for warning in reliability["warnings"])
        )

        runs = self.runs_for(state["run_group"])
        self.assertEqual(len(runs), len(results) + 1)
        entries = {entry["run_id"]: entry for entry in results}
        for run in runs:
            if run.data.params.get("model_role") == "task_2_meta_selection":
                self.assertEqual(run.info.status, "FINISHED")
                self.assertEqual(run.data.params["model_type"], state["task_2"]["predicted_algorithm"])
                continue
            entry = entries[run.info.run_id]
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.params["task_type"], "regression")
            self.assertEqual(run.data.params["model_type"], entry["model"])
            self.assertEqual(run.data.params["primary_metric"], "RMSE")
            self.assertEqual(run.data.params["metric_direction"], "minimize")
            self.assertEqual(run.data.params["dataset_fingerprint"], fingerprint)
            self.assertEqual(run.data.params["validation_rows"], "12")
            artifact_paths = {
                artifact.path
                for artifact in self.mlflow_client().list_artifacts(run.info.run_id)
            }
            self.assertNotIn("evaluation", artifact_paths)
            self.assertAlmostEqual(
                run.data.metrics["RMSE"],
                entry["metrics"]["RMSE"],
                delta=METRIC_COMPARISON_DELTA,
            )
            loaded_model = main.mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
            self.assertTrue(hasattr(loaded_model, "predict"))

        forest_run = next(
            run for run in runs
            if run.data.params.get("model_role") != "task_2_meta_selection"
            and entries[run.info.run_id]["model"] == "Random Forest Regressor"
        )
        self.assertEqual(forest_run.data.params["model_n_estimators"], "100")
        self.assertEqual(forest_run.data.params["model_random_state"], "42")

    def test_regression_prediction_serving(self):
        self.upload(build_regression_dataset())
        main.run_pipeline("target", "regression")

        prediction = predict_aspect(
            {"features": {"measurement": 0.5, "secondary": 0.3, "category": "warm"}}
        )

        self.assertEqual(prediction["status"], "success")
        self.assertIsInstance(prediction["prediction"], float)


class FailedCandidateTests(TrainingTestBase):
    def test_failed_candidate_is_reported_and_recorded_as_failed(self):
        self.upload(build_classification_dataset())
        original_candidates = main.candidate_models

        def broken_candidates(task_type):
            return (
                {
                    "name": "Broken Model",
                    "model": BrokenFittingModel(),
                    "role": "baseline",
                    "scale_numerical": False,
                },
                *original_candidates(task_type)[:2],
            )

        with patch.object(main, "candidate_models", broken_candidates):
            response = main.run_pipeline("diagnosis", "classification")

        self.assertEqual(response.status_code, 200)
        results = main.STATE["results"]
        self.assertEqual(len(results), 3)

        failed_entry = next(entry for entry in results if entry["status"] == "failed")
        self.assertEqual(failed_entry["model"], "Broken Model")
        self.assertEqual(failed_entry["error"]["type"], "RuntimeError")
        self.assertIn("synthetic fit BOOM failure", failed_entry["error"]["message"])
        self.assertEqual(failed_entry["primary"], None)
        self.assertEqual(failed_entry["hyperparameters"], {"broken": True})

        success_entries = [entry for entry in results if entry["status"] == "success"]
        self.assertEqual(len(success_entries), 2)
        self.assertEqual(main.STATE["best_model"], success_entries[0])
        reliability = main.STATE["reliability"]
        self.assertTrue(
            any(
                "failed during training" in warning
                for warning in reliability["warnings"]
            )
        )

        runs = self.runs_for(main.STATE["run_group"])
        self.assertEqual(len(runs), 3)
        entries = {entry["run_id"]: entry for entry in results}
        for run in runs:
            entry = entries[run.info.run_id]
            if entry["status"] == "failed":
                self.assertEqual(run.info.status, "FAILED")
                self.assertEqual(run.data.tags["error_type"], "RuntimeError")
                self.assertIn("BOOM", run.data.tags["error_message"])
                self.assertEqual(run.data.params["model_broken"], "True")
            else:
                self.assertEqual(run.info.status, "FINISHED")

    def test_all_candidates_failing_raises_and_records_failed_runs(self):
        self.upload(build_classification_dataset())

        def broken_candidates(task_type):
            return (
                {
                    "name": "Broken Model A",
                    "model": BrokenFittingModel(),
                    "role": "baseline",
                    "scale_numerical": False,
                },
                {
                    "name": "Broken Model B",
                    "model": BrokenFittingModel(),
                    "role": "linear",
                    "scale_numerical": True,
                },
            )

        with patch.object(main, "candidate_models", broken_candidates):
            with self.assertRaises(HTTPException) as raised:
                main.run_pipeline("diagnosis", "classification")

        self.assertEqual(raised.exception.status_code, 422)
        detail = raised.exception.detail
        self.assertIn("All candidate models failed", detail["message"])
        self.assertEqual(len(detail["failures"]), 2)
        for failure in detail["failures"]:
            self.assertEqual(failure["error"]["type"], "RuntimeError")

        runs = self.runs_for(detail["run_group"])
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertEqual(run.info.status, "FAILED")
            self.assertIn("BOOM", run.data.tags["error_message"])

        self.assertIsNone(main.STATE["best_model"])
        self.assertEqual(main.STATE["results"], [])
        self.assertIsNone(main.inference_pipeline)


class RestartRecoveryTests(TrainingTestBase):
    def test_predict_before_training_returns_409(self):
        with self.assertRaises(HTTPException) as raised:
            predict_aspect({"features": {"measurement": 0.5, "category": "cool"}})

        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("No active model", raised.exception.detail)

    def test_restart_rebuilds_pipeline_from_persisted_references(self):
        self.upload(build_regression_dataset())
        main.run_pipeline("target", "regression")
        trained_run_id = main.STATE["best_model"]["run_id"]

        main.inference_pipeline = None
        main.STATE.clear()
        main.STATE.update(state_template())
        main._restore_state()

        self.assertIsNotNone(main.inference_pipeline)
        prediction = predict_aspect(
            {"features": {"measurement": 0.5, "secondary": 0.3, "category": "warm"}}
        )
        self.assertEqual(prediction["status"], "success")
        self.assertEqual(prediction["model_version"], trained_run_id)

    def test_corrupt_state_file_recovers_from_model_registry(self):
        self.upload(build_classification_dataset())
        main.run_pipeline("diagnosis", "classification")
        trained_run_id = main.STATE["best_model"]["run_id"]
        trained_registry_version = main.STATE["active_model"]["registry_version"]

        main.STATE_FILE.write_bytes(b"{ not valid json")
        main.inference_pipeline = None
        main.STATE.clear()
        main.STATE.update(state_template())

        main._restore_state()  # must never crash the application

        self.assertIsNotNone(main.inference_pipeline)
        self.assertEqual(
            main.STATE["active_model"]["registry_version"],
            trained_registry_version,
        )
        self.assertEqual(main.STATE["best_model"]["run_id"], trained_run_id)
        prediction = predict_aspect(
            {"features": {"measurement": 0.5, "category": "cool"}}
        )
        self.assertEqual(prediction["status"], "success")

    def test_tampered_run_references_fall_back_to_registered_model(self):
        self.upload(build_classification_dataset())
        main.run_pipeline("diagnosis", "classification")
        trained_registry_version = main.STATE["active_model"]["registry_version"]

        tampered_state = dict(main.STATE)
        tampered_state["active_model"] = {
            **main.STATE["active_model"],
            "run_id": "f" * 32,
        }
        tampered_state["best_model"] = {
            **main.STATE["best_model"],
            "run_id": "f" * 32,
        }
        main._save_state(tampered_state)

        main.inference_pipeline = None
        main.STATE.clear()
        main.STATE.update(state_template())
        main._restore_state()

        self.assertIsNotNone(main.inference_pipeline)
        self.assertEqual(
            main.STATE["active_model"]["registry_version"],
            trained_registry_version,
        )

    def test_predict_returns_503_when_artifact_is_invalid(self):
        self.upload(build_classification_dataset())
        main.run_pipeline("diagnosis", "classification")
        main.inference_pipeline = None

        with patch.object(
            main.mlflow.sklearn,
            "load_model",
            side_effect=RuntimeError("artifact vanished"),
        ):
            with self.assertRaises(HTTPException) as raised:
                predict_aspect({"features": {"measurement": 0.5, "category": "cool"}})

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("missing or invalid", raised.exception.detail)


class PredictionValidationTests(unittest.TestCase):
    """Trains one regression model, then exercises the prediction contract."""

    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.directory = Path(cls.temporary_directory.name)
        cls.original_tracking_uri = main.mlflow.get_tracking_uri()
        cls.tracking_uri = f"sqlite:///{(cls.directory / 'tracking.db').as_posix()}"
        cls.patchers = [
            patch.object(main, "UPLOAD_DIR", cls.directory / "uploads"),
            patch.object(main, "STATE_FILE", cls.directory / "state.json"),
            patch.object(main, "MLFLOW_ARTIFACT_DIR", cls.directory / "mlruns"),
            patch.dict(main.STATE, state_template(), clear=True),
        ]
        for patcher in cls.patchers:
            patcher.start()
        main.mlflow.set_tracking_uri(cls.tracking_uri)
        main.inference_pipeline = None
        contents = build_regression_dataset().to_csv(index=False).encode("utf-8")
        asyncio.run(
            main.upload_dataset(
                UploadFile(filename="dataset.csv", file=BytesIO(contents))
            )
        )
        main.run_pipeline("target", "regression")

    @classmethod
    def tearDownClass(cls):
        client = MlflowClient(tracking_uri=cls.tracking_uri)
        tracking_client = client._tracking_client
        stores = [tracking_client.store]
        try:
            stores.append(tracking_client._get_registry_store())
        except Exception:
            pass
        for store in stores:
            engine = getattr(store, "engine", None)
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass
        for patcher in reversed(cls.patchers):
            patcher.stop()
        main.mlflow.set_tracking_uri(cls.original_tracking_uri)
        cls.temporary_directory.cleanup()

    def feature_row(self):
        return {"measurement": 0.5, "secondary": 0.3, "category": "warm"}

    def assert_prediction_error(self, payload, expected_status):
        with self.assertRaises(HTTPException) as raised:
            predict_aspect(payload)
        self.assertEqual(raised.exception.status_code, expected_status)
        return raised.exception.detail

    def test_single_row_response_shape_is_preserved(self):
        response = predict_aspect({"features": self.feature_row()})

        self.assertEqual(response["status"], "success")
        self.assertIsInstance(response["prediction"], float)
        self.assertIsInstance(response["model_version"], str)
        self.assertTrue(
            response["model_version"],
        )
        self.assertIsInstance(response["registry_version"], int)
        self.assertNotIn("predictions", response)

    def test_multiple_rows_are_supported(self):
        rows = [self.feature_row() for _ in range(3)]

        response = predict_aspect({"rows": rows})

        self.assertEqual(response["status"], "success")
        self.assertEqual(response["count"], 3)
        self.assertEqual(len(response["predictions"]), 3)
        self.assertTrue(
            all(isinstance(value, float) for value in response["predictions"])
        )

    def test_missing_values_are_accepted_for_imputation(self):
        response = predict_aspect(
            {"features": {**self.feature_row(), "measurement": None}}
        )

        self.assertEqual(response["status"], "success")
        self.assertIsInstance(response["prediction"], float)

    def test_missing_required_feature_is_reported(self):
        row = self.feature_row()
        del row["secondary"]

        detail = self.assert_prediction_error({"features": row}, 422)

        self.assertEqual(detail["missing_features"], ["secondary"])
        self.assertEqual(detail["unexpected_features"], [])
        self.assertEqual(detail["row_index"], 0)

    def test_unexpected_feature_is_reported(self):
        row = {**self.feature_row(), "mystery": 1}

        detail = self.assert_prediction_error({"features": row}, 422)

        self.assertEqual(detail["unexpected_features"], ["mystery"])
        self.assertEqual(detail["missing_features"], [])

    def test_incompatible_types_are_reported(self):
        string_row = {**self.feature_row(), "measurement": "not-a-number"}
        detail = self.assert_prediction_error({"features": string_row}, 422)
        self.assertEqual(detail["invalid_features"][0]["feature"], "measurement")
        self.assertIn("numeric", detail["invalid_features"][0]["reason"])

        boolean_row = {**self.feature_row(), "measurement": True}
        detail = self.assert_prediction_error({"features": boolean_row}, 422)
        self.assertIn("booleans", detail["invalid_features"][0]["reason"])

        nested_row = {**self.feature_row(), "measurement": {"nested": 1}}
        detail = self.assert_prediction_error({"features": nested_row}, 422)
        self.assertIn("scalar", detail["invalid_features"][0]["reason"])

    def test_batch_validation_reports_offending_row_index(self):
        broken_row = {
            "measurement": 0.4,
            "secondary": "oops",
            "category": "warm",
        }

        detail = self.assert_prediction_error(
            {"rows": [self.feature_row(), broken_row]}, 422
        )

        self.assertEqual(detail["row_index"], 1)
        self.assertEqual(detail["invalid_features"][0]["feature"], "secondary")

    def test_rows_contract_is_validated(self):
        self.assert_prediction_error({"rows": "not-a-list"}, 422)
        self.assert_prediction_error({"rows": []}, 422)
        self.assert_prediction_error({"rows": [["a"]]}, 422)
        self.assert_prediction_error(
            {"rows": [self.feature_row()], "features": self.feature_row()}, 422
        )
        self.assert_prediction_error({"data": []}, 422)
        self.assert_prediction_error([1, 2], 422)
        self.assert_prediction_error(
            {"rows": [self.feature_row()] * (main.MAX_PREDICTION_ROWS + 1)}, 413
        )

    def test_invalid_json_body_is_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(main.predict(PredictionRequest(ValueError("bad json"))))

        self.assertEqual(raised.exception.status_code, 400)

    def test_missing_schema_defers_rejection_to_the_pipeline(self):
        original_schema = main.STATE["feature_schema"]
        main.STATE["feature_schema"] = None
        try:
            detail = self.assert_prediction_error(
                {
                    "features": {
                        "measurement": "not-a-number",
                        "secondary": 0.3,
                        "category": "warm",
                    }
                },
                422,
            )
        finally:
            main.STATE["feature_schema"] = original_schema

        self.assertEqual(
            detail,
            "Feature values could not be processed by the active model.",
        )


class DashboardTemplateTests(unittest.TestCase):
    """The dashboard renders truthful server-side state in populated and empty modes."""

    def render_dashboard(self, state):
        environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(main.PROJECT_DIR / "templates"))
        )
        return environment.get_template("dashboard.html").render(state=state)

    def test_populated_state_renders_sections_with_real_values(self):
        populated = state_template("/datasets/uploads/stored.csv")
        populated.update(
            {
                "filename": "clinical.csv",
                "rows": 60,
                "columns_count": 4,
                "columns": [
                    {"name": "measurement", "type": "numeric"},
                    {"name": "category", "type": "categorical"},
                    {"name": "target", "type": "numeric"},
                ],
                "dataset_fingerprint": "a" * 64,
                "task_type": "regression",
                "target": "target",
                "primary_metric": "RMSE",
                "metric_direction": "minimize",
                "run_group": "autoflow_run_abc",
                "results": [
                    {
                        "model": "Random Forest Regressor",
                        "status": "success",
                        "role": "forest",
                        "primary": 2.4043,
                        "primary_metric": "RMSE",
                        "metric_direction": "minimize",
                        "metrics": {"MAE": 1.9, "RMSE": 2.4043, "R2": 0.97},
                        "hyperparameters": {"n_estimators": 100},
                        "confusion_matrix": None,
                        "error": None,
                        "run_id": "b" * 32,
                    },
                    {
                        "model": "Broken Model",
                        "status": "failed",
                        "role": "baseline",
                        "primary": None,
                        "primary_metric": "RMSE",
                        "metric_direction": "minimize",
                        "metrics": {},
                        "hyperparameters": {"broken": True},
                        "confusion_matrix": None,
                        "error": {"type": "RuntimeError", "message": "synthetic failure"},
                        "run_id": "c" * 32,
                    },
                ],
                "best_model": {
                    "model": "Random Forest Regressor",
                    "status": "success",
                    "primary": 2.4043,
                    "primary_metric": "RMSE",
                    "metric_direction": "minimize",
                    "run_id": "b" * 32,
                },
                "reliability": {
                    "status": "warning",
                    "warnings": ["Validation split contains only 12 rows."],
                },
                "active_model": {
                    "registry_name": main.ACTIVE_MODEL_REGISTRY_NAME,
                    "registry_version": 3,
                },
            }
        )

        html = self.render_dashboard(populated)

        for needle in (
            "Overview",
            "Dataset",
            "Training",
            "Meta-Learning",
            "Task Comparison",
            "Predictions",
            "clinical.csv",
            "60 rows",
            "data-has-dataset",
            "data-has-model",
            "2.4043",
            "Random Forest Regressor",
            "selected",
            "synthetic failure",
            "Evaluation notices",
            "Validation split contains only 12 rows.",
        ):
            self.assertIn(needle, html)
        self.assertIn('<option value="target" selected>', html)
        self.assertIn('<option value="regression" selected>', html)
        self.assertNotIn("Experiments", html)
        self.assertNotIn("Run ID", html)
        self.assertNotIn("fingerprint", html)
        self.assertNotIn("registry v", html)
        self.assertNotIn("autoflow_run_abc", html)

    def test_empty_state_renders_true_empty_states(self):
        html = self.render_dashboard(state_template())

        self.assertIn("Not uploaded", html)
        self.assertIn("No training has been run yet.", html)
        self.assertIn("No evaluated models yet. Run training to generate real results.", html)
        self.assertIn("No trained model yet. Train a model first", html)
        self.assertIn("No dataset available. Upload a CSV file", html)
        self.assertNotIn("data-has-model=\"true\"", html)

    def test_failed_css_states_use_text_badges(self):
        populated = state_template()
        populated.update(
            {
                "results": [
                    {
                        "model": "Broken",
                        "status": "failed",
                        "primary": None,
                        "metrics": {},
                        "error": {"type": "ValueError", "message": "x"},
                        "run_id": "d" * 32,
                        "confusion_matrix": None,
                        "hyperparameters": {},
                    }
                ],
                "best_model": None,
            }
        )
        html = self.render_dashboard(populated)
        self.assertIn("Failed", html)
        self.assertIn("row-failed", html)


class DatasetPreviewEndpointTests(TrainingTestBase):
    def test_preview_reports_shape_columns_and_first_rows(self):
        self.upload(build_regression_dataset())

        preview = main.dataset_preview(5)

        self.assertTrue(preview["available"])
        self.assertEqual(preview["shape"], {"rows": 60, "columns": 4})
        self.assertEqual(preview["preview_rows"], 5)
        kinds = {column["name"]: column for column in preview["columns"]}
        self.assertEqual(kinds["measurement"]["kind"], "numeric")
        self.assertEqual(kinds["category"]["kind"], "categorical")
        self.assertEqual(kinds["measurement"]["missing"], 1)
        self.assertEqual(kinds["secondary"]["missing"], 5)
        self.assertIsNone(preview["rows"][0]["measurement"])
        self.assertIn("target", preview["rows"][0])

    def test_preview_rejects_missing_dataset(self):
        with self.assertRaises(HTTPException) as raised:
            main.dataset_preview(5)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("No dataset is available", raised.exception.detail)

    def test_preview_respects_row_limit(self):
        self.upload(build_regression_dataset())

        small = main.dataset_preview(2)
        large = main.dataset_preview(999)

        self.assertEqual(small["preview_rows"], 2)
        self.assertEqual(large["preview_rows"], 50)


class ExperimentsEndpointTests(TrainingTestBase):
    def test_empty_history_is_reported_without_fabricated_data(self):
        response = main.get_experiments(limit=10)

        self.assertEqual(response["runs"], [])
        self.assertEqual(response["count"], 0)
        self.assertEqual(response["experiments"], 0)

    def test_completed_training_is_listed_with_real_run_information(self):
        self.upload(build_classification_dataset())
        main.run_pipeline("diagnosis", "classification")

        response = main.get_experiments(limit=25)

        self.assertEqual(response["count"], len(main.STATE["results"]) + 1)
        self.assertEqual(response["experiments"], 1)
        start_times = [run["start_time"] for run in response["runs"]]
        for index in range(len(start_times) - 1):
            self.assertGreaterEqual(start_times[index], start_times[index + 1])
        for run in response["runs"]:
            self.assertTrue(run["experiment"].startswith("autoflow_run_"))
            self.assertEqual(run["status"], "FINISHED")
            self.assertIsInstance(run["start_time"], int)
            self.assertIn(run["run_name"], {
                "Majority-Class Baseline",
                "Logistic Regression",
                "Decision Tree Classifier",
                "Random Forest Classifier",
                "Support Vector Machine",
                "K-Nearest Neighbors",
                "Naive Bayes",
                "Gradient Boosting Classifier",
            })
            self.assertEqual(run["primary_metric"], "F1")
            self.assertIsInstance(run["metrics"]["F1"], (int, float))
            self.assertTrue(run["dataset_fingerprint"])

        limited = main.get_experiments(limit=1)
        self.assertEqual(limited["count"], 1)

    def test_failed_runs_are_listed_with_status_and_error_details(self):
        self.upload(build_classification_dataset())

        def broken_candidates(task_type):
            return (
                {
                    "name": "Broken Model A",
                    "model": BrokenFittingModel(),
                    "role": "baseline",
                    "scale_numerical": False,
                },
                {
                    "name": "Broken Model B",
                    "model": BrokenFittingModel(),
                    "role": "linear",
                    "scale_numerical": True,
                },
            )

        with patch.object(main, "candidate_models", broken_candidates):
            with self.assertRaises(HTTPException):
                main.run_pipeline("diagnosis", "classification")

        response = main.get_experiments(limit=10)

        self.assertEqual(response["count"], 2)
        for run in response["runs"]:
            self.assertEqual(run["status"], "FAILED")
            self.assertIn("BOOM", run["error"])

    def test_limit_is_respected(self):
        self.upload(build_classification_dataset())
        main.run_pipeline("diagnosis", "classification")

        response = main.get_experiments(limit=1)

        self.assertEqual(response["count"], 1)


class ModelSchemaEndpointTests(TrainingTestBase):
    def test_schema_requires_a_trained_model(self):
        with self.assertRaises(HTTPException) as raised:
            main.model_schema()
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("No active model", raised.exception.detail)

    def test_schema_matches_the_persisted_training_contract(self):
        self.upload(build_classification_dataset())
        main.run_pipeline("diagnosis", "classification")

        schema = main.model_schema()

        self.assertTrue(schema["available"])
        self.assertEqual(schema["task_type"], "classification")
        self.assertEqual(schema["target"], "diagnosis")
        self.assertEqual(
            schema["features"],
            [
                {"name": "measurement", "kind": "numeric"},
                {"name": "category", "kind": "categorical"},
            ],
        )
        self.assertEqual(schema["model_version"], main.STATE["best_model"]["run_id"])
        self.assertIsInstance(schema["registry_version"], int)

        # Legacy-state fallback: without a persisted schema, the trained
        # pipeline's feature names are used with unspecified kinds.
        main.STATE["feature_schema"] = None
        legacy_schema = main.model_schema()
        legacy_names = [feature["name"] for feature in legacy_schema["features"]]
        self.assertEqual(legacy_names, ["measurement", "category"])
        self.assertTrue(
            all(feature["kind"] is None for feature in legacy_schema["features"])
        )


class DashboardJourneyTests(TrainingTestBase):
    """Upload -> inspect -> train -> review -> predict, against real handlers."""

    def test_complete_user_journey(self):
        # Step 1: upload a dataset with profiling feedback.
        contents = build_regression_dataset().to_csv(index=False).encode("utf-8")
        upload_response = asyncio.run(
            main.upload_dataset(
                UploadFile(filename="synthetic.csv", file=BytesIO(contents))
            )
        )
        self.assertEqual(upload_response.status_code, 200)
        profile = json.loads(upload_response.body)["profile"]
        self.assertEqual(profile["rows"], 60)
        self.assertEqual(profile["columns_count"], 4)

        # Step 2: inspect the dataset preview and column summaries.
        preview = main.dataset_preview(5)
        self.assertEqual(preview["shape"], {"rows": 60, "columns": 4})
        self.assertEqual(len(preview["rows"]), 5)
        self.assertEqual(preview["columns"][0]["name"], "measurement")

        # Step 3: train regression models.
        train_response = main.run_pipeline("target", "regression")
        self.assertEqual(train_response.status_code, 200)
        self.assertEqual(
            len(main.STATE["results"]),
            len(main.candidate_models("regression")),
        )
        self.assertEqual(main.STATE["primary_metric"], "RMSE")

        # Step 4: review real MLflow-backed experiments.
        experiments = main.get_experiments(limit=10)
        self.assertEqual(experiments["count"], len(main.STATE["results"]) + 1)
        self.assertTrue(
            all(run["status"] == "FINISHED" for run in experiments["runs"])
        )
        run_ids = {run["run_id"] for run in experiments["runs"]}
        self.assertNotIn(None, run_ids)

        # Step 5: predict through the real feature contract.
        schema = main.model_schema()
        feature_names = [feature["name"] for feature in schema["features"]]
        preview_columns = {column["name"] for column in preview["columns"]}
        self.assertEqual(set(feature_names), preview_columns - {"target"})

        payload = {feature["name"]: (0.5 if feature["kind"] == "numeric" else "warm")
                   for feature in schema["features"]}
        prediction = predict_aspect({"features": payload})
        self.assertEqual(prediction["status"], "success")
        self.assertIsInstance(prediction["prediction"], float)
        self.assertIn(prediction["model_version"], run_ids)

        # Invalid journey step: a missing required feature is rejected clearly.
        incomplete = dict(payload)
        del incomplete["measurement"]
        with self.assertRaises(HTTPException) as raised:
            predict_aspect({"features": incomplete})
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["missing_features"], ["measurement"])


if __name__ == "__main__":
    unittest.main()
