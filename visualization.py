import math
from typing import Any

import numpy as np
import pandas as pd

from meta_learning.meta_learner import evaluate_meta_reliability


MAX_CORRELATION_FEATURES = 20
MAX_SCATTER_POINTS = 500


def _finite_values(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.dropna().to_numpy(dtype=float)


def _histogram(values: np.ndarray, bins: int) -> dict[str, Any]:
    if not values.size:
        return {"bins": [], "count": 0}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bins": [
            {
                "start": float(edges[index]),
                "end": float(edges[index + 1]),
                "count": int(counts[index]),
            }
            for index in range(len(counts))
        ],
        "count": int(values.size),
    }


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _dataset_charts(
    dataframe: pd.DataFrame,
    target: str | None,
    task_type: str | None,
    selected_feature: str | None,
    bins: int,
) -> dict[str, Any]:
    target_distribution: dict[str, Any] = {"available": False}
    if target and target in dataframe.columns:
        target_values = dataframe[target].dropna()
        if task_type == "classification":
            counts = target_values.value_counts(dropna=False)
            total = int(counts.sum())
            target_distribution = {
                "available": True,
                "type": "classification",
                "classes": [
                    {
                        "label": str(label),
                        "count": int(count),
                        "percentage": float(count / total * 100) if total else 0.0,
                    }
                    for label, count in counts.items()
                ],
                "imbalance_ratio": (
                    float(counts.max() / counts.min()) if len(counts) and counts.min() else None
                ),
            }
        elif task_type == "regression":
            values = _finite_values(target_values)
            target_distribution = {
                "available": bool(values.size),
                "type": "regression",
                **_histogram(values, bins),
                "mean": _json_number(np.mean(values)) if values.size else None,
                "median": _json_number(np.median(values)) if values.size else None,
            }

    missing = []
    for column in dataframe.columns:
        count = int(dataframe[column].isna().sum())
        missing.append(
            {
                "column": str(column),
                "count": count,
                "percentage": float(count / max(len(dataframe), 1) * 100),
            }
        )
    missing.sort(key=lambda item: (-item["count"], item["column"]))

    numeric_columns = [
        str(column)
        for column in dataframe.select_dtypes(include=["number"]).columns
        if str(column) != target
    ]
    feature_distribution: dict[str, Any] = {"available": False}
    if selected_feature in numeric_columns:
        values = _finite_values(dataframe[selected_feature])
        if values.size:
            quartiles = np.percentile(values, [25, 50, 75])
            iqr = float(quartiles[2] - quartiles[0])
            lower_fence = float(quartiles[0] - 1.5 * iqr)
            upper_fence = float(quartiles[2] + 1.5 * iqr)
            outliers = values[(values < lower_fence) | (values > upper_fence)]
            feature_distribution = {
                "available": True,
                "feature": selected_feature,
                **_histogram(values, bins),
                "statistics": {
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(quartiles[1]),
                    "skewness": _json_number(pd.Series(values).skew()),
                    "q1": float(quartiles[0]),
                    "q3": float(quartiles[2]),
                    "lower_fence": lower_fence,
                    "upper_fence": upper_fence,
                    "outlier_count": int(outliers.size),
                },
                "boxplot": {
                    "minimum": float(np.min(values[values >= lower_fence]))
                    if np.any(values >= lower_fence)
                    else float(np.min(values)),
                    "q1": float(quartiles[0]),
                    "median": float(quartiles[1]),
                    "q3": float(quartiles[2]),
                    "maximum": float(np.max(values[values <= upper_fence]))
                    if np.any(values <= upper_fence)
                    else float(np.max(values)),
                    "outliers": [_json_number(value) for value in outliers[:100]],
                },
            }

    correlation_names = numeric_columns
    truncated = len(correlation_names) > MAX_CORRELATION_FEATURES
    if truncated:
        variances = dataframe[correlation_names].replace([np.inf, -np.inf], np.nan).var()
        correlation_names = variances.nlargest(MAX_CORRELATION_FEATURES).index.tolist()
    correlation_frame = (
        dataframe[correlation_names]
        .replace([np.inf, -np.inf], np.nan)
        .corr()
        if correlation_names
        else pd.DataFrame()
    )
    correlation = {
        "features": [str(name) for name in correlation_frame.columns],
        "matrix": [
            [_json_number(value) for value in row]
            for row in correlation_frame.to_numpy()
        ],
        "truncated": truncated,
        "maximum_features": MAX_CORRELATION_FEATURES,
        "note": "The target is excluded; values use Pearson correlation."
        if correlation_names
        else "No numeric feature columns are available for correlation analysis.",
    }
    return {
        "target_distribution": target_distribution,
        "missing_values": missing,
        "missing_value_total": int(sum(item["count"] for item in missing)),
        "feature_names": numeric_columns,
        "selected_feature": selected_feature,
        "feature_distribution": feature_distribution,
        "correlation": correlation,
    }


def _model_charts(
    state: dict[str, Any],
    task_type: str | None,
    target: str | None,
    selected_run: str | None,
) -> dict[str, Any]:
    matches_training = (
        task_type == state.get("task_type")
        and target == state.get("target")
    )
    if not matches_training:
        return {
            "available": False,
            "message": "Train this target and problem type to generate matching model visualizations.",
            "metrics": {},
            "metric_names": [],
            "primary_metric": None,
            "training_times": [],
            "cross_validation": [],
            "selected_model": {},
            "task_1_evaluation": {},
            "task_2_evaluation": {},
            "successful_algorithm_count": 0,
            "failed_algorithms": [],
            "comparison": {},
            "task_2": {},
        }
    results = state.get("results") if isinstance(state.get("results"), list) else []
    task_1_successful = [
        item for item in results
        if item.get("status") == "success"
    ]
    successful = list(task_1_successful)
    task_2 = state.get("task_2") if isinstance(state.get("task_2"), dict) else {}
    if task_2.get("status") == "success":
        successful.append(
            {
                **task_2,
                "model": task_2.get("predicted_algorithm"),
                "role": "task_2_meta_selection",
                "evaluation_data": task_2.get("evaluation_data"),
                "confusion_matrix": task_2.get("confusion_matrix"),
            }
        )
    selected = next((item for item in successful if item.get("run_id") == selected_run), None)
    if selected is None:
        selected = state.get("best_model") if isinstance(state.get("best_model"), dict) else None
    if selected is None:
        selected = successful[0] if successful else {}

    metrics = {}
    primary_metric = state.get("primary_metric")
    metric_names = (
        ["Accuracy", "Precision", "Recall", "F1", "Macro F1"]
        if task_type == "classification"
        else ["MAE", "RMSE", "R2"]
    )
    for name in metric_names:
        metrics[name] = [
            {
                "algorithm": item.get("model", "Unknown"),
                "value": _json_number((item.get("metrics") or {}).get(name)),
                "primary": name == primary_metric,
            }
            for item in task_1_successful
        ]
    training_times = [
        {
            "algorithm": item.get("model", "Unknown"),
            "seconds": _json_number(item.get("training_time_seconds")),
        }
        for item in task_1_successful
    ]
    cv_scores = [
        {
            "algorithm": item.get("model", "Unknown"),
            "mean": _json_number((item.get("cross_validation") or {}).get("mean")),
            "std": _json_number((item.get("cross_validation") or {}).get("std")),
            "metric": (item.get("cross_validation") or {}).get("metric"),
        }
        for item in task_1_successful
        if isinstance(item.get("cross_validation"), dict)
        and item["cross_validation"].get("mean") is not None
    ]
    evaluation_data = selected.get("evaluation_data") or {}
    comparison = state.get("comparison") if isinstance(state.get("comparison"), dict) else {}
    task_2_eval = task_2.get("evaluation_data") or {}
    task_1_eval = (state.get("best_model") or {}).get("evaluation_data") or {}
    return {
        "available": bool(successful),
        "available_models": [
            {
                "run_id": item.get("run_id"),
                "algorithm": item.get("model"),
                "role": item.get("role"),
            }
            for item in successful
            if item.get("run_id")
        ],
        "metrics": metrics,
        "metric_names": metric_names,
        "primary_metric": primary_metric,
        "training_times": training_times,
        "cross_validation": cv_scores,
        "selected_model": {
            "run_id": selected.get("run_id"),
            "algorithm": selected.get("model"),
            "confusion_matrix": selected.get("confusion_matrix"),
            "evaluation_data": evaluation_data,
        },
        "task_1_evaluation": task_1_eval,
        "task_2_evaluation": task_2_eval,
        "successful_algorithm_count": len(task_1_successful),
        "failed_algorithms": [
            {"algorithm": item.get("model"), "error": item.get("error")}
            for item in results
            if item.get("status") == "failed"
        ],
        "comparison": comparison,
        "task_2": task_2,
    }


def build_visualization_payload(
    dataframe: pd.DataFrame,
    state: dict[str, Any],
    historical_experiments: list[dict[str, Any]],
    feature: str | None = None,
    bins: int = 20,
    selected_run: str | None = None,
    target: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    task_type = task_type or state.get("task_type")
    target = target or state.get("target")
    numeric_features = [
        str(column)
        for column in dataframe.select_dtypes(include=["number"]).columns
        if str(column) != target
    ]
    selected_feature = feature if feature in numeric_features else (
        numeric_features[0] if numeric_features else None
    )
    dataset_charts = _dataset_charts(
        dataframe,
        target,
        task_type,
        selected_feature,
        min(max(int(bins), 5), 60),
    )
    model_charts = _model_charts(state, task_type, target, selected_run)

    historical_for_task = [
        item
        for item in historical_experiments
        if item.get("task_type") == task_type
        and item.get("actual_best_algorithm")
        and item.get("dataset_fingerprint")
    ]
    unique_history: dict[str, dict[str, Any]] = {}
    for item in historical_for_task:
        unique_history[str(item["dataset_fingerprint"])] = item
    winner_counts: dict[str, int] = {}
    for item in unique_history.values():
        algorithm = str(item["actual_best_algorithm"])
        winner_counts[algorithm] = winner_counts.get(algorithm, 0) + 1
    task_2 = model_charts.get("task_2", {})
    meta_status = task_2.get("meta_status") if isinstance(task_2.get("meta_status"), dict) else {}
    return {
        "task_type": task_type,
        "dataset_fingerprint": state.get("dataset_fingerprint"),
        "dataset": dataset_charts,
        "models": model_charts,
        "meta_learning": {
            "selection_method": task_2.get("selection_method"),
            "trained": bool(meta_status.get("trained")),
            "feature_importance": task_2.get("feature_importance", []),
            "probabilities": task_2.get("probabilities", []),
            "probability_type": task_2.get("probability_type"),
            "historical_winner_frequency": [
                {"algorithm": name, "datasets": count}
                for name, count in sorted(
                    winner_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "reliability": evaluate_meta_reliability(
                list(unique_history.values()),
                task_type,
            )
            if task_type in {"classification", "regression"}
            else {
                "available": False,
                "message": "Run a classification or regression experiment to evaluate reliability.",
                "evaluation_datasets": 0,
            },
        },
    }
