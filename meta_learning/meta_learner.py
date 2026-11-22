import os
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


META_MINIMUM_EXPERIMENTS = int(os.environ.get("AUTOFLOW_META_MINIMUM_EXPERIMENTS", "8"))
META_RELIABILITY_MINIMUM_DATASETS = int(
    os.environ.get("AUTOFLOW_META_RELIABILITY_MINIMUM_DATASETS", "10")
)
META_FEATURE_NAMES = [
    "rows",
    "columns",
    "numeric_features",
    "categorical_features",
    "missing_percentage",
    "duplicate_percentage",
    "mean_skewness",
    "mean_cardinality",
    "target_classes",
]


def distinct_experiments(
    experiments: list[dict[str, Any]],
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in experiments:
        fingerprint = item.get("dataset_fingerprint")
        if not fingerprint or not item.get("actual_best_algorithm"):
            continue
        if task_type is not None and item.get("task_type") != task_type:
            continue
        unique[str(fingerprint)] = item
    return list(unique.values())


def _meta_matrix(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                name: item.get("meta_features", {}).get(name, 0)
                for name in META_FEATURE_NAMES
            }
            for item in experiments
        ],
        columns=META_FEATURE_NAMES,
    )


def train_meta_model(
    experiments: list[dict[str, Any]],
    task_type: str,
    exclude_fingerprint: str | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    relevant = [
        item
        for item in distinct_experiments(experiments, task_type)
        if item.get("dataset_fingerprint") != exclude_fingerprint
    ]
    if len(relevant) < META_MINIMUM_EXPERIMENTS:
        return None, {
            "selection_method": "Cold-Start Rule-Based Prediction",
            "historical_experiments": len(relevant),
            "minimum_experiments": META_MINIMUM_EXPERIMENTS,
            "trained": False,
            "reliable": False,
        }

    X = _meta_matrix(relevant)
    y = pd.Series([item["actual_best_algorithm"] for item in relevant])
    if y.nunique() < 2:
        return None, {
            "selection_method": "Cold-Start Rule-Based Prediction",
            "historical_experiments": len(relevant),
            "minimum_experiments": META_MINIMUM_EXPERIMENTS,
            "trained": False,
            "reliable": False,
            "reason": "No sufficient algorithm diversity in historical experiments.",
        }
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, {
        "selection_method": "Meta-Learning Prediction",
        "historical_experiments": len(relevant),
        "minimum_experiments": META_MINIMUM_EXPERIMENTS,
        "trained": True,
        "reliable": False,
        "reliability_note": "Selection performance has not yet been validated across a held-out set of datasets.",
        "feature_names": META_FEATURE_NAMES,
    }


def predict_algorithm(
    meta_model: Any | None,
    meta_status: dict[str, Any],
    meta_features: dict[str, Any],
    task_type: str,
) -> tuple[str, float | None]:
    if meta_model is not None:
        feature_names = meta_status["feature_names"]
        row = pd.DataFrame(
            [[meta_features.get(name, 0) for name in feature_names]],
            columns=feature_names,
        )
        return str(meta_model.predict(row)[0]), None

    if task_type == "classification":
        if meta_features["categorical_features"] > meta_features["numeric_features"]:
            return "Random Forest Classifier", None
        if meta_features["rows"] >= 500:
            return "Logistic Regression", None
        return "Decision Tree Classifier", None
    if meta_features["rows"] >= 1000:
        return "Random Forest Regressor", None
    return "Gradient Boosting Regressor", None


def meta_model_explanation(
    model: Any | None,
    meta_status: dict[str, Any],
    meta_features: dict[str, Any],
) -> dict[str, Any]:
    if model is None:
        return {
            "feature_importance": [],
            "probabilities": [],
            "probability_type": None,
        }
    feature_names = meta_status.get("feature_names", META_FEATURE_NAMES)
    row = pd.DataFrame(
        [[meta_features.get(name, 0) for name in feature_names]],
        columns=feature_names,
    )
    probabilities = []
    if hasattr(model, "predict_proba"):
        values = model.predict_proba(row)[0]
        probabilities = [
            {"algorithm": str(label), "probability": float(value)}
            for label, value in zip(model.classes_, values)
        ]
    importances = getattr(model, "feature_importances_", [])
    feature_importance = [
        {"feature": name, "importance": float(value)}
        for name, value in zip(feature_names, importances)
    ]
    feature_importance.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "feature_importance": feature_importance,
        "probabilities": probabilities,
        "probability_type": "raw_uncalibrated_classifier_probability"
        if probabilities
        else None,
    }


def evaluate_meta_reliability(
    experiments: list[dict[str, Any]],
    task_type: str,
) -> dict[str, Any]:
    records = distinct_experiments(experiments, task_type)
    count = len(records)
    if count < META_RELIABILITY_MINIMUM_DATASETS:
        return {
            "available": False,
            "message": "Insufficient historical datasets to evaluate meta-learning reliability.",
            "evaluation_datasets": count,
            "minimum_datasets": META_RELIABILITY_MINIMUM_DATASETS,
            "split_method": "leave-one-dataset-out",
        }

    X = _meta_matrix(records)
    y = pd.Series([item["actual_best_algorithm"] for item in records]).astype(str)
    predictions: list[str] = []
    for held_out in range(count):
        train_indices = [index for index in range(count) if index != held_out]
        training_labels = y.iloc[train_indices]
        if training_labels.nunique() < 2:
            predictions.append(str(training_labels.iloc[0]))
            continue
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X.iloc[train_indices], training_labels)
        predictions.append(str(model.predict(X.iloc[[held_out]])[0]))

    labels = sorted(set(y), key=str)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y,
        predictions,
        labels=labels,
        zero_division=0,
    )
    return {
        "available": True,
        "evaluation_datasets": count,
        "minimum_datasets": META_RELIABILITY_MINIMUM_DATASETS,
        "split_method": "leave-one-dataset-out",
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
        "per_algorithm": [
            {
                "algorithm": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "datasets": int(support[index]),
            }
            for index, label in enumerate(labels)
        ],
    }
