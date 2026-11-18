from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


META_MINIMUM_EXPERIMENTS = 8


def train_meta_model(
    experiments: list[dict[str, Any]],
    task_type: str,
    exclude_fingerprint: str | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    relevant = [
        item for item in experiments
        if item.get("task_type") == task_type
        and item.get("actual_best_algorithm")
        and item.get("dataset_fingerprint") != exclude_fingerprint
    ]
    if len(relevant) < META_MINIMUM_EXPERIMENTS:
        return None, {
            "selection_method": "Cold-Start Rule-Based Prediction",
            "historical_experiments": len(relevant),
            "minimum_experiments": META_MINIMUM_EXPERIMENTS,
            "trained": False,
            "reliable": False,
        }

    feature_names = [
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
    X = pd.DataFrame([{name: item["meta_features"].get(name, 0) for name in feature_names} for item in relevant])
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
        "feature_names": feature_names,
    }


def predict_algorithm(
    meta_model: Any | None,
    meta_status: dict[str, Any],
    meta_features: dict[str, Any],
    task_type: str,
) -> tuple[str, float | None]:
    if meta_model is not None:
        feature_names = meta_status["feature_names"]
        row = pd.DataFrame([[meta_features.get(name, 0) for name in feature_names]], columns=feature_names)
        prediction = str(meta_model.predict(row)[0])
        return prediction, None

    if task_type == "classification":
        if meta_features["categorical_features"] > meta_features["numeric_features"]:
            return "Random Forest Classifier", None
        if meta_features["rows"] >= 500:
            return "Logistic Regression", None
        return "Decision Tree Classifier", None
    if meta_features["rows"] >= 1000:
        return "Random Forest Regressor", None
    return "Gradient Boosting Regressor", None
