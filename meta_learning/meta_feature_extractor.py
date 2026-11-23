import math
from typing import Any
import pandas as pd


def extract_meta_features(
    dataframe: pd.DataFrame,
    target: str,
    task_type: str,
) -> dict[str, Any]:
    features = dataframe.drop(columns=[target])
    numeric = features.select_dtypes(include=["number"]).replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )
    target_values = dataframe[target]
    missing_pct = float(features.isna().sum().sum() / max(features.size, 1) * 100)
    duplicate_pct = float(dataframe.duplicated().sum() / max(len(dataframe), 1) * 100)
    skewness = float(numeric.skew().mean()) if not numeric.empty else 0.0
    if not math.isfinite(skewness):
        skewness = 0.0
    if task_type == "classification":
        class_counts = target_values.value_counts(dropna=True)
        target_statistics: dict[str, Any] = {
            "target_distribution": {
                str(label): int(count) for label, count in class_counts.items()
            },
            "class_imbalance_ratio": (
                float(class_counts.max() / class_counts.min())
                if not class_counts.empty and class_counts.min() > 0
                else 0.0
            ),
            "target_mean": None,
            "target_median": None,
            "target_std": None,
            "target_skewness": None,
        }
    else:
        target_numeric = pd.to_numeric(target_values, errors="coerce")
        target_skewness = float(target_numeric.skew()) if target_numeric.notna().any() else 0.0
        if not math.isfinite(target_skewness):
            target_skewness = 0.0
        target_statistics = {
            "target_distribution": {},
            "class_imbalance_ratio": None,
            "target_mean": float(target_numeric.mean()),
            "target_median": float(target_numeric.median()),
            "target_std": float(target_numeric.std()) if target_numeric.notna().sum() > 1 else 0.0,
            "target_skewness": target_skewness,
        }
    return {
        "task_type": task_type,
        "rows": int(len(dataframe)),
        "columns": int(len(features.columns)),
        "numeric_features": int(len(numeric.columns)),
        "categorical_features": int(len(features.columns) - len(numeric.columns)),
        "missing_percentage": missing_pct,
        "duplicate_percentage": duplicate_pct,
        "mean_skewness": skewness,
        "mean_cardinality": float(features.nunique(dropna=True).mean())
        if len(features.columns)
        else 0.0,
        "target_classes": int(target_values.nunique(dropna=True))
        if task_type == "classification"
        else 0,
        **target_statistics,
    }
