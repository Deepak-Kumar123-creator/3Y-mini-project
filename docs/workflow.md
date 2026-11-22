# AutoFlow Workflow

## Task 1

The benchmark engine uses one held-out split, stratified for classification, and trains the configured baseline and candidate algorithms with leakage-safe preprocessing pipelines. Classification uses weighted F1 as the default selection metric; regression uses RMSE.

## Task 2

Task 2 extracts dataset-level meta-features and reads historical Task 1 experiments. It uses a trained Random Forest meta-model only after at least eight experiments for the relevant problem type and with at least two historical algorithm classes. Otherwise it transparently uses cold-start rule-based fallback logic and trains exactly one selected algorithm.

Meta-learning reliability should grow only with diverse, real historical datasets.

## Dashboard analytics

The dashboard's 19 native-SVG visualizations cover dataset target and feature
distributions, missingness and numeric correlations; Task 1 candidate metrics,
fit times, cross-validation, and held-out predictions; Task 2 model explanation,
winner frequency, and dataset-level reliability; and a Task 1 versus Task 2
comparison. Evaluation predictions are real held-out observations, bounded to
500 rows per run. ROC data comes from model probabilities or decision scores.
Meta-learning probabilities are raw and uncalibrated; reliability uses
leave-one-dataset-out evaluation and remains unavailable until at least 10
distinct dataset fingerprints are present.

`GET /api/visualizations` accepts optional `target`, `task_type`, `feature`,
`bins`, and `model_run_id` query parameters. Task 1 total computation time sums
candidate fit, validation, cross-validation, and evaluation-chart calculations;
Task 2 time includes meta-selection and one model's fit and evaluation. Both
exclude MLflow logging and artifact persistence; per-model fit time is reported
separately.
