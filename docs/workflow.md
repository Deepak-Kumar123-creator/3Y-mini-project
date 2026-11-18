# AutoFlow Workflow

## Task 1

The benchmark engine uses one held-out split, stratified for classification, and trains the configured baseline and candidate algorithms with leakage-safe preprocessing pipelines. Classification uses weighted F1 as the default selection metric; regression uses RMSE.

## Task 2

Task 2 extracts dataset-level meta-features and reads historical Task 1 experiments. It uses a trained Random Forest meta-model only after at least eight experiments for the relevant problem type and with at least two historical algorithm classes. Otherwise it transparently uses cold-start rule-based fallback logic and trains exactly one selected algorithm.

Meta-learning reliability should grow only with diverse, real historical datasets.
