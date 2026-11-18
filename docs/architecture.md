# AutoFlow Architecture

AutoFlow is a FastAPI and scikit-learn application with MLflow-backed persistence.

## End-to-end workflow

1. CSV upload and validation
2. Dataset profiling and problem-type detection
3. Statistical meta-feature extraction
4. Task 1: benchmark all configured algorithms
5. Task 2: select exactly one algorithm from cold-start rules or a trained meta-model
6. Independent training and evaluation of Task 1 and Task 2
7. Comparison, persistence, and prediction

Task 1 remains the measured reference. Task 2 is reported separately and never replaces the benchmark winner.

## Persistence

- Complete preprocessing-and-estimator pipelines are logged to MLflow.
- Task 1 and Task 2 run references are retained in application state.
- Uploaded data is isolated in `uploads/`.
- Historical valid Task 1 experiments are stored in `meta_data/experiments.json`.
