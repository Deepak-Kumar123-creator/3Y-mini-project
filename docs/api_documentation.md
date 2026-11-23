# AutoFlow API

The FastAPI application provides interactive Swagger docs at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `POST` | `/upload` | Validate and profile one CSV dataset |
| `GET` | `/api/dataset/preview?rows=10` | Preview rows and column statistics |
| `GET` | `/api/problem-detection?target=column` | Heuristic problem type suggestion |
| `POST` | `/train` | Run Task 1 benchmark and Task 2 single-model training |
| `GET` | `/api/experiments?limit=25` | List stored MLflow runs |
| `GET` | `/api/meta-learning` | Return latest Task 2 result/status |
| `GET` | `/api/comparison` | Compare the benchmark and prediction paths |
| `GET` | `/api/models` | List available Task 1/Task 2 artifacts |
| `GET` | `/api/models/{run_id}/download` | Download a trusted complete pipeline |
| `GET` | `/api/model/schema` | Get active Task 1 feature contract |
| `POST` | `/predict` | Predict from Task 1 or a selected saved model |

Example selected-model request:

```json
{
  "model_run_id": "32-character-MLflow-run-id",
  "features": {
    "measurement": 3.2,
    "category": "warm"
  }
}
```

Batch requests use `{"rows": [{...}, {...}]}` instead of `features`. The model
selector is optional; without it, inference uses the Task 1 benchmark winner.
Incoming features are validated against the selected run's persisted inference
schema, and the complete saved pipeline applies the training-time preprocessing.

Run IDs are accepted only when they identify a finished run in an AutoFlow-owned
MLflow experiment. User-supplied filesystem paths and model files are never
loaded by the API.
