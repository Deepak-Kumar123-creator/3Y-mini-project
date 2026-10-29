import os
import shutil
import pandas as pd
import numpy as np
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score
import mlflow

app = FastAPI(title="AutoFlow AI - Autonomous ML & MLOps Engine")

# Setup local storage
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Global tracking state for the active UI context
STATE: Dict[str, Any] = {
    "dataset_path": None,
    "filename": None,
    "rows": 0,
    "columns_count": 0,
    "columns": [],
    "target": None,
    "task_type": None,
    "run_group": None,
    "results": [],
    "best_model": None
}

# In-memory deployment tracking variable for immediate prediction availability
inference_pipeline = None

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="dashboard.html",
    context={"state": STATE}
)

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV datasets are currently supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")
        
    columns_summary = []
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = float((missing_count / len(df)) * 100) if len(df) > 0 else 0.0
        col_type = "numeric" if np.issubdtype(df[col].dtype, np.number) else "categorical"
        columns_summary.append({
            "name": col,
            "type": col_type,
            "missing": f"{missing_pct:.1f}% ({missing_count})"
        })
        
    STATE.update({
        "dataset_path": file_path,
        "filename": file.filename,
        "rows": len(df),
        "columns_count": len(df.columns),
        "columns": columns_summary,
        "target": None,
        "task_type": None,
        "run_group": None,
        "results": [],
        "best_model": None
    })
    
    return JSONResponse(content={"status": "success", "redirect": "/"})

@app.post("/train")
async def run_pipeline(target: str = Form(...), task_type: str = Form(...)):
    global inference_pipeline
    
    if not STATE["dataset_path"] or not os.path.exists(STATE["dataset_path"]):
        raise HTTPException(status_code=400, detail="No active dataset available. Please upload a file first.")
        
    df = pd.read_csv(STATE["dataset_path"])
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in the dataset.")
        
    STATE["target"] = target
    STATE["task_type"] = task_type
    STATE["run_group"] = f"{STATE['filename'].replace('.csv', '')}_run"
    
    X = df.drop(columns=[target])
    y = df[target]
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    if task_type == "classification":
        candidates = [
            ("Logistic Regression", LogisticRegression(max_iter=1000)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ]
    else:
        candidates = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ]
        
    mlflow.set_experiment(STATE["run_group"])
    results = []
    
    for name, model in candidates:
        with mlflow.start_run(run_name=name):
            clf = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            
            metrics = {}
            if task_type == "classification":
                metrics["Accuracy"] = float(accuracy_score(y_val, preds))
                metrics["Precision"] = float(precision_score(y_val, preds, average='weighted', zero_division=0))
                metrics["Recall"] = float(recall_score(y_val, preds, average='weighted', zero_division=0))
                metrics["F1"] = float(f1_score(y_val, preds, average='weighted', zero_division=0))
                primary_metric = "F1"
            else:
                metrics["MAE"] = float(mean_absolute_error(y_val, preds))
                metrics["RMSE"] = float(root_mean_squared_error(y_val, preds))
                metrics["R2"] = float(r2_score(y_val, preds))
                primary_metric = "RMSE"
                
            mlflow.log_params({"task_type": task_type, "model_type": name})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(clf, "model")
            
            run_info = mlflow.active_run().info
            results.append({
                "model": name,
                "metrics": metrics,
                "primary": metrics[primary_metric],
                "run_id": run_info.run_id
            })
            
    if not results:
        raise HTTPException(status_code=500, detail="Model execution completed but no valid results were captured.")
        
    if task_type == "classification":
        results.sort(key=lambda x: x["primary"], reverse=True)
    else:
        results.sort(key=lambda x: x["primary"], reverse=False)
        
    STATE["results"] = results
    STATE["best_model"] = results[0]
    
    inference_pipeline = mlflow.sklearn.load_model(f"runs:/{STATE['best_model']['run_id']}/model")
    return JSONResponse(content={"status": "success", "redirect": "/"})

@app.post("/predict")
async def predict(request: Request):
    global inference_pipeline
    if inference_pipeline is None:
        raise HTTPException(status_code=400, detail="No active model deployed. Please trigger an optimization pipeline run first.")
        
    payload = await request.json()
    try:
        features_input = payload.get("features", {})
        input_df = pd.DataFrame([features_input])
        
        # Verify alignment with expected structural input schema fields
        prediction = inference_pipeline.predict(input_df)
        
        # Handle array return types safely across metrics targets
        pred_value = prediction.tolist()[0]
        
        return {
            "prediction": pred_value,
            "model_version": STATE["best_model"]["run_id"] if STATE["best_model"] else "unknown",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference pipeline execution failed: {str(e)}")
